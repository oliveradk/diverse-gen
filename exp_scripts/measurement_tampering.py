import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import sys 
from collections import defaultdict
from typing import Optional, Literal
import copy
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from hydra import compose, initialize
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score


from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import get_scheduler

from diverse_gen.losses.divdis import DivDisLoss 
from diverse_gen.losses.ace import ACELoss, MixRateScheduler
from diverse_gen.losses.pass_through import PassThroughLoss
from diverse_gen.losses.src import SrcLoss
from diverse_gen.losses.loss_types import LossType

from diverse_gen.models.backbone import MultiHeadBackbone
from diverse_gen.utils.utils import batch_size, to_device 
from diverse_gen.utils.logger import Logger
from diverse_gen.utils.exp_utils import get_current_commit_hash


@dataclass 
class Config: 
    seed: int = 42
    # loss
    loss_type: LossType = LossType.TOPK
    one_sided_ace: bool = True
    ace_agree: bool = False
    pseudo_label_all_groups: bool = False
    source_weight: float = 1.0
    aux_weight: float = 1.0
    mix_rate_lower_bound: float = 0.1
    mix_rate_schedule: Optional[str] = "linear"
    mix_rate_t0: Optional[int] = 0
    mix_rate_t1: Optional[int] = 1
    # model
    model: str = "codegen-350M-mono-measurement_pred-diamonds-seed0"#"pythia-1_4b-deduped-measurement_pred-generated_stories"
    binary: bool = True
    heads: int = 2
    train: bool = True
    freeze_model: bool = False
    load_prior_probe: bool = False
    # data
    dataset: str = "diamonds-seed0" #"generated_stories"
    max_length: int = 1024
    feature_dim: int = 1024
    dataset_length: Optional[int] = None
    split_source_target: bool = True
    target_only_disagree: bool = False
    val_frac: float = 0.2
    source_labels: Optional[list[str| None]] = None # field(default_factory=lambda: ["sensors_agree"])
    target_labels: Optional[list[str| None]] = None # field(default_factory=lambda: ["sensors_agree"])
    # training
    lr: float = 2e-5 
    weight_decay: float = 2e-2
    epochs: int = 5
    scheduler: str = "cosine"
    frac_warmup_steps: float = 0.10
    num_epochs: int = 5
    effective_batch_size: int = 32
    forward_batch_size: int = 32
    micro_batch_size: int = 4
    # misc
    bootstrap_eval: bool = True
    n_bootstrap_samples: int = 100
    num_workers: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir: str = f"output/mtd/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


# load config
OmegaConf.register_new_resolver("div", lambda x, y: x // y)
conf = Config()
file_overrides = {}
print(sys.argv)
if sys.argv[1] == "--config_file": 
    config_path = sys.argv[2]
    with initialize(config_path=f"../configs/mtd", version_base=None):
        file_overrides = compose(config_name=config_path)
    cmd_line_args = sys.argv[3:] if len(sys.argv) > 3 else []
else: 
    cmd_line_args = sys.argv[1:]
cmd_line_overrides = OmegaConf.from_cli(cmd_line_args)
conf_dict = OmegaConf.merge(OmegaConf.structured(conf), file_overrides, cmd_line_overrides)
OmegaConf.resolve(conf_dict)
conf = Config(**conf_dict)
exp_dir = conf.exp_dir
os.makedirs(exp_dir, exist_ok=True)
print(f"Writing output to: {exp_dir}")
# save full config to exp_dir
with open(f"{exp_dir}/config.yaml", "w") as f:
    OmegaConf.save(config=conf, f=f)
# save commit hash
with open(f"{exp_dir}/commit_hash.txt", "w") as f:
    f.write(get_current_commit_hash())

# Model
model_path = f"oliverdk/{conf.model}"

config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True
)
pretrained_model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    config=config,
    trust_remote_code=True, 
    device_map=conf.device
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True, 
    padding_side="left", 
    truncation_side="left"
)
# set pad token and init sensor loc finder
pretrained_model.set_pad_token(tokenizer)
pretrained_model.init_sensor_loc_finder(tokenizer)


class MeasurementPredBackbone(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
    
    def forward(self, x):
        out = self.pretrained_model.base_model(x['input_ids'], attention_mask=x['attention_mask'])
        sensor_locs = self.pretrained_model.find_sensor_locs(x['input_ids'])
        sensor_embs = out.last_hidden_state.gather(
            1, sensor_locs.unsqueeze(-1).expand(-1, -1, out.last_hidden_state.size(-1))
        )
        assert sensor_embs.shape == (x['input_ids'].size(0), 4, out.last_hidden_state.size(-1))
        aggregate_sensor_embs = sensor_embs[:, -1, :].squeeze(1)
        assert aggregate_sensor_embs.shape == (x['input_ids'].size(0), out.last_hidden_state.size(-1))
        return aggregate_sensor_embs


# Dataset
dataset = load_dataset(f"redwoodresearch/{conf.dataset}")

class MeasurementDataset(Dataset):
    def __init__(self, dataset, max_length=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.measurements = torch.tensor(self.dataset['measurements'])
        self.ground_truth = torch.tensor(self.dataset['is_correct'])
        self.is_trusted = torch.tensor(self.dataset['is_clean'])

        self.encodings = tokenizer(
            self.dataset['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        encoding = {
            "input_ids": self.encodings['input_ids'][idx],
            "attention_mask": self.encodings['attention_mask'][idx], 
        }

        # labels
        gt = self.ground_truth[idx]
        all_sensors = torch.all(self.measurements[idx])
        sensors_agree = torch.all(self.measurements[idx] == self.measurements[idx][0]) 
        group_labels = torch.stack((gt, all_sensors, sensors_agree))

        # set labels to floats 
        label = gt.to(torch.float32)
        group_labels = group_labels.to(torch.float32)
        
        return encoding, label, group_labels

# truncate dataset
if conf.dataset_length is not None:
    for k, subset in dataset.items():
        # select random inidices 
        dataset[k] = subset.select(indices=np.random.choice(len(subset), min(conf.dataset_length, len(subset)), replace=False))

def all_same(ls):
    return all(x == ls[0] for x in ls)

if conf.target_only_disagree and not conf.split_source_target:
    # only clean examples or examples where sensors disagree filter out examples where not clean and sensors agree 
    dataset["train"] = dataset["train"].filter(lambda x: x["is_clean"] or not all_same(x['measurements']))

if conf.split_source_target: # standard split for diverse gen methods
    source_data = dataset["train"].filter(lambda x: x["is_clean"])
    splits = source_data.train_test_split(train_size=1-conf.val_frac, test_size=conf.val_frac, seed=conf.seed)
    dataset["source_train"] = splits['train']
    dataset["source_val"] = splits['test']

    # target (is not clean)
    target_data = dataset["train"].filter(lambda x: not x["is_clean"])
    if "train_for_val" in dataset:
        dataset["target_train"] = target_data
        dataset["target_val"] = dataset["train_for_val"]
    else:
        target_splits = target_data.train_test_split(train_size=1-conf.val_frac, test_size=conf.val_frac, seed=conf.seed)
        dataset["target_train"] = target_splits['train']
        dataset["target_val"] = target_splits['test']
else: 
    # TODO: should use train for val if present
    # combine source and target (trusted and untrusted) 
    # uses source labels, but None defaults to all sensors
    splits = dataset["train"].train_test_split(train_size=1-conf.val_frac, test_size=conf.val_frac, seed=conf.seed)
    dataset["source_train"] = splits['train']
    dataset["source_val"] = splits['test']

# test (validation)
dataset["test"] = dataset["validation"]
# only untrusted positive examples 
dataset["test"] = dataset["test"].filter(lambda x: not x['is_clean'] and all(x["measurements"]))

# remove train and train_for_val
dataset.pop("train")
if "train_for_val" in dataset:
    dataset.pop("train_for_val")
dataset.pop("validation")


source_train_ds = MeasurementDataset(dataset["source_train"], conf.max_length)
source_val_ds = MeasurementDataset(dataset["source_val"], conf.max_length)
if conf.split_source_target:
    target_train_ds = MeasurementDataset(dataset["target_train"], conf.max_length)
    target_val_ds = MeasurementDataset(dataset["target_val"], conf.max_length)
test_ds = MeasurementDataset(dataset["test"], conf.max_length)


# Train
pred_model = MeasurementPredBackbone(pretrained_model).to(conf.device)
net = MultiHeadBackbone(pred_model, classes=[1 for _ in range(conf.heads)], feature_dim=conf.feature_dim).to(conf.device)

if conf.freeze_model:
    for param in net.backbone.parameters():
        param.requires_grad = False

# load weights of pretrained model aggregate probe to second net head
if conf.load_prior_probe: # last head 
    net.heads.weight.data[-1, :] = pretrained_model.aggregate_probe.weight.data[0]
    net.heads.bias.data[-1] = pretrained_model.aggregate_probe.bias.data[0]

# dataloaders
source_train_loader = DataLoader(source_train_ds, batch_size=conf.micro_batch_size, num_workers=conf.num_workers)
source_val_loader = DataLoader(source_val_ds, batch_size=conf.micro_batch_size, num_workers=conf.num_workers)
if conf.split_source_target:
    target_train_loader = DataLoader(target_train_ds, batch_size=conf.effective_batch_size, num_workers=conf.num_workers)
    target_val_loader = DataLoader(target_val_ds, batch_size=conf.effective_batch_size, num_workers=conf.num_workers)
target_test_loader = DataLoader(test_ds, batch_size=conf.forward_batch_size, num_workers=conf.num_workers)

# optimizer and scheduler
opt = torch.optim.AdamW(net.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
num_training_steps = conf.num_epochs * len(source_train_loader) // (conf.effective_batch_size // conf.micro_batch_size)
scheduler = get_scheduler(
    name=conf.scheduler,
    optimizer=opt,
    num_warmup_steps=round(conf.frac_warmup_steps * num_training_steps),
    num_training_steps=num_training_steps
)

if conf.binary:
    classes_per_head = [1 for _ in range(conf.heads)]
else:
    classes_per_head = [2 for _ in range(conf.heads)]

# loss
if conf.loss_type == LossType.DIVDIS:
    loss_fn = DivDisLoss(heads=2)
elif conf.loss_type == LossType.ERM:
    loss_fn = PassThroughLoss()
elif conf.loss_type == LossType.TOPK:
    minority_groups = [(0,0)] if conf.ace_agree else [(0,1)]
    loss_fn = ACELoss(
        mix_rate=conf.mix_rate_lower_bound,
        classes_per_head=classes_per_head, 
        mode="topk", 
        minority_groups=minority_groups,
        device=conf.device
    )
    mix_rate_scheduler = None
    if conf.mix_rate_schedule == "linear":
        mix_rate_scheduler = MixRateScheduler(
            loss_fn=loss_fn,
            mix_rate_lb=conf.mix_rate_lower_bound,
            t0=conf.mix_rate_t0,
            t1=conf.mix_rate_t1
        )
# copy loss fn for validation (no scheduling)
val_loss_fn = copy.deepcopy(loss_fn)

src_loss_fn = SrcLoss(
    binary=conf.binary,
    classes_per_head=classes_per_head,
    use_group_labels=True
)


def get_src_gl(gl, label_type: Optional[list[Literal["all_sensors", "sensors_agree", None]]] = None):
    labels = []
    for i in range(conf.heads):
        if label_type is None or label_type[i] is None:
            labels.append(gl[:, 1])
        elif label_type[i] == "all_sensors":
            labels.append(gl[:, 1])
        elif label_type[i] == "sensors_agree":
            labels.append(gl[:, 2])
    labels = torch.stack(labels, dim=-1)
    assert labels.shape == (gl.shape[0], conf.heads), f"labels shape {labels.shape}, gl shape {gl.shape}"
    return labels


def compute_corrects(logits: torch.Tensor, head: int, y: torch.Tensor, binary: bool):
    if binary:
        preds = (logits[:, head] > 0).to(torch.float32)
        assert preds.shape == (logits.size(0), ), f"preds shape {preds.shape}, logits shape {logits.shape}"
        return ((preds == y.flatten()).sum().item())
    else:
        logits = logits.view(logits.size(0), conf.heads, -1)
        return (logits[:, head].argmax(dim=-1) == y).sum().item()
        

def slice_batch(batch, slice):
    if isinstance(batch, torch.Tensor):
        return batch[slice]
    elif isinstance(batch, dict):
        return {k: v[slice] for k, v in batch.items()}
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")


def in_slice(idx, slice):
    return idx >= slice.start and idx < slice.stop


def compute_labeled_target_loss(
        logits, gl, target_labels: list[Literal["all_sensors", "sensors_agree", None]]
    ):    
    # visible logits chunked
    logits_chunked = torch.chunk(logits, conf.heads, dim=-1)

    losses = []
    for i in range(conf.heads):
        if target_labels[i] is None: # no label for this head
            continue
        if target_labels[i] == "all_sensors":
            y_i = gl[:, 1]
        elif target_labels[i] == "sensors_agree":
            y_i = gl[:, 2]
        losses.append(F.binary_cross_entropy_with_logits(logits_chunked[i].squeeze(), y_i.squeeze().to(torch.float32)))
    if len(losses) == 0:
        return torch.tensor(0.0)
    return sum(losses)


def compute_target_loss( # TODO: look over again, maybe refactor
    logits, y, gl, loss_fn, loss_type, 
    target_labels: Optional[list[Literal["all_sensors", "sensors_agree", None]]] = None, 
    only_disagreeing_labels: bool = False
): 
    # separate instances based on whether they have disagreeing measurements (i.e. gl[:, 2] == 0)
    div_logits = logits 
    labeled_logits = logits 
    labeled_gl = gl 
    div_loss_kwargs = {}
    if loss_type == LossType.TOPK:
        div_loss_kwargs["virtual_bs"] = logits.shape[0]
    
    if only_disagreeing_labels: # for probing for evidence of tamper
        disagreeing_mask = gl[:, 2] == 0
        div_logits = logits[~disagreeing_mask]
        labeled_logits = logits[disagreeing_mask]
        labeled_gl = gl[disagreeing_mask]

    div_loss = loss_fn(div_logits, **div_loss_kwargs)
    labeled_loss = torch.tensor(0.0)    
    if target_labels is not None:
        labeled_loss = compute_labeled_target_loss(labeled_logits, labeled_gl, target_labels)
    return div_loss, labeled_loss


def eval(net, loader, conf, bootstrap=False, n_bootstrap_samples=100, bootstrap_seed=0, fraction=0.5): 
    net.eval()

    total_correct = torch.zeros(conf.heads)
    total_correct_groups = {
        "all_sensors": torch.zeros(conf.heads),
        "sensors_agree": torch.zeros(conf.heads)
    }
    total_samples = 0
    all_preds = [[] for _ in range(conf.heads)]
    all_labels = []
    all_group_labels = {
        "all_sensors": [],
        "sensors_agree": []
    }

    with torch.no_grad():
        for test_batch in tqdm(loader, desc="Target test"):
            test_x, test_y, test_gl = to_device(*test_batch, conf.device)
            test_logits = net(test_x)
            assert test_logits.shape == (batch_size(test_x), sum(classes_per_head))
            total_samples += test_y.size(0)

            # Store labels for AUROC
            all_labels.extend(test_y.cpu().numpy())
            all_group_labels["all_sensors"].extend(test_gl[:, 1].cpu().numpy())
            all_group_labels["sensors_agree"].extend(test_gl[:, 2].cpu().numpy())
            
            for i in range(conf.heads):
                total_correct[i] += compute_corrects(test_logits, i, test_y, conf.binary)
                total_correct_groups["all_sensors"][i] += compute_corrects(test_logits, i, test_gl[:, 1], conf.binary)
                total_correct_groups["sensors_agree"][i] += compute_corrects(test_logits, i, test_gl[:, 2], conf.binary)
                probs = torch.sigmoid(test_logits[:, i]).cpu().numpy()
                all_preds[i].extend(probs)

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = [np.array(preds) for preds in all_preds]
    all_group_labels = {k: np.array(v) for k, v in all_group_labels.items()}

    # Compute point estimates for accuracies
    head_accs = [(total_correct[i] / total_samples).item() for i in range(conf.heads)]
    head_accs_groups = {
        group: [(total_correct_groups[group][i] / total_samples).item() for i in range(conf.heads)]
        for group in ["all_sensors", "sensors_agree"]
    }

    # Compute AUROC (with or without bootstrapping)
    if not bootstrap:
        # Single AUROC computation per head, wrapped in a list
        head_aurocs = [[roc_auc_score(all_labels, preds)] for preds in all_preds]
    else:
        # Bootstrap AUROC computation
        head_aurocs = [[] for _ in range(conf.heads)]
        np_gen = np.random.default_rng(bootstrap_seed)

        correct_indices = np.where(all_labels == 1)[0]
        incorrect_indices = np.where(all_labels == 0)[0]
        correct_preds = [
            all_preds[i][correct_indices] for i in range(conf.heads)
        ]
        incorrect_preds = [
            all_preds[i][incorrect_indices] for i in range(conf.heads)
        ]
        
        for _ in range(n_bootstrap_samples):
            # split into corrects vs incorrects (based on y label)
            for i in range(conf.heads):
                correct_preds_sampled = np_gen.choice(correct_preds[i], size=round(len(correct_preds[i]) * fraction))
                incorrect_preds_sampled = np_gen.choice(incorrect_preds[i], size=round(len(incorrect_preds[i]) * fraction))
                gt = np.concatenate([np.ones_like(correct_preds_sampled), np.zeros_like(incorrect_preds_sampled)])
                scores = np.concatenate([correct_preds_sampled, incorrect_preds_sampled])
                try:
                    bootstrap_auroc = roc_auc_score(gt, scores)
                except ValueError:
                    bootstrap_auroc = np.nan
                head_aurocs[i].append(bootstrap_auroc)
    
    return head_accs, head_accs_groups, head_aurocs


def train_target(conf: Config):
    return conf.split_source_target and (conf.aux_weight > 0 or conf.target_labels is not None)


logger = Logger(conf.exp_dir)

def train(
    conf: Config, 
    net: nn.Module, 
    source_train_loader: DataLoader, 
    target_train_loader: DataLoader, 
    source_val_loader: DataLoader, 
    target_val_loader: DataLoader, 
    mix_rate_scheduler: MixRateScheduler, 
    logger: Logger
):
    # dataloader with effective batch size, then iterate over micro batches within batch 
    if conf.split_source_target:
        target_iter = iter(target_train_loader)
        target_batch = None
        target_logits = None

    for epoch in range(conf.epochs):
        target_logit_ls = []
        source_batch_loss = 0
        source_batch_corrects = {i: 0 for i in range(conf.heads)}
        target_batch_corrects = {(i, label): 0 for i in range(conf.heads) for label in ["y", "all_sensors", "sensors_agree"]}
        
        if mix_rate_scheduler is not None:
            mix_rate_scheduler.step()
        for batch_idx, (x, y, gl) in tqdm(enumerate(source_train_loader), desc="Train", total=len(source_train_loader)):
            # compute source logits with micro batch 
            x, y, gl = to_device(x, y, gl, conf.device)
            logits = net(x)
            losses = src_loss_fn(logits, y, get_src_gl(gl, conf.source_labels))
            xent = sum(losses)
            source_batch_loss += xent.item()

            # computer source acc 
            for i in range(conf.heads):
                source_batch_corrects[i] += compute_corrects(logits, i, y, conf.binary)
            # compute target logits with no grad on forward batch 
            div_loss = torch.tensor(0.0)
            labeled_target_loss = torch.tensor(0.0)
            if train_target(conf):
                if batch_idx % (conf.effective_batch_size // conf.micro_batch_size) == 0:
                    target_logits_ls = []
                    try: 
                        target_batch = next(target_iter)
                        if target_batch[1].shape[0] != conf.effective_batch_size:
                            raise StopIteration
                    except StopIteration:
                        target_iter = iter(target_train_loader)
                        target_batch = next(target_iter)
                    target_batch, target_y, target_gl = to_device(*target_batch, conf.device)
                    with torch.no_grad():
                        target_logits_ls.append(net(target_batch).detach())
                    target_logits = torch.cat(target_logits_ls, dim=0)
                # compute target logits with grad on micro batch
                micro_batch_idx = batch_idx % (conf.effective_batch_size // conf.micro_batch_size)
                micro_slice = slice(micro_batch_idx * conf.micro_batch_size, (micro_batch_idx + 1) * conf.micro_batch_size)
                target_micro_batch = slice_batch(target_batch, micro_slice)
                target_micro_logits = net(target_micro_batch)

                cloned_target_logits= target_logits.clone().requires_grad_(True)
                new_target_logits = torch.cat([
                    cloned_target_logits[i].unsqueeze(0) if 
                    not in_slice(i, micro_slice) else target_micro_logits[i - micro_slice.start].unsqueeze(0)
                    for i in range(len(cloned_target_logits))
                ])

                div_loss, labeled_target_loss = compute_target_loss(
                    new_target_logits, target_y, target_gl, loss_fn, conf.loss_type, conf.target_labels, 
                    only_disagreeing_labels=conf.target_only_disagree
                )

            # full loss (on micro batch)
            full_loss = conf.source_weight * xent + conf.aux_weight * div_loss + labeled_target_loss   
            full_loss.backward() 
            
            # update weights, clear gradients on effective batch
            if (batch_idx + 1) % (conf.effective_batch_size // conf.micro_batch_size) == 0:
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                opt.zero_grad()

                # compute target acc 
                if train_target(conf):
                    for i in range(conf.heads):
                        target_batch_corrects[(i, "y")] += compute_corrects(new_target_logits, i, target_y, conf.binary) 
                        target_batch_corrects[(i, "all_sensors")] += compute_corrects(new_target_logits, i, target_gl[:, 1], conf.binary)
                        target_batch_corrects[(i, "sensors_agree")] += compute_corrects(new_target_logits, i, target_gl[:, 2], conf.binary)

                source_batch_loss = source_batch_loss / conf.effective_batch_size
                # compute batch metrics 
                effective_batch_idx = batch_idx // (conf.effective_batch_size // conf.micro_batch_size)
                effective_num_batches = len(source_train_loader) // (conf.effective_batch_size // conf.micro_batch_size)
                logger.add_scalar("train", "source_loss", source_batch_loss, epoch * effective_num_batches + effective_batch_idx, to_metrics=False)
                if conf.aux_weight > 0:
                    logger.add_scalar("train", "div_loss", div_loss.item(), epoch * effective_num_batches + effective_batch_idx, to_metrics=False)
                if conf.target_labels is not None:
                    logger.add_scalar("train", "labeled_target_loss", labeled_target_loss.item(), epoch * effective_num_batches + effective_batch_idx, to_metrics=False)
                logger.add_scalar("train", "full_loss", source_batch_loss + conf.aux_weight * div_loss.item() + labeled_target_loss.item(), epoch * effective_num_batches + effective_batch_idx, to_metrics=False)
                
                for i in range(conf.heads):
                    logger.add_scalar("train", f"source_acc_{i}", source_batch_corrects[i] / conf.effective_batch_size, epoch * effective_num_batches + effective_batch_idx, to_metrics=False)
                    if train_target(conf):
                        for label in ["y", "all_sensors", "sensors_agree"]:
                            logger.add_scalar("train", f"target_acc_{i}_{label}", target_batch_corrects[(i, label)] / conf.effective_batch_size, epoch * effective_num_batches + effective_batch_idx, to_metrics=False)
                source_batch_loss = 0
                source_batch_corrects = {i: 0 for i in range(conf.heads)}
                target_batch_corrects = {(i, label): 0 for i in range(conf.heads) for label in ["y", "all_sensors", "sensors_agree"]}
        
        # validation and test
        if (epoch + 1) % 1 == 0:
            net.eval()
            # compute xent on source validation set
            xent_val = []
            with torch.no_grad():
                for batch in tqdm(source_val_loader, desc="Source val"):
                    x, y, gl = to_device(*batch, conf.device)
                    logits_val = net(x)
                    losses_val = src_loss_fn(logits_val, y, get_src_gl(gl, conf.source_labels))
                    xent_val.append(sum(losses_val).item())
            logger.add_scalar("val", "source_loss", np.mean(xent_val), epoch)
            
            # compute div loss on target validation set (used for model selection)
            if train_target(conf):
                div_losses_val = []
                labeled_target_losses_val = []
                with torch.no_grad():
                    for batch in tqdm(target_val_loader, desc="Target val"):
                        x, y, gl = to_device(*batch, conf.device)
                        logits_val = net(x)
                        div_loss, labeled_target_loss = compute_target_loss(
                            logits_val, y, gl, val_loss_fn, conf.loss_type, conf.target_labels, 
                            only_disagreeing_labels=conf.target_only_disagree
                        )
                        div_losses_val.append(div_loss.item())
                        labeled_target_losses_val.append(labeled_target_loss.item())
                
                logger.add_scalar("val", "target_div_loss", np.mean(div_losses_val), epoch)
                logger.add_scalar("val", "target_labeled_loss", np.mean(labeled_target_losses_val), epoch)
                logger.add_scalar("val", "target_weighted_div_loss", np.mean(div_losses_val) * conf.aux_weight, epoch)
                logger.add_scalar("val", "target_loss", np.mean(div_losses_val) * conf.aux_weight + np.mean(labeled_target_losses_val), epoch)
            
            # total validation loss
            val_loss = logger.metrics["val_source_loss"][-1]
            if train_target(conf):
                val_loss += logger.metrics["val_target_loss"][-1]
            logger.add_scalar("val", "loss", val_loss, epoch)
        
            # test evaluation (acc, acc_alt, auroc)
            head_accs, head_accs_groups, head_aurocs = eval(
                net, target_test_loader, conf, bootstrap=conf.bootstrap_eval, 
                n_bootstrap_samples=conf.n_bootstrap_samples, bootstrap_seed=conf.seed
            )
            test_groups = ["all_sensors"]
            for i in range(conf.heads):
                # acc 
                logger.add_scalar("test", f"acc_{i}", head_accs[i], epoch)
                for group in test_groups:
                    logger.add_scalar("test", f"acc_{i}_{group}", head_accs_groups[group][i], epoch)
                # auroc
                logger.add_scalar("test", f"auroc_{i}", np.array(head_aurocs[i]).mean(), epoch)
                logger.add_scalar("test", f"auroc_{i}_std", np.array(head_aurocs[i]).std(), epoch)              
            
            # print validation losses and test accs
            print(f"Epoch {epoch + 1} Test Accuracies:")
            print(f"Source val xent: {logger.metrics['val_source_loss'][-1]:.4f}")
            if train_target(conf):
                print(f"Target val div loss: {logger.metrics['val_target_div_loss'][-1]:.4f}")
                print(f"Target val weighted div loss: {logger.metrics['val_target_weighted_div_loss'][-1]:.4f}")
            print(f"val loss: {logger.metrics['val_loss'][-1]:.4f}")
            for i in range(conf.heads):
                print(
                    f"Head {i}: {logger.metrics[f'test_acc_{i}'][-1]:.4f}", 
                    *[f"{group}: {logger.metrics[f'test_acc_{i}_{group}'][-1]:.4f}" for group in test_groups]
                )
                print(f"Head {i} auroc: {logger.metrics[f'test_auroc_{i}'][-1]:.4f}")
            
            net.train()


# if not training, evaluate on test and exit
if not conf.train:
    head_accs, head_accs_groups, head_aurocs = eval(
        net, target_test_loader, conf, bootstrap=conf.bootstrap_eval, 
        n_bootstrap_samples=conf.n_bootstrap_samples, bootstrap_seed=conf.seed, fraction=0.5
    )

    # Initialize metrics lists
    test_groups = ["all_sensors"]
    for i in range(conf.heads):
        # acc
        logger.add_scalar("test", f"acc_{i}", head_accs[i], 0)
        for group in test_groups:
            logger.add_scalar("test", f"acc_{i}_{group}", head_accs_groups[group][i], 0)
        # auroc
        logger.add_scalar("test", f"auroc_{i}", np.array(head_aurocs[i]).mean(), 0)
        logger.add_scalar("test", f"auroc_{i}_std", np.array(head_aurocs[i]).std(), 0)

    logger.flush()
    exit() 


try: 
    train(
        conf=conf, 
        net=net, 
        source_train_loader=source_train_loader, 
        target_train_loader=(target_train_loader if conf.split_source_target else None), 
        source_val_loader=source_val_loader, 
        target_val_loader=(target_val_loader if conf.split_source_target else None), 
        mix_rate_scheduler=(mix_rate_scheduler if conf.loss_type == LossType.TOPK else None), 
        logger=logger
)
finally: 
    logger.flush()
    

