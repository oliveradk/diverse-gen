#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# set cuda visible devices
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

import os
if is_notebook():
    os.environ["CUDA_VISIBLE_DEVICES"] = "7" #"1"
    # os.environ['CUDA_LAUNCH_BLOCKING']="1"
    # os.environ['TORCH_USE_CUDA_DSA'] = "1"

import matplotlib 
if not is_notebook():
    matplotlib.use('Agg')


# In[ ]:


# TODO: separete target labeled and unlabeled loss 
# run experiment using source and target visisible (no unlabeled loss) # all vs any
# run experiment using source and target visible, with unlabeled loss # all vs any


# In[ ]:


import os
os.chdir("/nas/ucb/oliveradk/diverse-gen")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[ ]:


from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import random_split
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm


from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import get_scheduler

from losses.divdis import DivDisLoss 
from losses.divdis import DivDisLoss
from losses.ace import ACELoss
from losses.conf import ConfLoss
from losses.dbat import DBatLoss
from losses.smooth_top_loss import SmoothTopLoss
from losses.pass_through import PassThroughLoss
from losses.loss_types import LossType

from models.backbone import MultiHeadBackbone
from utils.utils import batch_size, to_device


# In[ ]:


from datetime import datetime
from dataclasses import dataclass
@dataclass 
class Config: 
    loss_type: LossType = LossType.TOPK
    one_sided_ace: bool = True
    dataset: str = "diamonds-seed0"
    lr: float = 2e-5
    weight_decay: float = 2e-2
    epochs: int = 5
    scheduler: str = "cosine"
    frac_warmup: float = 0.05
    num_epochs: int = 5
    effective_batch_size: int = 32
    forward_batch_size: int = 32
    micro_batch_size: int = 4
    use_visible_labels: bool = False
    use_negative_visible: bool = False
    all_measurements: bool = False
    seed: int = 42
    max_length: int = 1024
    dataset_len: Optional[int] = None
    binary: bool = True
    heads: int = 2
    train: bool = True
    freeze_model: bool = False
    load_prior_probe: bool = False
    source_weight: float = 1.0
    aux_weight: float = 1.0
    mix_rate_lower_bound: float = 0.1
    use_group_labels: bool = False
    num_workers: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir: str = f"output/mtd/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

def post_init(conf, overrride_keys):
    pass


# In[ ]:


conf = Config()


# In[ ]:


overrride_keys = []
if not is_notebook():
    import sys 
    overrides = OmegaConf.from_cli(sys.argv[1:])
    overrride_keys = overrides.keys()
    conf_dict = OmegaConf.merge(OmegaConf.structured(conf), overrides)
    conf = Config(**conf_dict)
post_init(conf, overrride_keys)


# In[ ]:


exp_dir = conf.exp_dir
os.makedirs(exp_dir, exist_ok=True)

# save full config to exp_dir
with open(f"{exp_dir}/config.yaml", "w") as f:
    OmegaConf.save(config=conf, f=f)


# # Model and Dataset

# In[ ]:


model_path = "oliverdk/codegen-350M-mono-measurement_pred"


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


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True, 
    padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

# dataset and model


# In[ ]:


from datasets import load_dataset

dataset = load_dataset(f"redwoodresearch/{conf.dataset}")


# In[ ]:


# dataset_dir = "data/diamonds/"
# os.makedirs(dataset_dir, exist_ok=True)
# def encode_dataset(examples):
#     return tokenizer(
#         examples['text'],
#         max_length=conf.max_length,
#         padding='max_length', 
#         truncation=True,
#         return_tensors='pt'
#     )

# dataset = dataset.map(
#     encode_dataset,
#     batched=True,
#     # cache_file_names={
#     #     split: f"{dataset_dir}/seed0_{split}_{conf.dataset_len}_{conf.max_length}.arrow"
#     #     for split in dataset.keys()
#     # }
# )


# In[ ]:


class DiamondsDataset(Dataset):
    def __init__(self, dataset, max_length=1024, negative_visible=False, all_measurements=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_visible = negative_visible
        self.all_measurements = all_measurements

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
        label = self.ground_truth[idx]
        tampering = torch.all(self.measurements[idx]) if self.all_measurements else torch.any(self.measurements[idx])
        
        # visible if failed tampering (some but not all measurements)
        failed_tampering = torch.any(self.measurements[idx]) & ~torch.all(self.measurements[idx])
        visible = failed_tampering 
        # visible if negative (implies no tampering b/c agent has no incentive to tamper)
        if self.negative_visible:
            negative = torch.tensor(not torch.any(self.measurements[idx]))
            visible = torch.logical_or(visible, negative)
        
        group_labels = torch.stack((label, tampering, visible))


        # set labels to floats 
        label = label.to(torch.float32)
        group_labels = group_labels.to(torch.float32)
        
        return encoding, label, group_labels


# In[ ]:


if conf.dataset_len is not None:
    for k, subset in dataset.items():
        dataset[k] = subset.select(range(conf.dataset_len))


# In[ ]:


# source (is clean)
val_frac = 0.2
source_data = dataset["train"].filter(lambda x: x["is_clean"])
splits = source_data.train_test_split(train_size=1-val_frac, test_size=val_frac, seed=conf.seed)
dataset["source_train"] = splits['train']
dataset["source_val"] = splits['test']

# target (is not clean)
target_data = dataset["train"].filter(lambda x: not x["is_clean"])
dataset["target_train"] = target_data
dataset["target_val"] = dataset["train_for_val"]

# test (validation)
dataset["test"] = dataset["validation"]
# only untrusted positive examples 
dataset["test"] = dataset["test"].filter(lambda x: not x['is_clean'] and all(x["measurements"]))

# remove train and train_for_val
dataset.pop("train")
dataset.pop("train_for_val")
dataset.pop("validation")


# In[ ]:


def is_tampering(x):
    return not x["is_correct"] and any(x["measurements"])

def fake_positive(x):
    return not x["is_correct"] and all(x["measurements"])

def split_tampering_rate(dataset):
    tampering_rate = len(dataset.filter(is_tampering)) / len(dataset)
    return tampering_rate
def split_fake_positive_rate(dataset):
    fake_positive_rate = len(dataset.filter(fake_positive)) / len(dataset)
    return fake_positive_rate
source_train_tampering_rate = split_tampering_rate(dataset["source_train"])
target_train_tampering_rate = split_tampering_rate(dataset["target_train"])
source_val_tampering_rate = split_tampering_rate(dataset["source_val"])
target_val_tampering_rate = split_tampering_rate(dataset["target_val"])
test_tampering_rate = split_tampering_rate(dataset["test"])   

source_train_fake_positive_rate = split_fake_positive_rate(dataset["source_train"])
target_train_fake_positive_rate = split_fake_positive_rate(dataset["target_train"])
source_val_fake_positive_rate = split_fake_positive_rate(dataset["source_val"])
target_val_fake_positive_rate = split_fake_positive_rate(dataset["target_val"])
test_fake_positive_rate = split_fake_positive_rate(dataset["test"])

print(f"source train: tampering {source_train_tampering_rate:.2f}, fake positive {source_train_fake_positive_rate:.2f}")
print(f"target train: tampering {target_train_tampering_rate:.2f}, fake positive {target_train_fake_positive_rate:.2f}")
print(f"source val: tampering {source_val_tampering_rate:.2f}, fake positive {source_val_fake_positive_rate:.2f}")
print(f"target val: tampering {target_val_tampering_rate:.2f}, fake positive {target_val_fake_positive_rate:.2f}")
print(f"test: tampering {test_tampering_rate:.2f}, fake positive {test_fake_positive_rate:.2f}")


# In[ ]:


source_train_ds = DiamondsDataset(dataset["source_train"], conf.max_length, conf.use_negative_visible, conf.all_measurements)
source_val_ds = DiamondsDataset(dataset["source_val"], conf.max_length, conf.use_negative_visible, conf.all_measurements)
target_train_ds = DiamondsDataset(dataset["target_train"], conf.max_length, conf.use_negative_visible, conf.all_measurements)
target_val_ds = DiamondsDataset(dataset["target_val"], conf.max_length, conf.use_negative_visible, conf.all_measurements)
test_ds = DiamondsDataset(dataset["test"], conf.max_length, conf.use_negative_visible, conf.all_measurements)


# In[ ]:


class MeasurementPredBackbone(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
    
    def forward(self, x):
        out = self.pretrained_model.base_model(x['input_ids'], attention_mask=x['attention_mask'])
        embd = out.last_hidden_state[:, -1, :]
        return embd


# # Train

# In[ ]:


pred_model = MeasurementPredBackbone(pretrained_model).to(conf.device)
net = MultiHeadBackbone(pred_model, n_heads=2, feature_dim=1024, classes=1).to(conf.device)

if conf.freeze_model:
    for param in net.backbone.parameters():
        param.requires_grad = False

# load weights of pretrained model aggregate probe to second net head
if conf.load_prior_probe:
    net.heads.weight.data[1, :] = pretrained_model.aggregate_probe.weight.data[0]
    net.heads.bias.data[1] = pretrained_model.aggregate_probe.bias.data[0]

source_train_loader = DataLoader(source_train_ds, batch_size=conf.micro_batch_size, num_workers=conf.num_workers)
target_train_loader = DataLoader(target_train_ds, batch_size=conf.effective_batch_size, num_workers=conf.num_workers)
source_val_loader = DataLoader(source_val_ds, batch_size=conf.micro_batch_size, num_workers=conf.num_workers)
target_val_loader = DataLoader(target_val_ds, batch_size=conf.effective_batch_size, num_workers=conf.num_workers)
target_test_loader = DataLoader(test_ds, batch_size=conf.forward_batch_size, num_workers=conf.num_workers)

opt = torch.optim.AdamW(net.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

num_training_steps = conf.num_epochs * len(source_train_loader) // (conf.effective_batch_size // conf.micro_batch_size)
scheduler = get_scheduler(
    name=conf.scheduler,
    optimizer=opt,
    num_warmup_steps=conf.frac_warmup * num_training_steps,
    num_training_steps=num_training_steps
)

if conf.loss_type == LossType.DIVDIS:
    loss_fn = DivDisLoss(heads=2)
elif conf.loss_type == LossType.ERM:
    loss_fn = PassThroughLoss()
elif conf.loss_type == LossType.TOPK:
    if conf.one_sided_ace:
        group_mix_rates = {(0, 1): conf.mix_rate_lower_bound}
        mix_rate = None
    else:
        mix_rate = conf.mix_rate_lower_bound
        group_mix_rates = None
    loss_fn = ACELoss(
        heads=2, 
        classes=2, 
        binary=True, 
        mode="topk", 
        group_mix_rates=group_mix_rates,  # TODO: should ignore visible labels
        mix_rate=mix_rate,
        pseudo_label_all_groups=False, 
        device=conf.device
    )


# In[ ]:


def compute_src_losses(logits, y, gl, binary, use_group_labels):
    logits_chunked = torch.chunk(logits, conf.heads, dim=-1)
    labels = torch.cat([y, y], dim=-1) if not use_group_labels else gl
    labels_chunked = torch.chunk(labels, conf.heads, dim=-1)
    if binary:
        losses = [F.binary_cross_entropy_with_logits(logit.squeeze(), y.squeeze().to(torch.float32)) for logit, y in zip(logits_chunked, labels_chunked)]
    else:
        losses = [F.cross_entropy(logit.squeeze(), y.squeeze().to(torch.long)) for logit, y in zip(logits_chunked, labels_chunked)]
    return losses

def compute_corrects(logits: torch.Tensor, head: int, y: torch.Tensor, binary: bool):
    if binary:
        return ((logits[:, head] > 0) == y.flatten()).sum().item()
    else:
        logits = logits.view(logits.size(0), conf.heads, -1)
        return (logits[:, head].argmax(dim=-1) == y).sum().item()
        


# In[ ]:


# TODO: change diciotary values to source loss, target loss

classes = 2
alt_index = 1

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

# metrics
metrics = defaultdict(list)
writer = SummaryWriter(log_dir=conf.exp_dir)


# In[ ]:


def slice_batch(batch, slice):
    if isinstance(batch, torch.Tensor):
        return batch[slice]
    elif isinstance(batch, dict):
        return {k: v[slice] for k, v in batch.items()}
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")


# In[ ]:


def in_slice(idx, slice):
    return idx >= slice.start and idx < slice.stop


# In[ ]:


# experiments to try: 
## using failed tampering instances as positive tampering (i.e. modify group labels to add a third, which is basically (visible), and the heads should be trained to ouptut the group labels on those)
## use labels on negative examples (we know no tampering)


# In[ ]:


def compute_visible_target_loss(visible_logits, visible_gl):    

    # visible logits chunked
    logits_chunked = torch.chunk(visible_logits, conf.heads, dim=-1)

    # visible group labels chunked
    gl_chunked = torch.chunk(visible_gl[:, :2], conf.heads, dim=-1)
    
    # compute losses for each head
    losses = [
        F.binary_cross_entropy_with_logits(
            logit.squeeze(), y_i.squeeze().to(torch.float32)
        )
        for logit, y_i in zip(logits_chunked, gl_chunked)
    ]
    return sum(losses)


# In[ ]:


def compute_target_loss(logits, y, gl, loss_fn, loss_type, use_visible_labels): 
    if not use_visible_labels:
        return loss_fn(logits), torch.tensor(0.0)
    
    visible_mask = gl[:, 2].bool()
    
    # compute div loss
    if loss_type == LossType.TOPK: 
        non_visible_logits = logits[~visible_mask]
        div_loss = loss_fn(non_visible_logits)
    else: 
        div_loss = loss_fn(logits)

    visible_loss = torch.tensor(0.0)
    if any(visible_mask):
        visible_loss = compute_visible_target_loss(logits[visible_mask], gl[visible_mask])

    return div_loss, visible_loss


# In[ ]:


# compute accuracy (both labels) and auroc (true vs fake positives) on test set
def eval(net, loader, conf): 
    net.eval()

    head_accs = []
    head_accs_alt = []
    head_aurocs = []
    total_correct = torch.zeros(conf.heads)
    total_correct_alt = torch.zeros(conf.heads)
    total_samples = 0
    all_preds = [[] for _ in range(conf.heads)]
    all_labels = []

    with torch.no_grad():
        for test_batch in tqdm(loader, desc="Target test"):
            test_x, test_y, test_gl = to_device(*test_batch, conf.device)
            test_logits = net(test_x)
            assert test_logits.shape == (batch_size(test_x), conf.heads * (1 if conf.binary else classes))
            total_samples += test_y.size(0)

                # Store labels for AUROC
            all_labels.extend(test_y.cpu().numpy())
            
            for i in range(conf.heads):
                total_correct[i] += compute_corrects(test_logits, i, test_y, conf.binary)
                total_correct_alt[i] += compute_corrects(test_logits, i, test_gl[:, 1], conf.binary)
                probs = torch.sigmoid(test_logits[:, i]).cpu().numpy()
                all_preds[i].extend(probs)

    # Compute and store AUROC for each head
    for i in range(conf.heads):
        auroc = roc_auc_score(all_labels, all_preds[i])
        head_aurocs.append(auroc)

    # compute and store accuracy for each head
    for i in range(conf.heads):
        head_accs.append((total_correct[i] / total_samples).item())
        head_accs_alt.append((total_correct_alt[i] / total_samples).item())
    return head_accs, head_accs_alt, head_aurocs


# In[ ]:


if not conf.train:
    head_accs, head_accs_alt, head_aurocs = eval(net, target_test_loader, conf)
    print(f"Test Accuracies:")
    for i in range(conf.heads):
        print(f"Head {i}: {head_accs[i]:.4f}, Alt: {head_accs_alt[i]:.4f}")
    print(f"Test AUROCs:")
    for i in range(conf.heads):
        print(f"Head {i}: {head_aurocs[i]:.4f}")
    # stop run all 
    raise ValueError("Stop run all (not an actual error)")


# In[ ]:


def train_target(conf: Config):
    return conf.aux_weight > 0 or conf.use_visible_labels


# In[ ]:


# dataloader with effective batch size, then iterate over micro batches within batch 
target_iter = iter(target_train_loader)
target_batch = None
target_logits = None

for epoch in range(conf.epochs):
    target_logit_ls = []
    source_batch_loss = 0
    source_batch_corrects = {i: 0 for i in range(conf.heads)}
    target_batch_corrects = {(i, label): 0 for i in range(conf.heads) for label in ["y", "gl"]}
    for batch_idx, (x, y, gl) in tqdm(enumerate(source_train_loader), desc="Source train", total=len(source_train_loader)):
        # compute source logits with micro batch 
        x, y, gl = to_device(x, y, gl, conf.device)
        logits = net(x)
        losses = compute_src_losses(logits, y, gl, conf.binary, conf.use_group_labels)
        xent = sum(losses)
        source_batch_loss += xent.item()

        # computer source acc 
        for i in range(conf.heads):
            source_batch_corrects[i] += compute_corrects(logits, i, y, conf.binary)
        # compute target logits with no grad on forward batch 
        div_loss = torch.tensor(0.0)
        visible_loss = torch.tensor(0.0)
        if train_target(conf):
            if batch_idx % (conf.effective_batch_size // conf.micro_batch_size) == 0:
                print("computing target logits")
                target_logits_ls = []
                try: 
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_train_loader)
                    target_batch = next(target_iter)
                target_batch, target_y, target_gl = to_device(*target_batch, conf.device)
                with torch.no_grad():
                    target_logits_ls.append(net(target_batch).detach())
                target_logits = torch.cat(target_logits_ls, dim=0)
                print("computed target logits")
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

            div_loss, visible_loss = compute_target_loss(new_target_logits, target_y, target_gl, loss_fn, conf.loss_type, conf.use_visible_labels)

        # full loss (on micro batch)
        full_loss = conf.source_weight * xent + conf.aux_weight * div_loss + visible_loss   
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
                    target_batch_corrects[(i, "gl")] += compute_corrects(new_target_logits, i, target_gl[:, 1], conf.binary)

            source_batch_loss = source_batch_loss / conf.effective_batch_size
            # compute batch metrics 
            effective_batch_idx = batch_idx // (conf.effective_batch_size // conf.micro_batch_size)
            effective_num_batches = len(source_train_loader) // (conf.effective_batch_size // conf.micro_batch_size)
            writer.add_scalar("train/source_loss", source_batch_loss, epoch * effective_num_batches + effective_batch_idx)
            if conf.aux_weight > 0:
                writer.add_scalar("train/div_loss", div_loss.item(), epoch * effective_num_batches + effective_batch_idx)
            if conf.use_visible_labels:
                writer.add_scalar("train/visible_loss", visible_loss.item(), epoch * effective_num_batches + effective_batch_idx)
            writer.add_scalar("train/full_loss", source_batch_loss + conf.aux_weight * div_loss.item() + visible_loss.item(), epoch * effective_num_batches + effective_batch_idx)
            
            for i in range(conf.heads):
                writer.add_scalar(f"train/source_acc_{i}", source_batch_corrects[i] / conf.effective_batch_size, epoch * effective_num_batches + effective_batch_idx)
                if train_target(conf):
                    for label in ["y", "gl"]:
                        writer.add_scalar(f"train/target_acc_{i}_{label}", target_batch_corrects[(i, label)] / conf.effective_batch_size, epoch * effective_num_batches + effective_batch_idx)
            source_batch_loss = 0
            source_batch_corrects = {i: 0 for i in range(conf.heads)}
            target_batch_corrects = {(i, label): 0 for i in range(conf.heads) for label in ["y", "gl"]}
    
    # validation and test
    if (epoch + 1) % 1 == 0:
        net.eval()
        # compute repulsion loss on target validation set (used for model selection)
        div_losses_val = []
        visible_losses_val = []
        with torch.no_grad():
            for batch in tqdm(target_val_loader, desc="Target val"):
                x, y, gl = to_device(*batch, conf.device)
                logits_val = net(x)
                div_loss, visible_loss = compute_target_loss(logits_val, y, gl, loss_fn, conf.loss_type, conf.use_visible_labels)
                div_losses_val.append(div_loss.item())
                visible_losses_val.append(visible_loss.item())
        
        metrics[f"val_target_div_loss"].append(np.mean(div_losses_val))
        metrics[f"val_target_visible_loss"].append(np.mean(visible_losses_val))
        metrics[f"val_target_weighted_div_loss"].append(np.mean(div_losses_val) * conf.aux_weight)
        metrics[f"val_target_loss"].append(np.mean(div_losses_val) * conf.aux_weight + np.mean(visible_losses_val))
        
        writer.add_scalar("val/div_loss", metrics[f"val_target_div_loss"][-1], epoch)
        writer.add_scalar("val/weighted_div_loss", metrics[f"val_target_weighted_div_loss"][-1], epoch)
        writer.add_scalar("val/visible_loss", metrics[f"val_target_visible_loss"][-1], epoch)
        writer.add_scalar("val/target_loss", metrics[f"val_target_loss"][-1], epoch)
        # compute xent on source validation set
        xent_val = []
        with torch.no_grad():
            for batch in tqdm(source_val_loader, desc="Source val"):
                x, y, gl = to_device(*batch, conf.device)
                logits_val = net(x)
                losses_val = compute_src_losses(logits_val, y, gl, conf.binary, conf.use_group_labels)
                xent_val.append(sum(losses_val).item())
        metrics[f"val_source_xent"].append(np.mean(xent_val))
        writer.add_scalar("val/source_loss", metrics[f"val_source_xent"][-1], epoch)
        
        metrics[f"val_loss"].append(metrics[f"val_target_loss"][-1] + metrics[f"val_source_xent"][-1])  
        writer.add_scalar("val/val_loss", metrics[f"val_loss"][-1], epoch)
        
        # test evaluation (acc, acc_alt, auroc)
        head_accs, head_accs_alt, head_aurocs = eval(net, target_test_loader, conf)
        for i in range(conf.heads):
            metrics[f"epoch_test_acc_{i}"].append(head_accs[i])
            metrics[f"epoch_test_acc_{i}_alt"].append(head_accs_alt[i])
            metrics[f"epoch_test_auroc_{i}"].append(head_aurocs[i])
            writer.add_scalar(f"val/test_acc_{i}", head_accs[i], epoch)
            writer.add_scalar(f"val/test_acc_{i}_alt", head_accs_alt[i], epoch)
            writer.add_scalar(f"val/test_auroc_{i}", head_aurocs[i], epoch)
        
        # print validation losses and test accs
        print(f"Epoch {epoch + 1} Test Accuracies:")
        print(f"Target val div loss: {metrics[f'val_target_div_loss'][-1]:.4f}")
        print(f"Target val weighted div loss: {metrics[f'val_target_weighted_div_loss'][-1]:.4f}")
        print(f"Source val xent: {metrics[f'val_source_xent'][-1]:.4f}")
        print(f"val loss: {metrics[f'val_loss'][-1]:.4f}")
        for i in range(conf.heads):
            print(f"Head {i}: {metrics[f'epoch_test_acc_{i}'][-1]:.4f}, Alt: {metrics[f'epoch_test_acc_{i}_alt'][-1]:.4f}")
            print(f"Head {i} auroc: {metrics[f'epoch_test_auroc_{i}'][-1]:.4f}")
        
        net.train()

metrics = dict(metrics)
# save metrics 
import json 
with open(f"{conf.exp_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
    

