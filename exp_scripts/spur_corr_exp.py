import os
import copy
from typing import Optional, Callable
from tqdm import tqdm
from functools import partial
from datetime import datetime
from dataclasses import dataclass 
from itertools import product, cycle
from dataclasses import field
import sys 

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

import torch.optim.lr_scheduler as lr_scheduler

from diverse_gen.losses.divdis import DivDisLoss 
from diverse_gen.losses.ace import ACELoss, MixRateScheduler
from diverse_gen.losses.conf import ConfLoss
from diverse_gen.losses.dbat import DBatLoss
from diverse_gen.losses.pass_through import PassThroughLoss
from diverse_gen.losses.src import SrcLoss
from diverse_gen.losses.loss_types import LossType

from diverse_gen.models.backbone import MultiHeadBackbone
from diverse_gen.models.multi_model import MultiNetModel
from diverse_gen.models.get_model import get_model

from diverse_gen.datasets.get_dataset import get_dataset

from diverse_gen.utils.divis_batch_sampler import DivisibleBatchSampler

from diverse_gen.utils.utils import to_device, feature_label_ls, str_to_tuple
from diverse_gen.utils.logger import Logger
from diverse_gen.utils.exp_utils import get_current_commit_hash


@dataclass
class Config():
    seed: int = 1
    dataset: str = "waterbirds"
    loss_type: LossType = LossType.TOPK
    # training 
    batch_size: int = 32
    target_batch_size: int = 64
    epochs: int = 5
    heads: int = 2
    binary: bool = False
    model: str = "Resnet50"
    shared_backbone: bool = True
    source_weight: float = 1.0
    aux_weight: float = 1.0
    use_group_labels: bool = False
    freeze_heads: bool = False
    head_1_epochs: int = 5
    # dataset
    source_cc: bool = True
    source_val_split: float = 0.2
    target_val_split: float = 0.2
    mix_rate: Optional[float] = 0.5
    shuffle_target: bool = True
    dataset_length: Optional[int] = None
    max_length: int = 128  # for text datasets
    combine_neut_entail: bool = True # for multi-nli
    contra_no_neg: bool = False # for multi-nli
    # topk # TODO: generalize properly configure group mix rates for MLI
    aggregate_mix_rate: bool = True
    mix_rate_lower_bound: Optional[float] = 0.5
    mix_rate_lower_bound_01: Optional[float] = None
    mix_rate_lower_bound_10: Optional[float] = None
    group_mix_rate_lower_bounds: Optional[dict[str, float]] = None # field(default_factory=lambda: {"0_1": 0.1, "1_0": 0.1})
    disagree_only: bool = True
    mix_rate_schedule: Optional[str] = "linear"
    mix_rate_t0: Optional[int] = 0
    mix_rate_t1: Optional[int] = 5
    mix_rate_interval_frac: Optional[float] = None # for mix rate updates within epoch
    # optimizer 
    lr: float = 1e-4
    weight_decay: float = 1e-3 # 1e-4
    optimizer: str = "adamw"
    lr_scheduler: Optional[str] = None 
    num_cycles: float = 0.5
    frac_warmup: float = 0.05
    # misc
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    exp_dir: str = f"output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    plot_activations: bool = False

def post_init(conf: Config, overrides: list[str]=[]):
    if conf.freeze_heads and "head_1_epochs" not in overrides:
        conf.head_1_epochs = round(conf.epochs / 2)
    
    # set group mix rate lower bounds based on 01 10 (kinda hacky for doing hparam searches)
    if conf.mix_rate_lower_bound_01 is not None or conf.mix_rate_lower_bound_10 is not None:
        assert conf.group_mix_rate_lower_bounds is None
        conf.group_mix_rate_lower_bounds = {
            "0_1": conf.mix_rate_lower_bound_01 if conf.mix_rate_lower_bound_01 is not None else 0,
            "1_0": conf.mix_rate_lower_bound_10 if conf.mix_rate_lower_bound_10 is not None else 0,
        }
    
    if conf.group_mix_rate_lower_bounds is not None:
        conf.group_mix_rate_lower_bounds = {str_to_tuple(k): v for k, v in conf.group_mix_rate_lower_bounds.items()}


# init config and get overrides
conf = Config()
overrride_keys = []
overrides = OmegaConf.from_cli(sys.argv[1:])
overrride_keys = overrides.keys()
conf_dict = OmegaConf.merge(OmegaConf.structured(conf), overrides)
conf = Config(**conf_dict)
exp_dir = conf.exp_dir
os.makedirs(exp_dir, exist_ok=True)

# save full config to exp_dir
with open(f"{exp_dir}/config.yaml", "w") as f:
    OmegaConf.save(config=conf, f=f)
post_init(conf, overrride_keys)

# save commit hash
with open(f"{exp_dir}/commit_hash.txt", "w") as f:
    f.write(get_current_commit_hash())


# check final configs
if conf.heads != 2:
    raise ValueError("Only 2 heads currently supported")


# init seeds
torch.manual_seed(conf.seed)
np.random.seed(conf.seed)


# get model
model_dict = get_model(conf.model)
model_builder = model_dict["model_builder"]
model_transform = model_dict["model_transform"]
feature_dim = model_dict["feature_dim"]
pad_sides = model_dict["pad_sides"]
tokenizer = model_dict["tokenizer"]

# get dataset
dataset_dict = get_dataset(
    dataset_name=conf.dataset, 
    mix_rate=conf.mix_rate, 
    source_cc=conf.source_cc, 
    source_val_split=conf.source_val_split, 
    target_val_split=conf.target_val_split, 
    model_transform=model_transform, 
    tokenizer=tokenizer, 
    dataset_length=conf.dataset_length, 
    pad_sides=pad_sides, 
    max_length=conf.max_length, 
    use_group_labels=conf.use_group_labels, 
    combine_neut_entail=conf.combine_neut_entail, 
    contra_no_neg=conf.contra_no_neg
)
source_train = dataset_dict["source_train"]
source_val = dataset_dict["source_val"]
target_train = dataset_dict["target_train"]
target_val = dataset_dict["target_val"]
target_test = dataset_dict["target_test"]
is_img = dataset_dict["is_img"]
classes_per_feat = dataset_dict["classes_per_feat"]

# set classes per head
# gt label + binary heads
classes_per_head = [classes_per_feat[0]] + [2 for _ in classes_per_feat[1:]]
# set ood groups
# TODO: make this overridable? and handle multi-class nli case?
ood_groups = [
    labels for labels in product(*[range(c) for c in classes_per_head])
    if len(set(labels)) > 1
]
if conf.binary:
    assert all([c == 2 for c in classes_per_head])
    classes_per_head = [1 for c in classes_per_head]

# set loaders
source_train_loader = DataLoader(
    source_train, batch_size=conf.batch_size, num_workers=conf.num_workers, 
    sampler=DivisibleBatchSampler(len(source_train), conf.batch_size, shuffle=True), 
)
if len(source_val) > 0:
    source_val_loader = DataLoader(
        source_val, batch_size=conf.batch_size, num_workers=conf.num_workers, 
        sampler=DivisibleBatchSampler(len(source_val), conf.batch_size, shuffle=False)
    )
# NOTE: shuffle "should" be true, but in divdis code its false, and this leads to substantial changes in worst goup result
target_train_loader = DataLoader(
    target_train, batch_size=conf.target_batch_size, num_workers=conf.num_workers, 
    sampler=DivisibleBatchSampler(len(target_train), conf.target_batch_size, shuffle=conf.shuffle_target)
)
if len(target_val) > 0:
    target_val_loader = DataLoader(
        target_val, batch_size=conf.target_batch_size, num_workers=conf.num_workers, 
        sampler=DivisibleBatchSampler(len(target_val), conf.target_batch_size, shuffle=False)
    )
target_test_loader = DataLoader(
    target_test, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=False
)

# model
if conf.shared_backbone:
    net = MultiHeadBackbone(model_builder(), classes_per_head, feature_dim)
else:
    net = MultiNetModel(model_builder=model_builder, classes_per_head=classes_per_head, feature_dim=feature_dim)
net = net.to(conf.device)

# optimizer
if conf.optimizer == "adamw":
    opt = torch.optim.AdamW(net.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
elif conf.optimizer == "sgd":
    opt = torch.optim.SGD(net.parameters(), lr=conf.lr, weight_decay=conf.weight_decay, momentum=0.9)
else: 
    raise ValueError(f"Optimizer {conf.optimizer} not supported")
num_steps = conf.epochs * len(source_train_loader)
if conf.lr_scheduler == "cosine":       
    scheduler = get_cosine_schedule_with_warmup(
        opt, 
        num_warmup_steps=num_steps * conf.frac_warmup, 
        num_training_steps=num_steps, 
        num_cycles=conf.num_cycles
    )
else: 
    # constant learning rate
    scheduler = None


# loss function
src_loss_fn = SrcLoss(
    binary=conf.binary, classes_per_head=classes_per_head, use_group_labels=conf.use_group_labels
)
if conf.loss_type == LossType.DIVDIS:
    loss_fn = DivDisLoss(heads=conf.heads)
elif conf.loss_type == LossType.DBAT:
    loss_fn = DBatLoss(heads=conf.heads, n_classes=classes_per_head[0])
elif conf.loss_type == LossType.CONF:
    loss_fn = ConfLoss()
elif conf.loss_type == LossType.ERM:
    loss_fn = PassThroughLoss()
elif conf.loss_type == LossType.TOPK:
    if conf.aggregate_mix_rate:
        group_mix_rates = None
    elif conf.group_mix_rate_lower_bounds is not None:
        group_mix_rates = conf.group_mix_rate_lower_bounds
    else:
        group_mix_rates = {
            group: conf.mix_rate_lower_bound / len(ood_groups) 
            for group in ood_groups
        }

    loss_fn = ACELoss(
        mix_rate=conf.mix_rate_lower_bound,
        classes_per_head=classes_per_head,
        mode=conf.loss_type.value, 
        minority_groups=ood_groups,
        group_mix_rates=group_mix_rates,
        disagree_only=conf.disagree_only,
        device=conf.device
    )
    mix_rate_scheduler = None
    if conf.mix_rate_schedule == "linear":
        mix_rate_scheduler = MixRateScheduler(
            loss_fn=loss_fn,
            mix_rate_lb=conf.mix_rate_lower_bound,
            t0=conf.mix_rate_t0,
            t1=conf.mix_rate_t1,
            interval_size=conf.mix_rate_interval_frac,
            total_steps=num_steps
        )
else:
    raise ValueError(f"Loss type {conf.loss_type} not supported")

# copy loss fn for validation (no scheduling)
valid_loss_fn = copy.deepcopy(loss_fn)


def compute_corrects(logits: torch.Tensor, y: torch.Tensor, binary: bool):
    if binary: # NOTE: not currently supported
        return ((logits.squeeze() > 0) == y.flatten()).sum().item()
    else:
        return (logits.argmax(dim=-1) == y).sum().item()


def eval(model, loader, device, loss_fn, use_labels=False, stage: str = "Evaluating"): 
    group_label_ls = feature_label_ls(classes_per_feat)
    
    # loss 
    losses = []

    # accuracy 
    total_corrects_by_groups = {
        group_label: torch.zeros(conf.heads)
        for group_label in group_label_ls
    }
    total_corrects_alt_by_groups = {
        group_label: torch.zeros(conf.heads)
        for group_label in group_label_ls
    }
    total_samples = 0
    total_samples_by_groups = {
        group_label: 0
        for group_label in group_label_ls
    }
    
    with torch.no_grad():
        for x, y, gl in tqdm(loader, desc=stage):
            x, y, gl = to_device(x, y, gl, conf.device)
            logits = model(x)
            # print(test_logits.shape)
            total_samples += logits.size(0)
            if use_labels and loss_fn is not None:
                loss = loss_fn(logits, y, gl)
            elif loss_fn is not None:
                loss = loss_fn(logits)
            else: 
                loss = torch.tensor(0.0, device=conf.device)
            if torch.isnan(loss):
                print(f"Warning: Nan Loss (likely due to batch size and target loss) "
                f"Batch size: {logits.size(0)}, Loss: {conf.loss_type}")
            else:
                losses.append(loss)

            # parition instances into groups based on group labels 
            logits_by_group = {}
            for group_label in group_label_ls:
                group_label_mask = torch.all(gl == torch.tensor(group_label).to(device), dim=1)
                logits_by_group[group_label] = logits[group_label_mask]
            
            for group_label, group_logits in logits_by_group.items():
                num_examples_group = group_logits.size(0)
                total_samples_by_groups[group_label] += num_examples_group
                group_labels = torch.tensor(group_label).repeat(num_examples_group, 1).to(device)
                group_logits_by_head = torch.split(group_logits, classes_per_head, dim=-1)
                for i in range(conf.heads):
                    if conf.use_group_labels:
                        total_corrects_by_groups[group_label][i] += compute_corrects(group_logits_by_head[i], group_labels[:, i], conf.binary)
                    else:
                        total_corrects_by_groups[group_label][i] += compute_corrects(group_logits_by_head[i], group_labels[:, 0], conf.binary)
                        total_corrects_alt_by_groups[group_label][i] += compute_corrects(group_logits_by_head[i], group_labels[:, 1], conf.binary)
    
    total_corrects = torch.stack([gl_corrects for gl_corrects in total_corrects_by_groups.values()], dim=0).sum(dim=0)
    if not conf.use_group_labels:
        total_corrects_alt = torch.stack([gl_corrects for gl_corrects in total_corrects_alt_by_groups.values()], dim=0).sum(dim=0)

    # average metrics
    metrics = {}
    for i in range(conf.heads):
        metrics[f"acc_{i}"] = (total_corrects[i] / total_samples).item()
        if not conf.use_group_labels:
            metrics[f"acc_alt_{i}"] = (total_corrects_alt[i] / total_samples).item()
    
    if loss_fn is not None:
        metrics["loss"] = torch.mean(torch.tensor(losses)).item()
    # group acc per head
    for group_label in group_label_ls:
        for i in range(conf.heads): 
            metrics[f"acc_{i}_{group_label}"] = (total_corrects_by_groups[group_label][i] / total_samples_by_groups[group_label]).item()
            if not conf.use_group_labels:
                metrics[f"acc_alt_{i}_{group_label}"] = (total_corrects_alt_by_groups[group_label][i] / total_samples_by_groups[group_label]).item()
    # worst group acc per head
    for i in range(conf.heads):
        metrics[f"worst_acc_{i}"] = min([metrics[f"acc_{i}_{group_label}"] for group_label in group_label_ls])

    # add group counts 
    for group_label in group_label_ls:
        metrics[f"count_{group_label}"] = total_samples_by_groups[group_label]

    return metrics

def train(
    conf: Config,
    net: nn.Module, 
    source_train_loader: DataLoader, 
    target_train_loader: DataLoader, 
    source_val_loader: DataLoader, 
    target_val_loader: DataLoader, 
    target_test_loader: DataLoader, 
    src_loss_fn: Callable,
    loss_fn: Callable,   
    valid_loss_fn: Callable, 
    classes_per_feat: list[int], 
    mix_rate_scheduler: Optional[MixRateScheduler], 
    logger: Logger
):
    if conf.freeze_heads: # freeze first head (for dbat)
        net.freeze_head(0)
    total_steps = 0
    for epoch in range(conf.epochs):
        train_loader = zip(source_train_loader, cycle(target_train_loader))
        loader_len = len(source_train_loader)
        ### Train
        # epoch mix rate schedule
        if conf.loss_type == LossType.TOPK and conf.mix_rate_schedule == "linear" and conf.mix_rate_interval_frac is None:
            mix_rate_scheduler.step()
        for batch_idx, (source_batch, target_batch) in tqdm(enumerate(train_loader), desc="Source train", total=loader_len):
            # step mix rate schedule
            if conf.loss_type == LossType.TOPK and conf.mix_rate_schedule == "linear" and conf.mix_rate_interval_frac is not None:
                mix_rate_scheduler.step()
            # freeze heads for dbat
            if conf.freeze_heads and epoch == conf.head_1_epochs: 
                net.unfreeze_head(0)
                net.freeze_head(1)
            # source
            x, y, gl = to_device(*source_batch, conf.device)
            logits = net(x)
            losses = src_loss_fn(logits, y, gl)
            xent = torch.mean(torch.stack(losses))
            logger.add_scalar("train", "source_loss", xent.item(), epoch * loader_len + batch_idx)
            
            # target
            target_x, target_y, target_gl = to_device(*target_batch, conf.device)
            target_logits = net(target_x)
            target_loss = loss_fn(target_logits)           
            
            logger.add_scalar("train", "target_loss", target_loss.item(), epoch * loader_len + batch_idx)
            logger.add_scalar("train", "weighted_target_loss", conf.aux_weight * target_loss.item(), epoch * loader_len + batch_idx, to_metrics=False, to_tb=True)
            # don't compute target loss before second head begins training
            if conf.freeze_heads and epoch < conf.head_1_epochs: 
                target_loss = torch.tensor(0.0, device=conf.device)
            
            # full loss 
            full_loss = conf.source_weight * xent + conf.aux_weight * target_loss
            logger.add_scalar("train", "loss", full_loss.item(), epoch * loader_len + batch_idx)
            
            # backprop
            opt.zero_grad()
            full_loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            total_steps += 1
        
        # eval
        if (epoch + 1) % 1 == 0:
            net.eval()
            ### Validation 
            # source
            total_val_loss = 0.0
            total_val_weighted_loss = 0.0
            if len(source_val) > 0:
                src_loss_sum_fn = lambda x, y, gl: sum(src_loss_fn(x, y, gl))
                source_val_metrics = eval(net, source_val_loader, conf.device, src_loss_sum_fn, use_labels=True, stage="Source Val")
                for k, v in source_val_metrics.items():
                    if 'count' not in k:
                        logger.add_scalar("val", f"source_{k}", v, epoch)
                total_val_loss += source_val_metrics["loss"]
                total_val_weighted_loss += total_val_loss
            # target
            weighted_target_val_loss = 0.0
            if len(target_val) > 0:  
                target_val_metrics = eval(net, target_val_loader, conf.device, valid_loss_fn, use_labels=False, stage="Target Val")
                for k, v in target_val_metrics.items():
                    if 'count' not in k:
                        logger.add_scalar("val", f"target_{k}", v, epoch)
                weighted_target_val_loss = target_val_metrics["loss"] * conf.aux_weight
                logger.add_scalar("val", "target_weighted_loss", weighted_target_val_loss, epoch)
                total_val_loss += target_val_metrics["loss"]    
                total_val_weighted_loss += weighted_target_val_loss
            # total val
            logger.add_scalar("val", "loss", total_val_loss, epoch)
            logger.add_scalar("val", "weighted_loss", total_val_weighted_loss, epoch)

            ### Test
            target_test_metrics = eval(net, target_test_loader, conf.device, None, use_labels=False, stage="Target Test")
            for k, v in target_test_metrics.items():
                if 'count' not in k:
                    logger.add_scalar("test", k, v, epoch)
            
            ### Print Results
            print(f"Epoch {epoch + 1} Eval Results:")
            # print validation losses
            if len(source_val) > 0:
                print(f"Source validation loss: {logger.metrics['val_source_loss'][-1]:.4f}")
            if len(target_val) > 0:
                print(f"Target validation loss {logger.metrics['val_target_loss'][-1]:.4f}")
            print(f"Validation loss: {logger.metrics['val_loss'][-1]:.4f}")
            print("\n=== Test Accuracies ===")
            # Overall accuracy for each head
            print("\nOverall Accuracies:")
            for i in range(conf.heads):
                print(f"Head {i}:  Main: {logger.metrics[f'test_acc_{i}'][-1]:.4f}" + \
                      (f"  |  Alt: {logger.metrics[f'test_acc_alt_{i}'][-1]:.4f}" if not conf.use_group_labels else ""))
            # Worst group accuracy for each head
            print("\nWorst Group Accuracies:")
            for i in range(conf.heads):
                print(f"Head {i}:  Worst: {logger.metrics[f'test_worst_acc_{i}'][-1]:.4f}")
            # Group-wise accuracies
            print("\nGroup-wise Accuracies:")
            for group_label in feature_label_ls(classes_per_feat):
                print(f"\nGroup {group_label}, count: {target_test_metrics[f'count_{group_label}']}:")
                for i in range(conf.heads):
                    print(f"Head {i}:  Main: {logger.metrics[f'test_acc_{i}_{group_label}'][-1]:.4f}" + \
                          (f"  |  Alt: {logger.metrics[f'test_acc_alt_{i}_{group_label}'][-1]:.4f}" if not conf.use_group_labels else ""))
            net.train()


logger = Logger(exp_dir)
try: 
    train(
        conf=conf,
        net=net, 
        source_train_loader=source_train_loader, 
        target_train_loader=target_train_loader, 
        source_val_loader=source_val_loader, 
        target_val_loader=target_val_loader, 
        target_test_loader=target_test_loader, 
        src_loss_fn=src_loss_fn, 
        loss_fn=loss_fn, 
        valid_loss_fn=valid_loss_fn, 
        mix_rate_scheduler=(mix_rate_scheduler if conf.loss_type == LossType.TOPK else None), 
        classes_per_feat=classes_per_feat, 
        logger=logger
    )
finally:
    logger.flush()

