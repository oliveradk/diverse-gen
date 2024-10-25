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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" #"1"
    # os.environ['CUDA_LAUNCH_BLOCKING']="1"
    # os.environ['TORCH_USE_CUDA_DSA'] = "1"

import matplotlib 
if not is_notebook():
    matplotlib.use('Agg')


# In[ ]:


import os
import math
import json
import random as rnd
from typing import Optional, Callable
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from datetime import datetime

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import matplotlib.pyplot as plt
import pandas as  pd
import torchvision.utils as vision_utils
from PIL import Image
import torchvision
from torchvision import transforms
from matplotlib.ticker import NullFormatter

from losses.divdis import DivDisLoss 
from losses.divdis import DivDisLoss
from losses.ace import ACELoss
from losses.dbat import DBatLoss
from losses.loss_types import LossType

from models.backbone import MultiHeadBackbone
from models.multi_model import MultiNetModel
from models.lenet import LeNet

from datasets.cifar_mnist import get_cifar_mnist_datasets
from utils.exp_utils import get_cifar_mnist_exp_dir


# In[ ]:


# TODO: add dbat 
# TODO: add other vision datasets 
# TODO: add language datasets 


# In[ ]:


from dataclasses import dataclass 
@dataclass
class Config():
    seed: int = 45
    loss_type: LossType = LossType.PROB
    batch_size: int = 128
    target_batch_size: int = 128
    epochs: int = 100
    heads: int = 2 
    model: str = "Resnet50"
    shared_backbone: bool = True
    aux_weight: float = 1.0
    mix_rate: Optional[float] = 0.5
    l_01_mix_rate: Optional[float] = None # TODO: geneneralize
    l_10_mix_rate: Optional[float] = None
    gamma: Optional[float] = 1.0
    mix_rate_lower_bound: Optional[float] = 0.5
    inbalance_ratio: Optional[bool] = True
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: Optional[str] = "cosine"# "cosine"
    num_cycles: float = 0.5
    frac_warmup: float = 0.05
    vertical: bool = True
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def post_init(conf: Config, overrides: list[str]=[]):
    if conf.l_01_mix_rate is not None and conf.l_10_mix_rate is None:
        conf.l_10_mix_rate = 0.0
        if conf.mix_rate is None:
            conf.mix_rate = conf.l_01_mix_rate
        assert conf.mix_rate == conf.l_01_mix_rate
    elif conf.l_01_mix_rate is None and conf.l_10_mix_rate is not None:
        conf.l_01_mix_rate = 0.0
        if conf.mix_rate is None:
            conf.mix_rate = conf.l_10_mix_rate
        assert conf.mix_rate == conf.l_10_mix_rate
    elif conf.l_01_mix_rate is not None and conf.l_10_mix_rate is not None:
        if conf.mix_rate is None:
            conf.mix_rate = conf.l_01_mix_rate + conf.l_10_mix_rate
        assert conf.mix_rate == conf.l_01_mix_rate + conf.l_10_mix_rate
    else: # both are none 
        assert conf.mix_rate is not None
        conf.l_01_mix_rate = conf.mix_rate / 2
        conf.l_10_mix_rate = conf.mix_rate / 2
    
    if conf.mix_rate_lower_bound is None:
        conf.mix_rate_lower_bound = conf.mix_rate
    
    if conf.loss_type == LossType.DIVDIS and "aux_weight" not in overrides:
        conf.aux_weight = 10.0
    if conf.model == "ClipViT" and "lr" not in overrides:
        conf.lr = 5e-5
    if conf.model == "ClipViT" and "lr_scheduler" not in overrides:
        conf.lr_scheduler = "cosine"


# In[ ]:


# initialize config 
conf = Config()
#get config overrides if runnign from command line
overrride_keys = []
if not is_notebook():
    import sys 
    overrides = OmegaConf.from_cli(sys.argv[1:])
    overrride_keys = overrides.keys()
    conf_dict = OmegaConf.merge(OmegaConf.structured(conf), overrides)
    conf = Config(**conf_dict)
post_init(conf, overrride_keys)


# In[ ]:


# create directory from config
dir_name = get_cifar_mnist_exp_dir(conf)
datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = f"{dir_name}/{datetime_str}"
os.makedirs(exp_dir, exist_ok=True)

# save full config to exp_dir
with open(f"{exp_dir}/config.yaml", "w") as f:
    OmegaConf.save(config=conf, f=f)


# In[ ]:


torch.manual_seed(conf.seed)
np.random.seed(conf.seed)


# In[ ]:


model_transform = None
if conf.model == "Resnet50":
    from torchvision import models
    from torchvision.models.resnet import ResNet50_Weights
    resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model_builder = lambda: torch.nn.Sequential(*list(resnet50.children())[:-1])
    resnet_50_transforms = ResNet50_Weights.DEFAULT.transforms()
    model_transform = transforms.Compose([
        # transforms.Resize(resnet_50_transforms.resize_size * 2, interpolation=resnet_50_transforms.interpolation),
        # transforms.CenterCrop(resnet_50_transforms.crop_size),
        transforms.Normalize(mean=resnet_50_transforms.mean, std=resnet_50_transforms.std)
    ])
    feature_dim = 2048
elif conf.model == "ClipViT":
    from models.clip_vit import ClipViT
    model_builder = lambda: ClipViT()
    feature_dim = 768
    input_size = 96
    model_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
    ])
elif conf.model == "LeNet":
    from models.lenet import LeNet
    from functools import partial
    model_builder = lambda: partial(LeNet, num_classes=1, dropout_p=0.0)
    feature_dim = 256
else: 
    raise ValueError(f"Model {conf.model} not supported")


# In[ ]:


source_train, source_val, target_train, target_val, target_test = get_cifar_mnist_datasets(
    vertical=conf.vertical, 
    mix_rate_0_9=conf.l_01_mix_rate, 
    mix_rate_1_1=conf.l_10_mix_rate, 
    transform=model_transform
)


# In[ ]:


assert sum([gl[0] == gl[1] for _, _, gl in target_test]) / len(target_test) == 0.5


# In[ ]:


# plot source images with vision_utils.make_grid
cifar_mnist_grid = torch.stack([source_train[i][0] for i in range(20)])
grid_img = vision_utils.make_grid(cifar_mnist_grid, nrow=10, normalize=True, padding=1)
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()


# In[ ]:


# plot target train images with vision_utils.make_grid
cifar_mnist_grid = torch.stack([target_train[i][0] for i in range(20)])
grid_img = vision_utils.make_grid(cifar_mnist_grid, nrow=10, normalize=True, padding=1)
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()


# In[ ]:


# data loaders 
source_train_loader = DataLoader(source_train, batch_size=conf.batch_size, shuffle=True)
target_train_loader = DataLoader(target_train, batch_size=conf.target_batch_size, shuffle=True)
target_val_loader = DataLoader(target_val, batch_size=conf.target_batch_size, shuffle=True)
target_test_loader = DataLoader(target_test, batch_size=conf.batch_size, shuffle=True)

# classifiers
from transformers import get_cosine_schedule_with_warmup
if conf.shared_backbone:
    net = MultiHeadBackbone(model_builder(), conf.heads, feature_dim)
else:
    net = MultiNetModel(heads=conf.heads, model_builder=model_builder)
net = net.to(conf.device)

# optimizer
opt = torch.optim.AdamW(net.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
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
if conf.loss_type == LossType.DIVDIS:
    loss_fn = DivDisLoss(heads=conf.heads)
else:
    loss_fn = ACELoss(
        heads=conf.heads, 
        mode=conf.loss_type.value, 
        gamma=conf.gamma,
        inbalance_ratio=conf.inbalance_ratio,
        l_01_rate=conf.mix_rate_lower_bound / 2, 
        l_10_rate=conf.mix_rate_lower_bound / 2, 
        device=conf.device
    )


# In[ ]:


from losses.ace import compute_head_losses

def get_orderings(logits: torch.Tensor):
    probs = torch.sigmoid(logits)
    head_0_0, head_0_1, head_1_0, head_1_1 = compute_head_losses(probs)
    loss_0_1 = head_0_0 + head_1_1
    loss_1_0 = head_1_0 + head_0_1
    loss_0_1, indices_0_1 = loss_0_1.sort()
    loss_1_0, indices_1_0 = loss_1_0.sort()
    return loss_0_1, loss_1_0, indices_0_1, indices_1_0



# In[ ]:


def compute_accs(logits: torch.Tensor, gl: torch.Tensor):
    with torch.no_grad():
        acc = torch.zeros(conf.heads)
        acc_alt = torch.zeros(conf.heads)
        for i in range(conf.heads):
            acc[i] += ((logits[:, i] > 0) == gl[:, 0].flatten()).to(torch.float32).mean().item()
            acc_alt[i] += ((logits[:, i] > 0) == gl[:, 1].flatten()).to(torch.float32).mean().item()
    return acc, acc_alt


# In[ ]:


metrics = defaultdict(list)
target_iter = iter(target_train_loader)
for epoch in range(conf.epochs):
    for x, y, gl in tqdm(source_train_loader, desc="Source train"):
        x, y, gl = x.to(conf.device), y.to(conf.device), gl.to(conf.device)
        logits = net(x)
        logits_chunked = torch.chunk(logits, conf.heads, dim=-1)
        # source loss 
        losses = [F.binary_cross_entropy_with_logits(logit.squeeze(), y) for logit in logits_chunked]
        xent = sum(losses)
        # target loss 
        try: 
            target_x, target_y, target_gl = next(target_iter)
        except StopIteration:
            target_iter = iter(target_train_loader)
            target_x, target_y, target_gl = next(target_iter)
        target_x, target_y, target_gl = target_x.to(conf.device), target_y.to(conf.device), target_gl.to(conf.device)
        target_logits = net(target_x)
        repulsion_loss_args = []
        repulsion_loss = loss_fn(target_logits, *repulsion_loss_args)
        # log orderings, false positive and false negative rates for each loss
        # fp = number of instances in top batch_size * mix_rate_lower_bound / 2 that don't have (0,1)/(1/0)
        # fn = number of instances not in top batch_size * mix_rate_lower_bound / 2 that have (0,1)/(1/0)
        loss_0_1, loss_1_0, indices_0_1, indices_1_0 = get_orderings(target_logits)
        # metrics[f"target_loss_0_1_ordering"].append(target_gl[indices_0_1].tolist())
        # metrics[f"target_loss_1_0_ordering"].append(target_gl[indices_1_0].tolist())
        k = conf.target_batch_size * conf.mix_rate_lower_bound / 2 
        acc, acc_alt = compute_accs(target_logits, target_gl)
        acc_0_1 = acc[0] + acc_alt[1]
        acc_1_0 = acc[1] + acc_alt[0]
        if acc_0_1 > acc_1_0:
            target_0_1 = torch.tensor([0, 1]).to(conf.device)
            target_1_0 = torch.tensor([1, 0]).to(conf.device)
        else:
            target_0_1 = torch.tensor([1, 0]).to(conf.device)
            target_1_0 = torch.tensor([0, 1]).to(conf.device)   

        fp_0_1 = (target_gl[indices_0_1[:int(k)]] != target_0_1).all(dim=1).float().mean().item()
        fp_1_0 = (target_gl[indices_1_0[:int(k)]] != target_1_0).all(dim=1).float().mean().item()
        fn_0_1 = (target_gl[indices_0_1[int(k):]] == target_0_1).all(dim=1).float().mean().item() 
        fn_1_0 = (target_gl[indices_1_0[int(k):]] == target_1_0).all(dim=1).float().mean().item()
        metrics[f"target_fp_0_1"].append(fp_0_1)
        metrics[f"target_fp_1_0"].append(fp_1_0)
        metrics[f"target_fn_0_1"].append(fn_0_1)
        metrics[f"target_fn_1_0"].append(fn_1_0)
        # full loss 
        full_loss = xent + conf.aux_weight * repulsion_loss
        opt.zero_grad()
        full_loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        metrics[f"xent"].append(xent.item())
        metrics[f"repulsion_loss"].append(repulsion_loss.item())
    # Compute loss on target validation set (used for model selection)
    # and aggregate metrics over the entire test set (should not really be using)
    if (epoch + 1) % 1 == 0:
        net.eval()
        # compute repulsion loss on target validation set (used for model selection)
        repulsion_losses_val = []
        weighted_repulsion_losses_val = []
        with torch.no_grad():
            for x, y, gl in tqdm(target_val_loader, desc="Target val"):
                x, y, gl = x.to(conf.device), y.to(conf.device), gl.to(conf.device)
                logits_val = net(x)
                repulsion_loss_val = loss_fn(logits_val, *repulsion_loss_args)
                repulsion_losses_val.append(repulsion_loss_val.item())
                weighted_repulsion_losses_val.append(conf.aux_weight * repulsion_loss_val.item())
        metrics[f"target_val_repulsion_loss"].append(np.mean(repulsion_losses_val))
        metrics[f"target_val_weighted_repulsion_loss"].append(np.mean(weighted_repulsion_losses_val))
        # compute xent on source validation set
        xent_val = []
        with torch.no_grad():
            for x, y, gl in tqdm(source_train_loader, desc="Source val"):
                x, y, gl = x.to(conf.device), y.to(conf.device), gl.to(conf.device)
                logits_val = net(x)
                logits_chunked_val = torch.chunk(logits_val, conf.heads, dim=-1)
                losses_val = [F.binary_cross_entropy_with_logits(logit.squeeze(), y) for logit in logits_chunked_val]
                xent_val.append(sum(losses_val).item())
        metrics[f"source_val_xent"].append(np.mean(xent_val))
        metrics[f"val_loss"].append(np.mean(repulsion_losses_val) + np.mean(xent_val))
        metrics[f"val_weighted_loss"].append(np.mean(weighted_repulsion_losses_val) + np.mean(xent_val))

        # compute accuracy over target test set (used to evaluate actual performance)
        total_correct = torch.zeros(conf.heads)
        total_correct_alt = torch.zeros(conf.heads)
        total_samples = 0
        
        with torch.no_grad():
            for test_x, test_y, test_gl in target_test_loader:
                test_x, test_y, test_gl = test_x.to(conf.device), test_y.to(conf.device), test_gl.to(conf.device)
                test_logits = net(test_x).squeeze()
                total_samples += test_y.size(0)
                
                for i in range(conf.heads):
                    total_correct[i] += ((test_logits[:, i] > 0) == test_y.flatten()).sum().item()
                    total_correct_alt[i] += ((test_logits[:, i] > 0) == test_gl[:, 1].flatten()).sum().item()
        
        for i in range(conf.heads):
            metrics[f"epoch_acc_{i}"].append((total_correct[i] / total_samples).item())
            metrics[f"epoch_acc_{i}_alt"].append((total_correct_alt[i] / total_samples).item())
        
        print(f"Epoch {epoch + 1} Test Accuracies:")
        # print validation losses
        print(f"Target val repulsion loss: {metrics[f'target_val_repulsion_loss'][-1]:.4f}")
        print(f"Target val weighted repulsion loss: {metrics[f'target_val_weighted_repulsion_loss'][-1]:.4f}")
        print(f"Source val xent: {metrics[f'source_val_xent'][-1]:.4f}")
        print(f"val loss: {metrics[f'val_loss'][-1]:.4f}")
        print(f"val weighted loss: {metrics[f'val_weighted_loss'][-1]:.4f}")
        for i in range(conf.heads):
            print(f"Head {i}: {metrics[f'epoch_acc_{i}'][-1]:.4f}, Alt: {metrics[f'epoch_acc_{i}_alt'][-1]:.4f}")
        
        net.train()

metrics = dict(metrics)
# save metrics 
import json 
with open(f"{exp_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# In[ ]:


plt.plot(metrics["xent"], label="xent", color="red")
plt.plot(metrics["repulsion_loss"], label="repulsion_loss", color="blue")
plt.legend()
plt.yscale("log")
plt.show()
if not is_notebook():
    plt.close()


# In[ ]:


# print loss
# plt.plot(metrics["xent"], label="xent", color="pink")
# plt.plot(metrics["repulsion_loss"], label="repulsion_loss", color="lightblue")
plt.plot(metrics["source_val_xent"], label="source_val_xent", color="red")
plt.plot(metrics["target_val_weighted_repulsion_loss"], label="target_val_repulsion_loss", color="blue")
plt.plot(metrics["val_weighted_loss"], label="val_loss", color="green")
plt.legend()
plt.yscale("log")
plt.show()
if not is_notebook():
    plt.close()
plt.savefig(f"{exp_dir}/val_loss.png")


# In[ ]:


# print metrics
# plot acc_0 and acc_1 and acc_0_alt and acc_1_alt
plt.plot(metrics["epoch_acc_0"], label="acc_0", color="blue")
plt.plot(metrics["epoch_acc_1"], label="acc_1", color="green")
plt.plot(metrics["epoch_acc_0_alt"], label="acc_0_alt", color="lightblue")
plt.plot(metrics["epoch_acc_1_alt"], label="acc_1_alt", color="lightgreen")
plt.legend()
plt.show()
if not is_notebook():
    plt.close()
plt.savefig(f"{exp_dir}/acc.png")


# In[ ]:


# plot false positive and false negative rates
plt.plot(metrics["target_fp_0_1"], label="target_fp_0_1", color="blue")
plt.plot(metrics["target_fp_1_0"], label="target_fp_1_0", color="green")
plt.plot(metrics["target_fn_0_1"], label="target_fn_0_1", color="lightblue")
plt.plot(metrics["target_fn_1_0"], label="target_fn_1_0", color="lightgreen")
plt.legend()
plt.show()
if not is_notebook():
    plt.close()
plt.savefig(f"{exp_dir}/false_positive_false_negative_rates.png")

# TODO: this looks very strange 


# In[ ]:


# find index of minimum val_weighted_loss, target_val_weighted_repulsion_loss 
min_val_weighted_loss_idx = np.argmin(metrics["val_weighted_loss"])
min_target_val_weighted_repulsion_loss_idx = np.argmin(metrics["target_val_weighted_repulsion_loss"])
# get maximum acc (max of max(acc_0, acc_1))
accs = np.maximum(np.array(metrics["epoch_acc_0"]), np.array(metrics["epoch_acc_1"]))
max_acc_idx = np.argmax(accs)
print(f"max_acc_idx: {max_acc_idx}")
print(f"min_val_weighted_loss_idx: {min_val_weighted_loss_idx}")
print(f"min_target_val_weighted_repulsion_loss_idx: {min_target_val_weighted_repulsion_loss_idx}")
# get accs for min val_weighted_loss and min target_val_weighted_repulsion_loss 
val_weighted_loss_acc = accs[min_val_weighted_loss_idx]
target_val_weighted_repulsion_loss_acc = accs[min_target_val_weighted_repulsion_loss_idx]
max_acc = accs[max_acc_idx]

# plot max_acc, val_weighted_loss_acc, target_val_weighted_repulsion_loss_acc as a bar chart 
plt.bar(["max", "weighted_loss", "weighted_repulsion_loss"], [max_acc, val_weighted_loss_acc, target_val_weighted_repulsion_loss_acc])
# show y ticks at every 0.1 
plt.yticks(np.arange(0, 1.1, 0.1))
plt.show()
if not is_notebook():
    plt.close()
plt.savefig(f"{exp_dir}/max_acc_model_selection.png")

