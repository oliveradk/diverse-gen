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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" #"1"
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
from losses.conf import ConfLoss
from losses.dbat import DBatLoss
from losses.loss_types import LossType

from models.backbone import MultiHeadBackbone
from models.multi_model import MultiNetModel
from models.lenet import LeNet

from datasets.cifar_mnist import get_cifar_mnist_datasets
from datasets.fmnist_mnist import get_fmnist_mnist_datasets
from datasets.toy_grid import get_toy_grid_datasets

from config import Config, post_init


# In[ ]:


# TODO: add dbat 
# TODO: add other vision datasets 
# TODO: add language datasets 


# In[ ]:


conf = Config(
    seed=45,
    dataset="cifar_mnist",
    loss_type=LossType.TOPK,
    batch_size=32,
    target_batch_size=128,
    epochs=10,
    heads=2,
    model="Resnet50",
    shared_backbone=True,
    source_weight=1.0,
    aux_weight=1.0,
    source_mix_rate=0.0,
    source_l_01_mix_rate=None,
    source_l_10_mix_rate=None,
    mix_rate=0.5,
    l_01_mix_rate=None,
    l_10_mix_rate=None,
    mix_rate_lower_bound=0.5,
    all_unlabeled=False,
    inbalance_ratio=False,
    lr=1e-3,
    weight_decay=1e-3,
    lr_scheduler=None,
    num_cycles=0.5,
    frac_warmup=0.05,
    device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    exp_dir=f"output/dominos/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
)


# In[ ]:


# # toy grid configs 
# conf.dataset = "toy_grid"
# conf.model = "toy_model"
# conf.epochs = 100


# In[ ]:


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
from dataclasses import asdict
exp_dir = conf.exp_dir
os.makedirs(exp_dir, exist_ok=True)

# save full config to exp_dir
with open(f"{exp_dir}/config.yaml", "w") as f:
    OmegaConf.save(config=conf, f=f)


# In[ ]:


torch.manual_seed(conf.seed)
np.random.seed(conf.seed)


# In[ ]:


model_transform = None
pad_sides = False
if conf.model == "Resnet50":
    from torchvision import models
    from torchvision.models.resnet import ResNet50_Weights
    resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model_builder = lambda: torch.nn.Sequential(*list(resnet50.children())[:-1])
    resnet_50_transforms = ResNet50_Weights.DEFAULT.transforms()
    model_transform = transforms.Compose([
        transforms.Resize(resnet_50_transforms.resize_size * 2, interpolation=resnet_50_transforms.interpolation),
        transforms.CenterCrop(resnet_50_transforms.crop_size),
        transforms.Normalize(mean=resnet_50_transforms.mean, std=resnet_50_transforms.std)
    ])
    pad_sides = True
    feature_dim = 2048
elif conf.model == "ClipViT":
    # from models.clip_vit import ClipViT
    # model_builder = lambda: ClipViT()
    # feature_dim = 768
    # input_size = 96
    # model_transform = transforms.Compose([
    #     transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
    # ])
    import clip 
    clip_model, preprocess = clip.load('ViT-B/32', device='cpu')
    model_builder = lambda: clip_model.visual
    model_transform = transforms.Compose([
        preprocess.transforms[0],
        preprocess.transforms[1],
        preprocess.transforms[4]
    ])
    feature_dim = 512
    pad_sides = True
elif conf.model == "toy_model":
    model_builder = lambda: nn.Sequential(
        nn.Linear(2, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU()
    )
    feature_dim = 40
elif conf.model == "LeNet":
    from models.lenet import LeNet
    from functools import partial
    model_builder = lambda: partial(LeNet, num_classes=1, dropout_p=0.0)
    feature_dim = 256
else: 
    raise ValueError(f"Model {conf.model} not supported")


# In[ ]:


if conf.dataset == "cifar_mnist":
    source_train, source_val, target_train, target_val, target_test = get_cifar_mnist_datasets(
        source_mix_rate_0_1=conf.source_l_01_mix_rate, 
        source_mix_rate_1_0=conf.source_l_10_mix_rate, 
        target_mix_rate_0_1=conf.l_01_mix_rate, 
        target_mix_rate_1_0=conf.l_10_mix_rate, 
        transform=model_transform, 
        pad_sides=pad_sides
    )
elif conf.dataset == "fmnist_mnist":
    source_train, source_val, target_train, target_val, target_test = get_fmnist_mnist_datasets(
        source_mix_rate_0_1=conf.source_l_01_mix_rate, 
        source_mix_rate_1_0=conf.source_l_10_mix_rate, 
        target_mix_rate_0_1=conf.l_01_mix_rate, 
        target_mix_rate_1_0=conf.l_10_mix_rate, 
        transform=model_transform, 
        pad_sides=pad_sides
    )
elif conf.dataset == "toy_grid":
    source_train, source_val, target_train, target_val, target_test = get_toy_grid_datasets(
        source_mix_rate_0_1=conf.source_l_01_mix_rate, 
        source_mix_rate_1_0=conf.source_l_10_mix_rate, 
        target_mix_rate_0_1=conf.l_01_mix_rate, 
        target_mix_rate_1_0=conf.l_10_mix_rate, 
    )
else:
    raise ValueError(f"Dataset {conf.dataset} not supported")


# In[ ]:


source_labels = [source_train[i][1:3] for i in range(len(source_train))]
assert all([l[1][0] == l[1][1] for l in source_labels])
target_labels = [target_train[i][1:3] for i in range(len(target_train))]
np.mean([l[1][0] == l[1][1] for l in target_labels])


# In[ ]:


# plot image 
img = source_train[0][0]
if img.dim() == 3:
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


# In[ ]:


# plot target train images with vision_utils.make_grid
if img.dim() == 3:
    cifar_mnist_grid = torch.stack([target_train[i][0] for i in range(20)])
    grid_img = vision_utils.make_grid(cifar_mnist_grid, nrow=10, normalize=True, padding=1)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()


# In[ ]:


# data loaders 
source_train_loader = DataLoader(source_train, batch_size=conf.batch_size, shuffle=True)
source_val_loader = DataLoader(source_val, batch_size=conf.batch_size, shuffle=True)
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
elif conf.loss_type == LossType.CONF:
    loss_fn = ConfLoss()
elif conf.loss_type in [LossType.TOPK, LossType.EXP, LossType.PROB]:
    loss_fn = ACELoss(
        heads=conf.heads, 
        mode=conf.loss_type.value, 
        inbalance_ratio=conf.inbalance_ratio,
        l_01_rate=conf.mix_rate_lower_bound / 2, 
        l_10_rate=conf.mix_rate_lower_bound / 2, 
        all_unlabeled=conf.all_unlabeled,
        device=conf.device
    )
else:
    raise ValueError(f"Loss type {conf.loss_type} not supported")


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


def get_acts_and_labels(model: nn.Module, loader: DataLoader):
    activations = []
    labels = []
    model = model_builder()
    model = model.to(conf.device)
    for x, y, gl in tqdm(loader):
        x, y, gl = x.to(conf.device), y.to(conf.device), gl.to(conf.device)
        acts = model(x)
        activations.append((acts.detach().cpu()))
        labels.append(gl)
    activations = torch.cat(activations, dim=0).squeeze()
    labels = torch.cat(labels, dim=0)
    labels = labels.squeeze()
    return activations, labels


# In[ ]:


# visualize data using first two principle componets of final layer activations
model = model_builder()
model = model.to(conf.device)
activations, labels = get_acts_and_labels(model, target_test_loader)


# In[ ]:


from sklearn.decomposition import PCA
if is_notebook():
    pca = PCA(n_components=2)
    pca.fit(activations)
    activations_pca = pca.transform(activations)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot first label
    scatter1 = ax1.scatter(activations_pca[:, 0], activations_pca[:, 1], c=labels[:, 0].to('cpu'), cmap="viridis")
    ax1.set_title('Label 0')

    # Plot second label
    scatter2 = ax2.scatter(activations_pca[:, 0], activations_pca[:, 1], c=labels[:, 1].to('cpu'), cmap="viridis")
    ax2.set_title('Label 1')

    plt.tight_layout()
    plt.show()


# In[ ]:


if is_notebook():
    group_labels = labels[:, 0] * 2 + labels[:, 1]
    plt.scatter(activations_pca[:, 0], activations_pca[:, 1], c=group_labels.to('cpu'), cmap="viridis")
    plt.title("Group labels")
    plt.show()


# In[ ]:


from sklearn.linear_model import LogisticRegression
if is_notebook():
    component_range = [2**i for i in range(1, 9)]
    component_range = [i for i in component_range if i <= feature_dim]
    n_components_accs = []
    for n_components in tqdm(component_range):
        pca = PCA(n_components=n_components)
        pca.fit(activations)
        activations_pca = pca.transform(activations)
        # fit probe 
        lr = LogisticRegression(max_iter=1000)
        lr.fit(activations_pca, labels[:, 0].to('cpu').numpy())
        acc = lr.score(activations_pca, labels[:, 0].to('cpu').numpy())
        n_components_accs.append(acc)
    plt.plot(component_range, n_components_accs, label="accuracy")
    plt.show()



# In[ ]:


# fit linear probe 
if is_notebook():
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=10000)
    lr.fit(activations.to('cpu').numpy(), labels[:, 0].to('cpu').numpy())
    # get accuracy 
    acc = lr.score(activations.to('cpu').numpy(), labels[:, 0].to('cpu').numpy())
    print(f"Accuracy: {acc:.4f}")


# In[ ]:


if is_notebook():
    fig = plt.figure(figsize=(12, 5))
    # Second 3D plot for group labels
    ax3 = fig.add_subplot(121, projection='3d')
    scatter3 = ax3.scatter(activations_pca[:, 0], activations_pca[:, 1], activations_pca[:,2], 
                        c=group_labels.to('cpu'), cmap="viridis")
    ax3.view_init(25, 210, 0)
    ax3.set_title('Group labels')


# In[ ]:


metrics = defaultdict(list)
target_iter = iter(target_train_loader)
for epoch in range(conf.epochs):
    for batch_idx, (x, y, gl) in tqdm(enumerate(source_train_loader), desc="Source train", total=len(source_train_loader)):
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
        repulsion_loss = loss_fn(target_logits)
        # full loss 
        full_loss = conf.source_weight * xent + conf.aux_weight * repulsion_loss
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
                repulsion_loss_val = loss_fn(logits_val)
                repulsion_losses_val.append(repulsion_loss_val.item())
                weighted_repulsion_losses_val.append(conf.aux_weight * repulsion_loss_val.item())
        metrics[f"target_val_repulsion_loss"].append(np.mean(repulsion_losses_val))
        metrics[f"target_val_weighted_repulsion_loss"].append(np.mean(weighted_repulsion_losses_val))
        # compute xent on source validation set
        xent_val = []
        with torch.no_grad():
            for x, y, gl in tqdm(source_val_loader, desc="Source val"):
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
                # test_logits = net(test_x).squeeze() #
                test_logits = net(test_x)
                assert test_logits.shape == (test_x.size(0), conf.heads)
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
plt.savefig(f"{exp_dir}/acc.png")


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
plt.savefig(f"{exp_dir}/max_acc_model_selection.png")
