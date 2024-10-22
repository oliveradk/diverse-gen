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
from losses.dbat import DBatLoss
from losses.loss_types import LossType

from models.backbone import MultiHeadBackbone
from models.multi_model import MultiNetModel
from models.lenet import LeNet


# In[ ]:


from dataclasses import dataclass 
@dataclass
class Config():
    seed: int = 45
    loss_type: LossType = LossType.PROB
    batch_size: int = 128
    target_batch_size: int = 128
    epochs: int = 20
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
    weight_decay: float = 1e-5
    vertical: bool = True
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
def post_init(conf: Config):
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

    if conf.loss_type == LossType.DIVDIS:
        conf.aux_weight = 10.0
    if conf.loss_type == LossType.EXP: 
        conf.target_batch_size = round(4 / conf.mix_rate)


# In[ ]:


# initialize config 
conf = Config()
#get config overrides if runnign from command line
if not is_notebook():
    import sys 
    conf_dict = OmegaConf.merge(OmegaConf.structured(conf), OmegaConf.from_cli(sys.argv[1:]))
    conf = Config(**conf_dict)
post_init(conf)


# In[ ]:


# create directory from config
def conf_to_dir_name(conf: Config):        
    dir_name = f"{conf.loss_type.name}_{conf.model}_{conf.mix_rate}_{conf.mix_rate_lower_bound}_{conf.epochs}_{conf.seed}"
    return dir_name
cifar_mnist_dir_name = "output/cifar_mnist"
dir_name = f"{cifar_mnist_dir_name}/{conf_to_dir_name(conf)}"
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
    feature_dim = 2048
elif conf.model == "ClipViT":
    from transformers import CLIPVisionModel
    model_builder = lambda: CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
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


# In[ ]:


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(x.repeat(3, 1, 1), (2, 2, 2, 2), 'constant', 0))
])


# get full datasets 
mnist_train = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True, transform=mnist_transform)
cifar_train = torchvision.datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST('./data/mnist/', train=False, download=True, transform=mnist_transform)
cifar_test = torchvision.datasets.CIFAR10('./data/cifar10/', train=False, download=True, transform=transform)

mnist_target, mnist_train, mnist_source_val, mnist_target_val = random_split(mnist_train, [10000, 45000, 2500, 2500], generator=torch.Generator().manual_seed(42))
cifar_target, cifar_train, cifar_source_val, cifar_target_val = random_split(cifar_train, [10000, 35000, 2500, 2500], generator=torch.Generator().manual_seed(42))


# In[ ]:


# function that generates dataset of concatenated cifar and mnist images
# 1-mix_rate is the number of cifar cars and 0's + cifar trucks and 1's (default)
# 0_9 corresponds to cifar trucks and mnist 0's 
# 1_1 corresponds to cifar cars and mnist 1's 
def generate_dataset(mnist_data, cifar_data, mix_rate_0_9, mix_rate_1_1, vertical=False, transform=None):
    # filter by labels
    mnist_0 = [(img, label) for img, label in mnist_data if label == 0]
    mnist_1 = [(img, label) for img, label in mnist_data if label == 1]
    cifar_1 = [(img, label) for img, label in cifar_data if label == 1]
    cifar_9 = [(img, label) for img, label in cifar_data if label == 9]
    # get number of samples
    num_samples = min(len(mnist_0), len(mnist_1), len(cifar_1), len(cifar_9))
    data_pairs = []
    num_clean = int(num_samples * (1-mix_rate_0_9 - mix_rate_1_1)) 
    num_mixed_0_9 = int(num_samples * mix_rate_0_9) 
    num_mixed_1_1 = int(num_samples * mix_rate_1_1) 
    i = 0
    for _ in range(num_clean // 2):
        # cars and 0's
        data_pairs.append(((cifar_1[i][0], mnist_0[i][0]), 0, (0, 0))) 
        # trucks and 1's
        data_pairs.append(((cifar_9[i][0], mnist_1[i][0]), 1, (1, 1)))
        i+=1
    for _ in range(num_mixed_0_9):
        # trucks and 0's
        data_pairs.append(((cifar_9[i][0], mnist_0[i][0]), 1, (1, 0)))
        i+=1
    for _ in range(num_mixed_1_1):
        # cars and 1's
        data_pairs.append(((cifar_1[i][0], mnist_1[i][0]), 0, (0, 1)))
        i+=1
    # construct dataset
    images, labels, group_labels = zip(*data_pairs)
    # concatenate images
    cat_dim = 1 if vertical else 2
    images = [torch.cat([mnist_img, cifar_img], dim=cat_dim) for cifar_img, mnist_img in images]
    images = torch.stack(images)
    # labels and group labels 
    labels = torch.tensor(labels).to(torch.float32)
    group_labels = torch.tensor([list(gl) for gl in group_labels]).to(torch.float32)
    # shuffle dataset
    shuffle = torch.randperm(len(images))
    images = images[shuffle]
    if transform is not None:
        images = transform(images)
    labels = labels[shuffle]
    group_labels = group_labels[shuffle]
    dataset = TensorDataset(images, labels, group_labels)
    return dataset

# generate datasets
source_train = generate_dataset(mnist_train, cifar_train, mix_rate_0_9=0.0, mix_rate_1_1=0.0, vertical=conf.vertical, transform=model_transform)
source_val = generate_dataset(mnist_source_val, cifar_source_val, mix_rate_0_9=0.0, mix_rate_1_1=0.0, vertical=conf.vertical, transform=model_transform)
target_train = generate_dataset(mnist_target, cifar_target, mix_rate_0_9=conf.l_01_mix_rate, mix_rate_1_1=conf.l_10_mix_rate, vertical=conf.vertical, transform=model_transform)
target_val = generate_dataset(mnist_target_val, cifar_target_val, mix_rate_0_9=conf.l_01_mix_rate, mix_rate_1_1=conf.l_10_mix_rate, vertical=conf.vertical, transform=model_transform)
target_test = generate_dataset(mnist_test, cifar_test, mix_rate_0_9=0.25, mix_rate_1_1=0.25, vertical=conf.vertical, transform=model_transform)



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


if conf.shared_backbone:
    net = MultiHeadBackbone(model_builder(), conf.heads, feature_dim)
else:
    net = MultiNetModel(heads=conf.heads, model_builder=model_builder)
net = net.to(conf.device)
opt = torch.optim.AdamW(net.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
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
# data loaders 
source_train_loader = DataLoader(source_train, batch_size=conf.batch_size, shuffle=True)
target_train_loader = DataLoader(target_train, batch_size=conf.target_batch_size, shuffle=True)
target_val_loader = DataLoader(target_val, batch_size=conf.target_batch_size, shuffle=True)
target_test_loader = DataLoader(target_test, batch_size=conf.batch_size, shuffle=True)


# In[ ]:


metrics = defaultdict(list)
target_iter = iter(target_train_loader)
for epoch in range(conf.epochs):
    for x, y, gl in tqdm(source_train_loader, desc="Source train"):
        x, y, gl = x.to(conf.device), y.to(conf.device), gl.to(conf.device)
        logits = net(x)
        logits_chunked = torch.chunk(logits, conf.heads, dim=-1)
        losses = [F.binary_cross_entropy_with_logits(logit.squeeze(), y) for logit in logits_chunked]
        xent = sum(losses)

        try: 
            target_x, target_y, target_gl = next(target_iter)
        except StopIteration:
            target_iter = iter(target_train_loader)
            target_x, target_y, target_gl = next(target_iter)
        target_x, target_y, target_gl = target_x.to(conf.device), target_y.to(conf.device), target_gl.to(conf.device)
        target_logits = net(target_x)

        repulsion_loss_args = []
        repulsion_loss = loss_fn(target_logits, *repulsion_loss_args)
        full_loss = xent + conf.aux_weight * repulsion_loss
        opt.zero_grad()
        full_loss.backward()
        opt.step()

        metrics[f"xent"].append(xent.item())
        metrics[f"repulsion_loss"].append(repulsion_loss.item())
    # Compute loss on target validation set (used for model selection)
    # and aggregate metrics over the entire test set (should not really be using)
    if (epoch + 1) % 1 == 0:
        net.eval()
        # compute repulsion loss on target validation set (used for model selection)
        repulsion_loss_val = []
        with torch.no_grad():
            for x, y, gl in tqdm(target_val_loader, desc="Target val"):
                x, y, gl = x.to(conf.device), y.to(conf.device), gl.to(conf.device)
                logits_val = net(x)
                repulsion_loss_val.append(conf.aux_weight * loss_fn(logits_val, * repulsion_loss_args).item())
        metrics[f"target_val_repulsion_loss"].append(np.mean(repulsion_loss_val))
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
        metrics[f"target_val_loss"].append(np.mean(repulsion_loss_val) + np.mean(xent_val))

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
        print(f"Target val xent: {metrics[f'source_val_xent'][-1]:.4f}")
        print(f"Target val loss: {metrics[f'target_val_loss'][-1]:.4f}")
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
plt.plot(metrics["target_val_repulsion_loss"], label="target_val_repulsion_loss", color="blue")
plt.plot(metrics["target_val_loss"], label="target_val_loss", color="green")
plt.legend()
plt.yscale("log")
plt.show()
if not is_notebook():
    plt.close()


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

