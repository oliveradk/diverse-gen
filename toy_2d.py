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
    os.environ["CUDA_VISIBLE_DEVICES"] = "" #"1"
    # os.environ['CUDA_LAUNCH_BLOCKING']="1"
    # os.environ['TORCH_USE_CUDA_DSA'] = "1"


# # 2D Grid

# In[ ]:


import os
import sys
from collections import defaultdict
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf


sys.path.insert(1, os.path.join(sys.path[0], ".."))
from losses.divdis import DivDisLoss
from losses.ace import ACELoss
from losses.dbat import DBatLoss
from losses.loss_types import LossType

from toy_data.grid import generate_data, plot_data, sample_minibatch, savefig


# In[ ]:


# TODO: add focal loss to ERM and ACE loss 
# for ACE, do (1-P_1*P_2)


# In[ ]:


from dataclasses import dataclass 
@dataclass
class Config():
    seed: int = 45 
    loss_type: LossType = LossType.PROB
    train_size: int = 500 
    test_size: int = 5000
    batch_size: int = 32 
    test_batch_size: int = 100 
    train_iter: int = 3000 
    heads: int = 2 
    aux_weight: int = 1.0
    mix_rate: Optional[float] = 1.0
    l_01_mix_rate: Optional[float] = None # TODO: geneneralize
    l_10_mix_rate: Optional[float] = None
    gamma: Optional[float] = 1.0
    lr: float = 1e-3
    train_separate: bool = False

    def __post_init__(self):
        # set mix rate
        if self.l_01_mix_rate is not None and self.l_10_mix_rate is None:
            self.l_10_mix_rate = 0.0
            if self.mix_rate is None:
                self.mix_rate = self.l_01_mix_rate
            assert self.mix_rate == self.l_01_mix_rate
        elif self.l_01_mix_rate is None and self.l_10_mix_rate is not None:
            self.l_01_mix_rate = 0.0
            if self.mix_rate is None:
                self.mix_rate = self.l_10_mix_rate
            assert self.mix_rate == self.l_10_mix_rate
        elif self.l_01_mix_rate is not None and self.l_10_mix_rate is not None:
            if self.mix_rate is None:
                self.mix_rate = self.l_01_mix_rate + self.l_10_mix_rate
            assert self.mix_rate == self.l_01_mix_rate + self.l_10_mix_rate
        else: # both are none 
            assert self.mix_rate is not None
            self.l_01_mix_rate = self.mix_rate / 2
            self.l_10_mix_rate = self.mix_rate / 2


# In[ ]:


# initialize config 
conf = Config()
#get config overrides if runnign from command line
if not is_notebook():
    import sys 
    conf_dict = OmegaConf.merge(OmegaConf.structured(conf), OmegaConf.from_cli(sys.argv[1:]))
    conf = Config(**conf_dict)


# In[ ]:


torch.manual_seed(conf.seed)
np.random.seed(conf.seed)


# In[ ]:


test_data = generate_data(5000, mix_rate=0.1)
plot_data(test_data)


# In[ ]:


def get_exp_name(conf: Config):
    mix_rate_str = f"mix_{conf.mix_rate}" if conf.mix_rate is not None else f"l01_{conf.l_01_mix_rate}_l10_{conf.l_10_mix_rate}"
    gamma_str = f"_gamma_{conf.gamma}" if conf.gamma is not None else ""
    return f"{conf.loss_type}_h{conf.heads}_w{conf.aux_weight}_{mix_rate_str}{gamma_str}_tr_s{conf.train_size}_te_s{conf.test_size}_b{conf.batch_size}_b_te{conf.test_batch_size}_lr{conf.lr}_separate{conf.train_separate}"


# In[ ]:


conf.mix_rate


# In[ ]:


exp_name = get_exp_name(conf)
os.makedirs(f"figures/temp/{exp_name}", exist_ok=True)

fig_save_times = sorted(
    [1, 2, 3, 4, 8, 16, 32, 64, 120, 128] + [50 * n for n in range(conf.train_iter // 50)]
)

training_data = generate_data(500, train=True)
quad_proportions = [conf.l_01_mix_rate, (1-conf.mix_rate)/2, conf.l_10_mix_rate, (1-conf.mix_rate)/2]
test_data = generate_data(5000, quadrant_proportions=quad_proportions)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU(), nn.Linear(40, conf.heads)
)
opt = torch.optim.Adam(net.parameters())
if conf.loss_type == LossType.DIVDIS:
    loss_fn = DivDisLoss(heads=conf.heads)
else:
    loss_fn = ACELoss(
        heads=conf.heads, 
        mode=conf.loss_type.value, 
        gamma=conf.gamma,
        l_01_rate=conf.l_01_mix_rate, 
        l_10_rate=conf.l_10_mix_rate, 
    )


# In[ ]:


def plot_pred_grid(time=""):
    N = 20
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    xv, yv = np.meshgrid(x, y)
    inpt = torch.tensor(np.stack([xv.reshape(-1), yv.reshape(-1)], axis=-1)).float()
    with torch.no_grad():
        preds = net(inpt).reshape(N, N, conf.heads).sigmoid().cpu()

    tr_x, tr_y = training_data
    for i in range(conf.heads):
        plt.figure(figsize=(4, 4))
        plt.contourf(xv, yv, preds[:, :, i], cmap="RdBu", alpha=0.75)
        for g, c in [(0, "#E7040F"), (1, "#00449E")]:
            tr_g = tr_x[tr_y.flatten() == g]
            plt.scatter(tr_g[:, 0], tr_g[:, 1], zorder=10, s=10, c=c, edgecolors="k")
        plt.xlim(-1.0, 1.0)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        savefig(f"temp/{exp_name}/{time}_h{i}", transparent=True)

def plot_head_disagreement(time=""):
    N = 20
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    xv, yv = np.meshgrid(x, y)
    inpt = torch.tensor(np.stack([xv.reshape(-1), yv.reshape(-1)], axis=-1)).float()
    with torch.no_grad():
        preds = net(inpt).reshape(N, N, conf.heads).sigmoid().cpu()
    plt.figure(figsize=(4, 4))
    plt.contourf(xv, yv, torch.abs(preds[:, :, 0] - preds[:, :, 1]), cmap="RdBu", alpha=0.75)
    plt.xlim(-1.0, 1.0)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    savefig(f"temp/{exp_name}/{time}_disagreement", transparent=True)


# In[ ]:


#%%
metrics = defaultdict(list)
for t in range(conf.train_iter):
    x, y = sample_minibatch(training_data, conf.batch_size)
    logits = net(x)
    logits_chunked = torch.chunk(logits, conf.heads, dim=-1)
    losses = [F.binary_cross_entropy_with_logits(logit, y) for logit in logits_chunked]
    xent = sum(losses)

    target_x, target_y = sample_minibatch(test_data, conf.test_batch_size)
    target_logits = net(target_x)
    repulsion_loss = loss_fn(target_logits)

    full_loss = xent + conf.aux_weight * repulsion_loss
    opt.zero_grad()
    full_loss.backward()
    opt.step()

    for i in range(conf.heads):
        corrects_i = (target_logits[:, i] > 0) == target_y.flatten()
        acc_i = corrects_i.float().mean()
        metrics[f"acc_{i}"].append(acc_i.item())
    metrics[f"xent"].append(xent.item())
    metrics[f"repulsion_loss"].append(repulsion_loss.item())

    if t in fig_save_times:
        print(f"Generating plots for {t}/{conf.train_iter}")
        plot_pred_grid(t)
        plot_head_disagreement(t)


# In[ ]:


#%% Train single ERM model (for comparison in learning curve)
net = nn.Sequential(
    nn.Linear(2, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU(), nn.Linear(40, conf.heads)
)
opt = torch.optim.Adam(net.parameters())

for t in range(conf.train_iter):
    x, y = sample_minibatch(training_data, conf.batch_size)
    logits = net(x)
    logits_chunked = torch.chunk(logits, conf.heads, dim=-1)
    losses = [F.binary_cross_entropy_with_logits(logit, y) for logit in logits_chunked]
    full_loss = sum(losses)
    opt.zero_grad()
    full_loss.backward()
    opt.step()

    target_x, target_y = sample_minibatch(test_data, conf.test_batch_size)
    target_logits = net(target_x)
    for i in range(conf.heads):
        corrects_i = (target_logits[:, i] > 0) == target_y.flatten()
        acc_i = corrects_i.float().mean()
        metrics[f"ERM_acc_{i}"].append(acc_i.item())
        print(acc_i.item())


# In[ ]:


#%% Draw learning curves
def draw_full_curve(t=None, with_erm=False):
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))
    N = 10
    uniform = np.ones(N) / N
    axs[0].set_xlim(-10, 1000)
    axs[0].set_ylim(0.45, 1.05)
    smooth = lambda x: np.convolve(x, uniform, mode="valid")
    for i in [0, 1]:
        axs[0].plot(smooth(metrics[f"acc_{i}"]), alpha=0.8, linewidth=2)
    if with_erm:
        axs[0].plot(smooth(metrics["ERM_acc_0"]), c="dimgray", alpha=0.5, linewidth=2)
    axs[1].plot(smooth(metrics["xent"]), c="dimgray")
    axs[2].plot(smooth(metrics["repulsion_loss"]), c="dimgray")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Cross-Entropy")
    axs[2].set_ylabel("MI")
    for ax in axs:
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["left"].set_linewidth(1.2)
        ax.xaxis.set_tick_params(width=1.2)
        ax.yaxis.set_tick_params(width=1.2)
        ax.spines["top"].set_color("none")
        ax.spines["right"].set_color("none")
    if t:
        for ax in axs:
            ax.axvline(x=t, c="k")


draw_full_curve()
savefig(f"temp/{exp_name}/learning_curve_full")

draw_full_curve(with_erm=True)
savefig(f"temp/{exp_name}/learning_curve_full_with_ERM")

for t in fig_save_times:
    draw_full_curve(t=t)
    savefig(f"temp/{exp_name}/learning_curve_full_{t}")

plt.figure(figsize=(8, 2))
N = 10
uniform = np.ones(N) / N
plt.ylim(0.45, 1.05)
smooth = lambda x: np.convolve(x, uniform, mode="valid")
ax = plt.gca()
for i in [0, 1]:
    ax.plot(smooth(metrics[f"acc_{i}"]), alpha=0.8, linewidth=2)
ax.plot(smooth(metrics["ERM_acc_0"]), c="dimgray", alpha=0.5, linewidth=2)
ax.set_ylabel("Accuracy")
ax.spines["bottom"].set_linewidth(1.2)
ax.spines["left"].set_linewidth(1.2)
ax.xaxis.set_tick_params(width=1.2)
ax.yaxis.set_tick_params(width=1.2)
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
savefig(f"temp/{exp_name}/learning_curve_with_ERM")


# In[ ]:


#%% Stitch figures into gifs
import imageio
os.makedirs("gifs", exist_ok=True)

filenames = [f"figures/temp/{exp_name}/{t}_h0.png" for t in fig_save_times]
images = [imageio.imread(filename) for filename in filenames]
gif_head_0_filename = f"gifs/{exp_name}_h0.gif"
imageio.mimsave(gif_head_0_filename, images)

filenames = [f"figures/temp/{exp_name}/{t}_h1.png" for t in fig_save_times]
images = [imageio.imread(filename) for filename in filenames]
gif_head_1_filename = f"gifs/{exp_name}_h1.gif"
imageio.mimsave(gif_head_1_filename, images)

filenames = [f"figures/temp/{exp_name}/{t}_disagreement.png" for t in fig_save_times]
images = [imageio.imread(filename) for filename in filenames]
gif_disagreement_filename = f"gifs/{exp_name}_disagreement.gif"
imageio.mimsave(gif_disagreement_filename, images)

filenames = [
    f"figures/temp/{exp_name}/learning_curve_full_{t}.png" for t in fig_save_times
]
images = [imageio.imread(filename) for filename in filenames]
gif_curve_filename = f"gifs/{exp_name}_curve.gif"
imageio.mimsave(gif_curve_filename, images)

print("GIF creation complete! Files are in:")
for fn in [gif_head_0_filename, gif_head_1_filename, gif_curve_filename]:
    print(fn)
