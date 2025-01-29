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
from dataclasses import dataclass 
from itertools import product

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
from sklearn.decomposition import PCA

from losses.divdis import DivDisLoss 
from losses.divdis import DivDisLoss
from losses.ace import ACELoss
from losses.conf import ConfLoss
from losses.dbat import DBatLoss
from losses.pass_through import PassThroughLoss
from losses.smooth_top_loss import SmoothTopLoss
from losses.loss_types import LossType

from models.backbone import MultiHeadBackbone
from models.multi_model import MultiNetModel, freeze_heads
from models.lenet import LeNet

from spurious_datasets.cifar_mnist import get_cifar_mnist_datasets
from spurious_datasets.fmnist_mnist import get_fmnist_mnist_datasets
from spurious_datasets.toy_grid import get_toy_grid_datasets
from spurious_datasets.waterbirds import get_waterbirds_datasets
from spurious_datasets.cub import get_cub_datasets
from spurious_datasets.camelyon import get_camelyon_datasets
from spurious_datasets.multi_nli import get_multi_nli_datasets
from spurious_datasets.civil_comments import get_civil_comments_datasets
from spurious_datasets.celebA import get_celebA_datasets

from utils.utils import to_device, batch_size, feature_label_ls
from utils.logger import Logger
from utils.act_utils import get_acts_and_labels, plot_activations, compute_probe_acc


# # Setup Experiment

# In[ ]:


from dataclasses import field

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
    aux_weight_schedule: Optional[str] = None
    aux_weight_t0: Optional[int] = None
    aux_weight_t1: Optional[int] = None
    use_group_labels: bool = False
    freeze_heads: bool = False
    head_1_epochs: int = 5
    # dataset
    source_cc: bool = True
    source_val_split: float = 0.2
    target_val_split: float = 0.2
    mix_rate: Optional[float] = None
    shuffle_target: bool = True
    dataset_length: Optional[int] = None
    max_length: int = 128  # for text datasets
    combine_neut_entail: bool = False # for multi-nli
    contra_no_neg: bool = True # for multi-nli
    # topk # TODO: generalize properly configure group mix rates for MLI
    aggregate_mix_rate: bool = False
    mix_rate_lower_bound: Optional[float] = 0.1
    group_mix_rate_lower_bounds: Optional[dict[str, float]] = None # field(default_factory=lambda: {"0_1": 0.1, "1_0": 0.1})
    disagree_only: bool = False
    mix_rate_schedule: Optional[str] = None
    mix_rate_t0: Optional[int] = None
    mix_rate_t1: Optional[int] = None
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
    plot_activations: bool = True

def str_to_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(i) for i in s.split("_"))

def post_init(conf: Config, overrides: list[str]=[]):
    if conf.freeze_heads and "head_1_epochs" not in overrides:
        conf.head_1_epochs = round(conf.epochs / 2)
    
    if conf.group_mix_rate_lower_bounds is not None:
        conf.group_mix_rate_lower_bounds = {str_to_tuple(k): v for k, v in conf.group_mix_rate_lower_bounds.items()}

    if conf.mix_rate_lower_bound is None:
        conf.mix_rate_lower_bound = conf.mix_rate


# In[ ]:


conf = Config()


# In[ ]:


# conf.dataset = "multi-nli"
# conf.lr = 1e-5 
# conf.lr_scheduler = "cosine"
# conf.model = "bert"
# conf.combine_neut_entail = False # True (done)
# conf.contra_no_neg = True # False
# conf.use_group_labels = False # True
# conf.source_cc = True # False
# conf.mix_rate = None # 0.5, 1.0, 0.1 
# conf.mix_rate_lower_bound = 0.1
# conf.dataset_length = 1024


# In[ ]:


# if conf.dataset in ["waterbirds", "celebA-0", "celebA-1", "celebA-2", "toy_grid"]:
#     if conf.loss_type == LossType.TOPK and conf.mix_rate_lower_bound == 0.1:
#         conf.aux_weight = 7.0 
#     else:
#         conf.aux_weight = 2.0


# In[ ]:


# conf.dataset = "toy_grid"
# conf.lr = 1e-3
# conf.optimizer = "adamw"
# conf.model = "toy_model"
# conf.batch_size = 32 
# conf.target_batch_size = 128
# conf.epochs = 100
# conf.loss_type = LossType.DIVDIS
# conf.mix_rate_lower_bound = 0.5
# conf.plot_activations = False


# In[ ]:


# if conf.loss_type == LossType.DBAT:
#     conf.shared_backbone = False 
#     conf.freeze_heads = True
#     conf.batch_size = 16
#     conf.target_batch_size = 32


# In[ ]:


# if conf.dataset in ["waterbirds", "cub"] and conf.loss_type == LossType.DIVDIS:
#     conf.lr = 1e-3
#     conf.weight_decay = 1e-4
#     conf.epochs = 100
#     conf.optimizer = "sgd"
#     conf.batch_size = 16 
#     conf.target_batch_size = 16
#     conf.aux_weight = 10.0
#     conf.shuffle_target = False



# In[ ]:


# if conf.dataset == "waterbirds" and conf.loss_type == LossType.TOPK:
#     conf.optimizer = "sgd"
#     conf.target_01_mix_rate_lower_bound = 0.38
#     conf.target_10_mix_rate_lower_bound = 0.10
#     conf.mix_rate_lower_bound = None


# In[ ]:


# # # toy grid configs 
# if conf.dataset == "toy_grid":
#     conf.model = "toy_model"
#     conf.epochs = 128
# if conf.model == "ClipViT":
#     # conf.epochs = 5
#     conf.lr = 1e-5
# Resnet50 Configs
# if conf.model == "Resnet50":
#     conf.lr = 1e-4 # probably too high, should be 1e-4
# if conf.dataset == "multi-nli" or conf.dataset == "civil_comments":
#     conf.model = "bert"
#     conf.lr = 1e-5
#     conf.lr_scheduler = "cosine"



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


if conf.heads != 2:
    raise ValueError("Only 2 heads currently supported")


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
tokenizer = None
if conf.model == "Resnet50":
    from torchvision import models
    from torchvision.models.resnet import ResNet50_Weights
    resnet_builder = lambda: models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)    
    model_builder = lambda: torch.nn.Sequential(*list(resnet_builder().children())[:-1])
    resnet_50_transforms = ResNet50_Weights.IMAGENET1K_V1.transforms()
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
    preprocess = clip.clip._transform(224)
    clip_builder = lambda: clip.load('ViT-B/32', device='cpu')[0]
    model_builder = lambda: clip_builder().visual
    model_transform = transforms.Compose([
        preprocess.transforms[0],
        preprocess.transforms[1],
        preprocess.transforms[4]
    ])
    feature_dim = 512
    pad_sides = True
elif conf.model == "bert":
    from transformers import BertModel, BertTokenizer
    from models.hf_wrapper import HFWrapper
    bert_builder = lambda: BertModel.from_pretrained('bert-base-uncased')
    model_builder = lambda: HFWrapper(bert_builder())
    feature_dim = 768
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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


collate_fn = None
# TODO: there should be varaible n_classes for each feature 
classes_per_head = [2, 2]
classes_per_feat = [2, 2]
is_img = True
alt_index = 1

if conf.dataset == "toy_grid":
    source_train, source_val, target_train, target_val, target_test = get_toy_grid_datasets(
        target_mix_rate_0_1=conf.mix_rate / 2 if conf.mix_rate is not None else None, 
        target_mix_rate_1_0=conf.mix_rate / 2 if conf.mix_rate is not None else None, 
    )
elif conf.dataset == "cifar_mnist":
    source_train, source_val, target_train, target_val, target_test = get_cifar_mnist_datasets(
        target_mix_rate_0_1=conf.mix_rate / 2 if conf.mix_rate is not None else None, 
        target_mix_rate_1_0=conf.mix_rate / 2 if conf.mix_rate is not None else None, 
        transform=model_transform, 
        pad_sides=pad_sides
    )

elif conf.dataset == "fmnist_mnist":
    source_train, source_val, target_train, target_val, target_test = get_fmnist_mnist_datasets(
        target_mix_rate_0_1=conf.mix_rate / 2 if conf.mix_rate is not None else None, 
        target_mix_rate_1_0=conf.mix_rate / 2 if conf.mix_rate is not None else None, 
        transform=model_transform, 
        pad_sides=pad_sides
    )
elif conf.dataset == "waterbirds":
    source_train, source_val, target_train, target_val, target_test = get_waterbirds_datasets(
        mix_rate=conf.mix_rate, 
        source_cc=conf.source_cc,
        transform=model_transform, 
        convert_to_tensor=True,
        val_split=conf.source_val_split,
        target_val_split=conf.target_val_split, 
        dataset_length=conf.dataset_length
    )
# elif conf.dataset == "cub":
#     source_train, target_train, target_test = get_cub_datasets()
#     source_val = []
#     target_val = []
elif conf.dataset.startswith("celebA"):
    if conf.dataset == "celebA-0":
        gt_feat = "Blond_Hair"
        spur_feat = "Male"
        inv_spur_feat = True
    elif conf.dataset == "celebA-1":
        gt_feat = "Mouth_Slightly_Open"
        spur_feat = "Wearing_Lipstick"
        inv_spur_feat = False
    elif conf.dataset == "celebA-2":
        gt_feat = "Wavy_Hair"
        spur_feat = "High_Cheekbones"
        inv_spur_feat = False
    else: 
        raise ValueError(f"Dataset {conf.dataset} not supported")
    source_train, source_val, target_train, target_val, target_test = get_celebA_datasets(
        mix_rate=conf.mix_rate, 
        source_cc=conf.source_cc,
        transform=model_transform, 
        gt_feat=gt_feat,
        spur_feat=spur_feat,
        inv_spur_feat=inv_spur_feat,
        dataset_length=conf.dataset_length
    )
elif conf.dataset == "camelyon":
    source_train, source_val, target_train, target_val, target_test = get_camelyon_datasets(
        transform=model_transform, 
        dataset_length=conf.dataset_length
    )
# elif conf.dataset == "civil_comments":
#     source_train, source_val, target_train, target_val, target_test = get_civil_comments_datasets(
#         tokenizer=tokenizer,
#         max_length=conf.max_length, 
#         dataset_length=conf.dataset_length
#     )
#     is_img = False

elif conf.dataset == "multi-nli":
    source_train, source_val, target_train, target_val, target_test = get_multi_nli_datasets(
        mix_rate=conf.mix_rate,
        source_cc=conf.source_cc,
        val_split=conf.source_val_split,
        target_val_split=conf.target_val_split,
        tokenizer=tokenizer,
        max_length=conf.max_length, 
        dataset_length=conf.dataset_length, 
        combine_neut_entail=conf.combine_neut_entail, 
        contra_no_neg=conf.contra_no_neg
    )
    is_img = False
    if not conf.combine_neut_entail:
        classes_per_feat = [3, 2]
        if conf.use_group_labels:
            classes_per_head = [3, 2] # [contradiction, entailment, neutral] x [no negation, negation]
        else:
            classes_per_head = [3, 3] # [contradiction, entailment, neutral] x 2

else:
    raise ValueError(f"Dataset {conf.dataset} not supported")

assert len(classes_per_head) == conf.heads
if conf.binary:
    assert all([c == 2 for c in classes_per_head])
    classes_per_head = [1 for c in classes_per_head]



# In[ ]:


# plot image 
img, y, gl = source_train[-1]
# pad 
# to PIL image 

# img = transforms.ToPILImage()(img)
# img
if is_img and img.dim() == 3 and is_notebook():
    plt.imshow(img.permute(1, 2, 0))
    # show without axis 
    plt.axis('off')
    plt.show()


# In[ ]:


# plot target train images with vision_utils.make_grid
if is_img and img.dim() == 3 and is_notebook():
    img_tensor_grid = torch.stack([target_train[i][0] for i in range(20)])
    grid_img = vision_utils.make_grid(img_tensor_grid, nrow=10, normalize=True, padding=1)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()


# In[ ]:


class DivisibleBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size: int, batch_size: int, shuffle: bool = True):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rnd.Random(42)
        
        # Calculate number of complete batches and total samples needed
        self.num_batches = math.ceil(dataset_size / batch_size)
        self.total_size = self.num_batches * batch_size

    def __iter__(self):
        # Generate indices for the entire dataset
        indices = list(range(self.dataset_size))
        
        if self.shuffle:
            # Shuffle all indices
            self.rng.shuffle(indices)
            
        # If we need more indices to make complete batches,
        # randomly sample from existing indices
        if self.total_size > self.dataset_size:
            extra_indices = self.rng.choices(indices, k=self.total_size - self.dataset_size)
            indices.extend(extra_indices)
            
        assert len(indices) == self.total_size
        return iter(indices)

    def __len__(self):
        return self.total_size


# In[ ]:


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

# classifiers
from transformers import get_cosine_schedule_with_warmup
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
if conf.loss_type == LossType.DIVDIS:
    loss_fn = DivDisLoss(heads=conf.heads)
elif conf.loss_type == LossType.DBAT:
    loss_fn = DBatLoss(heads=conf.heads, n_classes=classes_per_head[0])
elif conf.loss_type == LossType.CONF:
    loss_fn = ConfLoss()
elif conf.loss_type == LossType.SMOOTH:
    loss_fn = SmoothTopLoss(
        criterion=partial(F.binary_cross_entropy_with_logits, reduction='none'), 
        device=conf.device
    )
elif conf.loss_type == LossType.ERM:
    loss_fn = PassThroughLoss()
elif conf.loss_type in [LossType.TOPK, LossType.EXP, LossType.PROB]:
    def get_mix_rate(conf, mix_rate_lb_override=None, group_mix_rate_lb_override=None):
        mix_rate_lower_bound = mix_rate_lb_override if mix_rate_lb_override is not None else conf.mix_rate_lower_bound
        group_mix_rate_lower_bounds = group_mix_rate_lb_override if group_mix_rate_lb_override is not None else conf.group_mix_rate_lower_bounds
        if conf.aggregate_mix_rate:
            mix_rate = mix_rate_lower_bound
            group_mix_rates = None
        elif conf.group_mix_rate_lower_bounds is not None:
            mix_rate = None 
            group_mix_rates = group_mix_rate_lower_bounds
        else:
            mix_rate = None 
            if conf.dataset == "multi-nli" and conf.use_group_labels:
                ood_groups = [(0, 1), (1, 0), (2, 0)]
            else: 
                ood_groups = [gl for gl in feature_label_ls(classes_per_head)
                            if any(gl[0] != gl[i] for i in range(1, len(gl)))]
            group_mix_rates = {group: mix_rate_lower_bound / len(ood_groups) for group in ood_groups}
        return mix_rate, group_mix_rates
    
    mix_rate, group_mix_rates = get_mix_rate(conf)

    loss_fn = ACELoss(
        classes_per_head=classes_per_head,
        mode=conf.loss_type.value, 
        mix_rate=mix_rate,
        group_mix_rates=group_mix_rates,
        disagree_only=conf.disagree_only,
        device=conf.device
    )
else:
    raise ValueError(f"Loss type {conf.loss_type} not supported")


# # Plot Activations

# In[ ]:


# visualize data using first two principle componets of final layer activations
if conf.plot_activations and conf.shared_backbone:
    model = model_builder()
    model = model.to(conf.device)
    test_acts, test_labels = get_acts_and_labels(model, target_test_loader, conf.device)
    test_labels = test_labels.to('cpu')
    pca_fig, pca_acts, pca_reducer = plot_activations(
        activations=test_acts, labels=test_labels, 
        classes_per_feature=classes_per_feat, transform="pca"
    )
    umap_fig, umap_acts, umap_reducer = plot_activations(
        activations=test_acts, labels=test_labels, 
        classes_per_feature=classes_per_feat, transform="umap"
    )
    pca_fig.savefig(f"{exp_dir}/activations_pretrain_pca.png")
    pca_fig.savefig(f"{exp_dir}/activations_pretrain_pca.svg")
    umap_fig.savefig(f"{exp_dir}/activations_pretrain_umap.png")
    umap_fig.savefig(f"{exp_dir}/activations_pretrain_umap.svg")
    np.save(f"{exp_dir}/activations_pretrain_pca.npy", pca_acts)
    np.save(f"{exp_dir}/activations_pretrain_umap.npy", umap_acts)
    


# In[ ]:


# fit linear probe 
if conf.plot_activations and conf.shared_backbone:
    train_acts, train_labels = get_acts_and_labels(model, target_train_loader, conf.device)
    probe_acc, probe_acc_alt = compute_probe_acc(train_acts, train_labels, test_acts, test_labels, classes_per_feat)
    print(f"Accuracy: {probe_acc:.4f}")
    print(f"Alt Accuracy: {probe_acc_alt:.4f}")
    # nah I'll just try to picke the umap


# # Train

# In[ ]:


def compute_src_losses(logits, y, gl):
    logits_by_head = torch.split(logits, classes_per_head, dim=-1)
    labels_by_head = [y, y] if not conf.use_group_labels else [gl[:, 0], gl[:, 1]]

    if conf.binary: # NOTE: not currently supported
        losses = [F.binary_cross_entropy_with_logits(logit.squeeze(), y.squeeze().to(torch.float32)) 
                  for logit, y in zip(logits_by_head, labels_by_head)]
    else:
        assert logits_by_head[0].shape == (logits.size(0), classes_per_head[0]), logits_by_head[0].shape
        losses = [F.cross_entropy(logit.squeeze(), y.squeeze().to(torch.long)) 
                  for logit, y in zip(logits_by_head, labels_by_head)]
    return losses

# TODO: fix
def compute_corrects(logits: torch.Tensor, y: torch.Tensor, binary: bool):
    if binary: # NOTE: not currently supported
        return ((logits.squeeze() > 0) == y.flatten()).sum().item()
    else:
        return (logits.argmax(dim=-1) == y).sum().item()
        


# In[ ]:


def eval(model, loader, device, loss_fn, use_labels=False, stage: str = "Evaluating"): 
    group_label_ls = feature_label_ls(classes_per_feat)
    # e.g. for classes per_head = [3,2]
    # group_label_ls = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    
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
                # print("group label mask", group_label_mask.shape)
                logits_by_group[group_label] = logits[group_label_mask]
            # print("group logit shapes", [v.shape for v in logits_by_group.values()])
            
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


# In[ ]:


logger = Logger(exp_dir)
if conf.plot_activations and conf.shared_backbone:   
    logger.add_scalar("test", "probe_acc", probe_acc, -1)
    logger.add_scalar("test", "probe_acc_alt", probe_acc_alt, -1)


# In[ ]:


# TODO: change diciotary values to source loss, target loss
from itertools import cycle
try:
    if conf.freeze_heads:
        # freeze first head (for dbat)
        net.freeze_head(0)
    for epoch in range(conf.epochs):
        train_loader = zip(source_train_loader, cycle(target_train_loader))
        loader_len = len(source_train_loader)
        # train
        for batch_idx, (source_batch, target_batch) in tqdm(enumerate(train_loader), desc="Source train", total=loader_len):
            # freeze heads for dbat
            if conf.freeze_heads and epoch == conf.head_1_epochs: 
                net.unfreeze_head(0)
                net.freeze_head(1)
            
            # mix rate schedule 
            if isinstance(loss_fn, ACELoss):
                if conf.mix_rate_schedule == "linear":
                    if epoch < conf.mix_rate_t0: 
                        cur_mix_rate = 0
                    elif epoch >= conf.mix_rate_t1:
                        cur_mix_rate = conf.mix_rate_lower_bound
                    else:
                        cur_mix_rate = conf.mix_rate_lower_bound * (epoch - conf.mix_rate_t0) / (conf.mix_rate_t1 - conf.mix_rate_t0)
                    _mix_rate, group_mix_rates = get_mix_rate(conf, mix_rate_lb_override=cur_mix_rate)
                    loss_fn.group_mix_rates = group_mix_rates
            
            # source
            x, y, gl = to_device(*source_batch, conf.device)
            logits = net(x)
            losses = compute_src_losses(logits, y, gl)
            xent = torch.mean(torch.stack(losses))
            logger.add_scalar("train", "source_loss", xent.item(), epoch * loader_len + batch_idx)
            
            # target
            target_x, target_y, target_gl = to_device(*target_batch, conf.device)
            target_logits = net(target_x)
            target_loss = loss_fn(target_logits)
            # aux weight schedule
            aux_weight = conf.aux_weight 
            if conf.aux_weight_schedule == "linear":
                if epoch < conf.aux_weight_t0:
                    aux_weight = 0.0
                elif epoch >= conf.aux_weight_t1:
                    aux_weight = conf.aux_weight
                else:
                    aux_weight = conf.aux_weight * (epoch - conf.aux_weight_t0 + 1) / (conf.aux_weight_t1 - conf.aux_weight_t0)
            
            
            
            logger.add_scalar("train", "target_loss", target_loss.item(), epoch * loader_len + batch_idx)
            logger.add_scalar("train", "weighted_target_loss", aux_weight * target_loss.item(), epoch * loader_len + batch_idx, to_metrics=False, to_tb=True)
            # don't compute target loss before second head begins training
            if conf.freeze_heads and epoch < conf.head_1_epochs: 
                target_loss = torch.tensor(0.0, device=conf.device)
            
            # full loss 
            full_loss = conf.source_weight * xent + aux_weight * target_loss
            logger.add_scalar("train", "loss", full_loss.item(), epoch * loader_len + batch_idx)
            
            # backprop
            opt.zero_grad()
            full_loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
        
        # eval
        if (epoch + 1) % 1 == 0:
            net.eval()
            ### Validation 
            # reset mix rate 
            if isinstance(loss_fn, ACELoss):
                if conf.mix_rate_schedule is not None:
                    mix_rate, group_mix_rates = get_mix_rate(conf)
                    loss_fn.group_mix_rates = group_mix_rates
                    loss_fn.mix_rate = mix_rate
           
            # source
            total_val_loss = 0.0
            total_val_weighted_loss = 0.0
            if len(source_val) > 0:
                src_loss_fn = lambda x, y, gl: sum(compute_src_losses(x, y, gl))
                source_val_metrics = eval(net, source_val_loader, conf.device, src_loss_fn, use_labels=True, stage="Source Val")
                for k, v in source_val_metrics.items():
                    if 'count' not in k:
                        logger.add_scalar("val", f"source_{k}", v, epoch)
                total_val_loss += source_val_metrics["loss"]
                total_val_weighted_loss += total_val_loss
            # target
            weighted_target_val_loss = 0.0
            if len(target_val) > 0:  
                target_val_metrics = eval(net, target_val_loader, conf.device, loss_fn, use_labels=False, stage="Target Val")
                for k, v in target_val_metrics.items():
                    if 'count' not in k:
                        logger.add_scalar("val", f"target_{k}", v, epoch)
                weighted_target_val_loss = target_val_metrics["loss"] * conf.aux_weight
                logger.add_scalar("val", "target_weighted_loss", weighted_target_val_loss, epoch)
                total_val_loss += target_val_metrics["loss"]    
                total_val_weighted_loss += weighted_target_val_loss
            # total
            logger.add_scalar("val", "loss", total_val_loss, epoch)
            logger.add_scalar("val", "weighted_loss", total_val_weighted_loss, epoch)

            ### Test
            target_test_metrics = eval(net, target_test_loader, conf.device, None, use_labels=False, stage="Target Test")
            for k, v in target_test_metrics.items():
                if 'count' not in k:
                    logger.add_scalar("test", k, v, epoch)
            
            # probe acc
            if conf.plot_activations and conf.shared_backbone:
                train_acts, train_labels = get_acts_and_labels(model, target_train_loader, conf.device)
                test_acts, test_labels = get_acts_and_labels(net.backbone, target_test_loader, conf.device)
                probe_acc, probe_acc_alt = compute_probe_acc(train_acts, train_labels, test_acts, test_labels, classes_per_feat)
                logger.add_scalar("test", "probe_acc", probe_acc, epoch)
                logger.add_scalar("test", "probe_acc_alt", probe_acc_alt, epoch)


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


            # plot activations if lowest validation loss
            if logger.metrics["val_loss"][-1] == min(logger.metrics["val_loss"]):
                # plot activations 
                if conf.plot_activations and conf.shared_backbone:   
                    # get activations 
                    activations, labels = get_acts_and_labels(net.backbone, target_test_loader, conf.device)
                    labels = labels.to('cpu')
                    pca_fig, pca_acts, pca_reducer = plot_activations(
                        activations=activations, labels=labels, 
                        classes_per_feature=classes_per_feat, transform="pca"
                    )
                    umap_fig, umap_acts, umap_reducer = plot_activations(
                        activations=activations, labels=labels, 
                        classes_per_feature=classes_per_feat, transform="umap"
                    )
                    pca_fig.savefig(f"{exp_dir}/activations_{epoch}_pca.png")
                    pca_fig.savefig(f"{exp_dir}/activations_{epoch}_pca.svg")
                    umap_fig.savefig(f"{exp_dir}/activations_{epoch}_umap.png")
                    umap_fig.savefig(f"{exp_dir}/activations_{epoch}_umap.svg")
                    np.save(f"{exp_dir}/activations_{epoch}_pca.npy", pca_acts)
                    np.save(f"{exp_dir}/activations_{epoch}_umap.npy", umap_acts)
                    plt.close(pca_fig)
                    plt.close(umap_fig)
            
            net.train()
finally:
    logger.flush()

