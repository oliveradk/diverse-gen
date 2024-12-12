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


# TODO: fix gradient accumulation to work for batched loss, by computing logits  for 
# virtual batch without grad, then iterating over mini-batches and replacing the logits


# In[ ]:


import os
os.chdir("/nas/ucb/oliveradk/diverse-gen")


# In[ ]:


# ok so the basic setup would be pretraining on the whole thing (already done) 
# then finetuning on the source and target data using the standard losses


# In[ ]:


import torch
from torch.utils.data import random_split
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from losses.divdis import DivDisLoss 
from losses.divdis import DivDisLoss
from losses.ace import ACELoss
from losses.conf import ConfLoss
from losses.dbat import DBatLoss
from losses.smooth_top_loss import SmoothTopLoss
from losses.loss_types import LossType

from models.backbone import MultiHeadBackbone
from utils.utils import batch_size, to_device


# In[ ]:


conf = Config()


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

dataset = load_dataset("redwoodresearch/diamonds-seed0")


# In[ ]:


class DiamondsDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        # tokenize the text
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # convert label to tensor
        label = torch.tensor(item['is_correct'])
        all_measurements = torch.tensor(all(item['measurements'])).float()
        group_labels = torch.stack((label, all_measurements))
        encoding = {k:v.squeeze(0) for k, v in encoding.items()}

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
# remove unsuccessful tampering and real negatives from test 
dataset["test"] = dataset["test"].filter(lambda x: x["is_correct"] or all(x["measurements"]))

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


source_train_ds = DiamondsDataset(dataset["source_train"], tokenizer, conf.max_length)
source_val_ds = DiamondsDataset(dataset["source_val"], tokenizer, conf.max_length)
target_train_ds = DiamondsDataset(dataset["target_train"], tokenizer, conf.max_length)
target_val_ds = DiamondsDataset(dataset["target_val"], tokenizer, conf.max_length)
test_ds = DiamondsDataset(dataset["test"], tokenizer, conf.max_length)


# In[ ]:


from torch.utils.data import DataLoader
dataloader = DataLoader(source_train_ds, batch_size=conf.micro_batch_size)
x = next(iter(dataloader))


# In[ ]:


x[0], x[1], x[2]


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


from transformers import get_scheduler

pred_model = MeasurementPredBackbone(pretrained_model).to(conf.device)
net = MultiHeadBackbone(pred_model, n_heads=2, feature_dim=1024, classes=1).to(conf.device)

source_train_loader = DataLoader(source_train_ds, batch_size=conf.micro_batch_size)
target_train_loader = DataLoader(target_train_ds, batch_size=conf.effective_batch_size)
source_val_loader = DataLoader(source_val_ds, batch_size=conf.micro_batch_size)
target_val_loader = DataLoader(target_val_ds, batch_size=conf.effective_batch_size)
target_test_loader = DataLoader(test_ds, batch_size=conf.forward_batch_size)

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
elif conf.loss_type == LossType.TOPK:
    loss_fn = ACELoss(
        heads=2, 
        classes=2, 
        binary=True, 
        mode="topk", 
        mix_rate=conf.mix_rate_lower_bound, 
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

from collections import defaultdict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

# metrics
metrics = defaultdict(list)
writer = SummaryWriter(log_dir=conf.exp_dir)


# In[ ]:


# ok so I think it should be 
# 1. compute source logits with micro batch 
# if batch idx * micro_batch_size % effective_batch_size 
    # compute target logits (in forward passes in sets of forward_pass_batch_size) with no grad 
    # set current batch 
# compute target logits on current batch % 32 / micro_batch_size 
# replace target logits at the index current batch_idx * micro_batch_size : (current batch_idx + 1) * micro_batch_size


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


# how do do this: 

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
        if batch_idx % (conf.effective_batch_size // conf.micro_batch_size) == 0:
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
        # compute target logits with grad on micro batch
        micro_batch_idx = batch_idx % (conf.effective_batch_size // conf.micro_batch_size)
        micro_slice = slice(micro_batch_idx * conf.micro_batch_size, (micro_batch_idx + 1) * conf.micro_batch_size)
        target_micro_batch = slice_batch(target_batch, micro_slice)
        target_micro_logits = net(target_micro_batch)
        cloned_target_logits= target_logits.clone().requires_grad_(True)
        new_target_logits = torch.cat([
            cloned_target_logits[i].unsqueeze(0) if not in_slice(i, micro_slice) else target_micro_logits[i - micro_slice.start].unsqueeze(0)
            for i in range(len(cloned_target_logits))
        ])
        target_loss = loss_fn(new_target_logits)

        # compute target acc 
        for i in range(conf.heads):
            target_batch_corrects[(i, "y")] += compute_corrects(new_target_logits, i, target_y, conf.binary) 
            target_batch_corrects[(i, "gl")] += compute_corrects(new_target_logits, i, target_gl[:, 1], conf.binary)
        
        weighted_target_loss = conf.aux_weight * target_loss
        # full loss (on micro batch)
        full_loss = conf.source_weight * xent + conf.aux_weight * target_loss
        full_loss.backward() 
        
        # update weights, clear gradients on effective batch
        if (batch_idx + 1) % (conf.effective_batch_size // conf.micro_batch_size) == 0:
            opt.step()
            if scheduler is not None:
                scheduler.step()
            opt.zero_grad()

            # compute batch metrics 
            effective_batch_idx = batch_idx // (conf.effective_batch_size // conf.micro_batch_size)
            writer.add_scalar("train/source_loss", source_batch_loss / conf.effective_batch_size, epoch * len(source_train_loader) + effective_batch_idx)
            writer.add_scalar("train/target_loss", target_loss.item(), epoch * len(source_train_loader) + effective_batch_idx)
            for i in range(conf.heads):
                writer.add_scalar(f"train/source_acc_{i}", source_batch_corrects[i] / conf.effective_batch_size, epoch * len(source_train_loader) + effective_batch_idx)
                for label in ["y", "gl"]:
                    writer.add_scalar(f"train/target_acc_{i}_{label}", target_batch_corrects[(i, label)] / conf.effective_batch_size, epoch * len(source_train_loader) + effective_batch_idx)
            source_batch_loss = 0
            source_batch_corrects = {i: 0 for i in range(conf.heads)}
            target_batch_corrects = {(i, label): 0 for i in range(conf.heads) for label in ["y", "gl"]}
    
    # validation and test

    if (epoch + 1) % 1 == 0:
        net.eval()
        # compute repulsion loss on target validation set (used for model selection)
        target_losses_val = []
        weighted_target_losses_val = []
        with torch.no_grad():
            for batch in tqdm(target_val_loader, desc="Target val"):
                x, y, gl = to_device(*batch, conf.device)
                logits_val = net(x)
                target_loss_val = loss_fn(logits_val)
                target_losses_val.append(target_loss_val.item())
                weighted_target_losses_val.append(conf.aux_weight * target_loss_val.item())
        metrics[f"target_val_repulsion_loss"].append(np.mean(target_losses_val))
        metrics[f"target_val_weighted_repulsion_loss"].append(np.mean(weighted_target_losses_val))
        writer.add_scalar("val/target_loss", np.mean(target_losses_val), epoch)
        writer.add_scalar("val/weighted_target_loss", np.mean(weighted_target_losses_val), epoch)
        # compute xent on source validation set
        xent_val = []
        with torch.no_grad():
            for batch in tqdm(source_val_loader, desc="Source val"):
                x, y, gl = to_device(*batch, conf.device)
                logits_val = net(x)
                losses_val = compute_src_losses(logits_val, y, gl, conf.binary, conf.use_group_labels)
                xent_val.append(sum(losses_val).item())
        metrics[f"source_val_xent"].append(np.mean(xent_val))
        metrics[f"val_loss"].append(np.mean(target_losses_val) + np.mean(xent_val))
        metrics[f"val_weighted_loss"].append(np.mean(weighted_target_losses_val) + np.mean(xent_val))
        writer.add_scalar("val/source_loss", np.mean(xent_val), epoch)
        writer.add_scalar("val/val_loss", np.mean(target_losses_val) + np.mean(xent_val), epoch)
        writer.add_scalar("val/weighted_val_loss", np.mean(weighted_target_losses_val) + np.mean(xent_val), epoch)
        
        # compute AUROC betweeen fake positives and real positives (flter everything else )
        total_correct = torch.zeros(conf.heads)
        total_correct_alt = torch.zeros(conf.heads)
        total_samples = 0

        # store predictions
        all_preds = [[] for _ in range(conf.heads)]
        all_preds_alt = [[] for _ in range(conf.heads)]
        all_labels = []
        all_labels_alt = []

        with torch.no_grad():
            for test_batch in tqdm(target_test_loader, desc="Target test"):
                test_x, test_y, test_gl = to_device(*test_batch, conf.device)
                test_logits = net(test_x)
                assert test_logits.shape == (batch_size(test_x), conf.heads * (1 if conf.binary else classes))
                total_samples += test_y.size(0)

                 # Store labels for AUROC
                all_labels.extend(test_y.cpu().numpy())
                all_labels_alt.extend(test_gl[:, alt_index].cpu().numpy())
                
                for i in range(conf.heads):
                    total_correct[i] += compute_corrects(test_logits, i, test_y, conf.binary)
                    total_correct_alt[i] += compute_corrects(test_logits, i, test_gl[:, alt_index], conf.binary)
                    probs = torch.sigmoid(test_logits[:, i]).cpu().numpy()
                    all_preds[i].extend(probs)
        
        # Compute and store AUROC for each head
        for i in range(conf.heads):
            auroc = roc_auc_score(all_labels, all_preds[i])
            metrics[f"epoch_auroc_{i}"].append(auroc)
            writer.add_scalar(f"val/auroc_{i}", auroc, epoch)
            writer.add_scalar(f"val/auroc_{i}_alt", auroc_alt, epoch)
            print(f"Epoch {epoch + 1} AUROC {i}: {auroc:.4f}, Alt: {auroc_alt:.4f}")

        # compute and store accuracy for each head
        for i in range(conf.heads):
            metrics[f"epoch_acc_{i}"].append((total_correct[i] / total_samples).item())
            metrics[f"epoch_acc_{i}_alt"].append((total_correct_alt[i] / total_samples).item())
            writer.add_scalar(f"val/acc_{i}", (total_correct[i] / total_samples).item(), epoch)
            writer.add_scalar(f"val/acc_{i}_alt", (total_correct_alt[i] / total_samples).item(), epoch)
        
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
with open(f"{conf.exp_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
    


# In[ ]:


# grad_scaler = GradScaler()

# For now, I'll just compte all the logits with no grad, then iterate over in mini-batches, 
# and replace the logits at the index of the batch with the graident version
target_iter = iter(target_train_loader)
for epoch in range(conf.epochs):
    target_logit_ls = []
    for batch_idx, (x, y, gl) in tqdm(enumerate(source_train_loader), desc="Source train", total=len(source_train_loader)):
        x, y, gl = to_device(x, y, gl, conf.device)
        # with autocast(conf.device, enabled=conf.mixed_precision):
        logits = net(x)
        losses = compute_src_losses(logits, y, gl, conf.binary, conf.use_group_labels)
        xent = sum(losses)
        writer.add_scalar("train/source_loss", xent.item(), epoch * len(source_train_loader) + batch_idx)
        
        # # compute target logits with no grad 
        
        # target loss 
        try: 
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_train_loader)
            target_batch = next(target_iter)
        target_x, target_y, target_gl = to_device(*target_batch, conf.device)
        # with autocast(conf.device, enabled=conf.mixed_precision):
        target_logits = net(target_x)
        target_loss = loss_fn(target_logits)
        writer.add_scalar("train/target_loss", target_loss.item(), epoch * len(target_train_loader) + batch_idx)
        writer.add_scalar("train/weighted_target_loss", conf.aux_weight * target_loss.item(), epoch * len(target_train_loader) + batch_idx)
        # full loss 
        full_loss = conf.source_weight * xent + conf.aux_weight * target_loss
        # grad_scaler.scale(full_loss).backward()
        full_loss.backward()
        if (batch_idx + 1) % conf.gradient_accumulation_steps == 0:
            # unscale and clip gradients
            # grad_scaler.unscale_(opt)
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            # update weights, clear gradients 
            # grad_scaler.step(opt)
            opt.step()
            # grad_scaler.update()
            if scheduler is not None:
                scheduler.step()
            opt.zero_grad()

        metrics[f"xent"].append(xent.item())
        metrics[f"repulsion_loss"].append(target_loss.item())
    # Compute loss on target validation set (used for model selection)
    # and aggregate metrics over the entire test set (should not really be using)
    if (epoch + 1) % 1 == 0:
        net.eval()
        # compute repulsion loss on target validation set (used for model selection)
        target_losses_val = []
        weighted_target_losses_val = []
        with torch.no_grad():
            for batch in tqdm(target_val_loader, desc="Target val"):
                x, y, gl = to_device(*batch, conf.device)
                logits_val = net(x)
                target_loss_val = loss_fn(logits_val)
                if not target_loss_val.isnan():
                    target_losses_val.append(target_loss_val.item())
                    weighted_target_losses_val.append(conf.aux_weight * target_loss_val.item())
        metrics[f"target_val_repulsion_loss"].append(np.mean(target_losses_val))
        metrics[f"target_val_weighted_repulsion_loss"].append(np.mean(weighted_target_losses_val))
        writer.add_scalar("val/target_loss", np.mean(target_losses_val), epoch)
        writer.add_scalar("val/weighted_target_loss", np.mean(weighted_target_losses_val), epoch)
        # compute xent on source validation set
        xent_val = []
        with torch.no_grad():
            for batch in tqdm(source_val_loader, desc="Source val"):
                x, y, gl = to_device(*batch, conf.device)
                logits_val = net(x)
                losses_val = compute_src_losses(logits_val, y, gl, conf.binary, conf.use_group_labels)
                xent_val.append(sum(losses_val).item())
        metrics[f"source_val_xent"].append(np.mean(xent_val))
        metrics[f"val_loss"].append(np.mean(target_losses_val) + np.mean(xent_val))
        metrics[f"val_weighted_loss"].append(np.mean(weighted_target_losses_val) + np.mean(xent_val))
        writer.add_scalar("val/source_loss", np.mean(xent_val), epoch)
        writer.add_scalar("val/val_loss", np.mean(target_losses_val) + np.mean(xent_val), epoch)
        writer.add_scalar("val/weighted_val_loss", np.mean(weighted_target_losses_val) + np.mean(xent_val), epoch)
        
        # compute accuracy over target test set (used to evaluate actual performance)
        total_correct = torch.zeros(conf.heads)
        total_correct_alt = torch.zeros(conf.heads)
        total_samples = 0

        # store predictions
        all_preds = [[] for _ in range(conf.heads)]
        all_preds_alt = [[] for _ in range(conf.heads)]
        all_labels = []
        all_labels_alt = []

        with torch.no_grad():
            for test_batch in tqdm(target_test_loader, desc="Target test"):
                test_x, test_y, test_gl = to_device(*test_batch, conf.device)
                test_logits = net(test_x)
                assert test_logits.shape == (batch_size(test_x), conf.heads * (1 if conf.binary else classes))
                total_samples += test_y.size(0)

                 # Store labels for AUROC
                all_labels.extend(test_y.cpu().numpy())
                all_labels_alt.extend(test_gl[:, alt_index].cpu().numpy())
                
                for i in range(conf.heads):
                    total_correct[i] += compute_corrects(test_logits, i, test_y, conf.binary)
                    total_correct_alt[i] += compute_corrects(test_logits, i, test_gl[:, alt_index], conf.binary)
                    probs = torch.sigmoid(test_logits[:, i]).cpu().numpy()
                    all_preds[i].extend(probs)
        
        # Compute and store AUROC for each head
        for i in range(conf.heads):
            auroc = roc_auc_score(all_labels, all_preds[i])
            auroc_alt = roc_auc_score(all_labels_alt, all_preds[i])
            metrics[f"epoch_auroc_{i}"].append(auroc)
            metrics[f"epoch_auroc_{i}_alt"].append(auroc_alt)
            writer.add_scalar(f"val/auroc_{i}", auroc, epoch)
            writer.add_scalar(f"val/auroc_{i}_alt", auroc_alt, epoch)
            print(f"Epoch {epoch + 1} AUROC {i}: {auroc:.4f}, Alt: {auroc_alt:.4f}")

        # compute and store accuracy for each head
        for i in range(conf.heads):
            metrics[f"epoch_acc_{i}"].append((total_correct[i] / total_samples).item())
            metrics[f"epoch_acc_{i}_alt"].append((total_correct_alt[i] / total_samples).item())
            writer.add_scalar(f"val/acc_{i}", (total_correct[i] / total_samples).item(), epoch)
            writer.add_scalar(f"val/acc_{i}_alt", (total_correct_alt[i] / total_samples).item(), epoch)
        
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
with open(f"{conf.exp_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# In[ ]:


all_preds[i]


# In[ ]:


all_labels, all_preds[i]

