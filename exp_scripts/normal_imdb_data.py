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
    os.environ["CUDA_VISIBLE_DEVICES"] = "4" #"1"
    # os.environ['CUDA_LAUNCH_BLOCKING']="1"
    # os.environ['TORCH_USE_CUDA_DSA'] = "1"

import matplotlib 
if not is_notebook():
    matplotlib.use('Agg')


# In[ ]:


# Imports
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from omegaconf import OmegaConf

from losses.loss_types import LossType
from losses.ace import ACELoss
from losses.divdis import DivDisLoss
from losses.pass_through import PassThroughLoss



# In[ ]:


from dataclasses import dataclass
import torch

@dataclass
class Config:
    max_length: int = 256
    batch_size: int = 16
    target_batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    dataset_length: int | None = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_type: LossType = LossType.TOPK 
    aux_weight: float = 1.0 # Weight for auxiliary loss
    mix_rate_lower_bound: float = 0.1
    seed: int = 42
    exp_dir: str = f"output/normal_data/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"

def post_init(conf, overrride_keys):
    pass


# In[ ]:


conf = Config()
overrride_keys = []
if not is_notebook():
    import sys 
    overrides = OmegaConf.from_cli(sys.argv[1:])
    overrride_keys = overrides.keys()
    conf_dict = OmegaConf.merge(OmegaConf.structured(conf), overrides)
    conf = Config(**conf_dict)
post_init(conf, overrride_keys)
# Set random seed for reproducibility
torch.manual_seed(conf.seed)
os.makedirs(conf.exp_dir, exist_ok=True)


# In[ ]:


# Dataset class
class IMDBDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# In[ ]:


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions_head1 = 0
    correct_predictions_head2 = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model({'input_ids': input_ids, 'attention_mask': attention_mask})
            # Split outputs into two heads
            head1_output, head2_output = outputs.split(1, dim=1)
            
            # Calculate loss for each head
            loss1 = criterion(head1_output.squeeze(1), labels.float())
            loss2 = criterion(head2_output.squeeze(1), labels.float())
            total_loss += (loss1.item() + loss2.item()) / 2
            
            # Calculate accuracy for each head
            preds_head1 = (head1_output > 0.5).squeeze(1)
            preds_head2 = (head2_output > 0.5).squeeze(1)
            
            correct_predictions_head1 += (preds_head1 == labels).sum().item()
            correct_predictions_head2 += (preds_head2 == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_loss = total_loss / len(data_loader)
    acc_head1 = correct_predictions_head1 / total_predictions
    acc_head2 = correct_predictions_head2 / total_predictions
    
    return avg_loss, (acc_head1, acc_head2)


def evaluate_target(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Target Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model({'input_ids': input_ids, 'attention_mask': attention_mask})
            loss = loss_fn(outputs)
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


# In[ ]:


dataset = load_dataset("imdb")
if conf.dataset_length is not None:
    rng = torch.Generator().manual_seed(42)  # For reproducibility
    dataset['train'] = dataset['train'].shuffle(seed=42).select(range(conf.dataset_length))
    dataset['test'] = dataset['test'].shuffle(seed=42).select(range(conf.dataset_length))
    dataset['unsupervised'] = dataset['unsupervised'].shuffle(seed=42).select(range(conf.dataset_length))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# split training data into source_train and source_val
source_dataset_size = len(dataset['train'])
source_train_size = int(0.8 * source_dataset_size)  # 80-20 split
source_val_size = source_dataset_size - source_train_size

source_train_dataset, source_val_dataset = torch.utils.data.random_split(
    dataset['train'], 
    [source_train_size, source_val_size],
    generator=torch.Generator().manual_seed(42)
)

# Split unsupervised data into target_train and target_val
target_dataset_size = len(dataset['unsupervised'])
target_train_size = int(0.8 * target_dataset_size)  # 80-20 split
target_val_size = target_dataset_size - target_train_size

target_train_dataset, target_val_dataset = torch.utils.data.random_split(
    dataset['unsupervised'], 
    [target_train_size, target_val_size],
    generator=torch.Generator().manual_seed(42)
)

# Create datasets
source_train_dataset = IMDBDataset(source_train_dataset, tokenizer, conf.max_length)
source_val_dataset = IMDBDataset(source_val_dataset, tokenizer, conf.max_length)
target_train_dataset = IMDBDataset(target_train_dataset, tokenizer, conf.max_length)
target_val_dataset = IMDBDataset(target_val_dataset, tokenizer, conf.max_length)
target_test_dataset = IMDBDataset(dataset['test'], tokenizer, conf.max_length)

# Create data loaders
source_train_loader = DataLoader(source_train_dataset, batch_size=conf.batch_size, shuffle=True)
source_val_loader = DataLoader(source_val_dataset, batch_size=conf.batch_size)
target_train_loader = DataLoader(target_train_dataset, batch_size=conf.target_batch_size, shuffle=True)
target_val_loader = DataLoader(target_val_dataset, batch_size=conf.target_batch_size)
target_test_loader = DataLoader(target_test_dataset, batch_size=conf.batch_size)


# In[ ]:


from transformers import BertModel, BertTokenizer
from models.hf_wrapper import HFWrapper
bert_builder = lambda: BertModel.from_pretrained('bert-base-uncased')
model_builder = lambda: HFWrapper(bert_builder())
feature_dim = 768

# Initialize model, criterion, and optimizer
from models.backbone import MultiHeadBackbone
feature_dim = 768  # BERT's hidden size
classes_per_head = [1, 1]  # Binary classification for both heads

model = MultiHeadBackbone(model_builder(), classes_per_head, feature_dim).to(conf.device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)


# In[ ]:


# Initialize loss function
if conf.loss_type == LossType.TOPK:
    loss_fn = ACELoss(
        classes_per_head=[1, 1],  # Binary classification
        mode="topk",
        device=conf.device, 
        mix_rate=conf.mix_rate_lower_bound
    )
elif conf.loss_type == LossType.DIVDIS:
    loss_fn = DivDisLoss(heads=2)  # Using 2 heads
elif conf.loss_type == LossType.ERM:
    loss_fn = PassThroughLoss()
else:
    raise ValueError(f"Loss type {conf.loss_type} not supported")


# In[ ]:


from utils.logger import Logger
logger = Logger(exp_dir=conf.exp_dir)


# In[ ]:


from itertools import cycle
from utils.utils import to_device

# Training loop
best_val_loss = float('inf')

for epoch in tqdm(range(conf.epochs), desc="Epochs"):
    # Training
    model.train()
    total_loss = 0
    
    zipped_loaders = zip(source_train_loader, cycle(target_train_loader))
    train_iter = tqdm(zipped_loaders, total=len(source_train_loader), desc="Train")

    for source_batch, target_batch in train_iter:
        # Source forward pass
        input_ids = source_batch['input_ids'].to(conf.device)
        attention_mask = source_batch['attention_mask'].to(conf.device)
        labels = source_batch['label'].to(conf.device)
        
        source_outputs = model({'input_ids': input_ids, 'attention_mask': attention_mask})
        source_loss = sum([criterion(source_outputs[:, i], labels.float()) for i in range(2)])
        logger.add_scalar("train", "source_loss", source_loss.item(), step=len(train_iter), to_metrics=False)
        
        # Target forward pass
        target_input_ids = target_batch['input_ids'].to(conf.device)
        target_attention_mask = target_batch['attention_mask'].to(conf.device)
        
        target_outputs = model({'input_ids': target_input_ids, 'attention_mask': target_attention_mask})
        target_loss = loss_fn(target_outputs)
        logger.add_scalar("train", "target_loss", target_loss.item(), step=len(train_iter), to_metrics=False)
        # Combined loss
        loss = source_loss + conf.aux_weight * target_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    source_val_loss, (source_val_acc_1, source_val_acc_2) = evaluate(model, source_val_loader, criterion, conf.device)
    target_val_loss = evaluate_target(model, target_val_loader, loss_fn, conf.device)
    total_val_loss = source_val_loss + target_val_loss
    logger.add_scalar("val", "source_loss", source_val_loss, step=epoch)
    logger.add_scalar("val", "target_loss", target_val_loss, step=epoch)
    # Test Accuracy for each head
    test_loss, (test_acc_head1, test_acc_head2) = evaluate(model, target_test_loader, criterion, conf.device)
    logger.add_scalar("test", "acc_1", test_acc_head1, step=epoch)
    logger.add_scalar("test", "acc_2", test_acc_head2, step=epoch)
    
    print(f"Train Loss: {total_loss/len(source_train_loader):.4f}")
    print(f"Source Val Loss: {source_val_loss:.4f}, Source Val Acc: Head 1: {source_val_acc_1:.4f}, Head 2: {source_val_acc_2:.4f}")
    print(f"Target Val Loss: {target_val_loss:.4f}")
    print(f"Total Val Loss: {total_val_loss:.4f}")
    print(f"Test Accuracy - Head 1: {test_acc_head1:.4f}, Head 2: {test_acc_head2:.4f}")
    
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(model.state_dict(), f'{conf.exp_dir}/best_model.pt')
        print("Saved best model!")

logger.flush()

