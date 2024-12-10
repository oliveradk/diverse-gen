

import os
from functools import partial
from typing import Optional
import string 
import json

import torch
from torch.utils.data import Dataset, random_split
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset

class CustomCivilCommentsDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, y, _metadata = self.dataset[idx]
        
        # Tokenize the premise and hypothesis
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoding = {k:v.squeeze(0) for k, v in encoding.items()}
        gl = torch.tensor([y, y])
        return encoding, y,  gl

def get_civil_comments_datasets(
    val_split=0.2, 
    tokenizer=None,
    max_length=128,
    dataset_length=None
):
    dataset = CivilCommentsDataset(root_dir="./data/civilcomments", download=True)

    source = dataset.get_subset(split="train") 
    target = dataset.get_subset(split="val")
    test = dataset.get_subset(split="test")

    if dataset_length is not None:
        source = source.get_subset(split="train", frac=dataset_length / len(source))
        target = target.get_subset(split="val", frac=dataset_length / len(target))
        test = test.get_subset(split="test", frac=dataset_length / len(test))

    # train val splits 
    source_train, source_val = random_split(
        source, 
        [round(len(source) * (1 - val_split)), round(len(source) * val_split)],
        generator=torch.Generator().manual_seed(42)
    )
    target_train, target_val = random_split(
        target, 
        [round(len(target) * (1 - val_split)), round(len(target) * val_split)],
        generator=torch.Generator().manual_seed(42)
    )

    # construct datasets 
    source_train = CustomCivilCommentsDataset(source_train, tokenizer, max_length)
    source_val = CustomCivilCommentsDataset(source_val, tokenizer, max_length)
    target_train = CustomCivilCommentsDataset(target_train, tokenizer, max_length)
    target_val = CustomCivilCommentsDataset(target_val, tokenizer, max_length)
    test = CustomCivilCommentsDataset(test, tokenizer, max_length)

    return source_train, source_val, target_train, target_val, test

