from abc import abstractmethod
from typing import Optional, Callable
import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, random_split, Subset
from torchvision import transforms

from spurious_datasets.utils import (
    get_group_idxs, 
    update_idxs_from_mix_rate
)



class WaterbirdsDataset(Dataset):
    # download from https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/
    def __init__(self, data_dir: str, transform: Optional[Callable] = None, 
                 idxs: Optional[np.ndarray]=None):
        self.data_dir = data_dir
        self.transform = transform

        # load dataset data
        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))
        self.labels = self.metadata_df["y"].values
        self.feature_labels = self.metadata_df[["y", "place"]].values
        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values
        self.idxs = idxs
        if self.idxs is not None:
            self.labels = self.labels[idxs]
            self.feature_labels = self.feature_labels[idxs]
            self.filename_array = self.filename_array[idxs]
            self.split_array = self.split_array[idxs]
    
    def __getitem__(self, idx):
        filename = self.filename_array[idx]
        img_path = os.path.join(self.data_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        y = torch.tensor(self.labels[idx])
        feature_labels = torch.tensor(self.feature_labels[idx])
        return img, y, feature_labels
    
    def __len__(self):
        return len(self.labels)

def get_split(dataset, idxs):
    if dataset.idxs is not None:
        idxs = dataset.idxs[idxs]
    return WaterbirdsDataset(data_dir=dataset.data_dir, transform=dataset.transform, idxs=idxs)

def get_waterbirds_datasets(
    mix_rate: Optional[float] = 0.5,
    source_cc: bool = True,
    transform: Optional[Callable] = None, 
    convert_to_tensor: bool = True,
    val_split: float = 0.2, 
    target_val_split: float = 0.0, 
    dataset_length: Optional[int]=None,
):
    transform_list = []
    if convert_to_tensor:
        transform_list.append(transforms.ToTensor())
    if transform is not None:
        transform_list.append(transform)
    transform = transforms.Compose(transform_list)

    dataset = WaterbirdsDataset(data_dir="./data/waterbirds/waterbirds_v1.0", transform=transform)

    source_idxs = np.where(dataset.split_array == 0)[0]
    target_idxs = np.where(dataset.split_array == 1)[0]
    test_idxs = np.where(dataset.split_array == 2)[0]
    if dataset_length is not None:
        source_idxs = source_idxs[:dataset_length]
        target_idxs = target_idxs[:dataset_length]
        test_idxs = test_idxs[:dataset_length]
    source, target, test = get_split(dataset, source_idxs), get_split(dataset, target_idxs), get_split(dataset, test_idxs)

    if source_cc: 
        cc_groups = [np.array([0, 0]), np.array([1, 1])]
        cc_idxs = get_group_idxs(source.feature_labels, cc_groups)
        source = get_split(source, cc_idxs)
    
    # update target idxs based on mix rate (preserves group balance)
    if mix_rate is not None:
        target_idxs = update_idxs_from_mix_rate(target.feature_labels, mix_rate)
        target = get_split(target, target_idxs)

    # split datasets
    if val_split > 0:
        source_train_size = int(len(source) * (1 - val_split))
        source_train, source_val = random_split(
            source, 
            [source_train_size, len(source) - source_train_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        source_train = source 
        source_val = []
    if target_val_split > 0:
        target_train_size = int(len(target) * (1 - target_val_split))
        target_train, target_val = random_split(
            target, 
            [target_train_size, len(target) - target_train_size], 
            generator=torch.Generator().manual_seed(42)
        )
    else:
        target_train = target 
        target_val = []


    return source_train, source_val, target_train, target_val, test