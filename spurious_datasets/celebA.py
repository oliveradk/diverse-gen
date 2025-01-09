import os
from PIL import Image
from typing import Optional, Callable

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms

from spurious_datasets.utils import (
    get_group_idxs, 
    update_idxs_from_mix_rate
)


def load_celebA_dfs(root_dir: str):
    attr_path = os.path.join(root_dir, "list_attr_celeba.txt")
    splits_path = os.path.join(root_dir, "list_eval_partition.txt")
    attrs_df = pd.read_csv(attr_path, sep='\s+', header=0)
    attrs_df = attrs_df.applymap(lambda x: int(x)) # convert columns to int
    attrs_df = attrs_df.reset_index(drop=False) # sets filename as column, index as row number
    attrs_df = attrs_df.applymap(lambda x: 0 if x == -1 else x) # convert -1 to 0
    splits_df = pd.read_csv(splits_path, sep='\s+', header=None)
    return attrs_df, splits_df


class CelebA(Dataset):
    def __init__(self, 
        data_dir: str | None = None, gt_feat: str='Male', spur_feat: str='Blond_Hair', 
        inv_spur_feat: bool=False, transform: Optional[Callable]=None, 
        idxs: Optional[np.ndarray]=None, attrs_df: Optional[pd.DataFrame]=None, 
        splits_df: Optional[pd.DataFrame]=None
    ):
        self.data_dir = data_dir
        self.gt_feat = gt_feat
        self.spur_feat = spur_feat
        self.transform = transform

        # load attributes and splits
        if attrs_df is None or splits_df is None:
            attrs_df, splits_df = load_celebA_dfs(data_dir)
        if inv_spur_feat:
            attrs_df[spur_feat] = attrs_df[spur_feat].map(lambda x: 1-x)
        self.attrs_df = attrs_df
        self.splits_df = splits_df
        
        self.labels = self.attrs_df[gt_feat].values
        self.feature_labels = self.attrs_df[[gt_feat, spur_feat]].values
        self.filename_array = self.attrs_df.index.values
        self.split_array = self.splits_df.iloc[:, 1].values
        self.idxs = idxs
        if self.idxs is not None:
            self.labels = self.labels[idxs]
            self.feature_labels = self.feature_labels[idxs]
            self.filename_array = self.filename_array[idxs]
            self.split_array = self.split_array[idxs]


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        filename = self.filename_array[idx]
        img_path = os.path.join(self.data_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        y = torch.tensor(self.labels[idx])
        feature_labels = torch.tensor(self.feature_labels[idx])
        return img, y, feature_labels
    

def get_split(dataset, idxs):
    if dataset.idxs is not None:
        idxs = dataset.idxs[idxs]
    return CelebA(gt_feat=dataset.gt_feat, spur_feat=dataset.spur_feat, 
                  transform=dataset.transform, idxs=idxs, attrs_df=dataset.attrs_df, splits_df=dataset.splits_df)
    

def get_celebA_datasets(
    mix_rate: Optional[float]=None,
    source_cc: bool = True, 
    val_split=0.2,
    target_val_split=0.2,
    convert_to_tensor=True,
    transform=None,
    gt_feat: str='Blond_Hair',
    spur_feat: str='Male',
    inv_spur_feat: bool=False,
    dataset_length: Optional[int]=None,
):
    transform_list = []
    if convert_to_tensor:
        transform_list.append(transforms.ToTensor())
    if transform is not None:
        transform_list.append(transform)
    transform = transforms.Compose(transform_list)
    
    attrs_df, splits_df = load_celebA_dfs("./data/img_align_celeba")
    dataset = CelebA(gt_feat=gt_feat, spur_feat=spur_feat, inv_spur_feat=inv_spur_feat, 
                     transform=transform, attrs_df=attrs_df, splits_df=splits_df)

    source_idx = np.where(dataset.split_array == 0)[0]
    target_idx = np.where(dataset.split_array == 1)[0]
    test_idx = np.where(dataset.split_array == 2)[0]
    if dataset_length is not None:
        source_idx = source_idx[:dataset_length]
        target_idx = target_idx[:dataset_length]
        test_idx = test_idx[:dataset_length]
    source, target, test = get_split(dataset, source_idx), get_split(dataset, target_idx), get_split(dataset, test_idx)

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

    
