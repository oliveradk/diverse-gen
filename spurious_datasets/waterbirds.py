from typing import Optional, Callable
import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import random_split
from torchvision import transforms

from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset

# TODO: look into balanced target loader (see if DivDis, D-BAT uses one)

class CustomWaterbirdsDataset(WaterbirdsDataset):

    # def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
    #     self._version = version
    #     self._data_dir = self.initialize_data_dir(root_dir, download)

    #     if not os.path.exists(self.data_dir):
    #         raise ValueError(
    #             f'{self.data_dir} does not exist yet. Please generate the dataset first.')

    #     # Read in metadata
    #     # Note: metadata_df is one-indexed.
    #     metadata_df = pd.read_csv(
    #         os.path.join(self.data_dir, 'metadata.csv'))

    #     # Get the y values
    #     self._y_array = torch.LongTensor(metadata_df['y'].values)
    #     self._y_size = 1
    #     self._n_classes = 2

    #     self._metadata_array = torch.stack(
    #         (torch.LongTensor(metadata_df['place'].values), self._y_array),
    #         dim=1
    #     )
    #     self._metadata_fields = ['background', 'y']
    #     self._metadata_map = {
    #         'background': [' land', 'water'], # Padding for str formatting
    #         'y': [' landbird', 'waterbird']
    #     }

    #     # Extract filenames
    #     self._input_array = metadata_df['img_filename'].values
    #     self._original_resolution = (224, 224)

    #     # Extract splits
    #     self._split_scheme = split_scheme
    #     if self._split_scheme != 'official':
    #         raise ValueError(f'Split scheme {self._split_scheme} not recognized')
    #     self._split_array = metadata_df['split'].values

    #     self._eval_grouper = CombinatorialGrouper(
    #         dataset=self,
    #         groupby_fields=(['background', 'y']))

    #     super(WaterbirdsDataset, self).__init__(root_dir, download, split_scheme)
    
    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        metadata = metadata[:2] # remove "from source domain" metadata 
        # switch first and second element 
        metadata = metadata[[1, 0]] # y, background
        return x, y, metadata

def get_waterbirds_datasets(
    mix_rate: Optional[float] = 0.5,
    source_cc: bool = True,
    transform: Optional[Callable] = None, 
    convert_to_tensor: bool = True,
    val_split: float = 0.2, 
    target_val_split: float = 0.0, 
    reverse_order: bool = False,
    reverse_target: bool = False
):
    transform_list = []
    if convert_to_tensor:
        transform_list.append(transforms.ToTensor())
    if transform is not None:
        transform_list.append(transform)
    transform = transforms.Compose(transform_list)

    dataset = CustomWaterbirdsDataset(root_dir="./data/waterbirds", download=True)

    # source 
    source_mask = (dataset.split_array == 0)
    if source_cc and reverse_order:
        # use same indexing as with cub, grouping indexes by group (0, 0), (1, 1)
        # TODO: figure out why this is wrong / not aligned with cub
        source_idxs = np.where(source_mask)[0]
        group_0_idxs = np.where(torch.all(dataset.metadata_array[source_idxs][:, :2] == torch.tensor([0,0]), dim=1))[0]
        group_1_idxs = np.where(torch.all(dataset.metadata_array[source_idxs][:, :2] == torch.tensor([1,1]), dim=1))[0]
        source_cc_idxs = np.concatenate([group_0_idxs, group_1_idxs])
        source_idxs = source_idxs[source_cc_idxs]
    else: 
        if source_cc:
            source_mask = source_mask & (dataset.metadata_array[:, 0] == dataset.metadata_array[:, 1]).numpy()
        source_idxs = np.where(source_mask)[0]


    # target 
    if mix_rate is None: 
        target_idxs = np.where(dataset.split_array == 1)[0]
    else:
        target_mask = dataset.split_array == 1
        # compute current mix rate 
        num_ood = (dataset.metadata_array[target_mask][:, 0] != dataset.metadata_array[target_mask][:, 1]).sum().item()
        num_id = (dataset.metadata_array[target_mask][:, 0] == dataset.metadata_array[target_mask][:, 1]).sum().item()
        cur_mix_rate = num_ood / sum(target_mask)
        # if less than target, remove ood instances (iid = ood/mix_rate - ood)
        if cur_mix_rate < mix_rate:
            num_id_target = int((num_ood / mix_rate) - num_ood)
            id_idxs = np.where(target_mask & (dataset.metadata_array[:, 0] == dataset.metadata_array[:, 1]).numpy())[0]
            id_idxs = id_idxs[:num_id_target]
            ood_idxs = np.where(target_mask & (dataset.metadata_array[:, 0] != dataset.metadata_array[:, 1]).numpy())[0]
        else: # if greate than target, remove iid instances (ood = ood/mix_rate - id)
            # mix rate = (ood) / (ood + id)
            # -> (ood + id) * mix rate = ood 
            # -> mix_rate -1 * ood = - id * mix rate 
            # -> ood = id * mix rate / (1 - mix rate)
            num_ood_target = int(num_id * mix_rate / (1 - mix_rate))
            ood_idxs = np.where(target_mask & (dataset.metadata_array[:, 0] != dataset.metadata_array[:, 1]).numpy())[0]
            ood_idxs = ood_idxs[:num_ood_target]
            id_idxs = np.where(target_mask & (dataset.metadata_array[:, 0] == dataset.metadata_array[:, 1]).numpy())[0]
    
        target_idxs = np.concatenate([id_idxs, ood_idxs])
    
    if reverse_target:
        target_idxs = target_idxs[::-1]

    # test 
    test_mask = dataset.split_array == 2
    test_idxs = np.where(test_mask)[0]

    # create datasets
    source = WILDSSubset(dataset, source_idxs, transform)
    target = WILDSSubset(dataset, target_idxs, transform)
    test = WILDSSubset(dataset, test_idxs, transform)

    # split datasets
    if val_split > 0:
        source_train, source_val = random_split(
            source, 
            [round(len(source) * (1 - val_split)), round(len(source) * val_split)],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        source_train = source 
        source_val = []
    if target_val_split > 0:
        target_train, target_val = random_split(
            target, 
            [round(len(target) * (1 - target_val_split)), round(len(target) * target_val_split)], 
            generator=torch.Generator().manual_seed(42)
        )
    else:
        target_train = target 
        target_val = []


    return source_train, source_val, target_train, target_val, test