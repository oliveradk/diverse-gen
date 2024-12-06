import numpy as np

import torch
from torch.utils.data import random_split
from torchvision import transforms

from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset


class CustomWaterbirdsDataset(WaterbirdsDataset):
    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        metadata = metadata[:2] # remove "from source domain" metadata 
        # switch first and second element 
        metadata = metadata[[1, 0]] # y, background
        return x, y, metadata

def get_waterbirds_datasets(
    mix_rate=0.5,
    source_cc=True,
    transform=None, 
    val_split=0.2, 
):
    transform_list = [transforms.ToTensor()]
    if transform is not None:
        transform_list.append(transform)
    transform = transforms.Compose(transform_list)

    dataset = CustomWaterbirdsDataset(root_dir="./data/waterbirds", download=True)

    # source 
    source_mask = (dataset.split_array == 0)
    if source_cc:
        source_mask = source_mask & (dataset.metadata_array[:, 0] == dataset.metadata_array[:, 1]).numpy()
    source_idxs = np.where(source_mask)[0]

    # target 
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

    # test 
    test_mask = dataset.split_array == 2
    test_idxs = np.where(test_mask)[0]

    # create datasets
    source = WILDSSubset(dataset, source_idxs, transform)
    target = WILDSSubset(dataset, target_idxs, transform)
    test = WILDSSubset(dataset, test_idxs, transform)

    # split datasets
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


    return source_train, source_val, target_train, target_val, test