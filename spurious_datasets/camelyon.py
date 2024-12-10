import numpy as np

import torch
from torch.utils.data import random_split
from torchvision import transforms

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset


class CustomCamelyonDataset(Camelyon17Dataset):
    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        # no clear alt label, just duplicate gt label
        gl = torch.tensor([y, y])
        return x, y, gl


def get_camelyon_datasets(
    transform=None, 
    val_split=0.2, 
):
    transform_list = [transforms.ToTensor()]
    if transform is not None:
        transform_list.append(transform)
    transform = transforms.Compose(transform_list)

    dataset = CustomCamelyonDataset(root_dir="./data/camelyon17", download=True)

    source = dataset.get_subset(split="train", transform=transform)
    target = dataset.get_subset(split="val", transform=transform)
    test = dataset.get_subset(split="test", transform=transform)

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

    return source_train, source_val, target_train, target_val, test