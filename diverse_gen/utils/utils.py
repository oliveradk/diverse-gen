from enum import Enum
from typing import Union
from itertools import product
from copy import deepcopy
import torch



Input = Union[dict[str, torch.Tensor], torch.Tensor]

def conf_to_args(conf: dict):
    args = []
    conf = deepcopy(conf)
    config_file = conf.pop("--config_file", None)
    if config_file is not None:
        args.append("--config_file")
        args.append(config_file)
    for key, value in conf.items():
        # check if value is an enum 
        if isinstance(value, Enum):
            value = value.name 
        elif value is None:
            value = 'null'
        elif isinstance(value, list):
            value = str(value)
            value = value.replace("None", "null")
        args.append(f"{key}={value}")
    return args

def str_to_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(i) for i in s.split("_"))


def to_device(x: Input, y: torch.Tensor, gl: torch.Tensor, device: torch.device):
    if isinstance(x, dict):
        x = {k: v.to(device) for k, v in x.items()}
    else: 
        x = x.to(device)
    y, gl = y.to(device), gl.to(device)
    return x, y, gl


def batch_size(x: Input):
    if isinstance(x, dict):
        return next(iter(x.values())).shape[0]
    else: 
        return x.shape[0]


def feature_label_ls(classes_per_head: list[int]):
    return list(product(*[range(c) for c in classes_per_head]))


def group_labels_from_labels(classes_per_feature: list[int], labels: torch.Tensor):
    # Calculate the unique index for each combination
    # Using the principle: each position contributes its value multiplied by 
    # the product of the number of classes in previous positions
    
    multipliers = torch.ones(len(classes_per_feature), dtype=torch.int64)
    for i in range(len(classes_per_feature)-2, -1, -1):
        multipliers[i] = multipliers[i+1] * classes_per_feature[i+1]
    
    # Convert to unique indices
    unique_labels = (labels * multipliers).sum(dim=1)
    
    return unique_labels



    
