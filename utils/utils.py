from enum import Enum
from typing import Union
import torch


Input = Union[dict[str, torch.Tensor], torch.Tensor]

def conf_to_args(conf: dict):
    args = []
    for key, value in conf.items():
        # check if value is an enum 
        if isinstance(value, Enum):
            value = value.name 
        elif value is None:
            value = 'null'
        args.append(f"{key}={value}")
    return args


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