import torch
import torch.nn as nn
from typing import Callable


class MultiNetModel(nn.Module): # TODO: generalize this? either use this or add probes on top of common backbone
    def __init__(self, heads: int, model_builder: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.heads = nn.ModuleList([model_builder() for _ in range(heads)])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)