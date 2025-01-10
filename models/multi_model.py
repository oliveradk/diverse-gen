import torch
import torch.nn as nn
from typing import Callable, List
from contextlib import contextmanager

from utils.utils import batch_size

class MultiNetModel(nn.Module): 
    
    def __init__(self, 
                 model_builder: Callable[[], nn.Module], n_heads: int, feature_dim: int, 
                 classes: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.classes = classes
        self.backbones = nn.ModuleList([model_builder() for _ in range(n_heads)])
        self.heads = nn.ModuleList([nn.Linear(feature_dim, classes) for _ in range(n_heads)])
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = batch_size(x)
        features = [backbone(x).view(bs, -1) for backbone in self.backbones]
        outs = [head(feature) for head, feature in zip(self.heads, features)]
        out = torch.cat(outs, dim=-1)
        out = out.view(bs, self.n_heads * self.classes) # for parity with multi-head backbone
        return out
    
    def freeze_head(self, head_idx: int):
        self.backbones[head_idx].requires_grad_(False)
        self.heads[head_idx].requires_grad_(False)
    
    def unfreeze_head(self, head_idx: int):
        self.backbones[head_idx].requires_grad_(True)
        self.heads[head_idx].requires_grad_(True)


@contextmanager
def freeze_heads(model: MultiNetModel, head_idxs: List[int]):
    for head_idx in head_idxs:
        model.freeze_head(head_idx)
    try:
        yield
    finally:
        for head_idx in head_idxs:
            model.unfreeze_head(head_idx)
