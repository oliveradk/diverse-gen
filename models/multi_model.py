import torch
import torch.nn as nn
from typing import Callable, List
from contextlib import contextmanager

from utils.utils import batch_size

class MultiNetModel(nn.Module): 
    
    def __init__(self, 
        model_builder: Callable[[], nn.Module], classes: list[int], feature_dim: int, 
    ) -> None:
        """
        Multi-head model 

        model_builder: function to build a backbone
        classes: list of classes for each feature
            for binary classification with 2 heads, classes = [2, 2]
            (len(classes) = n_heads)
        feature_dim: dimension of the feature space
        """
        super().__init__()
        self.classes = classes
        self.backbones = nn.ModuleList([model_builder() for _ in range(len(classes))])
        self.heads = nn.ModuleList([nn.Linear(feature_dim, c) for c in classes])
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = batch_size(x)
        features = [backbone(x).view(bs, -1) for backbone in self.backbones]
        outs = [head(feature) for head, feature in zip(self.heads, features)]
        out = torch.cat(outs, dim=-1)
        out = out.view(bs, sum(self.classes)) # for parity with multi-head backbone
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
