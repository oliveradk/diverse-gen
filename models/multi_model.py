import torch
import torch.nn as nn
from typing import Callable, List
from contextlib import contextmanager

from utils.utils import batch_size

class MultiNetModel(nn.Module): 
    
    def __init__(self, 
        model_builder: Callable[[], nn.Module], classes_per_head: list[int], feature_dim: int, dropout_rate: float = 0.0
    ) -> None:
        """
        Multi-head model 

        model_builder: function to build a backbone
        classes_per_head: list of classes for each feature
            for binary classification with 2 heads, classes = [2, 2]
            (len(classes) = n_heads)
        feature_dim: dimension of the feature space
        """
        super().__init__()
        self.classes_per_head = classes_per_head
        self.dropout_rate = dropout_rate
        self.backbones = nn.ModuleList([model_builder() for _ in range(len(classes_per_head))])
        self.heads = nn.ModuleList([nn.Linear(feature_dim, c) for c in classes_per_head])
        self.dropout = nn.Dropout(dropout_rate)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = batch_size(x)
        features = [backbone(x).view(bs, -1) for backbone in self.backbones]
        features = [self.dropout(feature) for feature in features]
        outs = [head(feature) for head, feature in zip(self.heads, features)]
        out = torch.cat(outs, dim=-1)
        out = out.view(bs, sum(self.classes_per_head)) # for parity with multi-head backbone
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
