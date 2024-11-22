from typing import Optional, Literal

import torch
from torch import nn
from torch.nn import functional as F


class ConfLoss(nn.Module):
    def __init__(self, p: float = 0.5, reduction: Optional[str] = 'mean'):
        """
        p: probability of 1 in the target distribution
        """
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Input logits with shape [BATCH_SIZE, 1].
        """
        assert logits.ndim == 2 and logits.shape[1] == 1
        # set t based on p
        t = torch.quantile(logits, 1-self.p)
        # compute pseudo-labels 
        pseudo_labels = (logits > t).float()
        # compute cross entropy 
        loss  = F.binary_cross_entropy_with_logits(
            logits, 
            pseudo_labels, 
            reduction=self.reduction,
        )
        return loss
