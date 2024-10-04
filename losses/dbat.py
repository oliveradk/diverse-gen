import torch as t 
from typing import Literal, Optional

class DBatLoss(t.nn.Module):


    def __init__(self, heads=2):
        super().__init__()
        assert heads == 2
        self.heads = heads # assume 2 heads for now

    
    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS].
        """
        probs = t.sigmoid(logits)
        loss = - t.log(probs[:, 0] * (1 - probs[:, 1]) + probs[:, 1] * (1 - probs[:, 0]) +1e-7)
        return loss.mean()