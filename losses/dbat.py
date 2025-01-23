import torch as t 
from typing import Literal, Optional

class DBatLoss(t.nn.Module):


    def __init__(self, heads, n_classes):
        super().__init__()
        assert heads == 2
        self.heads = heads # assume 2 heads for now
        self.n_classes = n_classes

    
    def forward(self, logits, m=1):
        """
        Args:
            logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS * CLASSES].
            m (int): the index of the head currently being trained
        """
        logits_per_head = logits.split([self.n_classes for _ in range(self.heads)], dim=1)
        if self.n_classes == 1:
            probs = t.concat([t.sigmoid(head_logits) for head_logits in logits_per_head], dim=1)
        else:
            probs = t.concat([t.softmax(head_logits, dim=1) for head_logits in logits_per_head], dim=1)
        
        if self.n_classes > 2: # multclass classification (for 2 heads)
            argmax_prob = probs[:, m].argmax(dim=1)
            probs = probs[argmax_prob, :]
        
        loss = -t.log(probs[:, 0] * (1 - probs[:, 1]) + probs[:, 1] * (1 - probs[:, 0]) +1e-7)
        return loss.mean()
