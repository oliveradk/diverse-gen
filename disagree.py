import torch as t 
from typing import Literal, Optional

class DisagreeLoss(t.nn.Module):


    def __init__(self, heads=2, gamma=1, mode: Literal['focal', 'ace', 'conf'] = 'ace'):
        super().__init__()
        assert heads == 2
        self.heads = heads # assume 2 heads for now
        self.gamma = gamma
        self.mode = mode

    
    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS].
        """
        # L_{0,1} = -log(1-p_1) -log(p_2)
        # L_{1,0} = -log(p_1) - log(1-p_2)
        # L_{1, 0} + L_{0, 1} = -log((1-p_1) * p_2) + -log((1-p_2) * p_1)
        # focal weight (0, 1) = (1 - (1-p_1) * p_2) ^ gamma

        probs = t.sigmoid(logits)
        print(probs.max(), probs.min())
        assert probs.shape == logits.shape

        # I guess easier to just seperate them 
        head_0_0 = t.nn.functional.binary_cross_entropy(
            probs[:, 0], t.zeros_like(probs[:, 0]), reduction='none'
        )
        head_0_0 = t.nn.functional.binary_cross_entropy(
            probs[:, 0], t.zeros_like(probs[:, 0]), reduction='none'
        )
        head_0_1 = t.nn.functional.binary_cross_entropy(
            probs[:, 0], t.ones_like(probs[:, 0]), reduction='none'
        )
        head_1_0 = t.nn.functional.binary_cross_entropy(
            probs[:, 1], t.zeros_like(probs[:, 1]), reduction='none'
        )
        head_1_1 = t.nn.functional.binary_cross_entropy(
            probs[:, 1], t.ones_like(probs[:, 1]), reduction='none'
        )
        if self.mode == 'ace':
            loss_0_1 = head_0_0 + head_1_1
            loss_1_0 = head_1_0 + head_0_1
            # sort losses in ascending order
            loss_0_1, _ = loss_0_1.sort()
            loss_1_0, _ = loss_1_0.sort()
            # apply exponential weight [e^-0, ..., e^N]
            exp_weight = t.exp(-t.arange(loss_0_1.shape[0]))
            loss_0_1 = loss_0_1 * exp_weight
            loss_1_0 = loss_1_0 * exp_weight
            loss = loss_0_1 + loss_1_0
        elif self.mode == 'conf': # TODO: for weight, 
            # at first, encourages disagreement amoung uncertain samples 
            # then, encourages disaggreement amoung samples that disagree
            loss_0_1 = head_0_0 + head_1_1
            loss_1_0 = head_1_0 + head_0_1
            # uncertainty weight
            uncertainty_weight_0 = 0.5 - t.abs(probs[:, 0].detach() - 0.5)
            uncertainty_weight_1 = 0.5 - t.abs(probs[:, 1].detach() - 0.5)
            uncertainty_weight = uncertainty_weight_0 + uncertainty_weight_1
            # disagreement weight
            disaggrement_weight = t.abs(probs[:, 0] - probs[:, 1])
            # weight = t.max(t.stack([uncertainty_weight, disaggrement_weight], dim=-1), dim=-1).values
            weight = 1
            print("frac uncertainty weight", (uncertainty_weight > disaggrement_weight).float().mean())
            loss = weight * t.min(t.stack([loss_0_1, loss_1_0], dim=-1), dim=-1).values
        elif self.mode == 'focal':
            # old version (focal weight)
            focal_weight_0_1 = (1 - (1-probs[:, 0].detach()) * probs[:, 1].detach()) ** self.gamma
            focal_weight_1_0 = (1 - probs[:, 0].detach() * (1-probs[:, 1]).detach()) ** self.gamma
            loss_0_1 = (head_0_0 + head_1_1) * focal_weight_0_1
            loss_1_0 = (head_1_0 + head_0_1) * focal_weight_1_0
            loss = loss_0_1 + loss_1_0
        
        return loss.mean()


