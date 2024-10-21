import torch as t 
from typing import Literal, Optional

from scipy.stats import binom

def compute_head_losses(probs: t.Tensor):
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
    return head_0_0, head_0_1, head_1_0, head_1_1

class ACELoss(t.nn.Module):


    def __init__(
        self, 
        heads=2,
        mode: Literal['focal', 'exp', 'prob', 'conf'] = 'exp',
        gamma: Optional[int]=1, 
        inbalance_ratio: bool = False,
        normalize_prob: bool = True,
        l_01_rate: Optional[float]=0.25, 
        l_10_rate: Optional[float]=0.25, 
    ):
        super().__init__()
        assert heads == 2
        self.heads = heads # assume 2 heads for now
        self.gamma = gamma
        self.mode = mode
        self.inbalance_ratio = inbalance_ratio 
        self.normalize_prob = normalize_prob
        self.l_01_rate = l_01_rate
        self.l_10_rate = l_10_rate

    
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
        assert probs.shape == logits.shape

        # I guess easier to just seperate them 
        head_0_0, head_0_1, head_1_0, head_1_1 = compute_head_losses(probs)
        if self.mode == 'exp':
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
        elif self.mode == 'prob':
            # expected value of cross-term losses 
            # (assuming ordering is correct)
            loss_0_1 = head_0_0 + head_1_1
            loss_1_0 = head_1_0 + head_0_1
            # sort losses in ascending order
            loss_0_1, _ = loss_0_1.sort()
            loss_1_0, _ = loss_1_0.sort()
            bs = logits.shape[0]
            weights_01 = t.zeros(bs)
            weights_10 = t.zeros(bs)
            for i in range(1, bs+1):
                weight_update_01 = binom.pmf(i, bs, self.l_01_rate) / i
                weight_update_10 = binom.pmf(i, bs, self.l_10_rate) / i
                if self.inbalance_ratio:
                    weight_update_01 *= bs / i
                    weight_update_10 *= bs / i
                weights_01[:i] += weight_update_01
                weights_10[:i] += weight_update_10
            if self.normalize_prob:
                weights_01 *= 1 / (1 - binom.pmf(0, bs, self.l_01_rate))
                weights_10 *= 1 / (1 - binom.pmf(0, bs, self.l_10_rate))
            loss_0_1 = loss_0_1 * weights_01
            loss_1_0 = loss_1_0 * weights_10
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
            loss = weight * t.min(t.stack([loss_0_1, loss_1_0], dim=-1), dim=-1).values
        elif self.mode == 'focal':
            # old version (focal weight)
            focal_weight_0_1 = (1 - (1-probs[:, 0].detach()) * probs[:, 1].detach()) ** self.gamma
            focal_weight_1_0 = (1 - probs[:, 0].detach() * (1-probs[:, 1]).detach()) ** self.gamma
            loss_0_1 = (head_0_0 + head_1_1) * focal_weight_0_1
            loss_1_0 = (head_1_0 + head_0_1) * focal_weight_1_0
            loss = loss_0_1 + loss_1_0
        
        return loss.mean()


