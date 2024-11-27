import torch as t 
from typing import Literal, Optional

from scipy.stats import binom

def compute_head_losses(logits: t.Tensor):
    criterion = t.nn.functional.binary_cross_entropy_with_logits
    head_0_0 = criterion(
        logits[:, 0], t.zeros_like(logits[:, 0]), reduction='none'
    )
    head_0_1 = criterion(
        logits[:, 0], t.ones_like(logits[:, 0]), reduction='none'
    )
    head_1_0 = criterion(
        logits[:, 1], t.zeros_like(logits[:, 1]), reduction='none'
    )
    head_1_1 = criterion(
        logits[:, 1], t.ones_like(logits[:, 1]), reduction='none'
    )
    return head_0_0, head_0_1, head_1_0, head_1_1


class ACELoss(t.nn.Module):


    def __init__(
        self, 
        heads=2,
        mode: Literal['exp', 'prob', 'topk'] = 'exp',
        inbalance_ratio: bool = False,
        normalize_prob: bool = True,
        l_01_rate: Optional[float]=0.25, 
        l_10_rate: Optional[float]=0.25, 
        l_00_rate: Optional[float]=None,
        l_11_rate: Optional[float]=None,
        all_unlabeled: bool = False,
        device: str = "cpu"
    ):
        super().__init__()
        assert heads == 2
        self.heads = heads # assume 2 heads for now
        self.mode = mode
        self.inbalance_ratio = inbalance_ratio 
        self.normalize_prob = normalize_prob
        self.l_01_rate = l_01_rate
        self.l_10_rate = l_10_rate
        if l_00_rate is None and l_11_rate is None:
            iid_rate = 1 - l_01_rate - l_10_rate
            l_00_rate = l_11_rate = iid_rate / 2
        self.l_00_rate = l_00_rate
        self.l_11_rate = l_11_rate
        self.all_unlabeled = all_unlabeled
        self.device = device

    
    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS].
        """
        # L_{0,1} = -log(1-p_1) -log(p_2)
        # L_{1,0} = -log(p_1) - log(1-p_2)
        # L_{1, 0} + L_{0, 1} = -log((1-p_1) * p_2) + -log((1-p_2) * p_1)
        # focal weight (0, 1) = (1 - (1-p_1) * p_2) ^ gamma

        # I guess easier to just seperate them 
        head_0_0, head_0_1, head_1_0, head_1_1 = compute_head_losses(logits)
        loss_0_1 = head_0_0 + head_1_1
        loss_1_0 = head_0_1 + head_1_0 
        loss_0_0 = head_0_0 + head_1_0
        loss_1_1 = head_0_1 + head_1_1
        # sort losses in ascending order
        loss_0_1, _ = loss_0_1.sort()
        loss_1_0, _ = loss_1_0.sort()
        loss_0_0, _ = loss_0_0.sort()
        loss_1_1, _ = loss_1_1.sort()
        bs = logits.shape[0]
        if self.mode == 'exp':
            # apply exponential weight [e^-0, ..., e^N]
            exp_weight = t.exp(-t.arange(bs, device=self.device))
            loss_0_1 = loss_0_1 * exp_weight
            loss_1_0 = loss_1_0 * exp_weight
            loss_0_0 = loss_0_0 * exp_weight
            loss_1_1 = loss_1_1 * exp_weight
            losses = [loss_0_1, loss_1_0]
            if self.all_unlabeled:
                losses.extend([loss_0_0, loss_1_1])
            loss = t.stack(losses).mean()
            return loss
        elif self.mode == 'prob':
            weights_01 = t.zeros(bs, device=self.device)
            weights_10 = t.zeros(bs, device=self.device)
            weights_00 = t.zeros(bs, device=self.device)
            weights_11 = t.zeros(bs, device=self.device)
            for i in range(1, bs+1):
                weight_update_01 = binom.pmf(i, bs, self.l_01_rate) / i
                weight_update_10 = binom.pmf(i, bs, self.l_10_rate) / i
                weight_update_00 = binom.pmf(i, bs, self.l_00_rate) / i
                weight_update_11 = binom.pmf(i, bs, self.l_11_rate) / i
                if self.inbalance_ratio:
                    weight_update_01 *= bs / i
                    weight_update_10 *= bs / i
                    weight_update_00 *= bs / i
                    weight_update_11 *= bs / i
                weights_01[:i] += weight_update_01
                weights_10[:i] += weight_update_10
                weights_00[:i] += weight_update_00
                weights_11[:i] += weight_update_11
            if self.normalize_prob:
                weights_01 *= 1 / (1 - binom.pmf(0, bs, self.l_01_rate))
                weights_10 *= 1 / (1 - binom.pmf(0, bs, self.l_10_rate))
                weights_00 *= 1 / (1 - binom.pmf(0, bs, self.l_00_rate))
                weights_11 *= 1 / (1 - binom.pmf(0, bs, self.l_11_rate))
            loss_0_1 = loss_0_1 * weights_01
            loss_1_0 = loss_1_0 * weights_10
            loss_0_0 = loss_0_0 * weights_00
            loss_1_1 = loss_1_1 * weights_11
            loss = loss_0_1 + loss_1_0
            if self.all_unlabeled:
                loss += loss_0_0 + loss_1_1
            return loss
        elif self.mode == 'topk':
            n_0_1 = round(bs * self.l_01_rate)
            n_1_0 = round(bs * self.l_10_rate)
            n_0_0 = round(bs * self.l_00_rate)
            n_1_1 = round(bs * self.l_11_rate)
            loss_0_1 = loss_0_1[:n_0_1].mean()
            loss_1_0 = loss_1_0[:n_1_0].mean()
            loss_0_0 = loss_0_0[:n_0_0].mean()
            loss_1_1 = loss_1_1[:n_1_1].mean()
            loss = loss_0_1 + loss_1_0
            if self.all_unlabeled:
                loss += loss_0_0 + loss_1_1
            return loss


