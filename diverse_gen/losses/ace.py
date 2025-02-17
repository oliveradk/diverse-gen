import torch as t 
import numpy as np
from typing import Literal, Optional
from itertools import product

from scipy.stats import binom

def get_gls(classes_per_head: list[int]):
    return list(product(*[range(c) for c in classes_per_head]))

def compute_head_losses(logits_per_head: list[t.Tensor], classes_per_head: list[int], binary: bool):
    # all pairs of heads and labels, e.g. if classes_per_head = [3, 2],  
    # head_label_groups = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    head_label_groups = [(head, label) for head, classes in enumerate(classes_per_head) for label in range(classes)]
    
    device = logits_per_head[0].device

    # define the criterion and label shape based on whether heads are binary or not
    if binary: 
        assert all([c == 2 for c in classes_per_head])
        criterion = t.nn.functional.binary_cross_entropy_with_logits
        dtype = logits_per_head[0].dtype
        logits_per_head = [logit.squeeze(-1) for logit in logits_per_head]
        # label_shape = logits_per_head[0].shape
    else:
        criterion = t.nn.functional.cross_entropy
        dtype = t.long
    label_shape = (logits_per_head[0].shape[0],)

    # compute the loss for each head-label pair
    lossses = {}
    for (head, label) in head_label_groups:
        lossses[(head, label)] = criterion(
            logits_per_head[head], t.ones(label_shape, dtype=dtype, device=device) * label, reduction='none'
        )
    return lossses

def compute_group_losses(
    head_losses: dict[tuple[int, int], t.Tensor], classes_per_head: list[int]
):
    # get all groups of heads and labels
    # e.g. if classes_per_head = [3, 2], then 
    # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    gls = get_gls(classes_per_head)
    group_losses = {}
    for gl in gls:
        group_losses[gl] = sum(head_losses[head, label] for head, label in enumerate(gl))
    return group_losses

def compute_loss(
    losses: t.Tensor, mix_rate: float, mode: Literal['exp', 'prob', 'topk'], 
    virtual_bs: Optional[int] = None
):
    assert losses.ndim == 1
    bs = losses.shape[0]
    if mode == 'exp':
        exp_weight = t.exp(-t.arange(bs, device=losses.device))
        losses = t.sort(losses, dim=0, descending=False).values
        loss = (losses * exp_weight).mean()
    elif mode == 'prob':
        prob_weight = t.tensor([binom.pmf(i, bs, mix_rate) / i for i in range(bs)]).to(losses.device)
        losses = t.sort(losses, dim=0, descending=False).values
        loss = (losses * prob_weight).sum()
    elif mode == 'topk':
        topk_bs = virtual_bs if virtual_bs is not None else bs
        k = round(topk_bs * mix_rate)
        loss = t.topk(losses, k=k, largest=False).values.mean()
    return loss

def check_mix_rates_match(mix_rate: float, group_mix_rates: dict[tuple[int, ...], float]):
    assert np.isclose(sum(group_mix_rates.values()), mix_rate)

class ACELoss(t.nn.Module):

    def __init__(
        self, 
        mix_rate: float,
        classes_per_head: list[int] = [2, 2],
        mode: Literal['exp', 'prob', 'topk'] = 'exp',
        minority_groups: Optional[list[tuple[int, ...]]] = None,
        group_mix_rates: Optional[dict[tuple[int, ...], float]] = None,
        disagree_only: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        self.classes_per_head = classes_per_head
        self.mode = mode
        self.mix_rate = mix_rate
        self.disagree_only = disagree_only
        self.device = device
        self.group_mix_rates = group_mix_rates
        self.binary = all([c == 1 for c in self.classes_per_head])
        self.minority_groups = self._set_minority_groups(minority_groups)
        
        if self.group_mix_rates is not None:
            check_mix_rates_match(self.mix_rate, self.group_mix_rates)
    
    def _set_minority_groups(self, minority_groups: Optional[list[tuple[int, ...]]]):
        if minority_groups is not None:
            # check aligns with mix rates 
            if self.group_mix_rates is not None:
                assert set(minority_groups) == set(self.group_mix_rates.keys())
            # is disagree only, check all disagree  
            if self.disagree_only:
                assert all([len(set(group)) > 1 for group in minority_groups])
        elif self.group_mix_rates is not None:
            minority_groups = list(self.group_mix_rates.keys())
        elif self.disagree_only:
            minority_groups = [
                gl for gl in get_gls(self.classes_per_head) 
                if len(set(gl)) > 1
            ]
        else:
            minority_groups = get_gls(self.classes_per_head)
        return minority_groups
    
    def update_mix_rate(self, 
        mix_rate_update: float, 
        group_mix_rate_update: dict[tuple[int, ...], float] | None = None
    ):
        if group_mix_rate_update is not None: 
            self.mix_rate = mix_rate_update
            self.group_mix_rates = group_mix_rate_update
            check_mix_rates_match(self.mix_rate, self.group_mix_rates)
        else: 
            self.mix_rate = mix_rate_update
            if self.group_mix_rates is not None:
                n_groups = len(self.group_mix_rates)
                self.group_mix_rates = {group: mix_rate_update / n_groups 
                                        for group in self.group_mix_rates.keys()}
    
    def forward(self, logits, virtual_bs: Optional[int] = None):
        """
        Args:
            logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS * CLASSES].
            (or [BATCH_SIZE, HEADS] if binary)
        """
        assert logits.shape[1] == sum(self.classes_per_head)
        bs = logits.shape[0]


        # reshape logits to [BATCH_SIZE, HEADS, CLASSES] if not binary
        # if not self.binary:
        logits_per_head = logits.split(self.classes_per_head, dim=1)
        
        # compute losses for each head and each label (n_heads * n_classes)
        classes_per_head = [n_classes if n_classes != 1 else 2 for n_classes in self.classes_per_head]
        head_losses = compute_head_losses(logits_per_head, classes_per_head, self.binary)
        # compute losses for each group (a set of labels for each head)
        group_losses = compute_group_losses(head_losses, classes_per_head)
        # remove agreeing groups
        if self.disagree_only: 
            group_losses = {group: loss for group, loss in group_losses.items() if len(set(group)) > 1}
       
        # min over the group index (so each instance only gets one pseudo-label)
        if self.group_mix_rates is None:
            group_losses_stacked = t.stack(list(group_losses.values()), dim=0)
            assert group_losses_stacked.shape == (len(group_losses), logits.shape[0])
            losses = group_losses_stacked.min(dim=0).values

            loss = compute_loss(losses, self.mix_rate, self.mode, virtual_bs)
        # compute loss per group
        else: 
            loss = t.tensor([0.0], device=self.device) 
            for group, group_mix_rate in self.group_mix_rates.items():
                losses = group_losses[group]
                # TODO: ensure one pseudo-label per instance?
                loss += compute_loss(losses, group_mix_rate, self.mode, virtual_bs)
        return loss


class MixRateScheduler:
    """Scheduler for mix rates with linear annealing."""
    def __init__(
        self,
        loss_fn: ACELoss,
        mix_rate_lb: float | None = None,
        t0: int = 0,
        t1: int | None = None,
        interval_size: float | None = None,
        total_steps: int | None = None
    ):
        self.loss_fn = loss_fn
        self.mix_rate_lb = mix_rate_lb
        self.t0 = t0
        self.t1 = t1
        self.interval_size = interval_size
        self.total_steps = total_steps
        self.last_epoch = -1
        self.last_step = -1

    def step(self, epoch=None, step=None):
        """Update mix rates based on epoch or step."""
        if self.interval_size is not None:
            # Step-based scheduling
            if step is None:
                step = self.last_step + 1
            self.last_step = step
            
            if step % int(self.total_steps * self.interval_size) == 0:
                progress = step / self.total_steps
                self._update_rates(progress)
        else:
            # Epoch-based scheduling
            if epoch is None:
                epoch = self.last_epoch + 1
            self.last_epoch = epoch
            
            if epoch < self.t0:
                self._update_rates(0.0)
            elif epoch >= self.t1:
                self._update_rates(1.0)
            else:
                progress = (epoch - self.t0) / (self.t1 - self.t0)
                self._update_rates(progress)

    def _update_rates(self, progress):
        self.loss_fn.update_mix_rate(mix_rate_update=self.mix_rate_lb * progress)   

