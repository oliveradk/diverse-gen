import torch as t 
from typing import Literal, Optional
from itertools import product

from scipy.stats import binom

def compute_head_losses(logits_per_head: list[t.Tensor], classes_per_head: list[int], binary: bool):
    # all pairs of heads and labels 
    # e.g. if classes_per_head = [3, 2], then 
    # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
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
    # [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    feature_label_ls = list(product(*[range(c) for c in classes_per_head]))
    group_losses = {}
    for feature_label in feature_label_ls:
        group_losses[feature_label] = sum(head_losses[head, label] for head, label in enumerate(feature_label))
    return group_losses

def compute_loss(
    losses: t.Tensor, mix_rate: float, mode: Literal['exp', 'prob', 'topk'], 
    virtual_bs: Optional[int] = None
):
    assert losses.ndim == 1
    bs = losses.shape[0]
    if mode == 'exp':
        exp_weight = t.exp(-t.arange(bs, device=losses.device))
        loss = (losses * exp_weight).mean()
    elif mode == 'prob':
        prob_weight = t.tensor([binom.pmf(i, bs, mix_rate) / i for i in range(bs)])
        loss = (losses * prob_weight).sum()
    elif mode == 'topk':
        topk_bs = virtual_bs if virtual_bs is not None else bs
        k = round(topk_bs * mix_rate)
        loss = t.topk(losses, k=k, largest=False).values.mean()
    return loss

class ACELoss(t.nn.Module):

    def __init__(
        self, 
        classes_per_head: list[int] = [2, 2],
        mode: Literal['exp', 'prob', 'topk'] = 'exp',
        mix_rate: Optional[float] = None,
        group_mix_rates: Optional[dict[tuple[int, ...], float]] = None,
        disagree_only: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        self.classes_per_head = classes_per_head
        self.mode = mode
        self.mix_rate = mix_rate
        self.group_mix_rates = group_mix_rates
        self.disagree_only = disagree_only
        self.device = device
        self.binary = all([c == 1 for c in self.classes_per_head])

        if self.disagree_only:
            assert all([c == self.classes_per_head[0] for c in self.classes_per_head])

        assert (self.mix_rate is not None) ^ (self.group_mix_rates is not None) # carrot is xor
    
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
        if self.mix_rate is not None:
            group_losses_stacked = t.stack(list(group_losses.values()), dim=0)
            assert group_losses_stacked.shape == (len(group_losses), logits.shape[0])
            losses = group_losses_stacked.min(dim=0).values

            loss = compute_loss(losses, self.mix_rate, self.mode, virtual_bs)
        # compute loss per group
        else: 
            loss = 0 
            for group, group_mix_rate in self.group_mix_rates.items():
                losses = group_losses[group]
                # TODO: ensure one pseudo-label per instance?
                loss += compute_loss(losses, group_mix_rate, self.mode, virtual_bs)
        return loss
