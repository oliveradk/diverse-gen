import torch as t 
from typing import Literal, Optional
from itertools import product

from scipy.stats import binom

def compute_head_losses(
        logits: t.Tensor, heads: int, classes: int, binary: bool = False
):
    # all paris of heads and labels 
    # if heads=2, classes=3, then 
    # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    head_label_groups = list(product(range(heads), range(classes)))
    device = logits.device

    # define the criterion and label shape based on whether heads are binary or not
    if binary: 
        assert classes == 2
        criterion = t.nn.functional.binary_cross_entropy_with_logits
        label_shape = logits[:, 0].shape
        dtype = logits.dtype
    else:
        criterion = t.nn.functional.cross_entropy
        label_shape = (logits.shape[0],)
        dtype = t.long

    # compute the loss for each head-label pair
    lossses = {}
    for (head, label) in head_label_groups:
        lossses[(head, label)] = criterion(
            logits[:, head], t.ones(label_shape, dtype=dtype, device=device) * label, reduction='none'
        )
    return lossses

def compute_group_losses(
        head_losses: dict[tuple[int, int], t.Tensor], heads: int, classes: int
):
    # get all groups of heads and labels
    # e.g. if heads=2, classes=3, then groups are 
    # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    groups = list(product(range(classes), repeat=heads))
    group_losses = {}
    for group in groups:
        group_losses[group] = sum(head_losses[head, label] for head, label in enumerate(group))
    return group_losses

def compute_loss(losses: t.Tensor, mix_rate: float, mode: Literal['exp', 'prob', 'topk']):
    assert losses.ndim == 1
    bs = losses.shape[0]
    if mode == 'exp':
        exp_weight = t.exp(-t.arange(bs, device=losses.device))
        loss = (losses * exp_weight).mean()
    elif mode == 'prob':
        prob_weight = t.tensor([binom.pmf(i, bs, mix_rate) / i for i in range(bs)])
        loss = (losses * prob_weight).sum()
    elif mode == 'topk':
        k = round(bs * mix_rate)
        loss = t.topk(losses, k=k, largest=False).values.mean()
    return loss

class ACELoss(t.nn.Module):


    def __init__(
        self, 
        heads=2,
        classes=2,
        binary: bool = False,
        mode: Literal['exp', 'prob', 'topk'] = 'exp',
        inbalance_ratio: bool = False,
        mix_rate: Optional[float] = None,
        group_mix_rates: Optional[dict[tuple[int, ...], float]] = None,
        pseudo_label_all_groups: bool = False,
        device: str = "cpu"
    ):
        super().__init__()
        assert heads == 2
        self.heads = heads # assume 2 heads for now
        self.classes = classes
        self.binary = binary
        self.mode = mode
        self.inbalance_ratio = inbalance_ratio 
        self.mix_rate = mix_rate
        self.group_mix_rates = group_mix_rates
        self.pseudo_label_all_groups = pseudo_label_all_groups
        self.device = device
    
    def forward(self, logits, bs=None):
        """
        Args:
            logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS * CLASSES].
            (or [BATCH_SIZE, HEADS] if binary)
        """
        assert logits.shape[1] == self.heads * self.classes if not self.binary else self.heads
        if bs is None:
            bs = logits.shape[0]


        # reshape logits to [BATCH_SIZE, HEADS, CLASSES] if not binary
        if not self.binary:
            logits = logits.view(bs, self.heads, self.classes)
        
        # compute losses for each head and each label (n_heads * n_classes)
        head_losses = compute_head_losses(logits, self.heads, self.classes, self.binary)
        # compute losses for each group (a set of labels for each head)
        group_losses = compute_group_losses(head_losses, self.heads, self.classes)
        # remove agreeing groups
        if not self.pseudo_label_all_groups: 
            group_losses = {group: loss for group, loss in group_losses.items() if len(set(group)) > 1}
       
        # min over the group index (so each instance only gets one pseudo-label)
        if self.mix_rate is not None:
            group_losses_stacked = t.stack(list(group_losses.values()), dim=0)
            assert group_losses_stacked.shape == (len(group_losses), logits.shape[0])
            losses = group_losses_stacked.min(dim=0).values

            loss = compute_loss(losses, self.mix_rate, self.mode)
        # compute loss per group
        else: 
            loss = 0 
            for group, group_mix_rate in self.group_mix_rates.items():
                losses = group_losses[group]
                # TODO: ensure one pseudo-label per instance?
                loss += compute_loss(losses, group_mix_rate, self.mode)
        return loss
