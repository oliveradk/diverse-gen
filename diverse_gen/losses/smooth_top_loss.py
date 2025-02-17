import torch
import torch.nn as nn

class SmoothTopLoss(nn.Module):
    def __init__(self, criterion, one_cross_class=False, all_unlabeled_classes=False, device="cpu"):
        super().__init__()
        self.criterion = criterion
        self.device = device
        self.one_cross_class = one_cross_class
        self.all_unlabeled_classes = all_unlabeled_classes

    def forward(self, inputs):
        # NOTE this is modified from the original implementation
        # inputs is [batch_size, n_heads]
        assert inputs.ndim == 2, 'Inputs must be 2D'
        # swap axes to get [n_heads, ..., batch_size]
        inputs = inputs.permute(-1, 0)


        # top k cross terms loss
        batch_size = inputs.size(-1)

        # get losses against 0's and 1's
        losses = torch.stack(
            (
                self.criterion(inputs, torch.zeros_like(inputs)),
                self.criterion(inputs, torch.ones_like(inputs)),
            )
        )

        # compute topk for all combinations of features except the pure 1,1,1... and 0,0,0... (since those are equivalent with labeled data)
        n = len(inputs)  # number of heads
        if n == 1:
            return torch.Tensor([0]).to(self.device).mean()
        idx_sets = torch.cartesian_prod(
            *[torch.LongTensor([0, 1]).to(self.device) for _ in range(n)]
        )
        if self.one_cross_class:
            new_sets = []
            for idx_set in idx_sets:
                add = True
                for j in range(n-1):
                    if idx_set[j] < idx_set[j+1]: add=False
                if add:
                    new_sets.append(idx_set.detach().clone())
#            assert n == 2, 'One cross class only supported for two features'
            # remove one of the heterogeneous classes (ie. [1, 0] or [0, 1]
#            idx_sets = torch.cat([idx_sets[:1], idx_sets[2:]])
            idx_sets = torch.stack(new_sets)
        if not self.all_unlabeled_classes:
            idx_sets = idx_sets[1:-1]  # remove [0, 0, ..., 0] and [1, 1, ..., 1]
        cross_type_losses = losses[
            idx_sets, torch.arange(n).repeat(len(idx_sets), 1), :
        ].sum(
            1
        )  # for each cross_type, get losses for that cross type
        smoother = 1 / (torch.arange(batch_size, device=self.device).exp())
        smoothed_losses = torch.sort(cross_type_losses, dim=-1).values * smoother
        return (
            smoothed_losses.mean()
        )  # multiply by number of heads for backwards compatibility