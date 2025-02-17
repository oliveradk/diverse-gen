import torch 
from torch import nn
import torch.nn.functional as F


def compute_src_loss(logits, y, gl, binary: bool, classes_per_head: list[int], use_group_labels: bool):
    logits_by_head = torch.split(logits, classes_per_head, dim=-1)
    if not use_group_labels:
        labels_by_head = [y for _ in classes_per_head]
    else:
        labels_by_head = [gl[:, i].unsqueeze(-1) for i in range(len(classes_per_head))]

    if binary:
        losses = [
            F.binary_cross_entropy_with_logits(logit.squeeze(-1), y.squeeze(-1).to(torch.float32)) 
            for logit, y in zip(logits_by_head, labels_by_head)
        ]
    else:
        losses = [F.cross_entropy(logit.squeeze(-1), y.squeeze(-1).to(torch.long)) 
                  for logit, y in zip(logits_by_head, labels_by_head)]
    return losses


class SrcLoss(nn.Module):
    def __init__(self, binary: bool, classes_per_head: list[int], use_group_labels: bool):
        super().__init__()
        self.binary = binary
        self.use_group_labels = use_group_labels
        self.classes_per_head = classes_per_head

    def forward(self, logits, y, gl):
        return compute_src_loss(logits, y, gl, self.binary, self.classes_per_head, self.use_group_labels)
