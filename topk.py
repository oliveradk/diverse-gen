import torch

def topk_loss(logits, criterion=torch.nn.functional.binary_cross_entropy_with_logits, 
              r_01=0.25, r_10=0.25):
    """
    Args:
        logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS].
        criterion (torch.nn.Module): Loss criterion.
        r_01 (float): Expected proportion of samples labels (0,1).
        r_10 (float): Expected proportion of samples labels (1,0).
    """
    # get batch size
    bs = logits.shape[0]
    # compute head losses
    head_0_0 = criterion(
        logits[:, 0], torch.zeros_like(logits[:, 0]), reduction='none'
    )
    head_0_1 = criterion(
        logits[:, 0], torch.ones_like(logits[:, 0]), reduction='none'
    )
    head_1_0 = criterion(
        logits[:, 1], torch.zeros_like(logits[:, 1]), reduction='none'
    )
    head_1_1 = criterion(
        logits[:, 1], torch.ones_like(logits[:, 1]), reduction='none'
    )
    # compute disagreement losses
    loss_0_1 = head_0_0 + head_1_1
    loss_1_0 = head_0_1 + head_1_0 
    # sort losses in ascending order
    loss_0_1, _ = loss_0_1.sort()
    loss_1_0, _ = loss_1_0.sort()
    # compute top k losses
    k_01 = round(bs * r_01)
    k_10 = round(bs * r_10)
    loss_0_1 = loss_0_1[:k_01].mean()
    loss_1_0 = loss_1_0[:k_10].mean()
    # compute total loss
    loss = loss_0_1 + loss_1_0
    return loss