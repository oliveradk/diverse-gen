from torch import nn

class PassThroughLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.tensor(0.0)
