from torch import nn

class HFWrapper(nn.Module):
    def __init__(self, backbone, out_key: str="pooler_output"):
        super().__init__()
        self.backbone = backbone
        self.out_key = out_key
    
    def forward(self, x):
        out = self.backbone(**x)
        return out[self.out_key]