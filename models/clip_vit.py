from torch import nn
from transformers import CLIPVisionModel
class ClipViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    def forward(self, x):
        out = self.base(x)
        # CLS token output (could also mean-pool over all tokens, better performance https://arxiv.org/abs/2205.01580)
        return out.pooler_output 