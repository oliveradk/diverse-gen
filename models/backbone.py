import torch
import torch.nn as nn

class MultiHeadBackbone(nn.Module):
    def __init__(self, backbone, n_heads, feature_dim, classes):
        super(MultiHeadBackbone, self).__init__()
        self.backbone = backbone
        self.n_heads = n_heads
        self.classes = classes
        
        # Create a single matrix for all heads
        self.heads = nn.Linear(feature_dim, n_heads * classes)
    
    def forward(self, x):
        # Get features from the shared backbone
        features = self.backbone(x).view(x.size(0), -1)
        
        # Apply the heads to the features
        outputs = self.heads(features)
        
        # Reshape the output to separate the heads
        batch_size = features.size(0)
        outputs = outputs.view(batch_size, self.n_heads * self.classes)
        
        return outputs

