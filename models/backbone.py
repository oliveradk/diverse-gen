import torch
import torch.nn as nn

from utils.utils import batch_size

class MultiHeadBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, classes: list[int], feature_dim: int):
        """
        Multi-head backbone 

        backbone: backbone to use for feature extraction
        classes: list of classes for each feature 
            for binary classification with 2 heads, classes = [2, 2] 
            (len(classes) = n_heads)
        feature_dim: dimension of the feature space
        """
        super(MultiHeadBackbone, self).__init__()
        self.backbone = backbone
        self.classes = classes
        
        # Create a single matrix for all heads
        self.heads = nn.Linear(feature_dim, sum(classes))
    
    def forward(self, x):
        # Get features from the shared backbone
        bs = batch_size(x)
        features = self.backbone(x).view(bs, -1)
        
        # Apply the heads to the features
        outputs = self.heads(features)
        
        # Reshape the output to separate the heads # TODO: this doens't actually do that
        outputs = outputs.view(bs, sum(self.classes))
        
        return outputs

