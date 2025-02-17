import torch
import torch.nn as nn

from diverse_gen.utils.utils import batch_size

class MultiHeadBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, classes: list[int], feature_dim: int, dropout_rate: float = 0.0):
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
        self.dropout_rate = dropout_rate
        # Create a single matrix for all heads
        self.heads = nn.Linear(feature_dim, sum(classes))
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Get features from the shared backbone
        bs = batch_size(x)
        features = self.backbone(x).view(bs, -1)
        # Apply dropout
        features = self.dropout(features)
        # Apply the heads to the features
        outputs = self.heads(features)
        
        # Reshape the output to separate the heads # TODO: this doens't actually do that
        outputs = outputs.view(bs, sum(self.classes))
        
        return outputs

