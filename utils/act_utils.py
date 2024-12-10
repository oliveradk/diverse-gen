from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def get_acts_and_labels(model: nn.Module, loader: DataLoader, device: str):
    activations = []
    labels = []
    for x, y, gl in tqdm(loader):
        x, y, gl = x.to(device), y.to(device), gl.to(device)
        acts = model(x)
        activations.append((acts.detach().cpu()))
        labels.append(gl)
    activations = torch.cat(activations, dim=0).squeeze()
    labels = torch.cat(labels, dim=0)
    labels = labels.squeeze()
    return activations, labels

def transform_activations(activations: torch.Tensor):
    pca = PCA(n_components=2)
    pca.fit(activations)
    activations_pca = pca.transform(activations)
    return activations_pca, pca


def plot_activations(
    model: Optional[nn.Module] = None, 
    activations: Optional[torch.Tensor] = None,
    loader: Optional[DataLoader] = None, 
    device: Optional[str] = None,
    labels: Optional[torch.Tensor] = None
):
    assert activations is not None or (model is not None and loader is not None), "Either activations or model and loader must be provided"
    if activations is None:
        activations, labels = get_acts_and_labels(model, loader, device)
    activations_pca, pca_transform = transform_activations(activations)


    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot first label
    scatter1 = ax1.scatter(activations_pca[:, 0], activations_pca[:, 1], c=labels[:, 0].to('cpu'), cmap="viridis")
    ax1.set_title('Label 0')

    # Plot second label
    scatter2 = ax2.scatter(activations_pca[:, 0], activations_pca[:, 1], c=labels[:, 1].to('cpu'), cmap="viridis")
    ax2.set_title('Label 1')

    fig.tight_layout()
    return fig