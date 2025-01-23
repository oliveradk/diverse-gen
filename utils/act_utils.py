from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

from utils.utils import feature_label_ls, group_labels_from_labels, to_device


def get_acts_and_labels(model: nn.Module, loader: DataLoader, device: str):
    activations = []
    labels = []
    with torch.no_grad():
        for x, y, gl in tqdm(loader):
            x, y, gl = to_device(x, y, gl, device)
            acts = model(x)
            activations.append(acts.cpu())
            labels.append(gl)
    activations = torch.cat(activations, dim=0).squeeze()
    labels = torch.cat(labels, dim=0)
    labels = labels.squeeze()
    return activations, labels

def pca_transform(activations: torch.Tensor):
    pca = PCA(n_components=2)
    pca.fit(activations)
    activations_pca = pca.transform(activations)
    return activations_pca, pca

def umap_transform(activations: torch.Tensor):
    reducer = umap.UMAP(n_components=2, random_state=42)
    activations_umap = reducer.fit_transform(activations)
    return activations_umap, reducer


def plot_activations(
    model: Optional[nn.Module] = None, 
    activations: Optional[torch.Tensor] = None,
    loader: Optional[DataLoader] = None, 
    device: Optional[str] = None,
    labels: Optional[torch.Tensor] = None,
    transform: str = "pca", 
    classes_per_feature: list[int] = [2, 2]
):
    assert activations is not None or (model is not None and loader is not None), "Either activations or model and loader must be provided"
    if activations is None:
        activations, labels = get_acts_and_labels(model, loader, device)
    if transform == "pca":
        transformed_acts, reducer = pca_transform(activations)
    elif transform == "umap":
        transformed_acts, reducer = umap_transform(activations)
    else:
        raise ValueError(f"Invalid transform: {transform}")


    # Create a figure with two subplots side by side
    fig, ax = plt.subplots(1, 1)

    # generate group labels 
    group_labels = group_labels_from_labels(classes_per_feature, labels)

    # Plot
    scatter = ax.scatter(
        transformed_acts[:, 0], 
        transformed_acts[:, 1], 
        c=group_labels.to('cpu'), 
        cmap="viridis", 
        s=30,
        alpha=0.5
    )
    group_labels = [f"{f_l}" for f_l in feature_label_ls(classes_per_feature)]
    ax.legend(scatter.legend_elements()[0], group_labels, title="Feature Labels")

    ax.set_title(f"Activations {transform}")


    fig.tight_layout()
    return fig, transformed_acts, reducer

def compute_probe_acc(activations, labels, classes_per_feat):
    from sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression(
        max_iter=10000, 
        multi_class='multinomial' if len(classes_per_feat) > 2 else 'ovr'
    )
    label_classes = labels[:, 0].unique()
    if len(label_classes) > 1:
        lr.fit(activations.to('cpu').numpy(), labels[:, 0].to('cpu').numpy())
        acc = lr.score(activations.to('cpu').numpy(), labels[:, 0].to('cpu').numpy())
    else: 
        acc = 0
    
    lr_alt = LogisticRegression(
        max_iter=10000, 
        multi_class='multinomial' if len(classes_per_feat) > 2 else 'ovr'
    )
    alt_label_classes = labels[:, 1].unique()
    if len(alt_label_classes) > 1:
        lr_alt.fit(activations.to('cpu').numpy(), labels[:, 1].to('cpu').numpy())
        alt_acc = lr_alt.score(activations.to('cpu').numpy(), labels[:, 1].to('cpu').numpy())
    else: 
        alt_acc = 0
    return acc, alt_acc