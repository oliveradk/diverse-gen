import numpy as np
import torch as t
import os

from matplotlib import pyplot as plt


def generate_data(
    num_datapoints, 
    quadrant_proportions=None, 
    mix_rate=None, 
    train=False, 
    swap_y_meaning=False, 
    gaussian=False,
    std=1.0
):
    """
    Generate data with customizable proportions in each quadrant.
    
    :param num_datapoints: Total number of data points to generate.
    :param quadrant_proportions: List of 4 values representing the proportion of data in each quadrant.
                                 Order: [Q1, Q2, Q3, Q4]. Should sum to 1.
    :param train: If True, x2 is correlated with x1. If False, they are uncorrelated.
    :param mix_rate: If not None, the proportion of OOD data in the dataset.
    :param swap_y_meaning: If True, y is based on x2 > 0. If False, y is based on x1 < 0.
    :return: Tuple of (x, y) where x is a 2D array and y is a 1D array.
    """

    if train: 
        assert mix_rate is None 
        assert quadrant_proportions is None
        quadrant_proportions = [0, 0.5, 0, 0.5]
    if mix_rate is not None:
        assert quadrant_proportions is None 
        iid_props = (1-mix_rate)/2 
        ood_probs = mix_rate / 2 
        quadrant_proportions = [ood_probs, iid_props, ood_probs, iid_props]
    elif quadrant_proportions is None:
        quadrant_proportions = [0.25, 0.25, 0.25, 0.25]
    
    assert len(quadrant_proportions) == 4
    assert abs(sum(quadrant_proportions) - 1) < 1e-6, "Quadrant proportions must sum to 1"

    # Calculate number of points for each quadrant
    quadrant_points = [int(num_datapoints * prop) for prop in quadrant_proportions]
    quadrant_points[-1] += num_datapoints - sum(quadrant_points)  # Adjust for rounding errors

    x1, x2, y = [], [], []

    for q, n_points in enumerate(quadrant_points):
        if gaussian:
            if q == 0:  # Q1: x1 > 0 0, x2 > 0
                x1_q, x2_q = np.random.multivariate_normal(
                    mean=[0.5, 0.5], 
                    cov=[[std, 0], [0, std]], 
                    size=n_points
                ).T
            elif q == 1:  # Q2: x1 < 0, x2 > 0
                x1_q, x2_q = np.random.multivariate_normal(
                    mean=[-0.5, 0.5], 
                    cov=[[std, 0], [0, std]], 
                    size=n_points
                ).T
            elif q == 2:  # Q3: x1 < 0, x2 < 0
                x1_q, x2_q = np.random.multivariate_normal(
                    mean=[-0.5, -0.5], 
                    cov=[[std, 0], [0, std]], 
                    size=n_points
                ).T
            else:  # Q4: x1 > 0, x2 < 0
                x1_q, x2_q = np.random.multivariate_normal(
                    mean=[0.5, -0.5], 
                    cov=[[std, 0], [0, std]], 
                    size=n_points
                ).T
            x1_q = x1_q.reshape(-1, 1)
            x2_q = x2_q.reshape(-1, 1)
        else:
            if q == 0:  # Q1: x1 > 0 0, x2 > 0
                x1_q = np.random.uniform(0, 1, (n_points, 1))
                x2_q = np.random.uniform(0, 1, (n_points, 1))
            elif q == 1:  # Q2: x1 < 0, x2 > 0
                x1_q = np.random.uniform(-1, 0, (n_points, 1))
                x2_q = np.random.uniform(0, 1, (n_points, 1))
            elif q == 2:  # Q3: x1 < 0, x2 < 0
                x1_q = np.random.uniform(-1, 0, (n_points, 1))
                x2_q = np.random.uniform(-1, 0, (n_points, 1))
            else:  # Q4: x1 > 0, x2 < 0
                x1_q = np.random.uniform(0, 1, (n_points, 1))
                x2_q = np.random.uniform(-1, 0, (n_points, 1))

        x1.extend(x1_q)
        x2.extend(x2_q)
        y.extend((x1_q < 0).astype(int))

    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    x = np.concatenate([x1, x2], 1)
    
    if swap_y_meaning:
        y = (x2 > 0).astype(int)

    return x, y

def plot_data(data, title=""):
    x, y = data
    
    plt.figure(figsize=(6, 6))
    
    for g, c in [(0, "#E7040F"), (1, "#00449E")]:
        x_g = x[y.flatten() == g]
        plt.scatter(x_g[:, 0], x_g[:, 1], s=30, c=c, edgecolors="k", label=f"Class {g}")
    
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.legend()
    plt.title(title)
    
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')


def sample_minibatch(data, batch_size):
    x, y = data
    minibatch_idx = np.random.randint(0, x.shape[0], size=batch_size)
    return (
        t.tensor(x[minibatch_idx]).float(),
        t.tensor(y[minibatch_idx]).float(),
    )

def savefig(name, transparent=False, pdf=False):
    # FIG_ROOT = "figures"
    # os.makedirs(FIG_ROOT, exist_ok=True)
    modes = ["png"]
    if pdf:
        modes += ["pdf"]
    for mode in modes:
        file_name = f"{name}.{mode}"
        if transparent:
            plt.savefig(file_name, dpi=300, bbox_inches="tight", transparent=True)
        else:
            plt.savefig(file_name, dpi=300, facecolor="white", bbox_inches="tight")
    plt.clf()