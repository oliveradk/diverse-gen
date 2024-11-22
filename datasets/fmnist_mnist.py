import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from .dominos import gen_dominos_dataset, mnist_transform

def get_fmnist_mnist_datasets(
    vertical=True, 
    source_mix_rate_0_1=0.0,
    source_mix_rate_1_0=0.0,
    target_mix_rate_0_1=0.5, 
    target_mix_rate_1_0=0.5, 
    transform=None, 
    pad_sides=False, 
    seed=42
):
    # get full datasets 
    mnist_train = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True, transform=mnist_transform)
    fmnist_train = torchvision.datasets.FashionMNIST('./data/fashion_mnist/', train=True, download=True, transform=mnist_transform)
    mnist_test = torchvision.datasets.MNIST('./data/mnist/', train=False, download=True, transform=mnist_transform)
    fmnist_test = torchvision.datasets.FashionMNIST('./data/fashion_mnist/', train=False, download=True, transform=mnist_transform)
    # get splits
    mnist_target, mnist_train, mnist_source_val, mnist_target_val = random_split(mnist_train, [45000, 10000, 2500, 2500], generator=torch.Generator().manual_seed(seed))
    fmnist_target, fmnist_train, fmnist_source_val, fmnist_target_val = random_split(fmnist_train, [45000, 10000, 2500, 2500], generator=torch.Generator().manual_seed(seed))

    # generate datasets
    labels_a = [0, 1]
    labels_b = [0, 1]   
    source_train = gen_dominos_dataset(fmnist_train, mnist_train, mix_rate_0_1=source_mix_rate_0_1, mix_rate_1_0=source_mix_rate_1_0, vertical=vertical, transform=transform, pad_sides=pad_sides, labels_a=labels_a, labels_b=labels_b)
    source_val = gen_dominos_dataset(fmnist_source_val, mnist_source_val, mix_rate_0_1=source_mix_rate_0_1, mix_rate_1_0=source_mix_rate_1_0, vertical=vertical, transform=transform, pad_sides=pad_sides, labels_a=labels_a, labels_b=labels_b)
    target_train = gen_dominos_dataset(fmnist_target, mnist_target, mix_rate_0_1=target_mix_rate_0_1, mix_rate_1_0=target_mix_rate_1_0, vertical=vertical, transform=transform, pad_sides=pad_sides, labels_a=labels_a, labels_b=labels_b)
    target_val = gen_dominos_dataset(fmnist_target_val, mnist_target_val, mix_rate_0_1=target_mix_rate_0_1, mix_rate_1_0=target_mix_rate_1_0, vertical=vertical, transform=transform, pad_sides=pad_sides, labels_a=labels_a, labels_b=labels_b)
    target_test = gen_dominos_dataset(fmnist_test, mnist_test, mix_rate_0_1=0.25, mix_rate_1_0=0.25, vertical=vertical, transform=transform, pad_sides=pad_sides, labels_a=labels_a, labels_b=labels_b)
    return source_train, source_val, target_train, target_val, target_test

