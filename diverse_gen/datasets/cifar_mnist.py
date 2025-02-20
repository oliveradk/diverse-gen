import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from .dominos import gen_dominos_dataset, mnist_transform

# TODO: make pill dataset, not tensor, apply transforms when loading (use dataloader workers to speed up)

def get_cifar_mnist_datasets(
    vertical=True, 
    source_mix_rate_0_1: int | None = None,
    source_mix_rate_1_0: int | None = None,
    target_mix_rate_0_1: int | None = None, 
    target_mix_rate_1_0: int | None = None, 
    transform=None, 
    pad_sides=False, 
    seed: int = 42
):
    # set default mix rates
    if source_mix_rate_0_1 is None:
        source_mix_rate_0_1 = 0.0
    if source_mix_rate_1_0 is None:
        source_mix_rate_1_0 = 0.0
    if target_mix_rate_0_1 is None:
        target_mix_rate_0_1 = 0.25
    if target_mix_rate_1_0 is None:
        target_mix_rate_1_0 = 0.25

    # get full datasets 
    mnist_train = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True, transform=mnist_transform)
    cifar_train = torchvision.datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST('./data/mnist/', train=False, download=True, transform=mnist_transform)
    cifar_test = torchvision.datasets.CIFAR10('./data/cifar10/', train=False, download=True, transform=transforms.ToTensor())

    mnist_target, mnist_train, mnist_source_val, mnist_target_val = random_split(
        mnist_train, 
        [45000, 10000, 2500, 2500], 
        generator=torch.Generator().manual_seed(seed)
    )
    cifar_target, cifar_train, cifar_source_val, cifar_target_val = random_split(
        cifar_train, 
        [35000, 10000, 2500, 2500], 
        generator=torch.Generator().manual_seed(seed)
    )

    labels_a = [1, 9]
    labels_b = [0, 1]
    ds_name = "cifar-mnist"
    source_train = gen_dominos_dataset(ds_name, "source_train", cifar_train, mnist_train, mix_rate_0_1=source_mix_rate_0_1, mix_rate_1_0=source_mix_rate_1_0, vertical=vertical, transform=transform, pad_sides=pad_sides, labels_a=labels_a, labels_b=labels_b)
    source_val = gen_dominos_dataset(ds_name, "source_val", cifar_source_val, mnist_source_val, mix_rate_0_1=source_mix_rate_0_1, mix_rate_1_0=source_mix_rate_1_0, vertical=vertical, transform=transform, pad_sides=pad_sides, labels_a=labels_a, labels_b=labels_b)
    target_train = gen_dominos_dataset(ds_name, "target_train", cifar_target, mnist_target, mix_rate_0_1=target_mix_rate_0_1, mix_rate_1_0=target_mix_rate_1_0, vertical=vertical, transform=transform, pad_sides=pad_sides, labels_a=labels_a, labels_b=labels_b)
    target_val = gen_dominos_dataset(ds_name, "target_val", cifar_target_val, mnist_target_val, mix_rate_0_1=target_mix_rate_0_1, mix_rate_1_0=target_mix_rate_1_0, vertical=vertical, transform=transform, pad_sides=pad_sides, labels_a=labels_a, labels_b=labels_b)
    target_test = gen_dominos_dataset(ds_name, "target_test", cifar_test, mnist_test, mix_rate_0_1=0.25, mix_rate_1_0=0.25, vertical=vertical, transform=transform, pad_sides=pad_sides, labels_a=labels_a, labels_b=labels_b)
    return source_train, source_val, target_train, target_val, target_test



