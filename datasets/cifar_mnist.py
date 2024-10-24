import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

# TODO use Resnet50 normalization and resizing
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(
        x.repeat(3, 1, 1), 
        (2, 2, 2, 2), 
        fill=0, 
        padding_mode='constant')
    )
])


def generate_dataset(mnist_data, cifar_data, mix_rate_0_9, mix_rate_1_1, vertical=False, transform=None):
    # filter by labels
    mnist_0 = [(img, label) for img, label in mnist_data if label == 0]
    mnist_1 = [(img, label) for img, label in mnist_data if label == 1]
    cifar_1 = [(img, label) for img, label in cifar_data if label == 1]
    cifar_9 = [(img, label) for img, label in cifar_data if label == 9]
    # get number of samples
    num_samples = min(len(mnist_0), len(mnist_1), len(cifar_1), len(cifar_9))
    data_pairs = []
    num_clean = int(num_samples * (1-mix_rate_0_9 - mix_rate_1_1)) 
    num_mixed_0_9 = int(num_samples * mix_rate_0_9) 
    num_mixed_1_1 = int(num_samples * mix_rate_1_1) 
    i = 0
    for _ in range(num_clean // 2):
        # cars and 0's
        data_pairs.append(((cifar_1[i][0], mnist_0[i][0]), 0, (0, 0))) 
        # trucks and 1's
        data_pairs.append(((cifar_9[i][0], mnist_1[i][0]), 1, (1, 1)))
        i+=1
    for _ in range(num_mixed_0_9):
        # trucks and 0's
        data_pairs.append(((cifar_9[i][0], mnist_0[i][0]), 1, (1, 0)))
        i+=1
    for _ in range(num_mixed_1_1):
        # cars and 1's
        data_pairs.append(((cifar_1[i][0], mnist_1[i][0]), 0, (0, 1)))
        i+=1
    # construct dataset
    images, labels, group_labels = zip(*data_pairs)
    # concatenate images
    cat_dim = 1 if vertical else 2
    images = [torch.cat([mnist_img, cifar_img], dim=cat_dim) for cifar_img, mnist_img in images]
    images = torch.stack(images)
    # labels and group labels 
    labels = torch.tensor(labels).to(torch.float32)
    group_labels = torch.tensor([list(gl) for gl in group_labels]).to(torch.float32)
    # shuffle dataset
    shuffle = torch.randperm(len(images))
    images = images[shuffle]
    if transform is not None:
        images = transform(images)
    labels = labels[shuffle]
    group_labels = group_labels[shuffle]
    dataset = TensorDataset(images, labels, group_labels)
    return dataset

def get_cifar_mnist_datasets(vertical=True, mix_rate_0_9=0.5, mix_rate_1_1=0.5, transform=None):
    # get full datasets 
    mnist_train = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True, transform=mnist_transform)
    cifar_train = torchvision.datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST('./data/mnist/', train=False, download=True, transform=mnist_transform)
    cifar_test = torchvision.datasets.CIFAR10('./data/cifar10/', train=False, download=True, transform=transforms.ToTensor())

    mnist_target, mnist_train, mnist_source_val, mnist_target_val = random_split(mnist_train, [45000, 10000, 2500, 2500], generator=torch.Generator().manual_seed(42))
    cifar_target, cifar_train, cifar_source_val, cifar_target_val = random_split(cifar_train, [35000, 10000, 2500, 2500], generator=torch.Generator().manual_seed(42))

    source_train = generate_dataset(mnist_train, cifar_train, mix_rate_0_9=0.0, mix_rate_1_1=0.0, vertical=vertical, transform=transform)
    source_val = generate_dataset(mnist_source_val, cifar_source_val, mix_rate_0_9=0.0, mix_rate_1_1=0.0, vertical=vertical, transform=transform)
    target_train = generate_dataset(mnist_target, cifar_target, mix_rate_0_9=mix_rate_0_9, mix_rate_1_1=mix_rate_1_1, vertical=vertical, transform=transform)
    target_val = generate_dataset(mnist_target_val, cifar_target_val, mix_rate_0_9=mix_rate_0_9, mix_rate_1_1=mix_rate_1_1, vertical=vertical, transform=transform)
    target_test = generate_dataset(mnist_test, cifar_test, mix_rate_0_9=0.25, mix_rate_1_1=0.25, vertical=vertical, transform=transform)
    return source_train, source_val, target_train, target_val, target_test



