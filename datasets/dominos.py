import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset


mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(
        x.repeat(3, 1, 1), 
        (2, 2, 2, 2), 
        fill=0, 
        padding_mode='constant')
    )
])

def gen_dominos_dataset(
    dataset_a, 
    dataset_b, 
    mix_rate_0_1, 
    mix_rate_1_0, 
    vertical=True, 
    transform=None, 
    pad_sides=False, 
    labels_a=[0, 1], 
    labels_b=[0, 1]
):  
    # filter by labels
    dataset_a_0 = [(img, label) for img, label in dataset_a if label == labels_a[0]]
    dataset_a_1 = [(img, label) for img, label in dataset_a if label == labels_a[1]]
    dataset_b_0 = [(img, label) for img, label in dataset_b if label == labels_b[0]]
    dataset_b_1 = [(img, label) for img, label in dataset_b if label == labels_b[1]]

    num_samples = min(len(dataset_a_0), len(dataset_a_1), len(dataset_b_1), len(dataset_b_0))
    data_pairs = []
    num_clean = int(num_samples * (1-mix_rate_0_1 - mix_rate_1_0)) 
    num_mixed_0_1 = int(num_samples * mix_rate_0_1) 
    num_mixed_1_0 = int(num_samples * mix_rate_1_0) 
    i = 0
    for _ in range(num_clean // 2):
        # 0's
        data_pairs.append(((dataset_a_0[i][0], dataset_b_0[i][0]), 0, (0, 0))) 
        # 1's
        data_pairs.append(((dataset_a_1[i][0], dataset_b_1[i][0]), 1, (1, 1)))
        i+=1
    for _ in range(num_mixed_0_1):
        # 1 and 0's
        data_pairs.append(((dataset_a_1[i][0], dataset_b_0[i][0]), 1, (1, 0)))
        i+=1
    for _ in range(num_mixed_1_0):
        # 0's and 1's
        data_pairs.append(((dataset_a_0[i][0], dataset_b_1[i][0]), 0, (0, 1)))
        i+=1
    # construct dataset
    images, labels, group_labels = zip(*data_pairs)
    # concatenate images
    cat_dim = 1 if vertical else 2
    images = [torch.cat([dataset_b_img, dataset_a_img], dim=cat_dim) for dataset_a_img, dataset_b_img in images]
    images = torch.stack(images)
    if pad_sides: 
        padding = (16, 0) if vertical else (0, 16)
        images = F.pad(images, padding, fill=0, padding_mode='constant')
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