import numpy as np

from torchvision import transforms

from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset

def get_waterbirds_datasets(
    mix_rate=0.5,
    transform=None, 
    val_split=0.2, 
):
    transform_list = [transforms.ToTensor()]
    if transform is not None:
        transform_list.append(transform)
    transform = transforms.Compose(transform_list)

    dataset = WaterbirdsDataset(root_dir="./data/waterbirds", download=True)

    # source 
    source_mask = (dataset.split_array == 0) & (dataset.metadata_array[:, 0] == dataset.metadata_array[:, 1]).numpy()
    source_idxs = np.where(source_mask)[0]
    np.random.shuffle(source_idxs) # TODO: ideally remove this randomness
    source_train_idxs, source_val_idxs = np.split(source_idxs, [int(len(source_idxs) * (1 - val_split))])

    # target 
    target_mask = dataset.split_array == 1
    # compute current mix rate 
    num_ood = (dataset.metadata_array[target_mask][:, 0] != dataset.metadata_array[target_mask][:, 1]).sum().item()
    num_id = (dataset.metadata_array[target_mask][:, 0] == dataset.metadata_array[target_mask][:, 1]).sum().item()
    cur_mix_rate = num_ood / sum(target_mask)
    # if less than target, remove ood instances (iid = ood/mix_rate - ood)
    if cur_mix_rate < mix_rate:
        num_id_target = int((num_ood / mix_rate) - num_ood)
        id_idxs = np.where(target_mask & (dataset.metadata_array[:, 0] == dataset.metadata_array[:, 1]).numpy())[0]
        id_idxs = id_idxs[:num_id_target]
        ood_idxs = np.where(target_mask & (dataset.metadata_array[:, 0] != dataset.metadata_array[:, 1]).numpy())[0]
    else: # if greate than target, remove iid instances (ood = ood/mix_rate - id)
        # mix rate = (ood) / (ood + id)
        # -> (ood + id) * mix rate = ood 
        # -> mix_rate -1 * ood = - id * mix rate 
        # -> ood = id * mix rate / (1 - mix rate)
        num_ood_target = int(num_id * mix_rate / (1 - mix_rate))
        ood_idxs = np.where(target_mask & (dataset.metadata_array[:, 0] != dataset.metadata_array[:, 1]).numpy())[0]
        ood_idxs = ood_idxs[:num_ood_target]
        id_idxs = np.where(target_mask & (dataset.metadata_array[:, 0] == dataset.metadata_array[:, 1]).numpy())[0]
   
    target_idxs = np.concatenate([id_idxs, ood_idxs])
    np.random.shuffle(target_idxs) #TODO: ideally remove this randomness
    target_train_idxs, target_val_idxs = np.split(target_idxs, [int(len(target_idxs) * (1 - val_split))])

    # test 
    test_mask = dataset.split_array == 2
    test_idxs = np.where(test_mask)[0]
    
    source_train, source_val = WILDSSubset(dataset, source_train_idxs, transform), WILDSSubset(dataset, source_val_idxs, transform)
    target_train, target_val = WILDSSubset(dataset, target_train_idxs, transform), WILDSSubset(dataset, target_val_idxs, transform)
    test = WILDSSubset(dataset, test_idxs, transform)

    return source_train, source_val, target_train, target_val, test