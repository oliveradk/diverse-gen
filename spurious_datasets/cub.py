import numpy as np

from spurious_datasets.subpopulation.cub_dataset import CUBDataset
from spurious_datasets.subpopulation.dro_dataset import DRODataset
from spurious_datasets.subpopulation.folds import Subset



def get_cub_datasets():

    data_dir = "/nas/ucb/oliveradk/diverse-gen/data/waterbirds/waterbirds_v1.0"
    augment_data = False
    full_dataset = CUBDataset(data_dir, augment_data)

    # groups: 
    group_map = { # this could be wrong # TODO: check
        0: (0, 0), 
        1: (0, 1), 
        2: (1, 0), 
        3: (1, 1)
    }

    def process_item_fn(batch):
        x, y, g, _ = batch
        g = group_map[g]
        return x, y, np.array(g)

    splits = ["train", "val", "test"]
    subsets = full_dataset.get_splits(splits, train_frac=1.0)

    train_data, val_data, test_data = [
        DRODataset(
            subsets[split],
            process_item_fn=process_item_fn,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )
        for split in splits
    ]

    majority_groups = (0, 3) # originally (0, 3), seeing if changing lowers performance

    majority_idxs = [
        np.where(train_data.get_group_array() == i)[0] for i in majority_groups
    ]
    print("Majority idxs", majority_idxs)
    majority_idxs = np.concatenate(majority_idxs)
    temp_train_data = DRODataset(
        Subset(train_data, majority_idxs),
        process_item_fn=None,
        n_groups=train_data.n_groups,
        n_classes=train_data.n_classes,
        group_str_fn=train_data.group_str,
    )

    return temp_train_data, val_data, test_data