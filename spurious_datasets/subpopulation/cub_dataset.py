import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from .confounder_dataset import ConfounderDataset


class CUBDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(
        self,
        data_dir,
        augment_data=False,
        mix_up=False,
        group_id=None,
        dataset=None,
    ):
        self.mix_up = mix_up
        self.data_dir = data_dir
        self.augment_data = augment_data

        root_dir = os.path.dirname(self.data_dir)

        if not os.path.exists(self.data_dir) and not os.path.exists(
            os.path.join(root_dir, "features", "cub.npy")
        ):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )

        # Read in metadata
        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))

        # Get the y values
        self.y_array = self.metadata_df["y"].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df["place"].values
        self.n_confounders = 1

        # Extract filenames and splits
        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {"train": 0, "val": 1, "test": 2}

        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (
            self.y_array * (self.n_groups / 2) + self.confounder_array
        ).astype("int")

        # if args.group_by_label:
        #     idxes = np.where(self.split_array == self.split_dict["train"])[0]
        #     self.group_array[idxes] = self.y_array[idxes]

        # Set transform
        # if model_attributes[self.model_type]['feature_type']=='precomputed':
        self.precomputed = True
        self.pretransformed = True
        if os.path.exists(os.path.join(root_dir, "features", "cub.npy")):
            self.features_mat = torch.from_numpy(
                np.load(
                    os.path.join(root_dir, "features", "cub.npy"), allow_pickle=True
                )
            ).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = []
            self.train_transform = get_transform_cub(
                train=True, augment_data=augment_data
            )
            self.eval_transform = get_transform_cub(
                train=False, augment_data=augment_data
            )

            for idx in tqdm(range(len(self.y_array))):
                img_filename = os.path.join(self.data_dir, self.filename_array[idx])
                img = Image.open(img_filename).convert("RGB")
                # Figure out split and transform accordingly
                if (
                    self.split_array[idx] == self.split_dict["train"]
                    and self.train_transform
                ):
                    img = self.train_transform(img)
                elif (
                    self.split_array[idx]
                    in [self.split_dict["val"], self.split_dict["test"]]
                    and self.eval_transform
                ):
                    img = self.eval_transform(img)
                # # Flatten if needed
                # if model_attributes[self.model_type]["flatten"]:
                #     assert img.dim() == 3
                #     img = img.view(-1)
                x = img
                self.features_mat.append(x)

            self.features_mat = torch.cat(
                [x.unsqueeze(0) for x in self.features_mat], dim=0
            )
            os.makedirs(os.path.join(root_dir, "features"))
            np.save(
                os.path.join(root_dir, "features", "cub.npy"), self.features_mat.numpy()
            )

        self.RGB = True

        self.y_array_onehot = torch.zeros(len(self.y_array), self.n_classes)
        self.y_array_onehot = self.y_array_onehot.scatter_(
            1, torch.tensor(self.y_array).unsqueeze(1), 1
        ).numpy()
        self.original_train_length = len(self.features_mat)

        if group_id is not None:
            idxes = np.where(self.group_array == group_id)
            self.select_samples(idxes)

    def select_samples(self, idxes):
        self.y_array = self.y_array[idxes]
        self.group_array = self.group_array[idxes]
        self.split_array = self.split_array[idxes]
        self.features_mat = self.features_mat[idxes]
        self.y_array_onehot = self.y_array_onehot[idxes]

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array


def get_transform_cub(train, augment_data, target_resolution=(224, 224)):
    scale = 256.0 / 224.0
    assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        int(target_resolution[0] * scale),
                        int(target_resolution[1] * scale),
                    )
                ),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transform
