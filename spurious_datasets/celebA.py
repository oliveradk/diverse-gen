import os
from PIL import Image
from typing import Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms


class CelebA(Dataset):
    def __init__(self, root_dir: str, anno_df: pd.DataFrame, transform,
                 gt_feat: str='Male', spur_feats: list[str]=['Blond_Hair']):
        self.root_dir = root_dir
        self.anno_df = anno_df
        self.gt_feat = gt_feat
        self.spur_feats = spur_feats
        self.transform = transform

    
    def __len__(self):
        return len(self.anno_df)
    
    def __getitem__(self, idx):
        filename = self.anno_df.iloc[idx]['index']
        img_path = os.path.join(self.root_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        y = torch.tensor(self.anno_df.iloc[idx].loc[self.gt_feat])
        spur_feats = torch.tensor(self.anno_df.iloc[idx].loc[self.spur_feats].astype(int).to_numpy())
        gl = torch.cat((y.unsqueeze(0), spur_feats))
        return img, y, gl
    

def get_celebA_datasets(
    mix_rate: Optional[float]=None,
    val_split=0.2,
    transform=None,
    gt_feat: str='Male',
    spur_feat: str='Blond_Hair',
    inv_spur_feat: bool=True
):
    
    transform_list = [transforms.ToTensor()]
    if transform is not None:
        transform_list.append(transform)
    transform = transforms.Compose(transform_list)

    root_dir = "./data/img_align_celeba"
    anno_path = os.path.join(root_dir, "list_attr_celeba.txt")
    splits_path = os.path.join(root_dir, "list_eval_partition.txt")
    
    annotations = pd.read_csv(anno_path, sep='\s+', header=0)
    annotations = annotations.applymap(lambda x: int(x)) # convert columns to int
    annotations = annotations.reset_index(drop=False) # sets filename as column, index as row number
    annotations = annotations.applymap(lambda x: 0 if x == -1 else x) # convert -1 to 0
    if inv_spur_feat: # invert spur_feat if inverse correlation
        annotations[spur_feat] = annotations[spur_feat].map(lambda x: 1-x)
    splits = pd.read_csv(splits_path, sep='\s+', header=None)

    # split into train, val, test  
    anno_source = annotations.loc[splits[1] == 0]
    anno_target = annotations.loc[splits[1] == 1]
    anno_test = annotations.loc[splits[1] == 2]

    ### Source ###
    # remove train instances without gennder hair correlation 
    anno_source = anno_source[anno_source[spur_feat] == anno_source[gt_feat]]

    ### Target ###
    target_id_idxs = np.where(anno_target[spur_feat] == anno_target[gt_feat])[0]
    target_ood_idxs = np.where(anno_target[spur_feat] != anno_target[gt_feat])[0]
    num_ood = (anno_target[spur_feat] != anno_target[gt_feat]).sum()
    num_id = len(anno_target) - num_ood
    cur_mix_rate = num_ood / len(anno_target)
    if mix_rate is None: 
        pass
    elif cur_mix_rate < mix_rate: # remove iid 
        num_id_target = int((num_ood / mix_rate) - num_ood)
        target_id_idxs = target_id_idxs[:num_id_target]
    else: # remove ood 
        num_ood_target = int(num_id * mix_rate / (1 - mix_rate))
        target_ood_idxs = target_ood_idxs[:num_ood_target]
    anno_target = anno_target.iloc[np.concatenate((target_id_idxs, target_ood_idxs))]

    # create datasets
    source = CelebA(root_dir, anno_source, transform, gt_feat, [spur_feat])
    target = CelebA(root_dir, anno_target, transform, gt_feat, [spur_feat])
    test = CelebA(root_dir, anno_test, transform, gt_feat, [spur_feat])

    # train val split
    source_train, source_val = torch.utils.data.random_split(
        source, 
        [round(len(source) * (1 - val_split)), round(len(source) * val_split)], 
        generator=torch.Generator().manual_seed(42)
    )
    target_train, target_val = torch.utils.data.random_split(
        target, 
        [round(len(target) * (1 - val_split)), round(len(target) * val_split)], 
        generator=torch.Generator().manual_seed(42)
    )

    return source_train, source_val, target_train, target_val, test

    
