import os
from functools import partial
from typing import Optional
import string 
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset
from tqdm import tqdm

from diverse_gen.datasets.utils import get_group_idxs, update_idxs_from_mix_rate


class MultiNLIDataset(Dataset):
    # OG dataset entailment (0), neutral (1), contradiction (2)
    def __init__(self, dataset, tokenizer, max_length=128, combine_neut_entail=False, 
                 idxs: Optional[np.ndarray]=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.combine_neut_entail = combine_neut_entail
        
        labels = np.array(self.dataset["label"])
        if self.combine_neut_entail: # convert to binary classification
            labels = (labels == 2).astype(int) # 2:= contradiction
        negation = np.array(self.dataset["sentence2_has_negation"])
        
        self.labels = labels
        self.feature_labels = np.stack((labels, negation), axis=1)
        self.idxs = idxs
        if self.idxs is not None:
            self.dataset = self.dataset.select(self.idxs)
            self.labels = self.labels[self.idxs]
            self.feature_labels = self.feature_labels[idxs]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Tokenize the premise and hypothesis
        encoding = self.tokenizer(
            item['premise'],
            item['hypothesis'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert label to tensor
        label = torch.tensor(self.labels[idx]).unsqueeze(-1)
        group_label = torch.tensor(self.feature_labels[idx])

        encoding = {k:v.squeeze(0) for k, v in encoding.items()}
        
        return encoding, label, group_label


def get_split(dataset: MultiNLIDataset, idxs: Optional[np.ndarray]):
    if dataset.idxs is not None:
        idxs = dataset.idxs[idxs]
    return MultiNLIDataset(dataset.dataset, dataset.tokenizer, dataset.max_length, dataset.combine_neut_entail, idxs=idxs)


def tokenize(s):
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.lower()
    s = s.split(' ')
    return s


def get_setence_has_negation(dataset):
    negation_words = ['nobody', 'no', 'never', 'nothing']
    setence_has_negation = []
    for example in tqdm(dataset):
        setence_has_negation.append(any(word in tokenize(example["hypothesis"]) for word in negation_words))
    return setence_has_negation


def load_negation_metadata(dataset, filename, data_dir) -> np.ndarray:
    if not os.path.exists(os.path.join(data_dir, filename)):
        negations = get_setence_has_negation(dataset)
        try: 
            with open(os.path.join(data_dir, filename), "w") as f:
                for has_negation in negations:
                    f.write(json.dumps({"sentence2_has_negation": has_negation}))
                    f.write("\n")
        except Exception as e:
            print(e)
    else: 
        with open(os.path.join(data_dir, filename), "r") as f:
            negations = [json.loads(line)["sentence2_has_negation"] for line in f]
    return negations


def get_multi_nli_datasets(
    mix_rate: Optional[float]=None,
    source_cc: bool = True, 
    val_split=0.2,
    target_val_split=0.2,
    tokenizer=None,
    max_length=128,
    dataset_length=None,
    combine_neut_entail=False,
    contra_no_neg=True, 
    seed: int = 42
):
    assert tokenizer is not None, "Tokenizer is required"
    
    dataset = load_dataset("multi_nli")
    data_dir = "./data/multinli_1.0"
    neg_filename_train = "multinli_1.0_neg_train.jsonl"
    neg_filename_val = "multinli_1.0_neg_dev_matched.jsonl"
    neg_filename_test = "multinli_1.0_neg_dev_mismatched.jsonl"

    # get negation metadata
    negations_train = load_negation_metadata(dataset["train"], neg_filename_train, data_dir)
    negations_val = load_negation_metadata(dataset["validation_matched"], neg_filename_val, data_dir)
    negations_test = load_negation_metadata(dataset["validation_mismatched"], neg_filename_test, data_dir)
    
    # add negation metadata to dataset
    dataset["train"] = dataset["train"].add_column("sentence2_has_negation", negations_train)
    dataset["validation_matched"] = dataset["validation_matched"].add_column("sentence2_has_negation", negations_val)
    dataset["validation_mismatched"] = dataset["validation_mismatched"].add_column("sentence2_has_negation", negations_test)

    # filter dataset
    if dataset_length is not None:
        dataset['train'] = dataset['train'].select(random.sample(range(len(dataset['train'])), dataset_length))
        dataset['validation_matched'] = dataset['validation_matched'].select(random.sample(range(len(dataset['validation_matched'])), dataset_length))
        dataset['validation_mismatched'] = dataset['validation_mismatched'].select(random.sample(range(len(dataset['validation_mismatched'])), dataset_length))
    
    source = MultiNLIDataset(dataset['train'], tokenizer, max_length, combine_neut_entail=combine_neut_entail)
    target = MultiNLIDataset(dataset['validation_matched'], tokenizer, max_length, combine_neut_entail=combine_neut_entail)
    test = MultiNLIDataset(dataset['validation_mismatched'], tokenizer, max_length, combine_neut_entail=combine_neut_entail)

    classes_per_feature = [3,2] 
    cc_groups = [(0, 0), (1, 0), (2, 1)]
    if contra_no_neg:
        cc_groups.append((2, 0))
    if combine_neut_entail:
        classes_per_feature = [2, 2]
        cc_groups = [(0, 0), (1, 1)]
        if contra_no_neg:
            cc_groups.append((1, 0))
    
    if source_cc: 
        cc_idxs = get_group_idxs(source.feature_labels, [np.array(cc_group) for cc_group in cc_groups])
        source = get_split(source, cc_idxs)
    
    # update target idxs based on mix rate (preserves group balance)
    if mix_rate is not None:
        target_idxs = update_idxs_from_mix_rate(
            target.feature_labels, mix_rate, 
            cc_groups=cc_groups, classes_per_feature=classes_per_feature
        )
        target = get_split(target, target_idxs)

    # split datasets
    if val_split > 0:
        source_train_size = int(len(source) * (1 - val_split))
        source_train, source_val = random_split(
            source, 
            [source_train_size, len(source) - source_train_size],
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        source_train = source 
        source_val = []
    if target_val_split > 0:
        target_train_size = int(len(target) * (1 - target_val_split))
        target_train, target_val = random_split(
            target, 
            [target_train_size, len(target) - target_train_size], 
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        target_train = target 
        target_val = []

    return source_train, source_val, target_train, target_val, test

    
