import os
from functools import partial
from typing import Optional
import string 
import json

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm


class MultiNLIDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        label = torch.tensor(item['label'])
        # TODO: convert contradiction to 1
        has_negation = torch.tensor(item["sentence2_has_negation"])
        # contradition if negation else actual label
        alt_label = torch.tensor(2) if has_negation else label
        group_label = torch.stack((label, alt_label))

        encoding = {k:v.squeeze(0) for k, v in encoding.items()}
        
        return encoding, label, group_label




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

neg_filename_train = "multinli_1.0_neg_train.jsonl"
neg_filename_val = "multinli_1.0_neg_dev_matched.jsonl"
neg_filename_test = "multinli_1.0_neg_dev_mismatched.jsonl"

def load_negation_metadata(dataset, filename, data_dir):
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
    val_split=0.2,
    tokenizer=None,
    max_length=128,
    dataset_length=None,
):
    assert tokenizer is not None, "Tokenizer is required"
    
    dataset = load_dataset("multi_nli")
    data_dir = "./data/multinli_1.0"
    if dataset_length is not None:
        dataset['train'] = dataset['train'].select(range(dataset_length))
        dataset['validation_matched'] = dataset['validation_matched'].select(range(dataset_length))
        dataset['validation_mismatched'] = dataset['validation_mismatched'].select(range(dataset_length))

    # get negation metadata
    negations_train = load_negation_metadata(dataset["train"], neg_filename_train, data_dir)
    negations_val = load_negation_metadata(dataset["validation_matched"], neg_filename_val, data_dir)
    negations_test = load_negation_metadata(dataset["validation_mismatched"], neg_filename_test, data_dir)

    if dataset_length is not None:
        negations_train = negations_train[:dataset_length]
        negations_val = negations_val[:dataset_length]
        negations_test = negations_test[:dataset_length]
    
    # add negation metadata to dataset
    dataset["train"] = dataset["train"].add_column("sentence2_has_negation", negations_train)
    dataset["validation_matched"] = dataset["validation_matched"].add_column("sentence2_has_negation", negations_val)
    dataset["validation_mismatched"] = dataset["validation_mismatched"].add_column("sentence2_has_negation", negations_test)
    
    # remove neutral instances 
    dataset = dataset.filter(lambda ex: ex['label'] != 1)
    # set contradiction label to 1
    dataset = dataset.map(lambda x: {'label': 1 if x['label'] == 2 else x['label']})

    ### Source Distribution ###
    # filter source distribution (complete correlation)
    dataset['train'] = dataset['train'].filter(lambda ex: not (ex['label'] == 0 and ex["sentence2_has_negation"]))
    dataset['train'] = dataset['train'].filter(lambda ex: not (ex['label'] == 1 and not ex["sentence2_has_negation"]))

    # balance source dataset
    entailment_label_idxs = torch.where(torch.tensor(dataset['train']['label']) == 0)[0]
    contradiction_label_idxs = torch.where(torch.tensor(dataset['train']['label']) == 1)[0]

    num_entailments = entailment_label_idxs.shape[0]
    num_contradictions = contradiction_label_idxs.shape[0]
    group_size = min(num_entailments, num_contradictions)
    source_idxs = torch.cat((entailment_label_idxs[:group_size], contradiction_label_idxs[:group_size]))
    dataset['train'] = dataset['train'].select(source_idxs.tolist())

    # split into train/val/ 
    source_train, source_val = torch.utils.data.random_split(
        dataset['train'], 
        [round(len(dataset['train']) * (1 - val_split)), round(len(dataset['train']) * val_split)], 
        generator=torch.Generator().manual_seed(42)
    )

    ### Target Distribution ###
    is_ood = lambda ex: (ex['label'] == 0 and ex["sentence2_has_negation"]) or (ex['label'] == 1 and not ex["sentence2_has_negation"])

    target_labels = torch.tensor(dataset['validation_matched']['label'])
    target_negations = torch.tensor(dataset['validation_matched']["sentence2_has_negation"])

    target_id_idxs = torch.where((target_labels == 0) & (target_negations == 0) | (target_labels == 1) & (target_negations == 1))[0]
    target_ood_idxs = torch.where((target_labels == 0) & (target_negations == 1) | (target_labels == 1) & (target_negations == 0))[0]

    num_id = len(target_id_idxs)
    num_ood = len(target_ood_idxs)

    cur_mix_rate = num_ood / len(dataset['validation_matched'])

    if mix_rate is None: # keep fixed mix rate
        pass
    elif cur_mix_rate < mix_rate: # remove iid 
        num_id_target = int((num_ood / mix_rate) - num_ood)
        target_id_idxs = target_id_idxs[:num_id_target]
    else: # remove ood 
        num_ood_target = int(num_id * mix_rate / (1 - mix_rate))
        target_ood_idxs = target_ood_idxs[:num_ood_target]

    dataset['validation_matched'] = dataset['validation_matched'].select(torch.cat((target_id_idxs, target_ood_idxs)).tolist())
    target_train, target_val = torch.utils.data.random_split(
        dataset['validation_matched'], 
        [round(len(dataset['validation_matched']) * (1 - val_split)), round(len(dataset['validation_matched']) * val_split)], 
        generator=torch.Generator().manual_seed(42)
    )

    ### Test ###
    test = dataset['validation_mismatched']

    # convert to pytorch datasets
    source_train = MultiNLIDataset(source_train, tokenizer, max_length)
    source_val = MultiNLIDataset(source_val, tokenizer, max_length)
    target_train = MultiNLIDataset(target_train, tokenizer, max_length)
    target_val = MultiNLIDataset(target_val, tokenizer, max_length)
    test = MultiNLIDataset(test, tokenizer, max_length)

    return source_train, source_val, target_train, target_val, test

    
