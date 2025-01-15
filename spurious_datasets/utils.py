from typing import Optional
from itertools import product

import numpy as np




def get_group_idxs(feature_labels: np.ndarray, group_labels: list[np.ndarray]) -> np.ndarray:
    """
    Get the indices of the feature labels that match the group labels
    """
    idxs = []
    for group_label in group_labels:
        idxs.append(np.where(np.all(feature_labels == group_label, axis=1))[0])
    return np.concatenate(idxs)

def distribute_proportionally(target: int, fractions: list[float]) -> list[int]:
    """
    Distributes a target integer into parts proportional to given fractions.
    
    Args:
        target: The integer to be divided
        fractions: List of fractions that sum to 1
        
    Returns:
        List of integers that sum to target and are approximately proportional to fractions
        
    Raises:
        ValueError: If fractions don't sum to 1 (within floating point tolerance)
    """
    # Validate input
    if abs(sum(fractions) - 1.0) > 1e-10:
        print(fractions)
        raise ValueError("Fractions must sum to 1")
    
    # Calculate initial distribution using floating point multiplication
    float_parts = [target * f for f in fractions]
    
    # Round down to get initial integer parts
    int_parts = [int(p) for p in float_parts]
    
    # Calculate the remainder we need to distribute
    remainder = target - sum(int_parts)
    
    # Calculate fractional parts for priority in distributing remainder
    fractional_parts = [p - int(p) for p in float_parts]
    
    # Sort indices by fractional part in descending order
    indices = list(range(len(fractions)))
    indices.sort(key=lambda i: fractional_parts[i], reverse=True)
    
    # Distribute remainder by adding 1 to the parts with largest fractional components
    for i in range(remainder):
        int_parts[indices[i]] += 1
        
    return int_parts

def _get_group_n_instances(group_idxs: dict[tuple, np.ndarray], target_size: int) -> dict[tuple, int]:
   total_size = sum([len(idx) for idx in group_idxs.values()])
   fracs = [len(idx) / total_size for idx in group_idxs.values()]
   n_instances = distribute_proportionally(target_size, fracs)
   group_n_instances = {group_label: n_instances for group_label, n_instances in zip(group_idxs.keys(), n_instances)}
   return group_n_instances


def update_idxs_from_mix_rate(
    feature_labels: np.ndarray, mix_rate: float, 
    cc_groups: Optional[list[tuple[int]]]=None, classes_per_feature: Optional[list[int]]=None
) -> np.ndarray:
    
    if classes_per_feature is None:
        classes_per_feature = [2] * len(feature_labels.shape)
    
    group_label_ls = list(product(*[range(c) for c in classes_per_feature]))
    if cc_groups is None:
        cc_groups = [gl for gl in group_label_ls 
                           if all([gl[0] == gl[i] for i in range(len(gl))])]
    
    # get group idxs
    group_idxs = {}
    for group_label in group_label_ls:
        group_idxs[group_label] = get_group_idxs(feature_labels, [np.array(group_label)])
    
    # separate into iid and ood groups 
    ood_group_idxs = {k: idx for k, idx in group_idxs.items() if k not in cc_groups}
    iid_group_idxs = {k: idx for k, idx in group_idxs.items() if k in cc_groups}

    # compute number of ood, id instances
    n_ood = sum([len(idx) for idx in ood_group_idxs.values()])
    n_id = sum([len(idx) for idx in iid_group_idxs.values()])
    cur_mix_rate = (n_ood) / (n_ood + n_id)

    if cur_mix_rate < mix_rate:  # need to remove iid instances 
        n_id_target = round(n_ood / mix_rate) - n_ood
        # group group
        id_group_n_instances = _get_group_n_instances(iid_group_idxs, n_id_target)
        for group_label, idx in iid_group_idxs.items():
            n_group_target = id_group_n_instances[group_label]
            iid_group_idxs[group_label] = idx[:n_group_target]
    else: 
        n_ood_target = round(n_id * mix_rate / (1 - mix_rate))
        ood_group_n_instances = _get_group_n_instances(ood_group_idxs, n_ood_target)
        for group_label, idx in ood_group_idxs.items():
            n_group_target = ood_group_n_instances[group_label]
            ood_group_idxs[group_label] = idx[:n_group_target]
    
    id_idxs = np.concatenate(list(iid_group_idxs.values()))
    ood_idxs = np.concatenate(list(ood_group_idxs.values()))
    idxs = np.concatenate([id_idxs, ood_idxs])
    return idxs