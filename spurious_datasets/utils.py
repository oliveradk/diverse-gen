import numpy as np


def get_group_idxs(feature_labels: np.ndarray, group_labels: list[np.ndarray]) -> np.ndarray:
    """
    Get the indices of the feature labels that match the group labels
    """
    idxs = []
    for group_label in group_labels:
        idxs.append(np.where(np.all(feature_labels == group_label, axis=1))[0])
    return np.concatenate(idxs)

def update_idxs_from_mix_rate(feature_labels: np.ndarray, mix_rate: float) -> np.ndarray:
    # TODO: generalize to arbitary number of feature labels, n_classes_per feature label

    idxs_0_0 = get_group_idxs(feature_labels, [np.array([0, 0])])
    idxs_1_1 = get_group_idxs(feature_labels, [np.array([1, 1])])
    idxs_0_1 = get_group_idxs(feature_labels, [np.array([0, 1])])
    idxs_1_0 = get_group_idxs(feature_labels, [np.array([1, 0])])

    num_0_0 = len(idxs_0_0)
    num_1_1 = len(idxs_1_1)
    num_0_1 = len(idxs_0_1)
    num_1_0 = len(idxs_1_0)

    num_id = num_0_0 + num_1_1
    num_ood = num_0_1 + num_1_0

    cur_mix_rate = (num_ood) + (num_id + num_ood)
    # if less than target, remove ood instances (iid = ood/mix_rate - ood)
    if cur_mix_rate < mix_rate:
        # compute number of instances for each iid group
        num_id_target = int((num_ood / mix_rate) - num_ood) # so this is the number of instances we're targeting 
        num_0_0_target = int(num_0_0 / num_id * num_id_target)
        num_1_1_target = int(num_1_1 / num_id * num_id_target)

        idxs_0_0 = idxs_0_0[:num_0_0_target]
        idxs_1_1 = idxs_1_1[:num_1_1_target]
    else: # if greate than target, remove iid instances (ood = ood/mix_rate - id)
        # mix rate = (ood) / (ood + id)
        # -> (ood + id) * mix rate = ood 
        # -> mix_rate -1 * ood = - id * mix rate 
        # -> ood = id * mix rate / (1 - mix rate)
        num_ood_target = int(num_id * mix_rate / (1 - mix_rate))
        num_0_1_target = int(num_0_1 / num_ood * num_ood_target)
        num_1_0_target = int(num_1_0 / num_ood * num_ood_target)

        idxs_0_1 = idxs_0_1[:num_0_1_target]
        idxs_1_0 = idxs_1_0[:num_1_0_target]
    
    id_idxs = np.concatenate([idxs_0_0, idxs_1_1])
    ood_idxs = np.concatenate([idxs_0_1, idxs_1_0])
    idxs = np.concatenate([id_idxs, ood_idxs])
    return idxs