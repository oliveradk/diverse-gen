

from torchvision import transforms
from transformers import PreTrainedTokenizer

from diverse_gen.datasets.cifar_mnist import get_cifar_mnist_datasets
from diverse_gen.datasets.fmnist_mnist import get_fmnist_mnist_datasets
from diverse_gen.datasets.toy_grid import get_toy_grid_datasets
from diverse_gen.datasets.waterbirds import get_waterbirds_datasets
from diverse_gen.datasets.camelyon import get_camelyon_datasets
from diverse_gen.datasets.multi_nli import get_multi_nli_datasets
from diverse_gen.datasets.celebA import get_celebA_datasets




def get_dataset(
    dataset_name: str, 
    mix_rate: float | None = None, 
    source_cc: bool = True, 
    source_val_split: float = 0.2, 
    target_val_split: float = 0.2, 
    model_transform: transforms.Compose | None = None, 
    tokenizer: PreTrainedTokenizer | None = None, 
    dataset_length: int | None = None, 
    pad_sides: bool = False, 
    max_length: int = 128, 
    use_group_labels: bool = False, 
    # multi-nli
    combine_neut_entail: bool = True, 
    contra_no_neg: bool = False, 
):

    classes_per_feat = [2, 2]
    is_img = True

    if dataset_name == "toy_grid":
        source_train, source_val, target_train, target_val, target_test = get_toy_grid_datasets(
            target_mix_rate_0_1=mix_rate / 2 if mix_rate is not None else None, 
            target_mix_rate_1_0=mix_rate / 2 if mix_rate is not None else None, 
            gaussian=True,
            std=0.01
        )
    elif dataset_name == "cifar_mnist":
        source_train, source_val, target_train, target_val, target_test = get_cifar_mnist_datasets(
            target_mix_rate_0_1=mix_rate / 2 if mix_rate is not None else None, 
            target_mix_rate_1_0=mix_rate / 2 if mix_rate is not None else None, 
            transform=model_transform, 
            pad_sides=pad_sides
        )

    elif dataset_name == "fmnist_mnist":
        source_train, source_val, target_train, target_val, target_test = get_fmnist_mnist_datasets(
            target_mix_rate_0_1=mix_rate / 2 if mix_rate is not None else None, 
            target_mix_rate_1_0=mix_rate / 2 if mix_rate is not None else None, 
            transform=model_transform, 
            pad_sides=pad_sides
        )
    elif dataset_name == "waterbirds":
        source_train, source_val, target_train, target_val, target_test = get_waterbirds_datasets(
            mix_rate=mix_rate, 
            source_cc=source_cc,
            transform=model_transform, 
            convert_to_tensor=True,
            val_split=source_val_split,
            target_val_split=target_val_split, 
            dataset_length=dataset_length
        )
    # elif conf.dataset == "cub":
    #     source_train, target_train, target_test = get_cub_datasets()
    #     source_val = []
    #     target_val = []
    elif dataset_name.startswith("celebA"):
        if dataset_name == "celebA-0":
            gt_feat = "Blond_Hair"
            spur_feat = "Male"
            inv_spur_feat = True
        elif dataset_name == "celebA-1":
            gt_feat = "Mouth_Slightly_Open"
            spur_feat = "Wearing_Lipstick"
            inv_spur_feat = False
        elif dataset_name == "celebA-2":
            gt_feat = "Wavy_Hair"
            spur_feat = "High_Cheekbones"
            inv_spur_feat = False
        else: 
            raise ValueError(f"Dataset {dataset_name} not supported")
        source_train, source_val, target_train, target_val, target_test = get_celebA_datasets(
            mix_rate=mix_rate, 
            source_cc=source_cc,
            transform=model_transform, 
            gt_feat=gt_feat,
            spur_feat=spur_feat,
            inv_spur_feat=inv_spur_feat,
            dataset_length=dataset_length
        )
    elif dataset_name == "camelyon":
        source_train, source_val, target_train, target_val, target_test = get_camelyon_datasets(
            transform=model_transform, 
            dataset_length=dataset_length
        )

    elif dataset_name == "multi-nli":
        source_train, source_val, target_train, target_val, target_test = get_multi_nli_datasets(
            mix_rate=mix_rate,
            source_cc=source_cc,
            val_split=source_val_split,
            target_val_split=target_val_split,
            tokenizer=tokenizer,
            max_length=max_length, 
            dataset_length=dataset_length, 
            combine_neut_entail=combine_neut_entail, 
            contra_no_neg=contra_no_neg
        )
        is_img = False
        if not combine_neut_entail:
            classes_per_feat = [3, 2]
            if use_group_labels:
                classes_per_head = [3, 2] # [contradiction, entailment, neutral] x [no negation, negation]
            else:
                classes_per_head = [3, 3] # [contradiction, entailment, neutral] x 2

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    dataset_dict = {
        "source_train": source_train, 
        "source_val": source_val, 
        "target_train": target_train, 
        "target_val": target_val, 
        "target_test": target_test, 
        "is_img": is_img, 
        "classes_per_feat": classes_per_feat, 
    }
    return dataset_dict