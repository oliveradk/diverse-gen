import pytest
import subprocess
import itertools
from losses.loss_types import LossType

# ... existing code ...

DATASETS = [
    ("toy_grid", "toy_model"), 
    ("cifar_mnist", "Resnet50"),
    ("fmnist_mnist", "Resnet50"),
    ("waterbirds", "Resnet50"),
    ("celebA-0", "Resnet50"),
    # ("celebA-1", "Resnet50"),
    # ("celebA-2", "Resnet50"),
    # ("camelyon", "Resnet50"),
    ("multi-nli", "bert"),
]

LOSS_TYPES = [
    LossType.TOPK,
    LossType.DIVDIS,
    LossType.DBAT,
    LossType.ERM,
]

# Common test configurations
TEST_CONFIG = {
    "epochs": 1,
    "batch_size": 4,
    "target_batch_size": 4,
    "dataset_length": 8
}

@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_waterbirds_all_losses(loss_type):
    """Test all loss types on waterbirds dataset"""
    cmd = [
        "python", "spur_corr_exp.py",
        "dataset=waterbirds",
        f"loss_type={loss_type.name}",
    ]
    if loss_type == LossType.DBAT:
        cmd.extend([
            "freeze_heads=True", 
            "shared_backbone=False",
            "binary=True",
        ])
    # Add configuration parameters
    cmd.extend([f"{k}={v}" for k, v in TEST_CONFIG.items()])
    try: 
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error running command: {e}")
        raise e

@pytest.mark.parametrize("dataset", DATASETS)
def test_topk_all_datasets(dataset):
    """Test TOPK loss on all datasets"""
    cmd = [
        "python", "spur_corr_exp.py",
        f"dataset={dataset[0]}",
        f"model={dataset[1]}",
        "loss_type=TOPK",
    ]
    # Add configuration parameters
    cmd.extend([f"{k}={v}" for k, v in TEST_CONFIG.items()])

    try: 
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error running command: {e}")
        raise e

# If your script might print a lot, you can capture output to keep test logs clean:
    # result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)