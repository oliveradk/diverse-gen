import pytest
import subprocess
import itertools
from diverse_gen.losses.loss_types import LossType
from diverse_gen.utils.utils import conf_to_args

EXP_SCRIPT = "exp_scripts/spur_corr_exp.py"

DATASET_CONFIGS = {
    "toy_grid": {"dataset": "toy_grid", "model": "toy_model"},
    "cifar_mnist": {"dataset": "cifar_mnist", "model": "Resnet50"},
    "fmnist_mnist": {"dataset": "fmnist_mnist", "model": "Resnet50"},
    "waterbirds": {"dataset": "waterbirds", "model": "Resnet50"},
    "celebA-0": {"dataset": "celebA-0", "model": "Resnet50"},
    "multi-nli": {"dataset": "multi-nli", "model": "bert"},
}

LOSS_CONFIGS = {
    "TOPK": {
        "loss_type": LossType.TOPK, 
        "mix_rate_schedule": "linear", 
        "mix_rate_lower_bound": 0.5,
        "mix_rate_t0": 0,
        "mix_rate_t1": 1,
    },
    "DIVDIS": {"loss_type": LossType.DIVDIS},
    "DBAT": {
        "loss_type": LossType.DBAT,
        "freeze_heads": True,
        "shared_backbone": False,
        "binary": True,
    },
    "ERM": {"loss_type": LossType.ERM},
}

# Common test configurations
TEST_CONFIG = {
    "epochs": 1,
    "batch_size": 4,
    "target_batch_size": 4,
    "dataset_length": 8
}

@pytest.mark.parametrize("loss_name, loss_config", LOSS_CONFIGS.items())
def test_waterbirds_all_losses(loss_name, loss_config):
    """Test all loss types on waterbirds dataset"""
    conf = {**DATASET_CONFIGS["waterbirds"], **loss_config}
    cmd = [
        "python", EXP_SCRIPT,
        *conf_to_args(conf),
        *conf_to_args(TEST_CONFIG)
    ]
    try: 
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error running command: {e}")
        raise e

@pytest.mark.parametrize("dataset_name, dataset_config", DATASET_CONFIGS.items())
def test_topk_all_datasets(dataset_name, dataset_config):
    """Test TOPK loss on all datasets"""
    cmd = [
        "python", EXP_SCRIPT,
        *conf_to_args(dataset_config),
        *conf_to_args({"loss_type": LossType.TOPK}),
        *conf_to_args(TEST_CONFIG)
    ]

    try: 
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error running command: {e}")
        raise e