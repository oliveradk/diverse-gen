import pytest
import subprocess
import itertools
from diverse_gen.losses.loss_types import LossType
from diverse_gen.utils.utils import conf_to_args


EXP_SCRIPT = "exp_scripts/measurement_tampering.py"

DATASET_CONFIGS = {
    # "diamonds": {
    #     "dataset": "diamonds-seed0",
    #     "model": "codegen-350M-mono-measurement_pred-diamonds-seed0",
    # },
    "generated_stories": {
        "dataset": "generated_stories",
        "model": "pythia-1_4b-deduped-measurement_pred-generated_stories",
        "feature_dim": 2048
    },
}

LOSS_CONFIGS = {
    "TopK": {
        "loss_type": LossType.TOPK, 
        "mix_rate_schedule": "linear", 
        "mix_rate_t0": 0, 
        "mix_rate_t1": 1
    }, 
    # "DivDis": {"loss_type": LossType.DIVDIS}, 
    "FT Trusted": {
        "loss_type": LossType.ERM, 
        "aux_weight": 0.0, 
        "heads": 1
    }, 
    "Probe for Evidence of Tamper": {
        "loss_type": LossType.ERM,  
        "aux_weight": 0.0, 
        "heads": 1, "source_labels": ["sensors_agree"], 
        "split_source_target": False, 
        "target_only_disagree": True, 
        "freeze_model": True, 
    },
      "Measurement Predictor (baseline)": {
        "loss_type": LossType.ERM, 
        "heads": 1, 
        "train": False, 
        "load_prior_probe": True
    }
}

# Common test configurations
TEST_CONFIG = {
    "epochs": 1,
    "effective_batch_size": 2,
    "forward_batch_size": 2,
    "micro_batch_size": 1,
    "dataset_length": 32
}

@pytest.mark.parametrize("loss_name, loss_config", LOSS_CONFIGS.items())
def test_diamonds_all_losses(loss_name, loss_config):
    """Test all loss types on waterbirds dataset"""
    conf = {**DATASET_CONFIGS["diamonds"], **loss_config}
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