import pytest
import subprocess
import itertools
from diverse_gen.losses.loss_types import LossType
from diverse_gen.utils.utils import conf_to_args


EXP_SCRIPT = "exp_scripts/measurement_tampering.py"


METHODS = [
    "TopK_0.1", 
    "DivDis", 
    "FT_Trusted", 
    "Probe_for_Evidence_of_Tamper", 
    "Measurement_Predictor"
]

DATASETS = ["diamonds", "generated_stories"]

# Common test configurations
TEST_CONFIG = {
    "epochs": 1,
    "effective_batch_size": 2,
    "forward_batch_size": 2,
    "micro_batch_size": 1,
    "dataset_length": 32
}

@pytest.mark.parametrize("method_name", METHODS)
def test_diamonds_all_losses(method_name):
    """Test all loss types on diamonds dataset"""
    conf = {
        "--config_file": f"{method_name}_diamonds", 
        **TEST_CONFIG
    }
    cmd = [
        "python", EXP_SCRIPT,
        *conf_to_args(conf), 
    ]
    try: 
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error running command: {e}")
        raise e

@pytest.mark.parametrize("dataset_name", DATASETS)
def test_topk_all_datasets(dataset_name):
    """Test TOPK loss on all datasets"""
    conf = {
        "--config_file": f"TopK_0.1_{dataset_name}",
        **TEST_CONFIG
    }
    cmd = [
        "python", EXP_SCRIPT,
        *conf_to_args(conf),
    ]

    try: 
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error running command: {e}")
        raise e