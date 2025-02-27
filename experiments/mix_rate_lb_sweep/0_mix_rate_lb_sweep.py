import os 
os.chdir("/nas/ucb/oliveradk/diverse-gen/")

from itertools import product
from pathlib import Path
from datetime import datetime
import copy

import optuna
import numpy as np
from omegaconf import OmegaConf
from diverse_gen.losses.loss_types import LossType
from diverse_gen.utils.exp_utils import get_study_args_dict, get_executor, run_experiments
from diverse_gen.utils.run_study import get_storage_path


N_TRIALS = 120
NODES_PER_STUDY = 12 # NOTE: this should be 13 but oh well
SAMPLER = "quasi-random"
STUDY_SCRIPT_PATH = "diverse_gen/utils/run_study.py"

SCRIPT_NAME = "exp_scripts/spur_corr_exp.py"
EXP_DIR = Path("output/mix_rate_lb_sweep")
SUB_DIR = None
if SUB_DIR is None:
    SUB_DIR = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
EXP_DIR = Path(EXP_DIR, SUB_DIR)
EXP_DIR.mkdir(exist_ok=True, parents=True)


MIX_RATES = [0.1, 0.25, 0.5, 0.75, 1.0]

DATASETS = ["toy_grid", "fmnist_mnist", "cifar_mnist", "waterbirds"]

config_dir = Path("configs")
datasets = OmegaConf.load(config_dir / "datasets.yaml")
method_ds = OmegaConf.load(config_dir / "method_ds.yaml")

datasets = {k: v for k, v in datasets.items() if k in DATASETS}
method_ds = {k: v for k, v in method_ds.items() if k == "TopK_0.5"} # use topk 0.5 defaults


hparam_map = {
    "mix_rate_lower_bound_01": {"type": "float", "range": [0, 1], "log": False},
    "mix_rate_lower_bound_10": {"type": "float", "range": [0, 1], "log": False},
}

PARTITIONS = [
    # Bottom row (left to right)
    [0.0, 0.0, 0.25, 0.25],
    [0.25, 0.0, 0.5, 0.25],
    [0.5, 0.0, 0.75, 0.25],
    [0.75, 0.0, 1.0, 0.25],
    
    # Second row
    [0.0, 0.25, 0.25, 0.5],
    [0.25, 0.25, 0.5, 0.5],
    [0.5, 0.25, 0.75, 0.5],
    [0.75, 0.25, 1.0, 0.5],
    
    # Third row
    [0.0, 0.5, 0.25, 0.75],
    [0.25, 0.5, 0.5, 0.75],
    [0.5, 0.5, 0.75, 0.75],
    # [0.75, 0.5, 1.0, 0.75],
    
    # Top row (excluding rightmost corner partitions)
    [0.0, 0.75, 0.25, 1.0],
    [0.25, 0.75, 0.5, 1.0],
    # [0.5, 0.75, 0.75, 1.0],
    # [0.75, 0.75, 1.0, 1.0],
]

def update_hparam_map(hparam_map, idx):
    new_hparam_map = copy.deepcopy(hparam_map)
    parition = PARTITIONS[idx]
    new_hparam_map["mix_rate_lower_bound_01"]["range"] = [parition[0], parition[2]]
    new_hparam_map["mix_rate_lower_bound_10"]["range"] = [parition[1], parition[3]]
    return new_hparam_map

configs = {}
for (ds_name, ds_config), mix_rate in product(datasets.items(), MIX_RATES):
    configs[(ds_name, mix_rate)] = {
        **ds_config, 
        "mix_rate": mix_rate, 
        "loss_type": LossType.TOPK,
        "mix_rate_lower_bound": None, 
        "mix_rate_schedule": "linear", 
        "mix_rate_t0": 0, 
        "mix_rate_t1": ds_config["epochs"]
    }

def get_study_name(ds_name, mix_rate):
    return f"{ds_name}_{mix_rate}"

# run experiments
for (env_name, mix_rate), conf in configs.items(): 
    study_name = get_study_name(ds_name, mix_rate)
    study_dir = Path(EXP_DIR, study_name)
    study_dir.mkdir(exist_ok=True, parents=True)
    
    # create study (must create it here to nodes don't conflict)
    study = optuna.create_study(study_name=study_name, storage=get_storage_path(study_dir), direction="minimize", load_if_exists=True)  
    
    # run study
    n_trials_per_node = N_TRIALS // NODES_PER_STUDY
    cmds = [
        {
            **get_study_args_dict(conf, SCRIPT_NAME, update_hparam_map(hparam_map, i), n_trials_per_node, 0, study_name, study_dir), 
            "sampler_seed": i, 
            "sampler_type": SAMPLER
        } for i in range(NODES_PER_STUDY)
    ]
    executor = get_executor(study_dir, mem_gb=16, slurm_array_parallelism=NODES_PER_STUDY)

    jobs = run_experiments(executor, cmds, STUDY_SCRIPT_PATH)







