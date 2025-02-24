#%%
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
#%%
N_TRIALS = 32
N_PARTITIONS = 4 # no longer using multiple partitions
SAMPLER = "quasi-random"
STUDY_SCRIPT_PATH = "diverse_gen/utils/run_study.py"
#%%
SCRIPT_NAME = "exp_scripts/spur_corr_exp.py"
EXP_DIR = Path("output/dbat_aux_weight_sweep")
SUB_DIR = None
if SUB_DIR is None:
    SUB_DIR = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
EXP_DIR = Path(EXP_DIR, SUB_DIR)
EXP_DIR.mkdir(exist_ok=True, parents=True)
#%%
MIX_RATES = [0.1, 0.25, 0.5, 0.75, 1.0]
DATASETS = ["toy_grid", "fmnist_mnist", "cifar_mnist", "waterbirds"]
METHODS = ["DBAT"]


#%%
HPARAM_MAP = {
    "aux_weight": {"type": "float", "range": [1e-6, 1.], "log": True},
}

def update_hparam_map(hparam_map, idx):
    hparam_map = copy.deepcopy(hparam_map)
    for k, v in hparam_map.items():
        if v["type"] == "float":
            if v["log"]:
                inc_size = np.log10(v["range"][1] / v["range"][0]) / N_PARTITIONS
                new_range = [v["range"][0] * 10**(idx * inc_size), v["range"][0] * 10**((idx + 1) * inc_size)]
            else:
                inc_size = (v["range"][1] - v["range"][0]) / N_PARTITIONS
                new_range = [v["range"][0] + idx * inc_size, v["range"][0] + (idx + 1) * inc_size]
            hparam_map[k]["range"] = new_range
    return hparam_map

configs = {
    (ds_name, method_name, mix_rate): {
        "--config_file": f"{method_name}_{ds_name}", 
        "mix_rate": mix_rate, 
    } 
    for (ds_name, method_name, mix_rate) in 
    product(DATASETS, METHODS, MIX_RATES)
}
#%%
def get_study_name(ds_name, mix_rate):
    return f"{ds_name}_{mix_rate}"

# run experiments
for (ds_name, method_name, mix_rate), conf in configs.items(): 
    study_name = get_study_name(ds_name, mix_rate)
    study_dir = Path(EXP_DIR, study_name)
    study_dir.mkdir(exist_ok=True, parents=True)
    
    # create study (must create it here to nodes don't conflict)
    study = optuna.create_study(study_name=study_name, storage=get_storage_path(study_dir), direction="minimize", load_if_exists=True)  
    
    # run study
    n_trials_per_node = N_TRIALS // N_PARTITIONS
    cmds = [
        {
            **get_study_args_dict(
                conf, 
                SCRIPT_NAME, 
                HPARAM_MAP, 
                n_trials_per_node, 
                0, 
                study_name, 
                study_dir
            ), 
            "sampler_seed": i, 
            "sampler_type": SAMPLER
        } for i in range(N_PARTITIONS)
    ]
    executor = get_executor(study_dir, mem_gb=16, slurm_array_parallelism=N_PARTITIONS)

    jobs = run_experiments(executor, cmds, STUDY_SCRIPT_PATH)