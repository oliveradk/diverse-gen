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

N_TRIALS = 200
N_PARTITIONS = 4 # no longer using multiple partitions
SAMPLER = "grid"
STUDY_SCRIPT_PATH = "experiments/mix_rate_lb_sweep/run_mr_lb_study.py"

SCRIPT_NAME = "exp_scripts/spur_corr_exp.py"
EXP_DIR = Path("output/mix_rate_lb_sweep")
SUB_DIR = None
if SUB_DIR is None:
    SUB_DIR = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
EXP_DIR = Path(EXP_DIR, SUB_DIR)
EXP_DIR.mkdir(exist_ok=True, parents=True)

MIX_RATES = [0.1, 0.25, 0.5, 0.75, 1.0]
DATASETS = [
    "toy_grid", 
    "fmnist_mnist", 
    "cifar_mnist", 
    "waterbirds", 
    "celebA-0", 
    "multi-nli"
]

METHODS = ["TopK_0.5"]

HPARAM_MAP = {
    "mix_rate_lower_bound": {"type": "float", "range": [0.0, 1.0], "log": False},
}
search_space = {
    "mix_rate_lower_bound": np.linspace(0.0, 1.0, 20).tolist(), 
    "mix_rate_lower_bound_01": np.linspace(0.0, 1.0, 10).tolist(), 
}

def update_map_and_search_space(hparam_map, search_space, idx):
    hparam_map = copy.deepcopy(hparam_map)
    search_space = copy.deepcopy(search_space)
    k = "mix_rate_lower_bound"
    v = hparam_map[k]
    # compute new range
    inc_size = (v["range"][1] - v["range"][0]) / N_PARTITIONS
    new_range = [v["range"][0] + idx * inc_size, v["range"][0] + (idx + 1) * inc_size] 
    
    # update hparam map
    hparam_map[k]["range"] = new_range
    # update search space
    search_space[k] = np.linspace(new_range[0], new_range[1], 20).tolist()

    return hparam_map, search_space


# generate exp configs
configs = {
    (ds_name, method_name, mix_rate): {
        "--config_file": f"{method_name}_{ds_name}", 
        "mix_rate": mix_rate, 
    } 
    for (ds_name, method_name, mix_rate) in 
    product(DATASETS, METHODS, MIX_RATES)
}

# set aggregate mix rate to false for all datasets 
for conf in configs.values():
    conf["aggregate_mix_rate"] = False

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
    cmds = []
    for i in range(N_PARTITIONS):
        hparam_map, search_space = update_map_and_search_space(HPARAM_MAP, search_space, i)
        cmds.append(
            {
                **get_study_args_dict(
                conf, 
                SCRIPT_NAME, 
                hparam_map, 
                n_trials_per_node, 
                0, 
                study_name, 
                study_dir
            ), 
            "search_space": search_space, 
            "sampler_seed": i, 
            "sampler_type": SAMPLER
        })
    mem_gb = 16 if ds_name not in ["celebA-0", "multi-nli"] else 32
    executor = get_executor(study_dir, mem_gb=mem_gb, slurm_array_parallelism=N_PARTITIONS)

    jobs = run_experiments(executor, cmds, STUDY_SCRIPT_PATH)
