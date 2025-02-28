from pathlib import Path
from itertools import product
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf

from diverse_gen.utils.exp_utils import get_executor, get_conf_dir, run_experiments

SCRIPT_NAME = "exp_scripts/measurement_tampering.py"
EXP_DIR = Path("output/mtd")
SUB_DIR = None
if SUB_DIR is None:
    SUB_DIR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXP_DIR = Path(EXP_DIR, SUB_DIR)
EXP_DIR.mkdir(parents=True, exist_ok=True)

JOBS = 4
METHODS = [
    "TopK_0.1", 
    "DivDis", 
    "FT_Trusted", 
    "Probe_for_Evidence_of_Tamper", 
    "Measurement_Predictor"
]
DATASETS = ["diamonds", "generated_stories"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


# generate exp configs
configs = {
    (ds_name, method_name, seed): {
        "--config_file": f"{method_name}_{ds_name}", 
        "seed": seed
    } 
    for (ds_name, method_name, seed) in 
    product(DATASETS, METHODS, SEEDS)
}

# set exp dirs
for conf_name, conf in configs.items():
    conf["exp_dir"] = get_conf_dir(conf_name, Path(EXP_DIR))

# run experiments 
non_80gb_nodes = ["ddpg", "dqn", "gail", "gan","ppo", "vae"]
slurm_exclude = ",".join([f"{node}.ist.berkeley.edu" for node in non_80gb_nodes])
chunks = np.array_split(list(configs.values()), JOBS)
for i, chunk in enumerate(chunks):
    executor = get_executor(EXP_DIR, mem_gb=32, slurm_exclude=slurm_exclude)
    jobs = run_experiments(executor, chunk, SCRIPT_NAME)


