import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.chdir("/nas/ucb/oliveradk/diverse-gen/")

from itertools import product
from pathlib import Path
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf

from diverse_gen.losses.loss_types import LossType
from diverse_gen.utils.exp_utils import get_executor, run_experiments, get_conf_dir

# exp dir
SCRIPT_NAME = "exp_scripts/spur_corr_exp.py"
CONFIG_DIR = Path("configs/spur_corr")
EXP_DIR = Path("output/incomplete_waterbirds")
SUB_DIR = None
if SUB_DIR is None:
    SUB_DIR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXP_DIR = Path(EXP_DIR, SUB_DIR)
EXP_DIR.mkdir(parents=True, exist_ok=True)

NODES = 4

SEEDS = [1, 2, 3]
MIX_RATES = [None]
METHODS = [
    "TopK_0.1", 
    "TopK_0.5", 
    "ERM", 
    "DBAT", 
    "DivDis"
]
DATASETS = ["waterbirds"]

# generate exp configs
configs = {
    (ds_name, method_name, mix_rate, seed): {
        "--config_file": f"{method_name}_{ds_name}", 
        "mix_rate": mix_rate, 
        "seed": seed
    } 
    for (ds_name, method_name, mix_rate, seed) in 
    product(DATASETS, METHODS, MIX_RATES, SEEDS)
}

# set source cc to false 
for conf in configs.values():
    conf["source_cc"] = False

# set exp dirs
for conf_name, conf in configs.items():
    conf["exp_dir"] = get_conf_dir(conf_name, EXP_DIR)


# run experiments
chunks = np.array_split(list(configs.values()), NODES)
for i, chunk in enumerate(chunks):
    executor = get_executor(EXP_DIR, mem_gb=16)
    jobs = run_experiments(executor, chunk.tolist(), SCRIPT_NAME)
