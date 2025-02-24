import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.chdir("/nas/ucb/oliveradk/diverse-gen/")

from itertools import product
from pathlib import Path
from datetime import datetime
import numpy as np

from diverse_gen.utils.exp_utils import get_executor, run_experiments, get_conf_dir

# exp dir
SCRIPT_NAME = "exp_scripts/spur_corr_exp.py"
EXP_DIR = Path("output/random_network_baseline")
SUB_DIR = None
if SUB_DIR is None:
    SUB_DIR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXP_DIR = Path(EXP_DIR, SUB_DIR)
EXP_DIR.mkdir(parents=True, exist_ok=True)

# settings
NODES = 1
SEEDS = [1, 2, 3]
DATASETS = [
    "toy_grid", 
    "fmnist_mnist", 
    "cifar_mnist", 
    "waterbirds", 
    "celebA-0", 
    "multi-nli"
]
METHODS = ["ERM"]

# generate exp configs
configs = {}
for ds_name, seed in product(DATASETS, SEEDS):
    configs[(ds_name, "Random_Network", 0.0, seed)] = {
        "--config_file": f"ERM_{ds_name}", 
        "seed": seed, 
        "train": False
    }

# set exp dirs
for conf_name, conf in configs.items():
    conf["exp_dir"] = get_conf_dir(conf_name, EXP_DIR)

# low mem
low_mem_chunks = np.array_split(list(configs.values()), NODES)
for i, low_mem_chunk in enumerate(low_mem_chunks):
    executor = get_executor(EXP_DIR, mem_gb=16)
    jobs = run_experiments(executor, low_mem_chunk.tolist(), SCRIPT_NAME)
