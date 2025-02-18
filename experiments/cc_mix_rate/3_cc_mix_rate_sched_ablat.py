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
EXP_DIR = Path("output/cc_mix_rate")
SUB_DIR = None
if SUB_DIR is None:
    SUB_DIR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXP_DIR = Path(EXP_DIR, SUB_DIR)
EXP_DIR.mkdir(parents=True, exist_ok=True)


# settings
NODES = 4
SEEDS = [1, 2, 3]
MIX_RATES = [0.1, 0.25, 0.5, 0.75, 1.0]
DATASETS = [
    "toy_grid", 
    "fmnist_mnist", 
    "cifar_mnist", 
    "waterbirds", 
    "celebA-0", 
    "multi-nli"
]
METHODS = [
    "TopK_0.1", 
    "TopK_0.5", 
]

configs_dir = Path("configs")
methods = OmegaConf.load(configs_dir / "methods.yaml")
datasets = OmegaConf.load(configs_dir / "datasets.yaml")
method_ds = OmegaConf.load(configs_dir / "method_ds.yaml")

# filter configs 
datasets = {k: v for k, v in datasets.items() if k in DATASETS}
methods = {k: v for k, v in methods.items() if k in METHODS}

# remove schedule from topk configs
for method_name, method_conf in methods.items():
    if method_conf["loss_type"] == LossType.TOPK.name:
        method_conf["mix_rate_schedule"] = None
# append no schedule to method names
methods = {k + "_No_Sched": v for k, v in methods.items()}

# generate exp configs
configs = {
    (ds_name, method_name, mix_rate, seed): {**ds, **method, "mix_rate": mix_rate, "seed": seed} 
    for (ds_name, ds), (method_name, method), mix_rate, seed in 
    product(datasets.items(), methods.items(), MIX_RATES, SEEDS)
}
# dataset x method adjustments
for ((ds_name, method_name, mix_rate, seed), conf) in configs.items():
    update = method_ds.get(method_name, {}).get(ds_name, {})
    for k, v in update.items():
        conf[k] = v

# set exp dirs
for conf_name, conf in configs.items():
    conf["exp_dir"] = get_conf_dir(conf_name, EXP_DIR)


# run experiments
high_mem_ds = ["multi-nli", "celebA-0"]
low_mem_configs = {k: v for k, v in configs.items() if v["dataset"] not in high_mem_ds}
high_mem_configs = {k: v for k, v in configs.items() if v["dataset"] in high_mem_ds}

# low mem
low_mem_chunks = np.array_split(list(low_mem_configs.values()), NODES)
for i, low_mem_chunk in enumerate(low_mem_chunks):
    # low mem
    executor = get_executor(EXP_DIR, mem_gb=16)
    jobs = run_experiments(executor, low_mem_chunk.tolist(), SCRIPT_NAME)
# high mem
high_mem_chunks = np.array_split(list(high_mem_configs.values()), NODES)
for i, high_mem_chunk in enumerate(high_mem_chunks):
    # high mem
    executor = get_executor(EXP_DIR, mem_gb=32)
    jobs = run_experiments(executor, high_mem_chunk.tolist(), SCRIPT_NAME)    
