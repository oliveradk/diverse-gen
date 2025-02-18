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
EXP_DIR = Path("output/incomplete_waterbirds")
SUB_DIR = None
if SUB_DIR is None:
    SUB_DIR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXP_DIR = Path(EXP_DIR, SUB_DIR)
EXP_DIR.mkdir(parents=True, exist_ok=True)

NODES = 4


# settings
SEEDS = [1, 2, 3]
MIX_RATES = [None]

GLOBAL_CONFIGS = {
    "source_cc": False,
}

configs_dir = Path("configs")
methods = OmegaConf.load(configs_dir / "methods.yaml")
datasets = OmegaConf.load(configs_dir / "datasets.yaml")
method_ds = OmegaConf.load(configs_dir / "method_ds.yaml")

# filter for waterbirds only 
datasets = {k: v for k, v in datasets.items() if v["dataset"] == "waterbirds"}

# topk configs with no schedule
no_sched_topk_configs = {}
for method_name, method_conf in methods.items():
    if method_conf["loss_type"] == LossType.TOPK.name:
        conf = method_conf.copy()
        conf["mix_rate_schedule"] = None
        no_sched_topk_configs[method_name+"_No_Sched"] = conf
methods.update(no_sched_topk_configs)

# generate exp configs
configs = {
    (ds_name, method_name, mix_rate, seed): {**ds, **method, "mix_rate": mix_rate, "seed": seed} 
    for (ds_name, ds), (method_name, method), mix_rate, seed in 
    product(datasets.items(), methods.items(), MIX_RATES, SEEDS)
    if not (method_name == "ERM")
}
# # add ERM with mix rate 0.0 
# for (ds_name, ds), seed in product(datasets.items(), SEEDS):
#     configs[(ds_name, "ERM", 0.0, seed)] = {**ds, **methods["ERM"], "seed": seed}
# dataset x method adjustments
for ((ds_name, method_name, mix_rate, seed), conf) in configs.items():
    update = method_ds.get(method_name, {}).get(ds_name, {})
    for k, v in update.items():
        conf[k] = v
# update dbat batch size
for conf in configs.values():
    if conf["loss_type"] == LossType.DBAT.name: 
        conf["batch_size"] = int(conf["batch_size"] / 2)
        conf["target_batch_size"] = int(conf["target_batch_size"] / 2)
# update topk configs with schedule
for conf in configs.values():
    if conf["loss_type"] == LossType.TOPK.name and conf["mix_rate_schedule"] == "linear":
        conf["mix_rate_t0"] = 0
        conf["mix_rate_t1"] = 5

# set source cc to false 
for conf in configs.values():
    conf.update(GLOBAL_CONFIGS)

# set exp dirs
for conf_name, conf in configs.items():
    conf["exp_dir"] = get_conf_dir(conf_name, EXP_DIR)


# run experiments
chunks = np.array_split(list(configs.values()), NODES)
for i, chunk in enumerate(chunks):
    executor = get_executor(EXP_DIR, mem_gb=16)
    jobs = run_experiments(executor, chunk.tolist(), SCRIPT_NAME)
