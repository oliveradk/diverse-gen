from pathlib import Path
from itertools import product

from omegaconf import OmegaConf

from diverse_gen.utils.exp_utils import get_executor, get_conf_dir, run_experiments

SCRIPT_NAME = "exp_scripts/measurement_tampering.py"
EXP_DIR = "output/mtd_aux_weight_sweep"
n_trials = 16

config_dir = Path("configs/mtd")
methods = OmegaConf.load(config_dir / "mtd_methods.yaml")
datasets = OmegaConf.load(config_dir / "mt_datasets.yaml")
method_ds = OmegaConf.load(config_dir / "mtd_method_ds.yaml")


# build configs 
configs = {}
for (ds_name, ds_config), (method_name, method_config), seed in product(
    datasets.items(), 
    methods.items(), 
    range(n_trials)
):
    key = (ds_name, method_name, seed)
    conf = {
        **ds_config, 
        **method_config, 
        "seed": seed, 
        **method_ds[method_name].get(ds_name, {})
    } 
    if ds_name == "diamonds":
        conf["dataset"] += f"-seed{seed}"
        conf["model"] += f"-seed{seed}"
    conf["exp_dir"] = get_conf_dir(key, Path(EXP_DIR))
    configs[key] = conf

# run experiments 
non_80gb_nodes = ["ddpg", "dqn", "gail", "gan","ppo", "vae"]
slurm_exclude = ",".join([f"{node}.ist.berkeley.edu" for node in non_80gb_nodes])
executor = get_executor(EXP_DIR, mem_gb=32, slurm_exclude=slurm_exclude)
jobs = run_experiments(executor, list(configs.values()), SCRIPT_NAME)


