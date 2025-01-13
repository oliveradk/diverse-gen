from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from itertools import product
import json

import nevergrad as ng
import numpy as np

from losses.loss_types import LossType
from utils.exp_utils import get_executor, ExperimentCommandFunction


param_space = ng.p.Dict(
    aux_weight=ng.p.Log(lower=1e0, upper=1e1),
)

n_trials = 32
num_workers = 8

SCRIPT_NAME = "spur_corr_exp.py"

HPARM_PARENT_DIR = Path("output/subpopulation_aux_weight_sweep")
hparam_dir_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
hparam_dir = Path(HPARM_PARENT_DIR, hparam_dir_name)
hparam_dir.mkdir(exist_ok=True, parents=True)

loss_configs = {
    "DivDis": {"loss_type": LossType.DIVDIS},
    "DBAT": {"loss_type": LossType.DBAT, "shared_backbone": False, "freeze_heads": True, "batch_size": 16, "target_batch_size": 32},
    "TopK 0.1": {"loss_type": LossType.TOPK, "mix_rate_lower_bound": 0.1},
    "TopK 0.5": {"loss_type": LossType.TOPK, "mix_rate_lower_bound": 0.5},
    "TopK 1.0": {"loss_type": LossType.TOPK, "mix_rate_lower_bound": 1.0},
}

env_configs = {
    "toy_grid": {"dataset": "toy_grid", "model": "toy_model", "epochs": 100, "batch_size": 32, "target_batch_size": 128, "lr": 1e-3, "optim": "sgd"},
    "fmnist_mnist": {"dataset": "fmnist_mnist", "model": "Resnet50", "epochs": 5},
    "cifar_mnist": {"dataset": "cifar_mnist", "model": "Resnet50", "epochs": 5},
    "waterbirds": {"dataset": "waterbirds", "model": "Resnet50", "epochs": 5},
    "celebA-0": {"dataset": "celebA-0", "model": "Resnet50", "epochs": 5},
}

mix_rates = [0.1, 0.5, 1.0]

configs = list(product(env_configs.items(), loss_configs.items(), mix_rates))

results_file = Path(hparam_dir, "results.json")
results_file.touch()
results = {}

for (env_name, env_config), (loss_name, loss_config), mix_rate in tqdm(configs, desc="Sweeping"):
    conf = {**env_config, **loss_config, "mix_rate": mix_rate}
    exp_key = f"{env_name}_{loss_name}_{mix_rate}"
    sweep_dir = Path(hparam_dir, exp_key)
    executor = get_executor(sweep_dir, mem_gb=32)
    exp_cmd_func = ExperimentCommandFunction(SCRIPT_NAME, conf, "val_loss", sweep_dir)
    optimizer = ng.optimizers.RandomSearch(parametrization=param_space, budget=n_trials, num_workers=num_workers)
    try: 
        recommendation = optimizer.minimize(exp_cmd_func, executor=executor, batch_mode=True, verbosity=2)
    except Exception as e:
        optimizer.dump(Path(sweep_dir, "optimizer.pkl"))
        print(e)
        # save error message
        with open(Path(sweep_dir, "error.txt"), "w") as f:
            f.write(str(e))
        continue

    # store search results 
    sweep_results = [
        {"params": {k: v.value for k, v in value.parameter.items()}, 
         "loss": value.mean}
        for value in optimizer.archive.values()
    ]
    with open(Path(sweep_dir, "search_results.json"), "w") as f:
        json.dump(sweep_results, f, indent=4)

    # store experiment result (best parameters)
    results[exp_key] = {
        "params": recommendation.value,
        "loss": recommendation.loss
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
