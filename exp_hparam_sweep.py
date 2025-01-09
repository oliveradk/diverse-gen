import json
from functools import partial
from typing import Optional, Literal, Callable
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import submitit
from submitit.core.utils import CommandFunction
import nevergrad as ng
import numpy as np
import matplotlib.pyplot as plt


from losses.loss_types import LossType
from utils.exp_utils import get_executor, get_executor_local, run_experiments
from utils.utils import conf_to_args


param_space = ng.p.Dict(
    lr=ng.p.Log(lower=1e-5, upper=1e-2),
    weight_decay=ng.p.Log(lower=1e-5, upper=1e-0),
    aux_weight=ng.p.Log(lower=1e0, upper=1e2),
    optimizer=ng.p.Choice(["sgd", "adamw"]), 
)

n_trials = 64 
num_workers = 8 


SCRIPT_NAME = "run_exp.py"

loss_configs = {
    "DivDis": {"loss_type": LossType.DIVDIS},
    # "TopK": {"loss_type": LossType.TOPK},
    "TopK 0.1": {"loss_type": LossType.TOPK, "mix_rate_lower_bound": 0.1},
    "TopK 0.5": {"loss_type": LossType.TOPK, "mix_rate_lower_bound": 0.5},
    "DBAT": {"loss_type": LossType.DBAT, "shared_backbone": False, "freeze_heads": True, "batch_size": 16, "target_batch_size": 32},
    "ERM": {"loss_type": LossType.ERM}
}

env_configs = {
    "toy_grid": {"dataset": "toy_grid", "model": "toy_model", "epochs": 100, "batch_size": 32, "target_batch_size": 128},
    "fmnist_mnist": {"dataset": "fmnist_mnist", "model": "Resnet50", "epochs": 10},
    "cifar_mnist": {"dataset": "cifar_mnist", "model": "Resnet50", "epochs": 10},
    "waterbirds": {"dataset": "waterbirds", "model": "Resnet50", "epochs": 10, "aggregate_mix_rate": True},
    "celebA-0": {"dataset": "celebA-0", "model": "Resnet50", "epochs": 10, "aggregate_mix_rate": True},
    # {"dataset": "multi-nli", "model": "bert", "epochs": 10, "lr": 1e-5}
}

mix_rates = [0.1, 0.5]


# maye need to move this to utils to pickle properly 
class ExperimentCommandFunction(CommandFunction):
    def __init__(self, script_name: str, conf: dict, metric: str, parent_dir: Path):
        self.conf = conf
        self.metric = metric
        self.parent_dir = parent_dir
        assert "exp_dir" not in conf, "exp_dir should not be in conf"
        super().__init__(["python", script_name] + conf_to_args(conf))
    
    def __call__(self, params: dict):
        # set exp dir 
        exp_dir = Path(self.parent_dir, "_".join([f"{k}-{v}" for k, v in params.items()]))
        # randomly generate seed 
        seed = np.random.randint(10000)
        # convert to args
        param_args = conf_to_args({**params, "exp_dir": exp_dir, "seed": seed})
        # run experiment 
        _result = super().__call__(*param_args) # relying on this to return only when the experiment is complete 
        # load metrics 
        with open(exp_dir / "metrics.json", "r") as f:
            metrics = json.load(f)
        # get metric value
        metric_val = metrics[self.metric]
        return metric_val


optimizer = ng.optimizers.RandomSearch(parametrization=param_space, budget=n_trials, num_workers=num_workers)


# TODO: specify validation split for each dataset (probably 10%?)
HPARM_PARENT_DIR = Path("output/subpopulation_hparam_sweep")
hparam_dir = Path(HPARM_PARENT_DIR, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
hparam_dir.mkdir(exist_ok=True, parents=True)

results_file = Path(hparam_dir, "results.json")
results_file.touch()
results = {}


for env_name, env_config in env_configs.items():
    for loss_name, loss_config in loss_configs.items():
        for mix_rate in mix_rates:
            conf = {**env_config, **loss_config, "mix_rate": mix_rate}
            exp_key = f"{env_name}_{loss_name}_{mix_rate}"

            sweep_dir = Path(hparam_dir, exp_key)
            executor = get_executor(sweep_dir)
            exp_cmd_func = ExperimentCommandFunction(SCRIPT_NAME, conf, "val_loss", sweep_dir)
            recommendation = optimizer.minimize(exp_cmd_func, executor=executor, batch_mode=True)

            results[exp_key] = {
                "params": recommendation.value,
                "loss": recommendation.loss
            }

        with open(results_file, "w") as f:
            json.dump(results, f)