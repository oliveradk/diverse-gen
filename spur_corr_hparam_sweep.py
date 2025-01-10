from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from itertools import product
import json

import submitit
from submitit.core.utils import CommandFunction
import nevergrad as ng
import numpy as np

from losses.loss_types import LossType
from utils.exp_utils import get_executor
from utils.utils import conf_to_args


param_space = ng.p.Dict(
    lr=ng.p.Log(lower=1e-5, upper=1e-2),
    weight_decay=ng.p.Log(lower=1e-5, upper=1e-0),
    aux_weight=ng.p.Log(lower=1e0, upper=1e2),
    optimizer=ng.p.Choice(["sgd", "adamw"]), 
)

n_trials = 64
num_workers = 8


SCRIPT_NAME = "spur_corr_exp.py"

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

mix_rates = [0.5] # TODO: 0.1


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
        metric_val = np.nanmin(metrics[self.metric])
        return metric_val


# TODO: specify validation split for each dataset (probably 10%?)
HPARM_PARENT_DIR = Path("output/subpopulation_hparam_sweep")
hparam_dir = Path(HPARM_PARENT_DIR, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
hparam_dir.mkdir(exist_ok=True, parents=True)

results_file = Path(hparam_dir, "results.json")
results_file.touch()
results = {}

configs = list(product(env_configs.items(), loss_configs.items(), mix_rates))
for (env_name, env_config), (loss_name, loss_config), mix_rate in tqdm(configs, desc="Sweeping"):
    conf = {**env_config, **loss_config, "mix_rate": mix_rate}
    exp_key = f"{env_name}_{loss_name}_{mix_rate}"
    sweep_dir = Path(hparam_dir, exp_key)
    executor = get_executor(sweep_dir)
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

