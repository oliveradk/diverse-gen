from typing import Literal, Optional
from pathlib import Path
import json 
from collections import defaultdict

import numpy as np

from losses.loss_types import LossType

def get_exp_metrics(conf: dict):
    if not (Path(conf["exp_dir"]) / "metrics.json").exists():
        raise FileNotFoundError(f"Metrics file not found for experiment {conf['exp_dir']}")
    with open(Path(conf["exp_dir"]) / "metrics.json", "r") as f:
        exp_metrics = json.load(f)
    return exp_metrics

def get_max_acc(
    exp_metrics: dict,
    acc_metric: str = "test_acc",
    model_selection: str = "val_loss",
    head_1_epochs: Optional[int] = None, 
    max_model_select: bool = False,
    one_head: bool = False
):
    if head_1_epochs is not None:
        exp_metrics = {k: v[head_1_epochs:] for k, v in exp_metrics.items()}
    max_accs = np.array(exp_metrics[f'{acc_metric}_0'])
    if not one_head:
        max_accs = np.maximum(max_accs, np.array(exp_metrics[f'{acc_metric}_1']))
    if max_model_select: 
        max_acc_idx = np.argmax(exp_metrics[model_selection])
    else: 
        max_acc_idx = np.argmin(exp_metrics[model_selection])
    max_acc = max_accs[max_acc_idx]
    return max_acc

# data structure: dictionary with keys method types, values dict[mix_rate, list[len(seeds)]] of cifar accuracies (for now ignore case where mix_rate != mix_rate_lower_bound)
def get_acc_results(
    exp_configs: list[dict],
    acc_metric: Literal["test_acc", "test_worst_acc", "test_acc_alt"]="test_acc",
    model_selection: Literal["acc", "loss", "weighted_loss", "repulsion_loss"]="acc",
    verbose: bool=False, 
    mix_rates: bool = True
) -> dict | list:
    if mix_rates:
        results = defaultdict(list)
    else:
        results = []
    for conf in exp_configs:
        try:
            exp_metrics = get_exp_metrics(conf)
            head_1_epochs = round(conf["epochs"] / 2) if conf.get("loss_type", None) == LossType.DBAT else None
            max_acc = get_max_acc(exp_metrics, acc_metric, model_selection, head_1_epochs)
            if mix_rates:
                results[conf.get("mix_rate", 0.0)].append(max_acc)
            else:
                results.append(max_acc)
        except FileNotFoundError:
            if verbose:
                print(f"Metrics file not found for experiment {conf['exp_dir']}")
            continue
    if mix_rates:
        results = dict(results)
    return results