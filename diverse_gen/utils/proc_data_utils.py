from typing import Literal, Optional
from pathlib import Path
import json 
from collections import defaultdict
import yaml

import numpy as np

from diverse_gen.losses.loss_types import LossType

def get_exp_metrics(conf: Optional[dict] = None, exp_dir: Optional[str] = None):
    if exp_dir is None: 
        exp_dir = conf["exp_dir"]
    if not (Path(exp_dir) / "metrics.json").exists():
        raise FileNotFoundError(f"Metrics file not found for experiment {exp_dir}")
    with open(Path(exp_dir) / "metrics.json", "r") as f:
        exp_metrics = json.load(f)
    return exp_metrics


def get_max_acc(
    exp_metrics: dict,
    acc_metric: str = "test_acc",
    model_selection: str = "val_loss",
    head_1_epochs: Optional[int] = None, 
    max_model_select: bool = False,
    one_head: bool = False, 
    mask: Optional[np.ndarray] = None
):
    if head_1_epochs is not None:
        exp_metrics = {k: v[head_1_epochs:] for k, v in exp_metrics.items()}
        if mask is not None:
            mask = mask[head_1_epochs:]
    max_accs = np.array(exp_metrics[f'{acc_metric}_0'])
    if not one_head:
        max_accs = np.maximum(max_accs, np.array(exp_metrics[f'{acc_metric}_1']))
    selection_metric = exp_metrics[model_selection]
    if mask is not None:
        selection_metric = np.where(mask, selection_metric, np.inf if not max_model_select else -np.inf)
    if max_model_select: 
        max_acc_idx = np.argmax(selection_metric)
    else: 
        max_acc_idx = np.argmin(selection_metric)
    max_acc = max_accs[max_acc_idx]
    return max_acc



# TODO: fix edge case with dbat and mask (100 - 50)
# data structure: dictionary with keys method types, values dict[mix_rate, list[len(seeds)]] of cifar accuracies (for now ignore case where mix_rate != mix_rate_lower_bound)
def get_acc_results(
    exp_configs: Optional[list[dict]] = None,
    exp_dirs: Optional[list[str]] = None,
    acc_metric: str = "test_acc",
    model_selection: str = "val_loss",
    verbose: bool=False, 
    mix_rates: bool = True, 
    perf_source_acc: bool = False
) -> dict | list:
    if exp_configs is None and exp_dirs is not None: 
        exp_configs = []
        for exp_dir in exp_dirs:
            exp_conf = yaml.safe_load(open(Path(exp_dir) / "config.yaml", "r"))
            exp_configs.append(exp_conf)
    else: 
        exp_dirs = [conf["exp_dir"] for conf in exp_configs]
    if mix_rates:
        results = defaultdict(list)
    else:
        results = []
    for conf, exp_dir in zip(exp_configs, exp_dirs):
        try:
            exp_metrics = get_exp_metrics(exp_dir=exp_dir)
            head_1_epochs = round(conf["epochs"] / 2) if conf.get("loss_type", None) == LossType.DBAT else None
            # condition on perfect source validation accuracy in model selection
            mask = None 
            if perf_source_acc:
                head_0_acc = np.maximum(exp_metrics["val_source_acc_0"], exp_metrics["val_source_acc_alt_0"])
                head_1_acc = np.maximum(exp_metrics["val_source_acc_1"], exp_metrics["val_source_acc_alt_1"])
                mask = (head_0_acc == 1.0) & (head_1_acc == 1.0)
                # check if mask is all false 
                if not np.any(mask):
                    mask = None
            
            max_acc = get_max_acc(exp_metrics, 
                acc_metric=acc_metric, 
                model_selection=model_selection, 
                head_1_epochs=head_1_epochs, 
                mask=mask
            )
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