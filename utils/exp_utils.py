from typing import Optional
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import submitit
from submitit.helpers import CommandFunction

from utils.utils import conf_to_args


def get_executor(
    out_dir: Optional[Path] = None, 
    gpu_type: str | None = None, 
    mem_gb: int = 16, 
    timeout_min: int = 60 * 48,
    cpus_per_task: int = 4,
    nodes: int = 1,
    slurm_qos: str = "default",
    slurm_array_parallelism: int = 8,
    slurm_exclude: str = "ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu" # large sharded gpu's - often causes OOM issues
):
    if out_dir is None:
        out_dir = Path(f"output_logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        out_dir.mkdir(exist_ok=True, parents=True)
    executor = submitit.AutoExecutor(folder=out_dir)
    executor.update_parameters(
        timeout_min=timeout_min,
        mem_gb=mem_gb,
        slurm_gres=f"gpu{':' + gpu_type if gpu_type is not None else ''}:1",
        cpus_per_task=cpus_per_task,
        nodes=nodes,
        slurm_qos=slurm_qos,
        slurm_array_parallelism=slurm_array_parallelism, 
        slurm_exclude=slurm_exclude
    )
    return executor

def get_executor_local(out_dir: Path):
    executor = submitit.LocalExecutor(folder=out_dir)
    executor.update_parameters(
        timeout_min=60 * 48,
    )
    return executor

def run_experiments(executor, experiments: list, script_name: str):
    with executor.batch():
        jobs = []
        for exp in experiments:
            exp_dict = exp.__dict__ if hasattr(exp, '__dict__') else exp
            function = submitit.helpers.CommandFunction(
                ["python", script_name] + conf_to_args(exp_dict)
            )
            jobs.append(executor.submit(function))
    return jobs

def get_conf_dir(conf_name: tuple, exp_dir: Path):
    names, idx = conf_name[:-1], conf_name[-1]
    names = [str(name) for name in names]
    assert isinstance(idx, int), "idx must be an integer"
    return f"{exp_dir}/{'_'.join(names)}/{idx}"


class ExperimentCommandFunction(CommandFunction):
    def __init__(self, script_name: str, conf: dict, metric: str, parent_dir: Path):
        self.conf = conf
        self.metric = metric
        self.parent_dir = parent_dir
        assert "exp_dir" not in conf, "exp_dir should not be in conf"
        super().__init__(["python", script_name] + conf_to_args(conf))
    
    def __call__(self, params: dict):
        # randomly generate seed 
        seed = np.random.randint(10000)
        # set exp dir 
        exp_dir = Path(self.parent_dir) / str(seed)
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


