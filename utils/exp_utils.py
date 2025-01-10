from typing import Optional
from pathlib import Path
import submitit
from datetime import datetime
from dataclasses import dataclass
from utils.utils import conf_to_args

def get_executor(out_dir: Optional[Path] = None, gpu_type: str | None = None):
    if out_dir is None:
        out_dir = Path(f"output_logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        out_dir.mkdir(exist_ok=True, parents=True)
    executor = submitit.AutoExecutor(folder=out_dir)
    executor.update_parameters(
        timeout_min=60 * 48,
        mem_gb=16,
        slurm_gres=f"gpu{':' + gpu_type if gpu_type is not None else ''}:1",
        cpus_per_task=4,
        nodes=1,
        slurm_qos="high",
        slurm_array_parallelism=8, 
        slurm_exclude="ddpg.ist.berkeley.edu,dqn.ist.berkeley.edu" # large sharded gpu's - often causes OOM issues
    )
    return executor

def get_executor_local(out_dir: Path):
    executor = submitit.LocalExecutor(folder=out_dir)
    executor.update_parameters(
        timeout_min=60 * 48,
    )
    return executor

def run_experiments(executor, experiments: list, script_name: str):
    # with executor.batch():
    jobs = []
    for exp in experiments:
        executor.update_parameters(
            output_dir=exp.exp_dir
        )
        exp_dict = exp.__dict__ if hasattr(exp, '__dict__') else exp
        function = submitit.helpers.CommandFunction(
            ["python", script_name] + conf_to_args(exp_dict)
        )
        jobs.append(executor.submit(function))
    return jobs