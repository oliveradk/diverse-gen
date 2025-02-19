from typing import Optional, Dict
import subprocess
from pathlib import Path
import json
from datetime import datetime as dt
from dataclasses import dataclass, field
from functools import partial
from typing import Literal
import numpy as np
from omegaconf import OmegaConf
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler, RandomSampler, QMCSampler, GridSampler

from diverse_gen.utils.utils import conf_to_args
from diverse_gen.utils.exp_utils import get_conf_dir

def get_storage_path(study_dir: str):
    return f"sqlite:///{study_dir}/optuna_study.db"

@dataclass 
class HparamConfig: 
    type: str 
    range: Optional[list] = None
    choices: Optional[list] = None
    log: bool = False

@dataclass
class Config: 
    args: list[str] = field(default_factory=list)
    script_name: str = "exp_scripts/spur_corr_exp.py"
    hparams: Dict[str, HparamConfig] = field(default_factory=dict)
    n_trials: int = 64
    sampler_type: str = "quasi-random" # random, quasi-random, tse
    n_startup_trials: int = 10
    n_ei_candidates: int = 100
    search_space: dict[str, list[float]] = field(default_factory=dict)
    sampler_seed: int = 42
    study_name: str = "hparam_study"
    study_dir: str = f"output/hparam_study/{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}"

def objective(trial: Trial, conf: Config): 
    hparams = {}
    mix_rate_lb_range = conf.hparams["mix_rate_lower_bound"].range
    mix_rate_lb = trial.suggest_float("mix_rate_lower_bound", mix_rate_lb_range[0], mix_rate_lb_range[1])
    mr_lb_01_box = trial.suggest_float("mix_rate_lower_bound_01_box", 0.0, 1.0)
    mix_rate_lb_01 = mr_lb_01_box * (mix_rate_lb / 2)
    mix_rate_lb_10 = mix_rate_lb - mix_rate_lb_01
    hparams["mix_rate_lower_bound"] = mix_rate_lb
    hparams["mix_rate_lower_bound_01"] = mix_rate_lb_01
    hparams["mix_rate_lower_bound_10"] = mix_rate_lb_10
    
    seed = np.random.randint(0, 1000)
    exp_dir = Path(conf.study_dir) / str(trial.number)

    trial_conf = {
        "exp_dir": exp_dir,
        "seed": seed,
        **hparams,
    }

    cmd = ["python", conf.script_name] + conf.args + conf_to_args(trial_conf)

    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "stdout.log", "w") as out_f, open(exp_dir / "stderr.log", "w") as err_f:
        result = subprocess.run(
            cmd, 
            check=True,
            stdout=out_f,
            stderr=err_f, 
            text=True
        )
    if result.returncode != 0:
        raise ValueError(f"Experiment failed with return code {result.returncode}")

    # load metrics 
    with open(Path(exp_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)
    
    min_loss = min(metrics["val_loss"])

    return min_loss

def main():
    base_conf = OmegaConf.structured(Config())
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_conf, cli_conf)

    storage_path = get_storage_path(conf.study_dir)

    if conf.sampler_type == "tpe":
        sampler = TPESampler(
            seed=conf.sampler_seed,
            n_startup_trials=conf.n_startup_trials,
            n_ei_candidates=conf.n_ei_candidates
        )
    elif conf.sampler_type == "random":
        sampler = RandomSampler(
            seed=conf.sampler_seed,
        )
    elif conf.sampler_type == "quasi-random":
        sampler = QMCSampler(
            seed=conf.sampler_seed,
        )
    elif conf.sampler_type == "grid":
        sampler = GridSampler(
            search_space=conf.search_space,
            seed=conf.sampler_seed,
        )
    else:
        raise ValueError(f"Invalid sampler type: {conf.sampler_type}")

    # Create study with TPE sampler and pruner
    study = optuna.create_study(
        study_name=conf.study_name,
        storage=storage_path,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True
    )

    # Optimize
    study.optimize(partial(objective, conf=conf), n_trials=conf.n_trials)

    print("Best trial: ", study.best_trial)

if __name__ == "__main__":
    main()