from dataclasses import dataclass 
from typing import Optional
import torch
from datetime import datetime

from losses.loss_types import LossType

@dataclass
class Config():
    seed: int = 45
    dataset: str = "cifar_mnist"
    loss_type: LossType = LossType.TOPK
    batch_size: int = 32
    target_batch_size: int = 128
    epochs: int = 10
    heads: int = 2
    binary: bool = False
    model: str = "Resnet18"
    shared_backbone: bool = True
    source_weight: float = 1.0
    aux_weight: float = 1.0
    source_mix_rate: float = 0.0
    source_l_01_mix_rate: Optional[float] = None
    source_l_10_mix_rate: Optional[float] = None
    mix_rate: Optional[float] = 0.5
    aggregate_mix_rate: bool = False
    l_01_mix_rate: Optional[float] = None # TODO: geneneralize
    l_10_mix_rate: Optional[float] = None
    mix_rate_lower_bound: Optional[float] = 0.5
    l_01_mix_rate_lower_bound: Optional[float] = None
    l_10_mix_rate_lower_bound: Optional[float] = None
    all_unlabeled: bool = False
    inbalance_ratio: Optional[bool] = False
    lr: float = 1e-3
    weight_decay: float = 1e-3 
    lr_scheduler: Optional[str] = None 
    num_cycles: float = 0.5
    frac_warmup: float = 0.05
    max_length: int = 128
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    exp_dir: str = f"output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    plot_activations: bool = False

def post_init(conf: Config, overrides: list[str]=[]):
    if conf.l_01_mix_rate is not None and conf.l_10_mix_rate is None:
        conf.l_10_mix_rate = 0.0
        if conf.mix_rate is None:
            conf.mix_rate = conf.l_01_mix_rate
        assert conf.mix_rate == conf.l_01_mix_rate
    elif conf.l_01_mix_rate is None and conf.l_10_mix_rate is not None:
        conf.l_01_mix_rate = 0.0
        if conf.mix_rate is None:
            conf.mix_rate = conf.l_10_mix_rate
        assert conf.mix_rate == conf.l_10_mix_rate
    elif conf.l_01_mix_rate is not None and conf.l_10_mix_rate is not None:
        if conf.mix_rate is None:
            conf.mix_rate = conf.l_01_mix_rate + conf.l_10_mix_rate
        assert conf.mix_rate == conf.l_01_mix_rate + conf.l_10_mix_rate
    else: # both are none 
        assert conf.mix_rate is not None
        conf.l_01_mix_rate = conf.mix_rate / 2
        conf.l_10_mix_rate = conf.mix_rate / 2
    
    conf.source_l_01_mix_rate = conf.source_mix_rate / 2
    conf.source_l_10_mix_rate = conf.source_mix_rate / 2


    
    if conf.mix_rate_lower_bound is None:
        conf.mix_rate_lower_bound = conf.mix_rate

    if conf.l_01_mix_rate_lower_bound is None and conf.l_10_mix_rate_lower_bound is None:
        conf.l_01_mix_rate_lower_bound = conf.mix_rate_lower_bound / 2
        conf.l_10_mix_rate_lower_bound = conf.mix_rate_lower_bound / 2