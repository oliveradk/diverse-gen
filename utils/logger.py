import json
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.metrics = defaultdict(list)
        self.tb_writer = SummaryWriter(log_dir=exp_dir)
    
    def add_scalar(self, parition, name, value, step=None, to_metrics=True, to_tb=True):
        if to_metrics:
            self.metrics[f"{parition}_{name}"].append(value)
        if to_tb:
            self.tb_writer.add_scalar(f"{parition}/{name}", value, step)

    def flush(self):
        self.tb_writer.flush()
        # save metrics to json
        with open(f"{self.exp_dir}/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)