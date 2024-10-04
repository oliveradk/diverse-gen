from losses.loss_types import LossType
import subprocess
from tqdm import tqdm

# mix rates
mix_rates = []
mix_rates.extend([i * 1e-2 for i in range(1, 10)])
mix_rates.extend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

i = 0
for loss_type in [LossType.PROB, LossType.EXP, LossType.DIVDIS]:
    for mix_rate in tqdm(mix_rates, desc=f'loss_type={loss_type}'):
        command = [
            'python',
            'toy_2d.py',
            f'loss_type={loss_type.name}',
            f'mix_rate={mix_rate}',
        ]
        subprocess.run(command)