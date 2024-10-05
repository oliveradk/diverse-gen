from losses.loss_types import LossType
import subprocess
from tqdm import tqdm

# mix rates
mix_rates = []
mix_rates.extend([i * 1e-2 for i in range(1, 10)])
mix_rates.extend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

loss_types = [LossType.PROB, LossType.EXP, LossType.DIVDIS]

i = 0
for loss_type in loss_types:
    for mix_rate in tqdm(mix_rates, desc=f'loss_type={loss_type}'):
        command = [
            'python',
            'toy_2d.py',
            f'loss_type={loss_type.name}',
            f'mix_rate={mix_rate}',
        ]
        if loss_type == LossType.EXP: 
            # compute optimal batch size, assuming k=4 (or k=2 I guess?)
            batch_size = round(4 / (mix_rate))
            command.append(f'target_batch_size={batch_size}')

        subprocess.run(command)