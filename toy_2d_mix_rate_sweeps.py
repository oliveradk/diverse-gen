from losses.loss_types import LossType
import subprocess
from tqdm import tqdm

# mix rates
mix_rates = [i * 1e-2 for i in range(1, 10)] + [i * 1e-1 for i in range(1, 11)]

i = 0
for loss_type in [LossType.PROB, LossType.EXP, LossType.DIVDIS]:
    for mix_rate in tqdm(mix_rates, desc=f'loss_type={loss_type}'):
        # run python toy_2d_ace_prob_mix_rates.py --loss_type {loss_type} --mix_rate {mix_rate}
        command = [
            'python',
            'toy_2d.py',
            f'loss_type={loss_type.name}',
            f'mix_rate={mix_rate}',
        ]
        subprocess.run(command)