"""Continue the saved teacher with reference-consistent stance/phase rewards."""
from pathlib import Path
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import torch
from lytrobot_amp.env import LytRobotEnvCfg
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
from lytrobot_amp.ppo_amp import PPOAMPConfig, PPOAMPTrainer

source = ROOT / 'logs/left_leg_diagnosis/teacher_best.pt'
output = Path(__file__).resolve().parent
output.mkdir(exist_ok=True)
checkpoint = torch.load(source, map_location='cpu', weights_only=False)
env_cfg = LytRobotEnvCfg(**checkpoint['env_config'])
env_cfg.motion_file = str(output / 'reference.npz')
ppo_cfg = PPOAMPConfig(**checkpoint['ppo_amp_config'])
ppo_cfg.num_steps = 96
ppo_cfg.learning_rate = 1e-4
ppo_cfg.max_learning_rate = 1e-4
torch.set_num_threads(1)
torch.manual_seed(ppo_cfg.seed)
np.random.seed(ppo_cfg.seed)
torch.set_float32_matmul_precision('high')
env = LytRobotWarpAmpEnv(env_cfg, device='cuda')
trainer = PPOAMPTrainer(env, ppo_cfg, device='cuda', log_dir=output)
start = trainer.load(source)
print(json.dumps({'source': str(source), 'start_iteration': start, 'num_envs': env.num_envs,
                  'learning_rate': ppo_cfg.learning_rate, 'num_steps': ppo_cfg.num_steps}), flush=True)
started = time.perf_counter()
try:
    with (output / 'metrics.csv').open('w') as file:
        writer = None
        for update in range(500):
            metrics = trainer.train_iteration()
            row = {'update': update + 1, **metrics}
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            file.flush()
            if (update + 1) % 25 == 0:
                print(json.dumps({'update': update + 1, 'elapsed_s': round(time.perf_counter()-started, 1),
                                  **{key: value for key, value in metrics.items() if key in (
                                      'mean_reward', 'mean_task_reward', 'mean_episode_length',
                                      'termination_fraction', 'task_forward_velocity', 'task_left_swing_contact',
                                      'task_right_swing_contact', 'task_phase_symmetry', 'task_stance_foot')}}), flush=True)
            if (update + 1) % 250 == 0:
                trainer.save(output / f'after_{update+1}.pt', iteration=start+update)
finally:
    env.close()
