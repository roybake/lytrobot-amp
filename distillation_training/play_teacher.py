#!/usr/bin/env python3
"""Play a privileged distillation Teacher in the interactive MuJoCo viewer."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distillation_training.networks import SharedActor
from distillation_training.envs import DistillationForceCpuEnv, DistillationNoForceCpuEnv
from distillation_training.runtime import tensor, env_step, validate_checkpoint
from lytrobot_amp.env import LytRobotEnvCfg


def playback_env(checkpoint: dict, seed: int):
    """Replay with the saved task, including pre-profile Teacher checkpoints."""
    saved = checkpoint['env_config']
    valid_fields = {field.name for field in fields(LytRobotEnvCfg)}
    config = {key: value for key, value in saved.items() if key in valid_fields}
    motion_file = Path(saved['motion_file'])
    if not motion_file.is_file():
        motion_file = ROOT / 'robot' / 'reference_data' / motion_file.name
    config.update(
        model_xml=str(ROOT / 'robot' / 'MJCFmodelposition.xml'),
        motion_file=str(motion_file),
        num_envs=1,
        seed=seed,
        reset_noise_scale=0.0,
        randomize_episode_seed=False,
        episode_timeout_jitter_steps=0,
        domain_randomization=False,
        stand_start_prob=1.0,
        motion_reset_prob=0.0,
    )
    env_type = (DistillationForceCpuEnv if checkpoint['sensor_variant'] == 'phase_force'
                else DistillationNoForceCpuEnv)
    env = env_type(LytRobotEnvCfg(**config))
    env.task_reward_mode = checkpoint.get('task_reward_mode') or (
        'sensor_stability' if config.get('sensor_stability_reward') else 'stage1_reference'
    )
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=0, help="0 runs until the viewer closes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--camera-mode', choices=('fixed', 'follow'), default='fixed')
    parser.add_argument('--headless', action='store_true', help='Evaluate without opening a viewer.')
    args = parser.parse_args()
    if args.headless and args.steps <= 0:
        parser.error('--headless requires --steps > 0')

    checkpoint = validate_checkpoint(args.checkpoint, 'teacher',
                                     allow_legacy_reward_checkpoint=True)
    env = playback_env(checkpoint, args.seed)
    actor = SharedActor(checkpoint["sensor_dim"])
    actor.load_state_dict(checkpoint["shared_actor"])
    actor.eval()
    observation = tensor(env.reset(), "cpu")

    base = env.base_env
    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(base.model, base.datas[0])
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        viewer.cam.lookat[:] = (0.0, 0.0, 0.8)
        viewer.sync()
    rewards = []
    velocities = []
    falls = []
    step = 0
    try:
        while (viewer is None or viewer.is_running()) and (args.steps <= 0 or step < args.steps):
            with torch.no_grad():
                privileged = tensor(env.privileged_vector(), "cpu")
                action = actor.mean(observation, privileged)
            observation, reward, done, info = env_step(env, action)
            rewards.append(float(reward.mean()))
            velocities.append(float(info['forward_velocity'][0]))
            if bool(done[0]):
                falls.append(step + 1)
            if viewer is not None:
                if args.camera_mode == 'follow':
                    viewer.cam.lookat[:] = base.datas[0].qpos[:3]
                viewer.sync()
                time.sleep(max(float(base.dt), 0.0))
            step += 1
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
    mean_reward = sum(rewards) / max(len(rewards), 1)
    mean_velocity = sum(velocities) / max(len(velocities), 1)
    print(f'[TEACHER PLAY] steps={step} mean_reward={mean_reward:.5f} '
          f'mean_forward_velocity={mean_velocity:.4f}m/s reset_steps={falls}')


if __name__ == "__main__":
    main()
