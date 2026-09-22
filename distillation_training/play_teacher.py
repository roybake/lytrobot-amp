#!/usr/bin/env python3
"""Play a privileged distillation Teacher in the interactive MuJoCo viewer."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distillation_training.networks import SharedActor
from distillation_training.runtime import make_env, tensor, env_step, validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=0, help="0 runs until the viewer closes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = validate_checkpoint(args.checkpoint, "teacher")
    args.sensor_variant = checkpoint["sensor_variant"]
    args.num_envs = 1
    args.num_steps = 1
    args.device = "cpu"
    args.sim_backend = "cpu"

    env, _ = make_env(args)
    actor = SharedActor(checkpoint["sensor_dim"])
    actor.load_state_dict(checkpoint["shared_actor"])
    actor.eval()
    observation = tensor(env.reset(), "cpu")

    from mujoco import viewer as mujoco_viewer

    base = env.base_env
    viewer = mujoco_viewer.launch_passive(base.model, base.datas[0])
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -18.0
    viewer.sync()
    rewards = []
    step = 0
    try:
        while viewer.is_running() and (args.steps <= 0 or step < args.steps):
            with torch.no_grad():
                privileged = tensor(env.privileged_vector(), "cpu")
                action = actor.mean(observation, privileged)
            observation, reward, _, _ = env_step(env, action)
            rewards.append(float(reward.mean()))
            viewer.cam.lookat[:] = base.datas[0].qpos[:3]
            viewer.sync()
            time.sleep(max(float(base.dt), 0.0))
            step += 1
    finally:
        viewer.close()
        env.close()
    mean_reward = sum(rewards) / max(len(rewards), 1)
    print(f"[TEACHER PLAY] steps={step} mean_reward={mean_reward:.5f}")


if __name__ == "__main__":
    main()
