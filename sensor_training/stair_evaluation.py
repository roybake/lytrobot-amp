"""Deterministic frame-zero evaluation for fixed staircase checkpoints."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import torch

from .envs import SensorHistoryCpuAmpEnv


def evaluate_fixed_stair_route(actor, env_cfg) -> dict[str, float]:
    cfg = replace(env_cfg, num_envs=1, stair_phase_reset_prob=0.0,
                  stand_start_prob=0.0, reset_noise_scale=0.0,
                  action_noise_std=0.0, domain_randomization=False,
                  randomize_episode_seed=False, episode_timeout_jitter_steps=0)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = None
    try:
        policy = copy.deepcopy(actor).cpu().eval()
        env = SensorHistoryCpuAmpEnv(cfg)
        observation = env.reset()
        base = env.base_env
        start_x = float(base.datas[0].qpos[0])
        reference = base.motion.motions[0] if hasattr(base.motion, "motions") else base.motion
        distance = float(reference.reference_qpos[-1, 0] - reference.reference_qpos[0, 0])
        hidden = policy.initial_state(1)
        starts = torch.ones(1)
        progress = 0.0
        xy_errors = []
        success = 0.0
        with torch.inference_mode():
            for step in range(cfg.episode_steps):
                action, hidden = policy.act_inference_recurrent(
                    torch.as_tensor(observation), hidden, starts)
                previous_x = float(base.datas[0].qpos[0])
                observation, _, done, info = env.step(action.numpy())
                error = float(info["deepmimic_root_xy_error"][0])
                xy_errors.append(error)
                # step() autoresets terminals; reconstruct just that final
                # displacement from its pre-reset velocity instead of qpos.
                x = (previous_x + float(info["forward_velocity"][0]) * base.dt
                     if done[0] else float(base.datas[0].qpos[0]))
                if error <= cfg.deepmimic_route_max_xy_error:
                    progress = max(progress, float(np.clip((x-start_x) / max(distance, 1e-6), 0, 1)))
                starts.zero_()
                if done[0]:
                    success = float(info.get("successes", [0.0])[0])
                    break
        return {
            "eval/stair_route_success": success,
            "eval/stair_route_progress": progress,
            "eval/stair_episode_steps": float(step + 1),
            "eval/stair_mean_xy_error": float(np.mean(xy_errors)),
            "eval/stair_terminal_xy_error": float(xy_errors[-1]),
        }
    finally:
        if env is not None:
            env.close()
        torch.set_num_threads(threads)
