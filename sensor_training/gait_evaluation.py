"""Evaluate sustained walking and phase-aligned human gait tracking on CPU."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from scripts.play import OnnxPolicy, load_onnx_metadata
from sensor_training.envs import SensorHistoryCpuAmpEnv


def evaluate_gait_checkpoint(checkpoint: str | Path, *, seconds: float = 20.,
                             force_loop: bool = False,
                             seed: int = 42,
                             trace_path: Path | None = None) -> dict[str, float]:
    if seconds <= 0:
        raise ValueError('Gait evaluation duration must be positive.')
    path = Path(checkpoint).with_suffix('.onnx')
    metadata = load_onnx_metadata(path)
    allowed = {field.name for field in fields(LytRobotEnvCfg)}
    cfg = LytRobotEnvCfg(**{k: v for k, v in metadata['env_config'].items() if k in allowed})
    cfg.num_envs = 1
    cfg.seed = seed
    cfg.domain_randomization = False
    cfg.reset_noise_scale = cfg.action_noise_std = 0.
    cfg.randomize_episode_seed = False
    cfg.stand_start_prob = 1.
    cfg.motion_reset_prob = 0.
    cfg.episode_steps = max(1, round(seconds / (cfg.timestep * cfg.decimation)))
    cfg.episode_timeout_jitter_steps = 0
    if force_loop:
        cfg.motion_loop = (True,)
    sensor = bool(metadata.get('sensor_policy_architecture'))
    env = SensorHistoryCpuAmpEnv(cfg) if sensor else LytRobotAmpEnv(cfg)
    base = env.base_env if sensor else env
    reference = base.motion.motions[0] if hasattr(base.motion, 'motions') else base.motion
    if not reference.source_loop:
        env.close()
        raise ValueError('Continuous gait evaluation requires a looping reference.')
    policy = OnnxPolicy(path)
    policy.validate_dimensions(env.obs_dim, env.action_dim)
    obs = env.reset()
    starts = np.ones(1, dtype=np.float32)
    rows = []
    fell = False
    names = reference.common_joint_names
    knees = [i for i, name in enumerate(names) if 'knee_angle' in name]
    try:
        for step in range(cfg.episode_steps):
            action = policy.act_inference(obs, episode_start=starts)
            obs, _, done, info = env.step(action)
            starts = np.asarray(done, dtype=np.float32)
            if done[0] and float(info.get('termination', [0.])[0]) > .5:
                fell = True
                break
            if done[0]:
                # The only permissible non-fall terminal is the actual time
                # limit. A cycle-end reset must never pass as sustained gait.
                if step + 1 < cfg.episode_steps:
                    raise RuntimeError('Unexpected reset before the continuous evaluation time limit.')
                break
            data = base.datas[0]
            targets = base._targets_at_current_phase()
            actual = np.asarray(data.qpos[reference.robot_qpos_adrs])
            desired = targets.joint_pos[0]
            foot_error = base._foot_positions_local(data) - targets.foot_pos[0]
            row = {'time_s': (step+1)*base.dt,
                   'forward_velocity': float(info['forward_velocity'][0]),
                   'phase': float(base.motion_phase[0]),
                   'foot_error_squared': float(np.mean(foot_error**2))}
            for i, name in enumerate(names):
                row[name] = float(actual[i])
                row['reference_'+name] = float(desired[i])
            rows.append(row)
        duration = (step+1)*base.dt
        steady = [row for row in rows if row['time_s'] >= 2.]
        if not steady:
            steady = rows
        velocity = float(np.mean([r['forward_velocity'] for r in steady])) if steady else 0.
        knee_error = np.asarray([[r[names[i]]-r['reference_'+names[i]] for i in knees] for r in steady])
        knee_rmse = float(np.sqrt(np.mean(knee_error**2))) if knee_error.size else np.pi
        foot_rmse = float(np.sqrt(np.mean([r['foot_error_squared'] for r in steady]))) if steady else 1.
        reference_rom = np.ptp(reference.reference_joint_pos[:, knees], axis=0)
        knee_values = np.asarray([[r[names[i]] for i in knees] for r in steady])
        if len(knee_values) >= 2:
            actual_rom = np.percentile(knee_values, 95, axis=0) - np.percentile(knee_values, 5, axis=0)
            rom_ratio = float(np.min(np.clip(actual_rom / np.maximum(reference_rom, 1e-6), 0, 1)))
            lock_fraction = float(np.mean(np.abs(knee_values) < .12))
        else:
            rom_ratio, lock_fraction = 0., 1.
        survived = float(not fell and step+1 == cfg.episode_steps)
        velocity_track = float(np.exp(-((velocity-cfg.target_velocity)/.25)**2))
        tracking = float(np.exp(-(knee_rmse/.30)**2 - (foot_rmse/.08)**2))
        score = min(duration/seconds, 1.) * velocity_track * tracking * rom_ratio * (1-lock_fraction)
        accepted = survived and abs(velocity-cfg.target_velocity) <= .15 and knee_rmse <= .20 and foot_rmse <= .06 and rom_ratio >= .70 and lock_fraction <= .10
        metrics = {
            'eval/gait_accepted': float(accepted), 'eval/gait_score': score,
            'eval/gait_survived': survived, 'eval/gait_duration_s': duration,
            'eval/gait_forward_velocity': velocity,
            'eval/gait_knee_rmse_rad': knee_rmse,
            'eval/gait_foot_rmse_m': foot_rmse,
            'eval/gait_knee_rom_ratio': rom_ratio,
            'eval/gait_knee_lock_fraction': lock_fraction,
        }
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ['time_s', 'forward_velocity', 'phase', 'foot_error_squared'])
                writer.writeheader()
                writer.writerows(rows)
        return metrics
    finally:
        env.close()


def summarize_gait_evaluations(results: list[dict[str, float]]) -> dict[str, float]:
    """A failed start cannot be hidden by averaging its acceptance flag."""
    if not results:
        raise ValueError('At least one gait evaluation is required.')
    worst_high = {'eval/gait_knee_rmse_rad', 'eval/gait_foot_rmse_m',
                  'eval/gait_knee_lock_fraction'}
    aggregate = {}
    for name in results[0]:
        values = [result[name] for result in results]
        aggregate[name] = (float(np.mean(values)) if name == 'eval/gait_forward_velocity'
                           else max(values) if name in worst_high else min(values))
    aggregate['eval/gait_seed_count'] = float(len(results))
    return aggregate


def evaluate_gait_suite(checkpoint, *, seconds=20., seeds=(42, 123, 2026), trace_path=None):
    results = [evaluate_gait_checkpoint(checkpoint, seconds=seconds, seed=seed,
               trace_path=trace_path if index == 0 else None)
               for index, seed in enumerate(seeds)]
    return summarize_gait_evaluations(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--force-loop', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_gait_checkpoint(args.checkpoint, seconds=args.seconds,
                                      force_loop=args.force_loop,
                                      trace_path=args.output.with_suffix('.csv'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
