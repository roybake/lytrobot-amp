#!/usr/bin/env python3
"""Measure physical gait contact and motion scores for a sensor ONNX policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lytrobot_amp.gait_constraints import sensor_gait_scores_numpy
from scripts import play as playback
from sensor_training.envs import SensorHistoryCpuAmpEnv


def _score_state(dt: float):
    return SimpleNamespace(
        dt=float(dt),
        sensor_last_foot_contact=np.zeros((1, 2), dtype=bool),
        sensor_foot_roll_stage=np.zeros((1, 2), dtype=np.int32),
        sensor_toe_only_steps=np.zeros((1, 2), dtype=np.int32),
        sensor_last_touchdown_side=np.full(1, -1, dtype=np.int32),
        sensor_liftoff_x=np.zeros((1, 2), dtype=np.float32),
        sensor_had_liftoff=np.zeros((1, 2), dtype=bool),
        sensor_stride_ema=np.zeros((1, 2), dtype=np.float32),
        sensor_stride_seen=np.zeros((1, 2), dtype=bool),
        sensor_contact_duty_ema=np.zeros((1, 2), dtype=np.float32),
        sensor_single_support_ema=np.zeros(1, dtype=np.float32),
        sensor_time_since_touchdown=np.zeros(1, dtype=np.float32),
        sensor_step_interval_ema=np.zeros(1, dtype=np.float32),
        sensor_touchdown_seen=np.zeros(1, dtype=bool),
        sensor_cadence_seen=np.zeros(1, dtype=bool),
        sensor_swing_steps=np.zeros((1, 2), dtype=np.int32),
    )


def main() -> None:
    args = playback.parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if checkpoint_path.suffix.lower() != ".onnx":
        raise ValueError("Gait evaluation requires an exported .onnx sensor checkpoint.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive for gait evaluation.")
    if args.seed is None:
        args.seed = 42
    # Deployment starts from standing; random motion resets can conceal a
    # policy that falls during its stand-to-walk transition.
    if args.stand_start_prob is None:
        args.stand_start_prob = 1.0
    checkpoint = playback.load_onnx_metadata(checkpoint_path)
    stage = playback.infer_stage(args.checkpoint, checkpoint, args.stage)
    env_cfg = playback.make_env_cfg(args, checkpoint, stage)
    env_cfg.num_envs = 1
    env = SensorHistoryCpuAmpEnv(env_cfg)
    policy = playback.OnnxPolicy(checkpoint_path)
    policy.validate_dimensions(env.obs_dim, env.action_dim)
    rows: list[dict[str, float]] = []
    episode_distances: list[float] = []
    episode_steps: list[int] = []
    distance = 0.0
    length = 0
    try:
        obs = env.reset()
        episode_start = np.ones(1, dtype=np.float32)
        base = env.base_env
        state = _score_state(env.dt)
        previous_foot_world = base._foot_positions_world(base.datas[0])
        for step in range(args.steps):
            action = policy.act_inference(obs, episode_start=episode_start)
            obs, _reward, done, info = env.step(action)
            episode_start = np.asarray(done, dtype=np.float32)
            distance += float(np.asarray(info["forward_velocity"])[0]) * env.dt
            length += 1
            if done[0] > 0.5:
                episode_distances.append(distance)
                episode_steps.append(length)
                rows.append({"step": float(step), "reset": 1.0})
                distance = 0.0
                length = 0
                state = _score_state(env.dt)
                previous_foot_world = base._foot_positions_world(base.datas[0])
                continue
            data = base.datas[0]
            foot_world = base._foot_positions_world(data)
            foot_delta = (
                foot_world[[0, 2]] - previous_foot_world[[0, 2]]
            ) / max(env.dt, 1.0e-6)
            foot_local = base._foot_positions_local(data)
            scores = sensor_gait_scores_numpy(
                state,
                0,
                base._sensor_foot_roll_contacts(data),
                foot_world[[0, 2], 0],
                float(foot_local[0, 1] + foot_local[2, 1]),
                np.linalg.norm(foot_delta[:, :2], axis=1),
                foot_delta[:, 0],
                np.asarray(
                    [np.min(foot_world[0:2, 2]), np.min(foot_world[2:4, 2])],
                    dtype=np.float32,
                ),
            )
            row = {"step": float(step), **scores}
            for key in (
                "forward_velocity", "knee_flexion_track", "sensor_upright",
                "sensor_posture_track", "sensor_gait_height_track", "sensor_gait_quality",
                "gait_progress",
            ):
                if key in info:
                    row[key] = float(np.asarray(info[key])[0])
            row["reset"] = 0.0
            rows.append(row)
            previous_foot_world = foot_world.copy()
    finally:
        env.close()

    warmup = 500 if args.steps > 500 else args.steps // 2
    steady = [row for row in rows[warmup:] if row["reset"] < 0.5]
    if not steady:
        raise RuntimeError("No non-terminal gait samples remain after warmup.")
    mean = {
        key: float(np.mean([row[key] for row in steady]))
        for key in steady[0]
        if key not in ("step", "reset")
    }
    full_gait = [row for row in steady if row.get("gait_progress", 0.0) >= 0.95]
    full_gait_speed = (
        float(np.mean([row["forward_velocity"] for row in full_gait]))
        if full_gait else 0.0
    )
    # Calibrated against walk1_omniretarget.npz contact samples: roll=0.795,
    # mirror=0.750, alternation=0.813, hop=0, tiptoe=0.075.  A short toe-only
    # push-off is allowed; sustained toe-first support and flight are not.
    checks = {
        "no_resets": sum(row["reset"] for row in rows) == 0,
        "low_flight_fraction": mean["sensor_hop"] <= 0.05,
        "low_tiptoe_fraction": mean["sensor_tiptoe"] <= 0.08,
        "heel_sole_toe_roll": mean["sensor_foot_roll"] >= 0.75,
        "mirrored_steps": mean["sensor_mirror"] >= 0.72,
        "full_length_steps": mean["sensor_stride_length"] >= 0.75,
        "natural_cadence": mean["sensor_step_cadence"] >= 0.70,
        "swing_clearance": mean["sensor_swing_clearance"] >= 0.65,
        "alternating_support": mean["sensor_alternation"] >= 0.75,
        "upright_posture": mean.get("sensor_posture_track", 0.0) >= 0.70,
        "gait_height": mean.get("sensor_gait_height_track", 0.0) >= 0.70,
        "forward_speed": bool(full_gait) and abs(full_gait_speed - float(env_cfg.target_velocity)) <= 0.20,
    }
    report = {
        "checkpoint": str(checkpoint_path),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "warmup_steps": int(warmup),
        "resets": int(sum(row["reset"] for row in rows)),
        "completed_episode_distances_m": episode_distances,
        "completed_episode_steps": episode_steps,
        "current_episode_distance_m": distance,
        "current_episode_steps": length,
        "full_gait_samples": len(full_gait),
        "mean_full_gait_speed_mps": full_gait_speed,
        "mean": mean,
        "checks": checks,
        "natural_gait_pass": all(checks.values()),
    }
    report_path = checkpoint_path.parent / (
        f"gait_evaluation_{checkpoint_path.stem}_{args.steps}_{args.seed}.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[GAIT] saved {report_path}")


if __name__ == "__main__":
    main()
