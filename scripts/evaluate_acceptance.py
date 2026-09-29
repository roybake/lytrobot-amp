#!/usr/bin/env python3
"""Evaluate the sensor-only policy against the project acceptance criteria."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lytrobot_amp.benchmark import (  # noqa: E402
    AcceptanceResult,
    MAX_ROUGH_HEIGHT_DIFFERENCE_M,
    MAX_SLOPE_DEG,
    MIN_WALKING_SPEED_KMH,
    MIN_WALKING_SPEED_MPS,
    REQUIRED_SCENARIOS,
    STAIR_ASSESSMENT_CASES,
    STAIR_HEIGHT_RANGE_M,
    STAIR_TREAD_RANGE_M,
)
from lytrobot_amp.env import LytRobotEnvCfg  # noqa: E402
from lytrobot_amp.terrain import course_stage_scenarios  # noqa: E402
from sensor_training.envs import SensorHistoryCpuAmpEnv  # noqa: E402
from sensor_training.networks import SensorActorCritic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flat-steps", type=int, default=2000)
    parser.add_argument("--course-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def checkpoint_env_cfg(checkpoint: dict, *, seed: int) -> LytRobotEnvCfg:
    saved = checkpoint.get("env_config")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint has no env_config metadata.")
    valid = {field.name for field in fields(LytRobotEnvCfg)}
    cfg = LytRobotEnvCfg()
    for key, value in saved.items():
        if key in valid:
            setattr(cfg, key, value)
    cfg.num_envs = 1
    cfg.seed = int(seed)
    cfg.randomize_episode_seed = False
    cfg.domain_randomization = False
    cfg.domain_randomization_scale = 0.0
    cfg.reset_noise_scale = 0.0
    cfg.stand_start_prob = 0.0
    cfg.episode_timeout_jitter_steps = 0
    cfg.skill_conditioning = False
    return cfg


def load_policy(
    checkpoint: dict,
    env: SensorHistoryCpuAmpEnv,
    device: torch.device,
) -> SensorActorCritic:
    metadata = checkpoint.get("sensor_observation")
    if metadata is None or not bool(metadata.get("strict_sensor_only", False)):
        raise ValueError(
            "Acceptance evaluation requires a strict IMU+tendon sensor checkpoint."
        )
    if metadata != env.observation_metadata:
        raise ValueError("Checkpoint sensor layout does not match the 336-D evaluator layout.")
    architecture = checkpoint.get("sensor_policy_architecture")
    if not isinstance(architecture, dict) or not bool(architecture.get("recurrent", False)):
        raise ValueError(
            "Acceptance evaluation requires the recurrent proprioceptive MoE sensor policy."
        )
    ppo_cfg = checkpoint.get("ppo_amp_config", {})
    policy = SensorActorCritic(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        actor_hidden_dims=tuple(ppo_cfg.get("actor_hidden_dims", (256, 128))),
        critic_hidden_dims=tuple(ppo_cfg.get("critic_hidden_dims", (256, 128))),
        init_std=float(ppo_cfg.get("init_std", 0.8)),
        min_action_std=float(ppo_cfg.get("min_action_std", 0.05)),
        max_action_std=float(ppo_cfg.get("max_action_std", 0.8)),
        sensor_frame_dim=int(architecture["sensor_frame_dim"]),
        actor_history_frames=int(architecture.get("actor_history_frames", 1)),
        recurrent_hidden_dim=int(architecture["recurrent_hidden_dim"]),
        use_moe=bool(architecture.get("use_moe", True)),
        num_experts=int(architecture["num_experts"]),
        gate_hidden_dims=tuple(architecture["gate_hidden_dims"]),
        expert_hidden_dims=tuple(architecture["expert_hidden_dims"]),
        residual_scale=float(architecture["residual_scale"]),
        privileged_context_dim=int(architecture.get("privileged_context_dim", 0)),
    ).to(device)
    policy.load_state_dict(checkpoint["actor_critic"])
    policy.eval()
    return policy


def policy_action(
    policy: SensorActorCritic,
    observation: np.ndarray,
    device: torch.device,
    recurrent_state: torch.Tensor,
    episode_start: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor]:
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    start_tensor = torch.as_tensor(episode_start, dtype=torch.float32, device=device)
    action, next_state = policy.act_inference_recurrent(
        tensor,
        recurrent_state,
        start_tensor,
    )
    return action.cpu().numpy(), next_state


def flat_cfg(source: LytRobotEnvCfg) -> LytRobotEnvCfg:
    names = tuple(source.motion_names)
    try:
        walk_index = names.index("walk")
    except ValueError as exc:
        raise ValueError("Checkpoint motion library has no 'walk' clip.") from exc
    source.motion_files = (source.motion_files[walk_index],)
    source.motion_file = source.motion_files[0]
    source.motion_names = ("walk",)
    source.motion_weights = (1.0,)
    source.motion_loop = (True,)
    source.terrain_types = ()
    source.serial_terrain_course = False
    source.stair_terrain = False
    source.motion_reset_prob = 1.0
    return source


def assessment_course_cfg(
    source: LytRobotEnvCfg,
    *,
    stair_height: float = STAIR_HEIGHT_RANGE_M[1],
    stair_tread: float = STAIR_TREAD_RANGE_M[0],
) -> LytRobotEnvCfg:
    """Force evaluation onto the contractual worst-case terrain geometry."""

    if not STAIR_HEIGHT_RANGE_M[0] <= stair_height <= STAIR_HEIGHT_RANGE_M[1]:
        raise ValueError(f"stair height {stair_height} is outside {STAIR_HEIGHT_RANGE_M}.")
    if not STAIR_TREAD_RANGE_M[0] <= stair_tread <= STAIR_TREAD_RANGE_M[1]:
        raise ValueError(f"stair tread {stair_tread} is outside {STAIR_TREAD_RANGE_M}.")

    source.terrain_types = ("rough", "slope_up", "slope_down", "stairs")
    source.rough_terrain_height = MAX_ROUGH_HEIGHT_DIFFERENCE_M
    source.slope_angle_deg = MAX_SLOPE_DEG
    source.stair_step_height = float(stair_height)
    source.stair_step_depth = float(stair_tread)
    source.serial_terrain_course = True
    source.terrain_course_random_start_prob = 0.0
    source.terrain_course_variants = 1
    source.terrain_course_random_assignment = False
    return source


def evaluate_flat(
    checkpoint: dict,
    source_cfg: LytRobotEnvCfg,
    device: torch.device,
    steps: int,
) -> dict[str, float | int | bool]:
    env = SensorHistoryCpuAmpEnv(flat_cfg(source_cfg))
    try:
        policy = load_policy(checkpoint, env, device)
        observation = env.reset()
        recurrent_state = policy.initial_state(env.num_envs)
        episode_start = np.ones(env.num_envs, dtype=np.float32)
        velocity_sum = 0.0
        terminations = 0
        knee_indices = np.asarray(
            [
                index
                for index, name in enumerate(env.base_motion.common_joint_names)
                if "knee" in name
            ],
            dtype=np.int64,
        )
        if len(knee_indices) != 2:
            raise ValueError(
                f"Natural-gait evaluation requires two knee joints; found {len(knee_indices)}."
            )
        knee_qpos_adrs = env.base_motion.robot_qpos_adrs[knee_indices]
        knee_flexion_sum = np.zeros(2, dtype=np.float64)
        knee_track_sum = 0.0
        locked_knee_samples = 0
        gait_samples = 0
        knee_target = max(float(env.cfg.sensor_knee_flexion_target), 1.0e-6)
        for _ in range(max(int(steps), 1)):
            action, recurrent_state = policy_action(
                policy,
                observation,
                device,
                recurrent_state,
                episode_start,
            )
            observation, _reward, done, info = env.step(action)
            episode_start = np.asarray(done, dtype=np.float32)
            velocity = float(info["forward_velocity"][0])
            velocity_sum += velocity
            terminations += int(float(info["termination"][0]) > 0.5)
            if float(done[0]) <= 0.5:
                knee_flexion = np.maximum(
                    -np.asarray(env.base_env.datas[0].qpos[knee_qpos_adrs]),
                    0.0,
                )
                knee_track = float(
                    np.clip(np.min(knee_flexion) / knee_target, 0.0, 1.0)
                )
                knee_flexion_sum += knee_flexion
                knee_track_sum += knee_track
                locked_knee_samples += int(knee_track < 0.60)
                gait_samples += 1
        mean_speed = velocity_sum / max(int(steps), 1)
        mean_knee_flexion = knee_flexion_sum / max(gait_samples, 1)
        mean_knee_track = knee_track_sum / max(gait_samples, 1)
        locked_knee_fraction = locked_knee_samples / max(gait_samples, 1)
        return {
            "steps": int(steps),
            "duration_s": float(max(int(steps), 1) * env.dt),
            "mean_speed_mps": mean_speed,
            "mean_speed_kmh": mean_speed * 3.6,
            "terminations": terminations,
            "stable": terminations == 0,
            "mean_right_knee_flexion_rad": float(mean_knee_flexion[0]),
            "mean_left_knee_flexion_rad": float(mean_knee_flexion[1]),
            "mean_knee_flexion_track": float(mean_knee_track),
            "locked_knee_fraction": float(locked_knee_fraction),
            "natural_knee_gait": bool(
                mean_knee_track >= 0.90 and locked_knee_fraction <= 0.10
            ),
        }
    finally:
        env.close()


def evaluate_course(
    checkpoint: dict,
    cfg: LytRobotEnvCfg,
    device: torch.device,
    steps: int,
    *,
    stair_height: float = STAIR_HEIGHT_RANGE_M[1],
    stair_tread: float = STAIR_TREAD_RANGE_M[0],
) -> dict[str, object]:
    env = SensorHistoryCpuAmpEnv(
        assessment_course_cfg(
            cfg,
            stair_height=stair_height,
            stair_tread=stair_tread,
        )
    )
    try:
        policy = load_policy(checkpoint, env, device)
        observation = env.reset()
        recurrent_state = policy.initial_state(env.num_envs)
        episode_start = np.ones(env.num_envs, dtype=np.float32)
        order = env.base_env.more_terrain_cfg.normalized_course_orders[0]
        labels = course_stage_scenarios(order)
        visited: set[str] = set()
        terminated: set[str] = set()
        course_successes = 0
        for _ in range(max(int(steps), 1)):
            stage = int(env.base_env.course_stage_ids[0])
            scenario = labels[min(stage, len(labels) - 1)]
            if scenario in REQUIRED_SCENARIOS:
                visited.add(scenario)
            action, recurrent_state = policy_action(
                policy,
                observation,
                device,
                recurrent_state,
                episode_start,
            )
            observation, _reward, done, info = env.step(action)
            episode_start = np.asarray(done, dtype=np.float32)
            if float(info["termination"][0]) > 0.5:
                terminated.add(scenario)
            course_successes += int(float(info["course_complete"][0]) > 0.5)
        stable = sorted(visited - terminated)
        return {
            "steps": int(steps),
            "stair_height_m": float(stair_height),
            "stair_tread_m": float(stair_tread),
            "terrain_order": list(order),
            "stage_scenarios": list(labels),
            "visited_scenarios": sorted(visited),
            "terminated_scenarios": sorted(terminated),
            "stable_scenarios": stable,
            "course_successes": course_successes,
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device(
        args.device
        if not str(args.device).startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    cfg = checkpoint_env_cfg(checkpoint, seed=args.seed)
    flat = evaluate_flat(checkpoint, cfg, device, args.flat_steps)
    course_cases = []
    for stair_height, stair_tread in STAIR_ASSESSMENT_CASES:
        cfg = checkpoint_env_cfg(checkpoint, seed=args.seed)
        course_cases.append(
            evaluate_course(
                checkpoint,
                cfg,
                device,
                args.course_steps,
                stair_height=stair_height,
                stair_tread=stair_tread,
            )
        )
    required_course_scenarios = set(REQUIRED_SCENARIOS) - {"flat"}
    stable = required_course_scenarios.copy()
    for course_case in course_cases:
        stable.intersection_update(course_case["stable_scenarios"])
    if bool(flat["stable"]):
        stable.add("flat")
    course_successes = min(
        (int(course_case["course_successes"]) for course_case in course_cases),
        default=0,
    )
    worst_case_course = next(
        course_case
        for course_case in course_cases
        if course_case["stair_height_m"] == STAIR_HEIGHT_RANGE_M[1]
        and course_case["stair_tread_m"] == STAIR_TREAD_RANGE_M[0]
    )
    result = AcceptanceResult(
        mean_speed_mps=float(flat["mean_speed_mps"]),
        stable_scenarios=tuple(sorted(stable)),
        course_successes=course_successes,
    )
    natural_gait_passed = bool(flat["natural_knee_gait"])
    failures = list(result.failures)
    if not natural_gait_passed:
        failures.append(
            "locked-knee gait exceeds the natural-walking limit "
            f"({float(flat['locked_knee_fraction']):.1%} locked samples)"
        )
    report = {
        "passed": bool(result.passed and natural_gait_passed),
        "checkpoint": str(checkpoint_path),
        "sensor_input": checkpoint["sensor_observation"],
        "criteria": {
            "minimum_speed_kmh": MIN_WALKING_SPEED_KMH,
            "minimum_speed_mps": MIN_WALKING_SPEED_MPS,
            "maximum_rough_height_difference_m": MAX_ROUGH_HEIGHT_DIFFERENCE_M,
            "maximum_slope_deg": MAX_SLOPE_DEG,
            "stair_height_range_m": STAIR_HEIGHT_RANGE_M,
            "stair_tread_range_m": STAIR_TREAD_RANGE_M,
            "stair_assessment_cases": STAIR_ASSESSMENT_CASES,
            "required_scenarios": REQUIRED_SCENARIOS,
            "minimum_mean_knee_flexion_track": 0.90,
            "maximum_locked_knee_fraction": 0.10,
        },
        "flat": flat,
        "course": worst_case_course,
        "course_cases": course_cases,
        "stable_scenarios": list(result.stable_scenarios),
        "failures": failures,
        "scope": (
            "simulation pre-acceptance; physical prototype speed and system "
            "integration require hardware measurement"
        ),
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(serialized)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint_path.parent / "acceptance_report.json"
    )
    output.write_text(serialized + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
