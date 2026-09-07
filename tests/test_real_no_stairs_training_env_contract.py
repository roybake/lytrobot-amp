from __future__ import annotations

from pathlib import Path

import numpy as np

from lytrobot_amp.env import LytRobotEnvCfg
from scripts import train
from scripts.train_multimotion_no_stairs import default_train_args
from sensor_training.envs import SensorHistoryCpuAmpEnv


ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = ROOT / "robot" / "reference_data" / "multimotion"
MOTION_NAMES = ("walk", "rough_walk", "slope_up", "slope_down")


def test_real_no_stairs_training_env_contract() -> None:
    # Parse the exact canonical launcher list first; this test must not drift
    # into a hand-written approximation of the formal training environment.
    from unittest.mock import patch

    with patch.object(train.sys, "argv", ["train.py", *default_train_args()]):
        parsed = train.parse_args()
    assert parsed.training_task == "multimotion_no_stairs_amp_single"

    env = SensorHistoryCpuAmpEnv(
        LytRobotEnvCfg(
            model_xml=parsed.model_xml,
            motion_files=tuple(parsed.motion_files),
            motion_names=tuple(parsed.motion_names),
            motion_weights=tuple(parsed.motion_weights),
            motion_loop=tuple(parsed.motion_loop),
            skill_conditioning=parsed.skill_conditioning,
            use_reference_velocity=parsed.use_reference_velocity,
            target_velocity=parsed.target_velocity,
            terrain_types=tuple(parsed.terrain_types),
            terrain_asset_dir=parsed.terrain_asset_dir,
            stair_terrain=parsed.stair_terrain,
            serial_terrain_course=parsed.serial_terrain_course,
            serial_course_single_motion=parsed.serial_course_single_motion,
            terrain_course_variants=parsed.terrain_course_variants,
            terrain_course_random_assignment=parsed.terrain_course_random_assignment,
            serial_reference_state_init=parsed.serial_reference_state_init,
            motion_reset_prob=parsed.motion_reset_prob,
            stand_start_prob=parsed.stand_start_prob,
            reset_noise_scale=parsed.reset_noise_scale,
            domain_randomization=parsed.domain_randomization,
            amp_task_only=parsed.amp_task_only,
            reference_ctrl_blend=parsed.reference_ctrl_blend,
            forward_velocity_tracking_sigma=parsed.forward_velocity_tracking_sigma,
            forward_velocity_reward_coef=parsed.forward_velocity_reward_coef,
            forward_progress_reward_coef=parsed.forward_progress_reward_coef,
            forward_velocity_deficit_penalty_coef=parsed.forward_velocity_deficit_penalty_coef,
            forward_velocity_excess_penalty_coef=parsed.forward_velocity_excess_penalty_coef,
            foot_slip_velocity_threshold=parsed.foot_slip_velocity_threshold,
            foot_slip_penalty_coef=parsed.foot_slip_penalty_coef,
            terrain_course_random_start_prob=parsed.terrain_course_random_start_prob,
            terrain_course_phase_sync=parsed.terrain_course_phase_sync,
            terrain_course_phase_lookahead=parsed.terrain_course_phase_lookahead,
            terrain_course_stage_reward=parsed.terrain_course_stage_reward,
            terrain_course_transition_survival_steps=parsed.terrain_course_transition_survival_steps,
            terrain_course_transition_success_reward=parsed.terrain_course_transition_success_reward,
            terrain_course_run_out=parsed.terrain_course_run_out,
            rough_terrain_height=parsed.rough_terrain_height,
            slope_angle_deg=parsed.slope_angle_deg,
            motion_transition_blend_steps=parsed.motion_transition_blend_steps,
            contact_reward_coef=parsed.contact_reward_coef,
            alternating_step_reward_coef=parsed.alternating_step_reward_coef,
            terrain_penetration_tolerance=parsed.terrain_penetration_tolerance,
            terrain_penetration_penalty_coef=parsed.terrain_penetration_penalty_coef,
            num_envs=4,
            episode_timeout_jitter_steps=0,
        )
    )
    try:
        observation = env.reset()
        assert env.obs_dim == 56
        assert env.observation_metadata["actor_input_fields"] == (
            "imu",
            "tendon_length",
        )
        assert env.observation_metadata["command_fields"] == ()
        assert env.observation_metadata["command_dim"] == 0
        assert env.base_env.cfg.use_reference_velocity is False
        assert env.base_env.motion_count == 4
        assert "stairs" not in env.base_env.cfg.terrain_types
        assert observation.shape == (4, 56)
        assert np.isfinite(observation).all()

        action = np.zeros((4, env.action_dim), dtype=np.float32)
        next_observation, reward, done, info = env.step(action)
        assert next_observation.shape == (4, 56)
        assert np.isfinite(next_observation).all()
        assert np.isfinite(reward).all()
        assert np.isfinite(done).all()
        assert np.isfinite(info["terrain_scenario_id"]).all()
        assert np.isfinite(info["terrain_transition_from"]).all()
        assert np.isfinite(info["terrain_transition_to"]).all()
        assert set(np.unique(info["terrain_scenario_id"]).tolist()).issubset({0, 1, 2, 3})
    finally:
        env.close()
