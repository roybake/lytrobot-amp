from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import numpy as np

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from scripts import train
from scripts.train_multimotion_no_stairs_amp_teacher_lane_hold import (
    TASK_NAME,
    default_train_args,
)


ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = ROOT / "robot" / "reference_data" / "multimotion"


def test_lane_hold_launcher_enables_new_contract() -> None:
    args = default_train_args()
    with_args = ["train.py", *args, "--no-wandb", "--max-iterations", "1"]
    original_argv = sys.argv
    try:
        sys.argv = with_args
        parsed = train.parse_args()
    finally:
        sys.argv = original_argv

    assert parsed.training_task == TASK_NAME
    assert parsed.course_lane_hold is True
    assert parsed.serial_reference_state_init is True
    assert parsed.motion_reset_prob == 0.0
    assert parsed.stand_start_prob == 1.0
    assert parsed.reset_noise_scale == 0.0
    assert parsed.action_residual_scale == 0.30
    assert parsed.course_lane_position_reward_coef == 2.0
    assert parsed.course_lane_velocity_reward_coef == 0.5


def test_cpu_lane_hold_observation_and_boundary_termination() -> None:
    cfg = LytRobotEnvCfg(
        model_xml=str(ROOT / "robot" / "MJCFmodelposition_0806.xml"),
        motion_files=tuple(
            str(MOTION_DIR / filename)
            for filename in (
                "walk_omniretarget.npz",
                "rough_walk_omniretarget.npz",
                "slope_up_omniretarget.npz",
                "slope_down_omniretarget.npz",
            )
        ),
        motion_names=("walk", "rough_walk", "slope_up", "slope_down"),
        motion_weights=(0.25, 0.25, 0.25, 0.25),
        motion_loop=(True, False, False, False),
        terrain_types=("rough", "slope_up", "slope_down"),
        serial_terrain_course=True,
        terrain_course_variants=1,
        terrain_course_random_assignment=False,
        legacy_0806_terrain_model=True,
        serial_reference_state_init=True,
        motion_reset_prob=1.0,
        stand_start_prob=0.0,
        reset_noise_scale=0.0,
        reference_ctrl_blend=0.0,
        domain_randomization=False,
        episode_timeout_jitter_steps=0,
        robot_collision_profile="feet",
        num_envs=1,
        course_lane_hold=True,
        course_lane_position_reward_coef=2.0,
        course_lane_velocity_reward_coef=0.5,
        course_lane_position_sigma=0.35,
        course_lane_velocity_sigma=0.25,
        course_lane_termination_margin=0.35,
    )
    env = LytRobotAmpEnv(cfg)
    try:
        observation = env.reset()
        assert observation.shape == (1, env.obs_dim)
        assert env.obs_dim == 133

        env.datas[0].qpos[1] = 1.30
        mujoco.mj_forward(env.model, env.datas[0])
        _, reward, done, info = env.step(
            np.zeros((1, env.action_dim), dtype=np.float32)
        )

        assert np.isfinite(reward).all()
        np.testing.assert_array_equal(done, [1.0])
        np.testing.assert_array_equal(info["course_lane_outside"], [1.0])
    finally:
        env.close()
