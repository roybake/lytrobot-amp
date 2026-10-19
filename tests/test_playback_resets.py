"""Playback uses the training environment's reset and termination rules."""

from dataclasses import asdict, replace
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from scripts.play import make_env_cfg, parse_args
from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def playback_cfg(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "play.py", "--checkpoint", "model_final.onnx", "--no-viewer", "--seed", "42",
    ])
    args = parse_args()
    return make_env_cfg(args, {"env_config": {
        "model_xml": str(ROOT / "robot/MJCFmodelfinal.xml"),
        "motion_file": str(ROOT / "robot/reference_data/walk1_mjcf_repaired.npz"),
        "motion_loop": [False],
        "episode_steps": 1,
        "episode_timeout_jitter_steps": 0,
        "termination_mode": "fall",
        "motion_reset_prob": 0.0,
        "stand_start_prob": 0.0,
        "domain_randomization": False,
        "action_noise_std": 0.0,
        "min_forward_progress_speed": 100.0,
        "forward_progress_grace_steps": 0,
        "forward_progress_window_steps": 1,
    }})


def test_playback_continues_past_timeout_and_reference_end(playback_cfg):
    env = LytRobotAmpEnv(playback_cfg)
    try:
        env.motion_phase[0] = env.motion.frame_count - 0.25
        action = np.zeros((1, env.action_dim), dtype=np.float32)
        _, _, done, info = env.step(action)
        assert not done[0]
        assert env.motion_phase[0] < 1.0
        assert info["insufficient_progress"][0] == 1.0
        for _ in range(4):
            _, _, done, info = env.step(action)
            assert not done[0]
            assert info["time_outs"][0] == 0.0
        assert env.episode_step[0] == 5
    finally:
        env.close()


def test_playback_resets_at_training_posture_threshold(playback_cfg):
    env = LytRobotAmpEnv(playback_cfg)
    try:
        data = env.datas[0]
        data.qpos[3:7] = [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, data)
        _, _, done, info = env.step(np.zeros((1, env.action_dim), dtype=np.float32))
        assert done[0]
        assert info["termination"][0] == 1.0
        assert env.episode_step[0] == 0
    finally:
        env.close()


def test_playback_resets_when_body_reaches_ground(playback_cfg):
    env = LytRobotAmpEnv(playback_cfg)
    try:
        data = env.datas[0]
        data.qpos[2] = 0.05
        data.qpos[3:7] = [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, data)
        _, _, done, info = env.step(np.zeros((1, env.action_dim), dtype=np.float32))
        assert done[0]
        assert info["termination"][0] == 1.0
        assert info["time_outs"][0] == 0.0
        assert env.episode_step[0] == 0
        assert data.qpos[2] > 0.5
    finally:
        env.close()


def test_timed_evaluation_resets_at_episode_time_limit(playback_cfg):
    env = LytRobotAmpEnv(replace(
        playback_cfg, termination_mode="task", episode_steps=1,
        episode_timeout_jitter_steps=0, min_forward_progress_speed=0.0,
    ))
    try:
        _, _, done, info = env.step(np.zeros((1, env.action_dim), dtype=np.float32))
        assert done[0]
        assert info["time_outs"][0] == 1.0
        assert env.episode_step[0] == 0
    finally:
        env.close()


@pytest.mark.parametrize('stage,robot_count', [('stage1', 1), ('stage2', 1), ('multimotion', 3)])
def test_playback_preserves_saved_configuration(monkeypatch, stage, robot_count):
    saved = LytRobotEnvCfg(
        num_envs=32, seed=2026, termination_mode='fall', target_velocity=1.28,
        stand_start_prob=1., motion_reset_prob=0., reset_noise_scale=.03,
        randomize_episode_seed=True, domain_randomization=True,
        domain_randomization_scale=.25, action_noise_std=.04,
        stair_phase_reset_prob=.2, terrain_course_random_start_prob=.3,
        serial_terrain_course=False,
    )
    monkeypatch.setattr(sys, 'argv', ['play.py', '--checkpoint', 'model.onnx', '--no-viewer'])
    actual = make_env_cfg(parse_args(), {'env_config': asdict(saved)}, stage=stage)
    assert asdict(actual) == asdict(replace(saved, num_envs=robot_count))
