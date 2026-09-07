from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
from lytrobot_amp.terrain import MoreTerrainCfg, load_model, terrain_height_from_model
from scripts.audit_independent_references import launcher_config
from scripts.play import make_env_cfg


@pytest.mark.parametrize("contact,speed,expected", [
    ([1, 0], 2.0, 1.0),  # left swing with right support
    ([1, 0], 0.0, 0.0),  # standing
    ([1, 1], 2.0, 0.0),  # shuffling both feet on the surface
    ([0, 0], 2.0, 0.0),  # jumping
    ([1, 0], -2.0, 0.0),  # backward swing
    ([0, 1], 2.0, 0.0),  # wrong side of reference gait
])
def test_supported_forward_swing_cpu_warp_agree(contact, speed, expected):
    cpu = object.__new__(LytRobotAmpEnv)
    gpu = object.__new__(LytRobotWarpAmpEnv)
    cpu.cfg = gpu.cfg = LytRobotEnvCfg(target_velocity=1.0)
    velocity = np.zeros((4, 3), dtype=np.float32)
    velocity[2:, 0] = speed
    contact = np.asarray(contact, dtype=np.float32)
    reference = np.asarray([1, 0], dtype=np.float32)
    assert cpu._forward_step_reward(velocity, contact, reference) == expected
    assert gpu._forward_step_reward(torch.tensor(velocity[None]), torch.tensor(contact[None]),
                                    torch.tensor(reference[None])).item() == expected


@pytest.mark.parametrize("key", ["flat", "rough", "slope_up", "slope_down"])
def test_playback_preserves_independent_training_contract(key):
    cfg = launcher_config(key)
    args = SimpleNamespace(model_xml=None, motion_file=cfg.motion_file, num_robots=1,
                           seed=42, target_velocity=None, serial_terrain_course=None,
                           checkpoint="model.pt", stage="stage1")
    played = make_env_cfg(args, {"training_task": cfg.training_task, "env_config": asdict(cfg)}, "stage1")
    for field in ("terrain_types", "standalone_field_x_min", "standalone_field_y_min",
                  "standalone_field_tile_width", "motion_files", "motion_loop", "episode_steps",
                  "rough_terrain_height",
                  "forward_step_reward_coef", "side_foot_pos_coef", "step_gap_coef",
                  "swing_advance_coef", "reference_ctrl_blend", "skill_conditioning"):
        assert getattr(played, field) == getattr(cfg, field)
    assert played.stand_start_prob == 1.0
    args.episode_steps = 500
    assert make_env_cfg(args, {"env_config": asdict(cfg)}, "stage1").episode_steps == 500


def test_rough_field_preserves_source_heightmap_scale_and_origin():
    import mujoco

    cfg = launcher_config("rough")
    assert cfg.rough_terrain_height == 0.05
    env = LytRobotAmpEnv(cfg)
    original = load_model(cfg.model_xml, None, MoreTerrainCfg(enabled=True, terrain_types=("rough",)))
    data = mujoco.MjData(original)
    mujoco.mj_forward(original, data)
    try:
        mujoco.mj_forward(env.model, env.datas[0])
        x = np.linspace(-0.4, 4.4, 60)
        y = np.linspace(5.1, 6.9, 60)
        expected = terrain_height_from_model(original, data, x, y)
        actual = terrain_height_from_model(env.model, env.datas[0], x, y)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        assert actual.min() >= -1e-6
        assert actual.max() <= 0.05 + 1e-6
        assert env.model.hfield_size[0, 2] == 0.05
    finally:
        env.close()


@pytest.mark.parametrize("height", ["0.051", "0.1", "nan", "-0.05"])
def test_independent_rough_rejects_invalid_relief(height):
    import sys
    from unittest.mock import patch
    from scripts.independent_terrain_task import default_train_args
    from scripts.train import parse_args

    with patch.object(sys, "argv", ["train.py", *default_train_args("rough"),
                                    "--rough-terrain-height", height]):
        with pytest.raises(SystemExit):
            parse_args()


@pytest.mark.parametrize("key", ["flat", "rough", "slope_up", "slope_down"])
def test_sensor_training_config_short_cpu_smoke(key):
    from sensor_training.envs import SensorHistoryCpuAmpEnv

    env = SensorHistoryCpuAmpEnv(launcher_config(key))
    try:
        env.reset()
        for _ in range(3):
            obs, reward, done, info = env.step(np.zeros((1, env.action_dim), dtype=np.float32))
            assert np.isfinite(obs).all()
            assert np.isfinite(reward).all()
    finally:
        env.close()
