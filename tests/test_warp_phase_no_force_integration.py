from pathlib import Path

import pytest
import torch

from foot_sensor_training.envs import PhaseNoForceWarpAmpEnv
from lytrobot_amp import env_warp as warp_env_module
from lytrobot_amp.env import LytRobotEnvCfg


ROOT = Path(__file__).resolve().parents[1]


def test_real_cuda_warp_phase_no_force_integration(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable; GPU Warp integration must not fall back to CPU')
    cfg = LytRobotEnvCfg(
        model_xml=str(ROOT/'robot/MJCFmodelfinal.xml'),
        motion_file=str(ROOT/'robot/reference_data/walk1_mjcf_repaired.npz'),
        num_envs=4,
        seed=123,
        reference_ctrl_blend=0.0,
        domain_randomization=False,
        sensor_stability_reward=True,
        stand_start_prob=1.0,
        motion_reset_prob=0.0,
        stand_start_transition_min_steps=120,
        stand_start_transition_max_steps=120,
    )
    env = PhaseNoForceWarpAmpEnv(cfg, device='cuda')
    try:
        obs = env.reset()
        assert obs.shape == (4, 58) and obs.is_cuda and torch.isfinite(obs).all()

        stale_reset_called = False

        def stale_reset_data(*args, **kwargs):
            nonlocal stale_reset_called
            stale_reset_called = True
            raise RuntimeError(warp_env_module._WARP_DYNAMIC_SOURCE_ERROR)

        monkeypatch.setattr(warp_env_module.mjwarp, 'reset_data', stale_reset_data)
        obs = env.reset()
        assert stale_reset_called
        assert obs.shape == (4, 58) and torch.isfinite(obs).all()

        for step in range(32):
            obs, reward, _, _ = env.step(torch.zeros(4, 14, device='cuda'))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            if step == 0:
                # A second forward pass should not move geometry: observations
                # and rewards must already use transforms at the new state.
                foot_pos = env.geom_xpos[:, env.contact_geom_ids].clone()
                with warp_env_module.wp.ScopedDevice(env.wp_device):
                    env._warp_forward()
                torch.testing.assert_close(
                    env.geom_xpos[:, env.contact_geom_ids], foot_pos,
                    rtol=1e-5, atol=1e-5,
                )
            torch.testing.assert_close(
                obs[:, 56].square() + obs[:, 57].square(),
                torch.ones(4, device='cuda'), rtol=1e-4, atol=1e-4,
            )
    finally:
        env.close()
