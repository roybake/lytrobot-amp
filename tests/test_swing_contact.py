"""Swing contacts remain diagnostics with the original reference reward."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
from lytrobot_amp.gait_constraints import swing_contact_violation
from lytrobot_amp.motion import reference_swing_interior
from sensor_training.gait_evaluation import (
    FIDELITY_LIMITS,
    gait_quality_failures,
    reference_fidelity_metrics,
    summarize_gait_evaluations,
)

ROOT = Path(__file__).resolve().parents[1]


def test_reference_swing_excludes_50ms_at_each_boundary_and_wraps():
    contact = np.ones((12, 2), dtype=np.float32)
    contact[2:10, 0] = 0.
    contact[[10, 11, 0, 1, 2, 3], 1] = 0.
    mask = reference_swing_interior(contact, 40., loop=True)
    np.testing.assert_array_equal(np.flatnonzero(mask[:, 0]), [4, 5, 6, 7])
    np.testing.assert_array_equal(np.flatnonzero(mask[:, 1]), [0, 1])
    non_looping = reference_swing_interior(contact, 40., loop=False)
    assert not non_looping[:, 1].any()
    assert not reference_swing_interior(np.ones((12, 2)), 40., loop=True).any()


def test_foot_toe_contacts_count_once_per_side_with_numpy_cuda_parity():
    # Toe-only, foot-only, both colliders, other-side support and double touch.
    contacts = np.asarray([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0],
                           [0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.float32)
    mask = np.asarray([[1, 0], [1, 0], [1, 0], [1, 0], [1, 1]], dtype=np.float32)
    expected = np.asarray([[1, 0], [1, 0], [1, 0], [0, 0], [1, 1]], dtype=np.float32)
    np.testing.assert_array_equal(swing_contact_violation(contacts, mask), expected)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actual = swing_contact_violation(torch.tensor(contacts, device=device),
                                    torch.tensor(mask, device=device))
    torch.testing.assert_close(actual, torch.tensor(expected, device=device))


def config():
    return LytRobotEnvCfg(
        model_xml=str(ROOT / 'robot/MJCFmodelfinal.xml'),
        motion_file=str(ROOT / 'robot/reference_data/walk1_mjcf_repaired.npz'),
        motion_loop=(True,), num_envs=1, task_reward_mode='stage1_reference',
        stand_start_prob=0., motion_reset_prob=0., reset_noise_scale=0.,
        randomize_episode_seed=False, action_noise_std=0., obs_noise_std=0.,
        domain_randomization=False, symmetry_reward_coef=0.,
    )


def test_cpu_swing_diagnostics_do_not_change_reward_and_startup_has_no_swing():
    env = LytRobotAmpEnv(config())
    try:
        data = env.datas[0]
        mujoco.mj_forward(env.model, data)
        targets = env._targets_at_current_phase()
        targets = replace(targets, foot_swing_interior=np.asarray([[1., 0.]], dtype=np.float32))
        foot_pos = env._foot_positions_local(data)
        contact = np.asarray([0., 1., 1., 0.], dtype=np.float32)
        action = np.zeros(env.action_dim, dtype=np.float32)

        def reward():
            return env._compute_reward_done(
                data, action, action, 0, env.cfg.target_velocity, targets,
                foot_pos, np.zeros_like(foot_pos), np.zeros_like(foot_pos),
                contact, env._side_contacts_from_geom(contact),
            )

        before, _, _, info = reward()
        assert info['right_swing_contact'] == 1. and info['left_swing_contact'] == 0.
        targets = replace(targets, foot_swing_interior=np.zeros((1, 2), dtype=np.float32))
        after, _, _, info = reward()
        assert after == pytest.approx(before)
        assert info['swing_contact'] == 0.
        targets = replace(targets, foot_swing_interior=np.asarray([[1., 0.]], dtype=np.float32))
        env.stand_start[:] = True
        env.episode_step[:] = 0
        assert not env._blend_start_targets(targets).foot_swing_interior.any()
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA Warp is required')
def test_warp_reward_and_reference_masks_match_cpu():
    cpu = LytRobotAmpEnv(config())
    gpu = LytRobotWarpAmpEnv(config(), device='cuda')
    try:
        phases = np.arange(cpu.motion.frame_count, dtype=np.float32) + .4
        cpu_mask = cpu.motion.targets_at_phase(phases).foot_swing_interior
        gpu_mask = gpu._targets_at_phase(torch.tensor(phases, device='cuda'))['foot_swing_interior']
        np.testing.assert_array_equal(cpu_mask, gpu_mask.cpu().numpy())
        targets = gpu._targets_at_current_phase()
        targets['foot_swing_interior'] = torch.tensor([[1., 0.]], device='cuda')
        contact = torch.tensor([[0., 1., 1., 0.]], device='cuda')
        foot_pos = gpu._foot_positions_local()
        action = torch.zeros((1, gpu.action_dim), device='cuda')

        def reward():
            return gpu._compute_reward_done(
                action, action, torch.tensor([gpu.cfg.target_velocity], device='cuda'),
                targets, foot_pos, torch.zeros_like(foot_pos), torch.zeros_like(foot_pos),
                contact, gpu._side_contacts_from_geom(contact),
            )

        before, _, _, info = reward()
        assert info['right_swing_contact'].item() == 1.
        targets['foot_swing_interior'].zero_()
        after, _, _, info = reward()
        torch.testing.assert_close(before, after)
        assert info['swing_contact'].item() == 0.
        targets['foot_swing_interior'][:] = torch.tensor([[1., 0.]], device='cuda')
        gpu.stand_start[:] = True
        gpu.episode_step[:] = 0
        assert not gpu._blend_start_targets(targets)['foot_swing_interior'].any()
        # Exercise the actual simulation step and its reward diagnostics.
        obs, reward_value, _, info = gpu.step(action)
        assert torch.isfinite(obs).all() and torch.isfinite(reward_value).all()
        assert 'swing_contact' in info
    finally:
        cpu.close()
        gpu.close()


def test_swing_contact_is_diagnostic_and_original_acceptance_is_preserved():
    clean = {name: limit * .5 for name, limit in FIDELITY_LIMITS.items()}
    clean.update({'eval/gait_accepted': 1., 'eval/swing_contact_fraction': 0.,
                  'eval/right_swing_contact_fraction': 0., 'eval/left_swing_contact_fraction': 0.})
    assert not gait_quality_failures(clean)
    touching = dict(clean, **{'eval/swing_contact_fraction': .01,
                             'eval/right_swing_contact_fraction': .02})
    assert not gait_quality_failures(touching)
    assert reference_fidelity_metrics(touching) == reference_fidelity_metrics(clean)
    summary = summarize_gait_evaluations([clean, touching, clean])
    assert summary['eval/swing_contact_fraction'] == .01
    assert summary['eval/right_swing_contact_fraction'] == .02
