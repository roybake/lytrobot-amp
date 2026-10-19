"""Stage1 tracks human walking speed with the historical reference reward."""

from dataclasses import fields
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv


ROOT = Path(__file__).resolve().parents[1]


def config():
    return LytRobotEnvCfg(
        model_xml=str(ROOT / "robot/MJCFmodelfinal.xml"),
        motion_file=str(ROOT / "robot/reference_data/walk1.npz"),
        motion_loop=(True,), num_envs=1, task_reward_mode="stage1_reference",
        target_velocity=1.28, forward_velocity_tracking_sigma=.25,
        forward_velocity_reward_coef=2., forward_velocity_deficit_penalty_coef=2.5,
        symmetry_reward_coef=0., stand_start_prob=0., motion_reset_prob=0.,
        reset_noise_scale=0., randomize_episode_seed=False,
        domain_randomization=False, termination_mode="fall", episode_steps=0,
    )


def cpu_reward(env, velocity, target_velocity=1.28):
    data = env.datas[0]
    data.qvel[0] = velocity
    targets = env._targets_at_current_phase()
    foot_pos = env._foot_positions_local(data)
    contacts = env._foot_geom_contacts(data)
    action = np.zeros(env.action_dim, dtype=np.float32)
    return env._compute_reward_done(
        data, action, action, 0, target_velocity, targets, foot_pos,
        np.zeros_like(foot_pos), np.zeros_like(foot_pos), contacts,
        env._side_contacts_from_geom(contacts),
    )


def expected_root_velocity(env, target_velocity):
    targets = env._targets_at_current_phase()
    linear_target = targets.root_lin_vel[0].copy()
    linear_target[0] = target_velocity
    linear_error = env.datas[0].qvel[:3] - linear_target
    angular_error = env.datas[0].qvel[3:6] - targets.root_ang_vel[0]
    return .65 * np.exp(-np.mean(linear_error**2) / .75**2) + .35 * np.exp(
        -np.mean(angular_error**2) / 1.5**2
    )


def test_root_velocity_uses_original_linear_angular_blend_and_limb_feedback():
    env = LytRobotAmpEnv(config())
    try:
        scores = []
        for speed in (0., .65, .9, 1.28, 1.66, 1.91):
            _, _, _, info = cpu_reward(env, speed)
            expected = expected_root_velocity(env, 1.28)
            assert info["root_vel"] == pytest.approx(expected)
            # Verify the velocity term changes the actual reward.
            env.cfg.reference_root_velocity_reward_coef += 1.
            raised, _, _, _ = cpu_reward(env, speed)
            env.cfg.reference_root_velocity_reward_coef -= 1.
            original, _, _, _ = cpu_reward(env, speed)
            assert raised - original == pytest.approx(expected)
            # Pose feedback stays dense even when the robot is too slow.
            env.cfg.reference_joint_position_reward_coef += 1.
            raised, _, _, changed = cpu_reward(env, speed)
            env.cfg.reference_joint_position_reward_coef -= 1.
            original, _, _, _ = cpu_reward(env, speed)
            assert raised - original == pytest.approx(changed["pose_track"])
            scores.append(expected)
        assert scores[0] < scores[1] < scores[2] < scores[3]
        assert scores[2] == pytest.approx(scores[4])
        assert scores[1] == pytest.approx(scores[5])
        # A standing startup command still receives reference feedback.
        _, _, _, standing = cpu_reward(env, 0., target_velocity=0.)
        assert standing["root_vel"] == pytest.approx(scores[3])
    finally:
        env.close()


def test_velocity_reward_favors_the_command_during_startup_and_walking():
    env = LytRobotAmpEnv(config())
    try:
        for command in (0., .64, 1.28):
            speeds = (command - .3, command, command + .3)
            rewards = [cpu_reward(env, speed, target_velocity=command)[0] for speed in speeds]
            assert rewards[1] > rewards[0]
            assert rewards[1] > rewards[2]
    finally:
        env.close()


def test_fall_penalty_is_deducted_from_original_reference_reward():
    env = LytRobotAmpEnv(config())
    try:
        data = env.datas[0]
        data.qpos[2] = .1
        mujoco.mj_forward(env.model, data)
        env.cfg.fall_penalty = 0.
        before, done, _, info = cpu_reward(env, 1.28)
        assert done and info["fallen"] == 1.
        assert info["root_vel"] == pytest.approx(expected_root_velocity(env, 1.28))
        env.cfg.fall_penalty = 2.
        after, _, _, _ = cpu_reward(env, 1.28)
        assert before - after == pytest.approx(2.)
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Warp is required")
def test_cpu_and_warp_use_the_same_original_reference_reward():
    cpu = LytRobotAmpEnv(config())
    gpu = LytRobotWarpAmpEnv(config(), device="cuda")
    try:
        targets = cpu._targets_at_current_phase()
        gpu_targets = {field.name: torch.as_tensor(
            getattr(targets, field.name), device="cuda") for field in fields(targets)}
        gpu.qpos[:] = torch.as_tensor(cpu.datas[0].qpos[None], device="cuda")
        gpu.qvel[:] = torch.as_tensor(cpu.datas[0].qvel[None], device="cuda")
        gpu.ctrl[:] = torch.as_tensor(cpu.datas[0].ctrl[None], device="cuda")
        gpu._warp_forward()
        contacts_np = cpu._foot_geom_contacts(cpu.datas[0])[None]
        contacts = torch.as_tensor(contacts_np, device="cuda")
        foot_pos = gpu._foot_positions_local()
        action = torch.zeros((1, gpu.action_dim), device="cuda")
        for speed in (.65, .9, 1.28, 1.91):
            expected, _, _, cpu_info = cpu_reward(cpu, speed)
            gpu.qvel[:, 0] = speed
            actual, _, _, gpu_info = gpu._compute_reward_done(
                action, action, torch.tensor([1.28], device="cuda"),
                gpu_targets, foot_pos, torch.zeros_like(foot_pos),
                torch.zeros_like(foot_pos), contacts,
                gpu._side_contacts_from_geom(contacts),
            )
            assert actual.item() == pytest.approx(expected, abs=2e-5)
            for name in ("forward_velocity", "root_vel", "stable_forward_velocity_track"):
                assert gpu_info[name].item() == pytest.approx(cpu_info[name], abs=2e-5)
        # Exercise physics and the reward diagnostics through the training API.
        observation, reward, _, info = gpu.step(action)
        assert torch.isfinite(observation).all() and torch.isfinite(reward).all()
        assert ((info["root_vel"] >= 0.) & (info["root_vel"] <= 1.)).all()
    finally:
        cpu.close()
        gpu.close()
