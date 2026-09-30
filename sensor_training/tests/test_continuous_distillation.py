"""Regressions for cycle-end resets, startup labels and off-policy roll-in."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from sensor_training.distillation_config import SensorDistillationPPOAMPConfig
from sensor_training.distillation_env import SensorDistillationWarpAmpEnv
from sensor_training.distillation_trainer import SensorDistillationPPOAMPTrainer
from sensor_training.gait_evaluation import summarize_gait_evaluations

ROOT = Path(__file__).resolve().parents[2]


def test_failed_start_prevents_a_multi_seed_gait_pass():
    good = {'eval/gait_accepted': 1., 'eval/gait_score': .8,
            'eval/gait_forward_velocity': .87, 'eval/gait_knee_rmse_rad': .1}
    failed = dict(good, **{'eval/gait_accepted': 0., 'eval/gait_score': .02,
                         'eval/gait_knee_rmse_rad': .4})
    result = summarize_gait_evaluations([good, failed, good])
    assert result['eval/gait_accepted'] == 0.
    assert result['eval/gait_score'] == .02
    assert result['eval/gait_knee_rmse_rad'] == .4


def test_walking_cycle_boundary_does_not_reset_the_robot():
    cfg = LytRobotEnvCfg(motion_file=str(ROOT/'robot/reference_data/walk1_omniretarget.npz'),
                         motion_loop=(True,), num_envs=1, stand_start_prob=0,
                         motion_reset_prob=0, reset_noise_scale=0,
                         action_noise_std=0, domain_randomization=False)
    env = LytRobotAmpEnv(cfg)
    try:
        env.reset()
        env.datas[0].qpos[:] = env.motion.reference_qpos[0]
        env.datas[0].qvel[:] = 0
        mujoco.mj_forward(env.model, env.datas[0])
        env.motion_phase[0] = env.motion.frame_count-.1
        _, _, done, info = env.step(np.zeros((1, env.action_dim), dtype=np.float32))
        assert not done[0]
        assert info['time_outs'][0] == 0
        assert env.motion_phase[0] < 1
        assert env.episode_step[0] == 1
    finally:
        env.close()


def test_teacher_label_applies_equal_startup_ramps_once():
    trainer = SensorDistillationPPOAMPTrainer.__new__(SensorDistillationPPOAMPTrainer)
    trainer.teacher = SimpleNamespace(metadata=SimpleNamespace(
        startup_action_ramp_steps=20, reference_ctrl_blend=1.,
        action_residual_scale=.5, distal_action_scale=1.))
    trainer._teacher_act = lambda observation: torch.tensor([[.4, -.4]])
    trainer.env = SimpleNamespace(
        cfg=SimpleNamespace(startup_action_ramp_steps=20),
        teacher_observe=lambda: torch.zeros(1, 6),
        teacher_action_ramp=lambda steps: torch.tensor([[.05]]),
        remap_teacher_action=lambda action, **kwargs: action)
    label = trainer._teacher_label()
    torch.testing.assert_close(label, torch.tensor([[.4, -.4]]))
    torch.testing.assert_close(label*.05, torch.tensor([[.02, -.02]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Warp regression requires CUDA')
def test_teacher_rollin_is_supervised_and_excluded_from_policy_likelihood_loss(tmp_path):
    # A real privileged checkpoint is deliberately reused only for this
    # implementation regression; production handoff has a separate gait gate.
    teacher = ROOT/'logs/teachers/stage1_gait/20260929_sensor_walk_v2/model_best.pt'
    if not teacher.is_file():
        pytest.skip('Local privileged fixture is unavailable')
    cfg = LytRobotEnvCfg(motion_file=str(ROOT/'robot/reference_data/walk1_omniretarget.npz'),
                         motion_loop=(True,), num_envs=4, stand_start_prob=1,
                         reset_noise_scale=0, action_noise_std=0, domain_randomization=False)
    env = SensorDistillationWarpAmpEnv(cfg, device='cuda')
    ppo = SensorDistillationPPOAMPConfig(
        teacher_checkpoint=str(teacher), amp_enabled=False, num_steps=4,
        update_epochs=1, num_minibatches=1, compile_rollout=False,
        compile_policy_update=False, teacher_action_probability=1.,
        teacher_action_probability_final=1., teacher_action_loss_coef=1.,
        teacher_action_loss_scale=.1)
    try:
        trainer = SensorDistillationPPOAMPTrainer(env, ppo, 'cuda', tmp_path)
        rollout = trainer.collect_rollout()
        assert rollout['policy_sample_mask'].sum() == 0
        torch.testing.assert_close(rollout['actions'], rollout['teacher_actions'])
        before = [p.detach().clone() for p in trainer.actor_critic.actor.parameters()]
        metrics = trainer.update_policy(rollout)
        assert metrics['policy_loss'] == 0
        assert metrics['approx_kl'] == 0
        assert metrics['student_on_policy_fraction'] == 0
        assert metrics['teacher_action_loss'] > 0
        assert any(not torch.equal(old, new) for old, new in zip(before, trainer.actor_critic.actor.parameters()))
    finally:
        env.close()
