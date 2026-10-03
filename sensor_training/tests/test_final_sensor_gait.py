"""Deployment-channel, shared BPTT and Stage1→Stage2 regressions."""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import copy
import json
import tempfile

import numpy as np
import pytest
import torch

from lytrobot_amp.env import LytRobotEnvCfg
from sensor_training.config import SensorObservationConfig
from sensor_training.measurement import CausalMeasurement, MeasurementSpec
from sensor_training.envs import SensorHistoryCpuAmpEnv
from sensor_training.final_profile import FINAL_SENSOR_OPTIONS, apply_final_profile, final_checkpoint_errors
from sensor_training.networks import SensorActorCritic
from sensor_training.distillation_config import SensorDistillationPPOAMPConfig
from sensor_training.distillation_trainer import SensorDistillationPPOAMPTrainer
from sensor_training.trainer import SensorHistoryPPOAMPTrainer
from sensor_training.ppo_config import SensorPPOAMPConfig
from sensor_training.train_serial_distillation import _phase_command, _phase_paths
from sensor_training.export_sensor_policy import compute_profile, export_checkpoint, load_policy
from sensor_training.gait_evaluation import evaluate_gait_checkpoint, gait_quality_failures

ROOT = Path(__file__).resolve().parents[2]
TEACHER = ROOT/'logs/teachers/stage1_gait/20260930_sensor_walk_v3/model_best.pt'


def observation_config(**overrides):
    names = {field.name for field in fields(SensorObservationConfig)}
    return SensorObservationConfig(**{**{key: value for key, value in FINAL_SENSOR_OPTIONS.items() if key in names}, **overrides})


def base_config(**overrides):
    values = dict(model_xml=str(ROOT/'robot/MJCFmodelposition.xml'),
                  motion_file=str(ROOT/'robot/reference_data/walk1_mjcf_repaired.npz'),
                  motion_loop=(True,), num_envs=2, reset_noise_scale=0.,
                  action_noise_std=0., obs_noise_std=0., domain_randomization=False,
                  sensor_root_imu_only=False, episode_steps=12, episode_timeout_jitter_steps=0)
    return LytRobotEnvCfg(**{**values, **overrides})


def policy_config(config_class=SensorPPOAMPConfig, **overrides):
    names = {field.name for field in fields(config_class)}
    values = {key: value for key, value in FINAL_SENSOR_OPTIONS.items() if key in names}
    values.update(num_steps=96, num_minibatches=1, update_epochs=1,
                  compile_rollout=False, compile_policy_update=False)
    values.update(overrides)
    return config_class(**values)


def test_measurement_cpu_torch_noise_bias_scale_quantization_clipping_and_delay():
    spec = MeasurementSpec(noise_std=.01, constant_bias=.001, max_delay_ticks=2, quantization=.005)
    initial = np.full((2, 14), .3, dtype=np.float32)
    outputs = []
    for tensor in (False, True):
        value = torch.from_numpy(initial.copy()) if tensor else initial.copy()
        zero = torch.zeros_like(value) if tensor else np.zeros_like(value)
        model = CausalMeasurement(value, spec, center=zero, lower=zero+.1, upper=zero+.4)
        model.reset(value, [0, 1], scale=1.)
        model.gain[:] = 1.1; model.bias[:] = .001; model.delay[:] = 2
        before = model.output.clone() if tensor else model.output.copy()
        model.history[:] = before[None]
        results = []
        for t in range(4):
            truth = value + .001*t
            result = model.advance(truth, scale=1., noise_sample=zero+1)
            results.append(np.asarray(result).copy())
        # First two samples are cached reset output, then delayed noisy/quantized data.
        np.testing.assert_array_equal(results[0], np.asarray(before))
        np.testing.assert_array_equal(results[1], np.asarray(before))
        np.testing.assert_allclose(results[2], .34, atol=1e-7)
        outputs.append(results[2:])
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-7)


def test_tendon_channels_share_noisy_cached_measurement_and_never_re_read_truth():
    env = SensorHistoryCpuAmpEnv(base_config(domain_randomization=True, domain_randomization_scale=1.),
         observation_config(tendon_measurement_noise_std=.0005, tendon_velocity_filter_alpha=1.))
    try:
        truth = np.broadcast_to(env.ctrl_mid, (2, 14)).copy()
        with mock.patch.object(env, '_true_tendon_lengths', return_value=truth):
            env.reset()
            previous = env.measured_tendon_lengths().copy()
            env._advance_measurements()
            measured = env.measured_tendon_lengths().copy()
            frame = env._normalized_sensor_frame()
            assert env.sensor_frame_dim == 144 and env.obs_dim == 864
            reconstructed = frame[:, 42:56]*env.ctrl_half_range+env.ctrl_mid
            np.testing.assert_allclose(reconstructed, measured, atol=1e-7)
            np.testing.assert_allclose(frame[:, 56:70], (measured-previous)/env.dt, atol=1e-6)
            reference = env._gait_command_frame()[:, 4:18]*env.ctrl_half_range+env.ctrl_mid
            np.testing.assert_allclose(frame[:, 70:84], (measured-reference)/.05, atol=1e-6)
            assert not np.allclose(measured, truth)
        # Neither raw truth nor privileged labels can affect a re-read of the same tick.
        with mock.patch.object(env, '_true_tendon_lengths', side_effect=AssertionError('truth leakage')):
            env.state_supervision_observe(True)
            np.testing.assert_array_equal(frame, env._normalized_sensor_frame())
        env.reset([0])
        np.testing.assert_array_equal(env.observe().reshape(2, 6, 144)[0, :, 56:70], 0.)
    finally:
        env.close()


def test_metadata_matches_actual_state_layout_33_and_45():
    env = SensorHistoryCpuAmpEnv(base_config(), observation_config())
    try:
        for foot, dim in ((False, 33), (True, 45)):
            env.configure_state_supervision(foot)
            assert env.state_supervision_metadata['dim'] == dim
            assert sum(env.state_supervision_metadata['layout'].values()) == dim
            assert env.state_supervision_observe().shape[-1] == dim
            assert ('foot_local_position' in env.state_supervision_metadata['layout']) == foot
    finally:
        env.close()


def test_shared_sequence_gradients_equal_independent_loss_forwards():
    torch.manual_seed(12)
    first = SensorActorCritic(30, 14, recurrent_hidden_dim=16, privileged_context_dim=25,
        state_estimator_enabled=True, state_estimator_include_foot_position=True, actor_use_estimated_state=True)
    second = copy.deepcopy(first)
    obs, actions, context = torch.randn(6, 2, 30), torch.randn(6, 2, 14), torch.randn(6, 2, 25)
    hidden, starts = torch.randn(2, 16), torch.zeros(6, 2)
    starts[3, 0] = 1.
    shared = first.evaluate_sequence_shared(obs, hidden, starts, actions, context)
    loss1 = shared[0].mean()+shared[8].square().mean()+shared[9].square().mean()
    loss1.backward()
    legacy = second.evaluate_sequence(obs, hidden, starts, actions, context)
    means = second.action_mean_sequence(obs, hidden, starts)
    states, _ = second.auxiliary_sequence(obs, hidden, starts)
    flat = torch.cat(tuple(states.values()), dim=-1)
    loss2 = legacy[0].mean()+means.square().mean()+flat.square().mean()
    loss2.backward()
    torch.testing.assert_close(loss1, loss2)
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5, msg=name)
    assert first.joint_position_head[0].weight.grad.abs().sum() > 0


def test_bptt_96_one_gru_pass_and_stage2_resume(tmp_path):
    env = SensorHistoryCpuAmpEnv(base_config(), observation_config())
    try:
        trainer = SensorHistoryPPOAMPTrainer(env, policy_config(), 'cpu', tmp_path)
        rollout = trainer.collect_rollout()
        counts = []
        hook = trainer.actor_critic.recurrent.register_forward_hook(lambda *args: counts.append(1))
        with mock.patch.object(trainer.actor_critic, 'auxiliary_sequence', side_effect=AssertionError('repeated GRU')):
            metrics = trainer.update_policy(rollout)
        hook.remove()
        assert len(counts) == 96
        assert np.isfinite(metrics['state/foot_position_mae'])
        payload = trainer._checkpoint_payload(0)
        payload['distillation'] = {'enabled': True, 'teacher_behavior_probability': 0.,
                                  'teacher': {'checkpoint': str(TEACHER)}}
        path = tmp_path/'stage1.pt'; torch.save(payload, path)
        stage2_env = SensorHistoryCpuAmpEnv(base_config(domain_randomization=True,
            actuator_strength_scale_min=.9, actuator_strength_scale_max=1.1), observation_config())
        try:
            stage2 = SensorHistoryPPOAMPTrainer(stage2_env, policy_config(), 'cpu', tmp_path/'stage2')
            stage2.load(path, load_optimizers=False)
            assert torch.count_nonzero(stage2.recurrent_state) == 0
            torch.testing.assert_close(stage2.actor_critic.recurrent.weight_ih, trainer.actor_critic.recurrent.weight_ih)
        finally:
            stage2_env.close()
        assert not final_checkpoint_errors(payload)
        # Export doesn't need physics or a Teacher process/checkpoint read.
        export_checkpoint(path, tmp_path/'deployment.onnx')
        profile = compute_profile(load_policy(payload))
        assert profile['deployment_parameters'] > 1_000_000
        assert profile['macs_per_step'] < 2_000_000
        quality = evaluate_gait_checkpoint(tmp_path/'deployment.onnx', seconds=.1)
        assert quality['eval/teacher_execution_probability'] == 0.
        assert 'eval/joint_position_rmse_rad' in quality
        assert 'eval/stride_length_m' in quality
        assert 'eval/torso_orientation_rmse_rad' in quality
    finally:
        env.close()


def test_final_recipe_stage1_and_stage2_commands_have_fixed_architecture():
    args = SimpleNamespace(final_sensor_gait=True, student_num_steps=None, student_num_envs=2,
        teacher_stage1_checkpoint=TEACHER, student_stage1_checkpoint=Path('stage1.pt'),
        teacher_stage1_iterations=None, teacher_stage2_iterations=None,
        num_envs=2, decimation=2, stage1_iterations=30000, stage2_iterations=30000,
        device='cuda', sim_backend='warp', no_wandb=True, wandb_log_interval=10)
    apply_final_profile(args)
    assert args.student_num_steps == 96
    for name in ('student_stage1', 'student_stage2'):
        command = _phase_command(name, args, _phase_paths('09-30-10:47:29'), resume_checkpoint=None)
        for flag in ('--include-tendon-velocity', '--include-tendon-tracking-error',
                     '--state-estimator-enabled', '--state-estimator-include-foot-position',
                     '--actor-use-estimated-state', '--no-future-prediction-enabled', '--no-sensor-root-imu-only'):
            assert flag in command
        assert command[command.index('--recurrent-hidden-dim')+1] == '256'
        assert command[command.index('--num-steps')+1] == '96'
        assert command[command.index('--teacher-checkpoint')+1] == str(TEACHER)
        assert '--no-stop-on-gait-acceptance' in command
        if name == 'student_stage2':
            assert '--student-checkpoint' in command and '--domain-randomization' in command
            assert command[command.index('--dr-curriculum-mode')+1] == 'linear'
            assert '--warp-global-domain-randomization' in command
        else:
            assert '--no-domain-randomization' in command


def test_teacher_quality_rejects_good_speed_but_bad_reference():
    metrics = {'eval/gait_accepted': 1., 'eval/gait_forward_velocity': .87}
    assert any('joint_position_rmse' in failure for failure in gait_quality_failures(metrics))


@pytest.mark.skipif(not torch.cuda.is_available() or not TEACHER.exists(), reason='CUDA/local Teacher fixture required')
@pytest.mark.parametrize('backend', ['warp', 'mjlab'])
def test_backend_measured_signals_teacher_zero_single_forward_and_dr_resume(tmp_path, backend):
    from sensor_training.distillation_env import SensorDistillationWarpAmpEnv, SensorDistillationMjlabAmpEnv
    cls = SensorDistillationWarpAmpEnv if backend == 'warp' else SensorDistillationMjlabAmpEnv
    cfg = base_config(num_envs=2)
    env = cls(cfg, device='cuda', observation_cfg=observation_config())
    try:
        ppo = policy_config(SensorDistillationPPOAMPConfig, teacher_checkpoint=str(TEACHER),
                            teacher_action_probability=0., teacher_action_probability_final=0.,
                            teacher_action_loss_coef=1., amp_enabled=False)
        trainer = SensorDistillationPPOAMPTrainer(env, ppo, 'cuda', tmp_path)
        rollout = trainer.collect_rollout()
        assert rollout['policy_sample_mask'].sum() == 96*2
        counts = []
        hook = trainer.actor_critic.recurrent.register_forward_hook(lambda *args: counts.append(1))
        with (mock.patch.object(trainer.actor_critic, 'auxiliary_sequence', side_effect=AssertionError('repeated GRU')),
              mock.patch.object(trainer.actor_critic, 'action_mean_sequence', side_effect=AssertionError('repeated GRU'))):
            metrics = trainer.update_policy(rollout)
        hook.remove()
        assert len(counts) == 96
        assert metrics['teacher_action_loss'] > 0.
        assert np.isfinite(metrics['state/foot_position_mae'])
        measured = env.measured_tendon_lengths().clone()
        frame = env._normalized_sensor_frame()
        torch.testing.assert_close(frame[:, 42:56]*env.ctrl_half_range+env.ctrl_mid, measured)
        reference = env._gait_command_frame()[:, 4:18]*env.ctrl_half_range+env.ctrl_mid
        torch.testing.assert_close(frame[:, 70:84], ((measured-reference)/.05).clamp(-20, 20))
        checkpoint = tmp_path/'stage1.pt'; trainer.save(checkpoint, 0)
    finally:
        env.close()
    stage2 = cls(base_config(num_envs=2, domain_randomization=True, warp_global_domain_randomization=True,
            actuator_strength_scale_min=.9, actuator_strength_scale_max=1.1),
            device='cuda', observation_cfg=observation_config())
    try:
        trainer2 = SensorDistillationPPOAMPTrainer(stage2, ppo, 'cuda', tmp_path/'stage2')
        trainer2.load(checkpoint, load_optimizers=False)
        assert stage2.sensor_frame_dim == 144
        stage2.set_domain_randomization_scale(.5)
        trainer2.reset_training_env()
        metrics = trainer2.train_iteration()
        assert np.isfinite(metrics['state/joint_position_mae'])
    finally:
        stage2.close()


def test_compiled_shared_chunk_outputs_and_gradients(tmp_path):
    env = SensorHistoryCpuAmpEnv(base_config(), observation_config())
    try:
        trainer = SensorHistoryPPOAMPTrainer(env, policy_config(), 'cpu', tmp_path)
        policy = trainer.actor_critic
        obs = torch.randn(17, 2, 864, requires_grad=True)
        starts = torch.zeros(17, 2); starts[9, 0] = 1.
        actions = torch.randn(17, 2, 14); context = torch.randn(17, 2, 25)
        state = policy.initial_state(2)
        expected = policy.evaluate_sequence_shared(obs, state, starts, actions, context)
        trainer.cfg.compile_policy_chunk_steps = 8
        # Capture the full graph while keeping this regression fast and independent
        # of CUDA compiler caches; integration tests separately exercise GPU updates.
        trainer._compiled_evaluate_sequence = torch.compile(policy.evaluate_sequence_shared,
                                                            backend='eager', fullgraph=True)
        actual = trainer._evaluate_policy_sequence(obs, state, starts, actions, context)
        assert len(actual) == 11
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target)
        actual[8][-1].sum().backward()
        assert obs.grad[0, 1].abs().sum() > 0  # gradient crosses compiled chunk boundaries
        assert torch.count_nonzero(obs.grad[:9, 0]) == 0  # reset blocks old episode gradients
    finally:
        env.close()


def test_final_pipeline_blocks_bad_teacher_before_student_execution(tmp_path, monkeypatch):
    from sensor_training import train_serial_distillation as serial
    teacher = tmp_path/'teacher.pt'
    torch.save({'env_config': dict(motion_file=str(ROOT/'robot/reference_data/walk1_mjcf_repaired.npz'),
                                  decimation=2, reference_ctrl_blend=1.)}, teacher)
    with mock.patch('sys.argv', ['final', '--final-sensor-gait', '--phase', 'stage1',
                                '--teacher-stage1-checkpoint', str(teacher), '--no-wandb']):
        args = serial.parse_args()
    monkeypatch.setattr(serial, 'REPO_ROOT', tmp_path)
    import sensor_training.gait_evaluation as evaluation
    monkeypatch.setattr(evaluation, 'evaluate_gait_suite', lambda *a, **k: {'eval/gait_accepted': 1.,
                                                                        'eval/joint_position_rmse_rad': .5})
    with mock.patch.object(serial.subprocess, 'run') as launch:
        with pytest.raises(RuntimeError, match='Teacher reference fidelity is insufficient'):
            serial.run_final_pipeline(args)
        launch.assert_not_called()
    manifest = json.loads((tmp_path/'logs/serial_distillation'/args.run_name/'pipeline.json').read_text())
    assert manifest['teacher_quality']['eval/joint_position_rmse_rad'] == .5
