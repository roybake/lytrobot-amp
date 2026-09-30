from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch

from lytrobot_amp.env import LytRobotEnvCfg
from sensor_training.config import SensorObservationConfig
from sensor_training.envs import SensorHistoryCpuAmpEnv
from sensor_training.experiment_options import observation_config_from_metadata, estimator_kwargs
from sensor_training.networks import SensorActorCritic, RecurrentSensorActorExport
from sensor_training.ppo_config import SensorPPOAMPConfig
from sensor_training.state_estimation import state_layout, state_estimation_loss, future_prediction_loss
from sensor_training.tendon_features import CausalTendonFeatures
from sensor_training.trainer import SensorHistoryPPOAMPTrainer, _sensor_metadata_compatible
from sensor_training.train_ablation import training_command

ROOT = Path(__file__).resolve().parents[2]


def make_env(features=False, **overrides):
    cfg = LytRobotEnvCfg(model_xml=str(ROOT/'robot/MJCFmodelposition.xml'),
        motion_file=str(ROOT/'robot/reference_data/walk1_omniretarget.npz'),
        motion_loop=(True,), num_envs=2, sensor_root_imu_only=True,
        episode_steps=12, episode_timeout_jitter_steps=0, reset_noise_scale=0.,
        domain_randomization=False, **overrides)
    return SensorHistoryCpuAmpEnv(cfg, SensorObservationConfig(
        include_tendon_velocity=features, include_tendon_tracking_error=features))


class StateEstimationTests(unittest.TestCase):
    def test_feature_dimensions_and_training_label_layout(self):
        old, new = make_env(), make_env(True)
        try:
            self.assertEqual(old.sensor_frame_dim, 80)
            self.assertEqual(new.sensor_frame_dim, 108)
            self.assertEqual(old.obs_dim, 480)
            self.assertEqual(new.obs_dim, 648)
            self.assertEqual(new.state_supervision_observe().shape, (2, 33))
            self.assertEqual(new.state_supervision_observe(True).shape, (2, 45))
            self.assertNotIn('terrain', str(state_layout(True)))
            np.testing.assert_array_equal(new.state_supervision_observe()[:, 31:33],
                np.stack([new.base_env._foot_contacts(data) for data in new.base_env.datas]))
            data = new.base_env.datas[0]
            np.testing.assert_allclose(new.state_supervision_observe()[0, :14],
                data.qpos[new.base_motion.robot_qpos_adrs])
        finally:
            old.close(); new.close()

    def test_tendon_velocity_is_causal_and_reset_safe_numpy_and_torch(self):
        cfg = SensorObservationConfig(include_tendon_velocity=True, tendon_velocity_filter_alpha=.5)
        for use_torch in (False, True):
            lengths = torch.zeros(2, 14) if use_torch else np.zeros((2, 14), dtype=np.float32)
            features = CausalTendonFeatures(lengths, cfg)
            features.update(lengths + .01, .01)
            np.testing.assert_allclose(np.asarray(features.velocity), .5)
            saved = features.frame(lengths+.01, lengths).clone() if use_torch else features.frame(lengths+.01, lengths).copy()
            # Re-reading a frame is pure: no extra differentiation/filter update.
            np.testing.assert_array_equal(np.asarray(saved), np.asarray(features.frame(lengths+.01, lengths)))
            features.update(lengths+.02, .01)
            np.testing.assert_allclose(np.asarray(features.velocity), .75)
            features.reset(lengths, [0])
            self.assertEqual(float(features.velocity[0].sum()), 0)
            features.update(lengths, .01)
            self.assertEqual(float(features.velocity[0].sum()), 0)

    def test_tracking_error_uses_onboard_current_reference_and_is_read_only(self):
        env = make_env(True)
        try:
            env.reset()
            command = env._gait_command_frame()
            reference = command[:, 4:18] * env.ctrl_half_range + env.ctrl_mid
            expected = np.clip((env._measured_tendon_lengths()-reference)/.05, -20, 20)
            frame = env.observe().reshape(2, 6, 108)[:, -1]
            np.testing.assert_allclose(frame[:, 34:48], expected, atol=1e-6)
            obs = env.observe().copy()
            env.state_supervision_observe(True)
            np.testing.assert_array_equal(obs, env.observe())
            env.step(np.zeros((2, 14), dtype=np.float32))
            before = env._tendon_features.velocity.copy()
            env._normalized_sensor_frame(); env._normalized_sensor_frame()
            np.testing.assert_array_equal(before, env._tendon_features.velocity)
            env.reset([0])
            np.testing.assert_array_equal(env.observe().reshape(2, 6, 108)[0, :, 20:34], 0.)
        finally:
            env.close()

    def test_contact_bce_and_physical_scale_losses(self):
        cfg = SensorPPOAMPConfig(state_estimator_enabled=True)
        truth = torch.zeros(4, 33)
        truth[:, 14:28] = 10.
        truth[:, 31:] = torch.tensor([0., 1.])
        predicted = {k: torch.zeros(4, v, requires_grad=True) for k, v in state_layout().items()}
        loss, metrics = state_estimation_loss(predicted, truth, cfg)
        self.assertAlmostEqual(float(metrics['state/contact_bce']), np.log(2), places=6)
        self.assertAlmostEqual(float(metrics['state/joint_velocity_loss']), .5, places=6)
        self.assertAlmostEqual(float(metrics['state/joint_velocity_mae']), 10., places=6)
        loss.backward()
        self.assertTrue((predicted['foot_contact'].grad[:, 0] > 0).all())
        self.assertTrue((predicted['foot_contact'].grad[:, 1] < 0).all())

    def test_estimator_shapes_actor_privileged_independence_and_reset(self):
        for hidden_dim in (128, 256, 384):
            for foot in (False, True):
                policy = SensorActorCritic(30, 14, recurrent_hidden_dim=hidden_dim,
                    privileged_context_dim=25, state_estimator_enabled=True,
                    state_estimator_include_foot_position=foot, actor_use_estimated_state=True)
                obs, hidden = torch.randn(2, 30), torch.randn(2, hidden_dim)
                first = policy.recurrent_action_and_value(obs, hidden,
                    privileged_context=torch.randn(2, 25))
                second = policy.recurrent_action_and_value(obs, hidden,
                    privileged_context=torch.randn(2, 25)*100)
                # Means/hidden depend on sensor input alone; critic may use true context.
                torch.testing.assert_close(first[5], second[5])
                torch.testing.assert_close(policy._actor_outputs(first[5])[0], policy._actor_outputs(second[5])[0])
                for key, value in policy.predict_state(first[5]).items():
                    self.assertEqual(value.shape, (2, state_layout(foot)[key]))
                a, h = policy.act_inference_recurrent(obs, hidden, torch.ones(2))
                b, j = policy.act_inference_recurrent(obs, torch.zeros_like(hidden), torch.zeros(2))
                torch.testing.assert_close(a, b); torch.testing.assert_close(h, j)
                # Estimator has an actual path into the actor, with trainable gradients.
                mean = policy._actor_outputs(first[5])[0]
                mean.sum().backward()
                self.assertGreater(float(policy.joint_position_head[0].weight.grad.abs().sum()), 0)

    def test_all_disabled_network_and_old_metadata_compatible(self):
        torch.manual_seed(11); old = SensorActorCritic(20, 14)
        torch.manual_seed(11); new = SensorActorCritic(20, 14,
            state_estimator_enabled=False, actor_use_estimated_state=False, future_prediction_enabled=False)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for key in old.state_dict():
            torch.testing.assert_close(old.state_dict()[key], new.state_dict()[key], rtol=0, atol=0)
        env = make_env()
        try:
            current = env.observation_metadata
            legacy = dict(current, config=dict(current['config']))
            for key in ('include_tendon_velocity', 'include_tendon_tracking_error',
                        'tendon_velocity_scale', 'tendon_velocity_filter_alpha', 'tendon_tracking_error_scale'):
                legacy['config'].pop(key)
            self.assertTrue(_sensor_metadata_compatible(legacy, current))
            changed = dict(current, config=dict(current['config'], include_tendon_velocity=True))
            self.assertFalse(_sensor_metadata_compatible(legacy, changed))
        finally:
            env.close()

    def test_future_loss_masks_episode_edges_and_missing_future(self):
        predictions = [torch.zeros(6, 2, 3, requires_grad=True)]
        frames = torch.zeros(6, 2, 3); frames[3:, 0] = 100.
        starts = torch.zeros(6, 2); starts[3, 0] = 1.
        # Three-step predictions from episode 0 cannot see episode 1 labels.
        loss = future_prediction_loss(predictions, frames, starts, (3,))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertEqual(float(predictions[0].grad.abs().sum()), 0.)
        starts.zero_()
        self.assertGreater(float(future_prediction_loss(predictions, frames, starts, (3,))), 0.)
        self.assertEqual(float(future_prediction_loss(predictions, frames, starts, (9,))), 0.)

    def test_new_onnx_external_inputs_and_torch_parity(self):
        policy = SensorActorCritic(30, 14, recurrent_hidden_dim=256,
            state_estimator_enabled=True, actor_use_estimated_state=True,
            state_estimator_include_foot_position=True, future_prediction_enabled=True,
            future_prediction_dim=20).eval()
        obs, hidden, starts = torch.randn(2, 30), torch.randn(2, 256), torch.tensor([1., 0.])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'actor.onnx'
            torch.onnx.export(RecurrentSensorActorExport(policy), (obs, hidden, starts), path,
                input_names=['observation', 'recurrent_state', 'episode_start'],
                output_names=['action', 'next_recurrent_state'], opset_version=17, dynamo=False)
            model = onnx.load(path); onnx.checker.check_model(model)
            self.assertEqual([i.name for i in model.graph.input], ['observation', 'recurrent_state', 'episode_start'])
            self.assertFalse(any('future_heads' in n.name for n in model.graph.node))
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            result = session.run(None, dict(observation=obs.numpy(), recurrent_state=hidden.numpy(), episode_start=starts.numpy()))
            expected = policy.act_inference_recurrent(obs, hidden, starts)
            for actual, target in zip(result, expected):
                np.testing.assert_allclose(actual, target.numpy(), atol=2e-6)

    def test_train_update_checkpoint_export_and_incompatible_load(self):
        env = make_env(True)
        try:
            with tempfile.TemporaryDirectory() as directory:
                cfg = SensorPPOAMPConfig(num_steps=8, num_minibatches=1, update_epochs=1,
                    compile_rollout=False, compile_policy_update=False, recurrent_hidden_dim=256,
                    state_estimator_enabled=True, state_estimator_include_foot_position=True,
                    actor_use_estimated_state=True, future_prediction_enabled=True)
                trainer = SensorHistoryPPOAMPTrainer(env, cfg, 'cpu', directory)
                rollout = trainer.collect_rollout()
                self.assertEqual(rollout['state_targets'].shape, (8, 2, 45))
                metrics = trainer.update_policy(rollout)
                for key in ('state/joint_position_mae', 'state/joint_velocity_mae', 'state/contact_accuracy',
                            'state/foot_position_loss', 'future/prediction_loss'):
                    self.assertTrue(np.isfinite(metrics[key]), key)
                checkpoint = trainer._checkpoint_payload(0)
                architecture = checkpoint['sensor_policy_architecture']
                reconstructed = SensorActorCritic(env.obs_dim, 14, sensor_frame_dim=env.sensor_frame_dim,
                    actor_history_frames=6, recurrent_hidden_dim=256, privileged_context_dim=env.privileged_context_dim,
                    **estimator_kwargs(architecture))
                reconstructed.load_state_dict(checkpoint['actor_critic'])
                self.assertEqual(observation_config_from_metadata(checkpoint), env.sensor_observation_cfg)
                path = Path(directory)/'checkpoint.pt'; torch.save(checkpoint, path)
                trainer.load(path)
                checkpoint['sensor_policy_architecture']['actor_use_estimated_state'] = False
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, 'actor_use_estimated_state'):
                    trainer.load(path)
                trainer._export_onnx(Path(directory)/'actor.onnx', trainer._checkpoint_payload(0))
        finally:
            env.close()

    def test_ablation_physics_and_reward_invariants(self):
        def options(command):
            result = {}
            for i, item in enumerate(command):
                if item.startswith('--'):
                    result[item] = command[i+1] if i+1 < len(command) and not command[i+1].startswith('--') else True
            return result
        common = None
        for experiment in 'ABCDEFG':
            for bptt in (64, 96, 128):
                args = options(training_command(experiment, 42, bptt, 100, 'teacher.pt'))
                self.assertEqual(int(args['--num-envs'])*int(args['--num-steps'])*2, 393216)
                shared = {key: value for key, value in args.items() if 'reward' in key or 'penalty' in key
                          or key in ('--motion-file', '--decimation', '--reference-ctrl-blend', '--gamma', '--gae-lambda')}
                if common is None:
                    common = shared
                self.assertEqual(shared, common)


class WarpStateEstimationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for Warp integration')
    def test_warp_features_state_labels_and_distillation_update(self):
        from sensor_training.distillation_env import SensorDistillationWarpAmpEnv
        from sensor_training.distillation_config import SensorDistillationPPOAMPConfig
        from sensor_training.distillation_trainer import SensorDistillationPPOAMPTrainer
        teacher = ROOT/'logs/teachers/stage1_gait/20260930_sensor_walk_v3/model_best.pt'
        if not teacher.is_file():
            self.skipTest('Local teacher integration fixture unavailable')
        cfg = LytRobotEnvCfg(model_xml=str(ROOT/'robot/MJCFmodelposition.xml'),
            motion_file=str(ROOT/'robot/reference_data/walk1_omniretarget.npz'),
            motion_loop=(True,), num_envs=4, sensor_root_imu_only=True,
            stand_start_prob=1., motion_reset_prob=0., reset_noise_scale=0.,
            domain_randomization=False, episode_steps=8, episode_timeout_jitter_steps=0)
        env = SensorDistillationWarpAmpEnv(cfg, device='cuda', observation_cfg=SensorObservationConfig(
            include_tendon_velocity=True, include_tendon_tracking_error=True))
        try:
            self.assertEqual(env.sensor_frame_dim, 108)
            labels = env.state_supervision_observe(True)
            self.assertEqual(tuple(labels.shape), (4, 45))
            torch.testing.assert_close(labels[:, :14], env.base_env.qpos[:, env.robot_qpos_adrs])
            torch.testing.assert_close(labels[:, 14:28], env.base_env.qvel[:, env.robot_qvel_adrs])
            torch.testing.assert_close(labels[:, 31:33], env.base_env._foot_contacts())
            torch.testing.assert_close(labels[:, 33:], env.base_env._foot_positions_local().reshape(4, 12))
            with tempfile.TemporaryDirectory() as directory:
                ppo = SensorDistillationPPOAMPConfig(teacher_checkpoint=str(teacher),
                    num_steps=10, num_minibatches=1, update_epochs=1, amp_enabled=False,
                    compile_rollout=False, compile_policy_update=False, recurrent_hidden_dim=256,
                    state_estimator_enabled=True, actor_use_estimated_state=True,
                    future_prediction_enabled=True, teacher_action_probability=1.,
                    teacher_action_probability_final=1., teacher_action_loss_coef=1., teacher_action_loss_scale=.1)
                trainer = SensorDistillationPPOAMPTrainer(env, ppo, 'cuda', directory)
                rollout = trainer.collect_rollout()
                torch.testing.assert_close(rollout['actions'], rollout['teacher_actions'])
                self.assertEqual(float(rollout['policy_sample_mask'].sum()), 0.)
                metrics = trainer.update_policy(rollout)
                self.assertGreater(metrics['teacher_action_loss'], 0.)
                self.assertIn('state/joint_position_mae', metrics)
                self.assertIn('future/prediction_loss', metrics)
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
                self.assertEqual(metrics['policy_loss'], 0.)
                # Reset filters isolate episodes and preserve the unaffected worlds.
                env.reset([0])
                torch.testing.assert_close(env._tendon_features.velocity[0], torch.zeros(14, device='cuda'))
                expected = (env._measured_tendon_lengths() -
                    (env._gait_command_frame()[:, 4:18]*env.ctrl_half_range+env.ctrl_mid))/.05
                torch.testing.assert_close(env._normalized_sensor_frame()[:, 34:48], expected.clamp(-20, 20))
                trainer.save(Path(directory)/'student.pt', 0)
                trainer._export_onnx(Path(directory)/'actor.onnx', trainer._checkpoint_payload(0))
        finally:
            env.close()


class ReferenceQualityTests(unittest.TestCase):
    def test_reference_metrics_units_contact_timing_and_teacher_gap(self):
        from sensor_training.gait_evaluation import reference_quality_metrics, comparison_metrics, summarize_gait_evaluations
        rows = [dict(joint_error_squared=.04, joint_velocity_error_squared=4.,
                     foot_error_squared=.0025, root_velocity_error_squared=.09,
                     right_contact=r, left_contact=l,
                     reference_right_contact=r, reference_left_contact=l)
                for r, l in ((0, 1), (1, 0), (0, 1), (1, 0))]
        metrics = reference_quality_metrics(rows, .1, False, True)
        self.assertAlmostEqual(metrics['eval/joint_position_rmse_rad'], .2)
        self.assertAlmostEqual(metrics['eval/joint_velocity_rmse_rad_s'], 2.)
        self.assertAlmostEqual(metrics['eval/foot_local_position_rmse_m'], .05)
        self.assertAlmostEqual(metrics['eval/root_velocity_rmse_m_s'], .3)
        self.assertEqual(metrics['eval/contact_timing_error'], 0.)
        self.assertAlmostEqual(metrics['eval/step_frequency_hz'], 10.)
        self.assertEqual(metrics['eval/left_stance_ratio'], .5)
        self.assertEqual(metrics['eval/fall_rate'], 0.)
        failed = dict(metrics, **{'eval/fall_rate': 1., 'eval/episode_completion': 0.})
        suite = summarize_gait_evaluations([metrics, failed])
        self.assertEqual(suite['eval/fall_rate'], .5)
        self.assertEqual(suite['eval/episode_completion'], .5)
        gap = comparison_metrics(metrics, failed)
        self.assertEqual(gap['student_teacher_gap/fall_rate'], 1.)
