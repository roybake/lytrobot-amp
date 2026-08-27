from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile

import numpy as np
import onnx
import torch

from sensor_training.networks import SensorActorCritic
from sensor_training.ppo_config import SensorPPOAMPConfig
from sensor_training.trainer import SensorHistoryPPOAMPTrainer
from scripts.play import OnnxPolicy


class _DummySensorEnv:
    num_envs = 2
    obs_dim = 6
    sensor_frame_dim = 3
    action_dim = 2
    amp_obs_dim = 4
    privileged_context_dim = 5
    is_torch_env = True
    privileged_context_metadata = {
        "fields": tuple(f"context_{index}" for index in range(5))
    }
    observation_metadata = {
        "strict_sensor_only": True,
        "privileged_training_context": {
            "actor_input": False,
            "dim": privileged_context_dim,
        },
    }

    def __init__(self) -> None:
        self.step_count = 0
        self.cfg = SimpleNamespace(decimation=1)

    def reset(self):
        self.step_count = 0
        return torch.zeros(self.num_envs, self.obs_dim)

    def amp_observe(self):
        return torch.zeros(self.num_envs, self.amp_obs_dim)

    def privileged_observe(self):
        return torch.full(
            (self.num_envs, self.privileged_context_dim),
            0.1 * float(self.step_count),
        )

    def step(self, action):
        self.step_count += 1
        observation = torch.full(
            (self.num_envs, self.obs_dim),
            float(self.step_count),
        )
        reward = torch.ones(self.num_envs)
        done = torch.tensor(
            [float(self.step_count == 1), 0.0],
            dtype=torch.float32,
        )
        info = {
            "episode_step": torch.full((self.num_envs,), float(self.step_count)),
            "time_outs": torch.zeros(self.num_envs),
            "successes": torch.zeros(self.num_envs),
            "termination": done.clone(),
        }
        return observation, reward, done, info


class RecurrentSensorPolicyTests(unittest.TestCase):
    def test_resume_layout_check_ignores_squat_reward_only_metadata(self) -> None:
        saved = {
            "actor_obs_dim": 338,
            "sensor_history_obs_dim": 336,
            "actor_input_fields": (
                "imu",
                "tendon_length",
                "feedback_corrected_phase",
            ),
            "phase_feedback": {
                "reference_frequency_hz": 40.0,
                "bottom_depth_threshold": 0.85,
                "cycle_completion_bonus": 5.0,
            },
            "success_definition": "survived_episode_with_complete_cycle",
        }
        current = {
            **saved,
            "phase_feedback": {
                **saved["phase_feedback"],
                "cycle_completion_bonus": 25.0,
                "terminate_on_cycle_completion": True,
            },
            "success_definition": "completed_cycle",
        }

        self.assertEqual(
            SensorHistoryPPOAMPTrainer._sensor_observation_layout_signature(saved),
            SensorHistoryPPOAMPTrainer._sensor_observation_layout_signature(current),
        )

        incompatible = {
            **current,
            "actor_obs_dim": 336,
        }
        self.assertNotEqual(
            SensorHistoryPPOAMPTrainer._sensor_observation_layout_signature(saved),
            SensorHistoryPPOAMPTrainer._sensor_observation_layout_signature(
                incompatible
            ),
        )

    def _policy(self) -> SensorActorCritic:
        torch.manual_seed(7)
        return SensorActorCritic(
            obs_dim=18,
            action_dim=4,
            sensor_frame_dim=3,
            recurrent_hidden_dim=8,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(8,),
        )

    def test_recurrent_encoder_uses_only_latest_sensor_frame(self) -> None:
        policy = self._policy()
        first = torch.randn(2, 18)
        second = torch.randn(2, 18)
        second[:, -3:] = first[:, -3:]
        hidden = torch.randn(2, 8)

        first_next = policy._next_hidden(first, hidden, torch.zeros(2))
        second_next = policy._next_hidden(second, hidden, torch.zeros(2))

        torch.testing.assert_close(first_next, second_next)

    def test_episode_start_clears_hidden_state_before_gru_step(self) -> None:
        policy = self._policy()
        observation = torch.randn(2, 18)
        nonzero_hidden = torch.randn(2, 8)
        zero_hidden = torch.zeros(2, 8)

        reset_next = policy._next_hidden(observation, nonzero_hidden, torch.ones(2))
        clean_next = policy._next_hidden(observation, zero_hidden, torch.zeros(2))

        torch.testing.assert_close(reset_next, clean_next)

    def test_single_actor_mlp_keeps_action_dimension_without_experts(self) -> None:
        policy = self._policy()
        hidden = torch.randn(5, 8)

        mean = policy._actor_mean(hidden)

        self.assertEqual(tuple(mean.shape), (5, 4))
        self.assertTrue(torch.all(mean.abs() <= 1.0))
        self.assertFalse(hasattr(policy, "gate"))
        self.assertFalse(hasattr(policy, "experts"))

    def test_sequence_backpropagates_through_recurrent_state(self) -> None:
        policy = self._policy()
        observations = torch.randn(4, 3, 18)
        actions = torch.zeros(4, 3, 4)
        starts = torch.zeros(4, 3)
        starts[0] = 1.0
        hidden = policy.initial_state(3)

        log_prob, _entropy, values, _median, _next, _context = (
            policy.evaluate_sequence(observations, hidden, starts, actions)
        )
        loss = -(log_prob.mean() + values.mean())
        loss.backward()

        self.assertIsNotNone(policy.recurrent.weight_hh.grad)
        self.assertTrue(torch.isfinite(policy.recurrent.weight_hh.grad).all())
        self.assertGreater(float(policy.recurrent.weight_hh.grad.abs().sum()), 0.0)

    def test_trainer_preserves_reset_masks_and_updates_sequences(self) -> None:
        env = _DummySensorEnv()
        cfg = SensorPPOAMPConfig(
            num_steps=3,
            update_epochs=1,
            num_minibatches=1,
            amp_enabled=False,
            recurrent_hidden_dim=8,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(8,),
        )
        trainer = SensorHistoryPPOAMPTrainer(env, cfg, "cpu", ".")

        rollout = trainer.collect_rollout()

        self.assertEqual(tuple(rollout["obs"].shape), (3, 2, 6))
        self.assertEqual(tuple(rollout["initial_recurrent_state"].shape), (2, 8))
        torch.testing.assert_close(rollout["episode_starts"][0], torch.ones(2))
        torch.testing.assert_close(
            rollout["episode_starts"][1],
            torch.tensor([1.0, 0.0]),
        )
        metrics = trainer.update_policy(rollout)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertIn("privileged_context_loss", metrics)
        self.assertEqual(
            tuple(rollout["privileged_contexts"].shape),
            (3, 2, env.privileged_context_dim),
        )

    def test_privileged_context_trains_estimator_and_critic_but_not_actor_input(self) -> None:
        policy = SensorActorCritic(
            obs_dim=18,
            action_dim=4,
            sensor_frame_dim=3,
            recurrent_hidden_dim=8,
            privileged_context_dim=5,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(8,),
        )
        observations = torch.randn(3, 2, 18)
        actions = torch.zeros(3, 2, 4)
        starts = torch.zeros(3, 2)
        contexts = torch.randn(3, 2, 5).tanh()
        hidden = policy.initial_state(2)

        result = policy.evaluate_sequence(
            observations,
            hidden,
            starts,
            actions,
            contexts,
        )
        context_prediction = result[-1]
        loss = torch.nn.functional.smooth_l1_loss(
            context_prediction,
            contexts,
        )
        loss.backward()

        self.assertEqual(policy.actor[0].in_features, 8)
        self.assertEqual(policy.critic[0].in_features, 13)
        self.assertEqual(tuple(context_prediction.shape), (3, 2, 5))
        self.assertIsNotNone(policy.recurrent.weight_hh.grad)
        self.assertGreater(
            float(policy.recurrent.weight_hh.grad.abs().sum()),
            0.0,
        )

    def test_recurrent_onnx_keeps_and_resets_hidden_state(self) -> None:
        policy = self._policy()
        trainer = SensorHistoryPPOAMPTrainer.__new__(SensorHistoryPPOAMPTrainer)
        trainer.device = torch.device("cpu")
        trainer.env = SimpleNamespace(obs_dim=18)
        trainer.actor_critic = policy
        checkpoint = {
            "iteration": 0,
            "global_step": 0,
            "obs_dim": 18,
            "action_dim": 4,
            "amp_obs_dim": 5,
            "ppo_amp_config": {},
            "env_config": {"timestep": 0.005, "decimation": 2},
            "common_joint_names": [],
            "sensor_observation": {},
            "sensor_policy_architecture": {
                "name": "proprioceptive_gru_context_mlp",
                "recurrent": True,
            },
        }
        observation = np.random.default_rng(4).normal(size=(2, 18)).astype(np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recurrent.onnx"
            trainer._export_onnx(path, checkpoint)
            exported_model = onnx.load(path)
            self.assertEqual(
                [value.name for value in exported_model.graph.input],
                ["observation", "recurrent_state", "episode_start"],
            )
            onnx_policy = OnnxPolicy(path)
            onnx_policy.validate_dimensions(18, 4)

            first = onnx_policy.act_inference(
                observation,
                episode_start=np.ones(2, dtype=np.float32),
            )
            continued = onnx_policy.act_inference(
                observation,
                episode_start=np.zeros(2, dtype=np.float32),
            )
            reset = onnx_policy.act_inference(
                observation,
                episode_start=np.ones(2, dtype=np.float32),
            )

        self.assertFalse(np.allclose(first, continued))
        np.testing.assert_allclose(first, reset, rtol=1.0e-5, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
