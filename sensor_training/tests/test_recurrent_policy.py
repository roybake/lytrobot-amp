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


class _TimeoutSensorEnv(_DummySensorEnv):
    """Autoresets one environment and supplies its pre-reset sensor state."""

    def step(self, action):
        self.step_count += 1
        observation = torch.full((self.num_envs, self.obs_dim), -3.0)
        reward = torch.ones(self.num_envs)
        done = torch.tensor([1.0, 0.0], dtype=torch.float32)
        info = {
            "episode_step": torch.full((self.num_envs,), float(self.step_count)),
            "time_outs": torch.tensor([1.0, 0.0]),
            "successes": torch.zeros(self.num_envs),
            "termination": torch.zeros(self.num_envs),
            # These deliberately differ from the reset observation above.
            "terminal_observation": torch.full((1, self.obs_dim), 2.0),
            "terminal_privileged_context": torch.full(
                (1, self.privileged_context_dim), 0.25
            ),
            "terminal_env_ids": torch.tensor([0]),
        }
        return observation, reward, done, info


class _GaitGateSensorEnv(_DummySensorEnv):
    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["gait_progress"] = torch.full(
            (self.num_envs,),
            min(max((self.step_count - 1) * 0.5, 0.0), 1.0),
        )
        return observation, reward, done, info


class RecurrentSensorPolicyTests(unittest.TestCase):
    def _policy(self) -> SensorActorCritic:
        torch.manual_seed(7)
        return SensorActorCritic(
            obs_dim=18,
            action_dim=4,
            sensor_frame_dim=3,
            actor_history_frames=6,
            recurrent_hidden_dim=8,
            use_moe=True,
            num_experts=3,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(8,),
            gate_hidden_dims=(8,),
            expert_hidden_dims=(8,),
            residual_scale=0.25,
        )

    def test_recurrent_encoder_uses_older_sensor_frames(self) -> None:
        policy = self._policy()
        first = torch.randn(2, 18)
        second = first.clone()
        second[:, :3] += 1.0
        hidden = torch.randn(2, 8)

        first_next = policy._next_hidden(first, hidden, torch.zeros(2))
        second_next = policy._next_hidden(second, hidden, torch.zeros(2))

        self.assertFalse(torch.allclose(first_next, second_next))
        self.assertEqual(policy.recurrent.input_size, 18)

    def test_episode_start_clears_hidden_state_before_gru_step(self) -> None:
        policy = self._policy()
        observation = torch.randn(2, 18)
        nonzero_hidden = torch.randn(2, 8)
        zero_hidden = torch.zeros(2, 8)

        reset_next = policy._next_hidden(observation, nonzero_hidden, torch.ones(2))
        clean_next = policy._next_hidden(observation, zero_hidden, torch.zeros(2))

        torch.testing.assert_close(reset_next, clean_next)

    def test_residual_moe_is_softly_weighted_and_keeps_action_dimension(self) -> None:
        policy = self._policy()
        hidden = torch.randn(5, 8)

        mean, weights, residual = policy._actor_outputs(hidden)

        self.assertEqual(tuple(mean.shape), (5, 4))
        self.assertEqual(tuple(weights.shape), (5, 3))
        self.assertEqual(tuple(residual.shape), (5, 4))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(5))
        self.assertTrue(torch.all((weights > 0.0) & (weights < 1.0)))
        self.assertTrue(torch.all(mean.abs() <= 1.0))

    def test_sequence_backpropagates_through_recurrent_state(self) -> None:
        policy = self._policy()
        observations = torch.randn(4, 3, 18)
        actions = torch.zeros(4, 3, 4)
        starts = torch.zeros(4, 3)
        starts[0] = 1.0
        hidden = policy.initial_state(3)

        log_prob, _entropy, values, _median, _next, weights, residual, _context = (
            policy.evaluate_sequence(observations, hidden, starts, actions)
        )
        loss = -(log_prob.mean() + values.mean()) + weights.square().mean() + residual.square().mean()
        loss.backward()

        self.assertIsNotNone(policy.recurrent.weight_hh.grad)
        self.assertTrue(torch.isfinite(policy.recurrent.weight_hh.grad).all())
        self.assertGreater(float(policy.recurrent.weight_hh.grad.abs().sum()), 0.0)

    def test_rollout_hot_path_matches_full_recurrent_evaluation(self) -> None:
        policy = self._policy()
        observations = torch.randn(3, 18)
        hidden = policy.initial_state(3)
        starts = torch.tensor([1.0, 0.0, 0.0])

        torch.manual_seed(29)
        rollout_action, rollout_log_prob, rollout_value, rollout_hidden = (
            policy.rollout_action_and_value(observations, hidden, starts)
        )
        torch.manual_seed(29)
        full = policy.recurrent_action_and_value(observations, hidden, starts)

        torch.testing.assert_close(rollout_action, full[0])
        torch.testing.assert_close(rollout_log_prob, full[1])
        torch.testing.assert_close(rollout_value, full[3])
        torch.testing.assert_close(rollout_hidden, full[5])

    def test_trainer_preserves_reset_masks_and_updates_sequences(self) -> None:
        env = _DummySensorEnv()
        cfg = SensorPPOAMPConfig(
            num_steps=3,
            update_epochs=1,
            num_minibatches=1,
            amp_enabled=False,
            recurrent_hidden_dim=8,
            num_experts=2,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(8,),
            gate_hidden_dims=(8,),
            expert_hidden_dims=(8,),
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
        self.assertIn("gate_change_loss", metrics)
        self.assertIn("expert_load_balance_loss", metrics)
        self.assertIn("residual_l2_loss", metrics)
        self.assertIn("privileged_context_loss", metrics)
        self.assertEqual(
            tuple(rollout["privileged_contexts"].shape),
            (3, 2, env.privileged_context_dim),
        )

    def test_direct_amp_reward_fades_in_with_startup_speed(self) -> None:
        for gate_enabled, expected in (
            (True, [0.0, 0.5, 1.0]),
            (False, [1.0, 1.0, 1.0]),
        ):
            with self.subTest(gate_enabled=gate_enabled):
                env = _GaitGateSensorEnv()
                cfg = SensorPPOAMPConfig(
                    num_steps=3,
                    amp_enabled=True,
                    amp_startup_gate=gate_enabled,
                    recurrent_hidden_dim=8,
                    actor_hidden_dims=(8,),
                    critic_hidden_dims=(8,),
                    discriminator_hidden_dims=(8,),
                )
                trainer = SensorHistoryPPOAMPTrainer(env, cfg, "cpu", ".")
                trainer._rollout_amp_reward = lambda _features: torch.ones(
                    env.num_envs
                )
                rollout = trainer.collect_rollout()
                torch.testing.assert_close(
                    rollout["amp_rewards"][:, 0],
                    torch.tensor(expected),
                )

    def test_compiled_chunks_preserve_full_bptt_evaluation(self) -> None:
        """Chunk boundaries must not alter recurrent outputs or hidden state."""

        env = _DummySensorEnv()
        cfg = SensorPPOAMPConfig(
            num_steps=5,
            update_epochs=1,
            num_minibatches=1,
            amp_enabled=False,
            compile_policy_chunk_steps=2,
            recurrent_hidden_dim=8,
            num_experts=2,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(8,),
            gate_hidden_dims=(8,),
            expert_hidden_dims=(8,),
        )
        trainer = SensorHistoryPPOAMPTrainer(env, cfg, "cpu", ".")
        torch.manual_seed(17)
        observations = torch.randn(5, env.num_envs, env.obs_dim)
        actions = torch.randn(5, env.num_envs, env.action_dim)
        starts = torch.zeros(5, env.num_envs)
        starts[0] = 1.0
        starts[3, 0] = 1.0
        contexts = torch.randn(5, env.num_envs, env.privileged_context_dim)
        initial_hidden = trainer.actor_critic.initial_state(env.num_envs)

        expected = trainer.actor_critic.evaluate_sequence(
            observations, initial_hidden, starts, actions, contexts
        )
        # Use the eager callable as a deterministic test double for the
        # compiled chunk callable; CUDA integration chooses torch.compile.
        trainer._compiled_evaluate_sequence = trainer.actor_critic.evaluate_sequence
        actual = trainer._evaluate_policy_sequence(
            observations, initial_hidden, starts, actions, contexts
        )

        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_timeout_bootstrap_uses_pre_reset_terminal_sensor_state(self) -> None:
        env = _TimeoutSensorEnv()
        cfg = SensorPPOAMPConfig(
            num_steps=1,
            update_epochs=1,
            num_minibatches=1,
            amp_enabled=False,
            recurrent_hidden_dim=8,
            num_experts=2,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(8,),
            gate_hidden_dims=(8,),
            expert_hidden_dims=(8,),
        )
        trainer = SensorHistoryPPOAMPTrainer(env, cfg, "cpu", ".")
        initial_obs = torch.zeros(1, env.obs_dim)
        initial_context = torch.zeros(1, env.privileged_context_dim)
        with torch.no_grad():
            terminal_hidden = trainer.actor_critic.recurrent_action_and_value(
                initial_obs,
                trainer.actor_critic.initial_state(1),
                torch.ones(1),
                privileged_context=initial_context,
            )[5]
            expected_terminal_value = trainer.actor_critic.recurrent_action_and_value(
                torch.full((1, env.obs_dim), 2.0),
                terminal_hidden,
                torch.zeros(1),
                privileged_context=torch.full((1, env.privileged_context_dim), 0.25),
            )[3]
        rollout = trainer.collect_rollout()

        torch.testing.assert_close(
            rollout["timeout_bootstrap_values"][0, :1],
            expected_terminal_value,
        )
        self.assertEqual(float(rollout["timeout_bootstrap_values"][0, 1]), 0.0)

    def test_privileged_context_trains_estimator_and_critic_but_not_actor_input(self) -> None:
        policy = SensorActorCritic(
            obs_dim=18,
            action_dim=4,
            sensor_frame_dim=3,
            recurrent_hidden_dim=8,
            privileged_context_dim=5,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(8,),
            gate_hidden_dims=(8,),
            expert_hidden_dims=(8,),
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
            "env_config": {},
            "common_joint_names": [],
            "sensor_observation": {},
            "sensor_policy_architecture": {"recurrent": True},
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


def test_acceptance_loader_restores_privileged_critic_architecture():
    from scripts.evaluate_acceptance import load_policy

    env = _DummySensorEnv()
    cfg = SensorPPOAMPConfig(
        actor_hidden_dims=(8,), critic_hidden_dims=(8,),
        recurrent_hidden_dim=8, use_moe=True, num_experts=2,
        gate_hidden_dims=(8,), expert_hidden_dims=(8,),
    )
    trainer = SensorHistoryPPOAMPTrainer(env, cfg, 'cpu', '/tmp')
    checkpoint = {
        'sensor_observation': env.observation_metadata,
        'sensor_policy_architecture': {
            'recurrent': True, 'sensor_frame_dim': env.sensor_frame_dim,
            'actor_history_frames': env.obs_dim // env.sensor_frame_dim,
            'recurrent_hidden_dim': cfg.recurrent_hidden_dim, 'use_moe': cfg.use_moe,
            'num_experts': cfg.num_experts,
            'gate_hidden_dims': cfg.gate_hidden_dims, 'expert_hidden_dims': cfg.expert_hidden_dims,
            'residual_scale': cfg.residual_scale, 'privileged_context_dim': env.privileged_context_dim,
        },
        'ppo_amp_config': {'actor_hidden_dims': cfg.actor_hidden_dims, 'critic_hidden_dims': cfg.critic_hidden_dims},
        'actor_critic': trainer.actor_critic.state_dict(),
    }
    policy = load_policy(checkpoint, env, torch.device('cpu'))
    assert policy.privileged_context_dim == env.privileged_context_dim
    obs = torch.randn(env.num_envs, env.obs_dim)
    hidden = policy.initial_state(env.num_envs)
    starts = torch.ones(env.num_envs)
    torch.testing.assert_close(
        policy.act_inference_recurrent(obs, hidden, starts)[0],
        trainer.actor_critic.act_inference_recurrent(obs, hidden, starts)[0],
    )
