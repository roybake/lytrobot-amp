from __future__ import annotations

import unittest

import torch

from lytrobot_amp.networks import ActorCritic
from sensor_training.networks import SensorActorCritic
from sensor_training.ppo_config import SensorPPOAMPConfig


class SensorPolicyDistributionTests(unittest.TestCase):
    def test_full_state_training_action_keeps_unclipped_gaussian_sample(self) -> None:
        torch.manual_seed(7)
        policy = ActorCritic(
            obs_dim=3,
            action_dim=2,
            actor_hidden_dims=(),
            critic_hidden_dims=(),
            init_std=0.6,
            max_action_std=0.6,
        )
        with torch.no_grad():
            policy.actor[0].weight.zero_()
            policy.actor[0].bias.zero_()

        obs = torch.zeros((4096, 3))
        action, log_prob, _, _ = policy.get_action_and_value(obs)

        self.assertTrue(torch.any(action.abs() > 1.0))
        expected = policy._distribution(obs).log_prob(action).sum(dim=-1)
        torch.testing.assert_close(log_prob, expected)

    def test_training_action_keeps_unclipped_gaussian_sample(self) -> None:
        torch.manual_seed(7)
        policy = SensorActorCritic(
            obs_dim=3,
            action_dim=2,
            actor_hidden_dims=(),
            critic_hidden_dims=(),
            init_std=0.6,
            max_action_std=0.6,
        )
        with torch.no_grad():
            policy.actor[0].weight.zero_()
            policy.actor[0].bias.zero_()

        obs = torch.zeros((4096, 3))
        action, log_prob, _, _ = policy.get_action_and_value(obs)

        self.assertTrue(torch.any(action.abs() > 1.0))
        expected = policy._distribution(obs).log_prob(action).sum(dim=-1)
        torch.testing.assert_close(log_prob, expected)

    def test_stability_defaults_are_sensor_scoped(self) -> None:
        cfg = SensorPPOAMPConfig()
        self.assertEqual(cfg.clip_coef, 0.2)
        self.assertEqual(cfg.bpo_lambda, 0.001)
        self.assertEqual(cfg.bpo_tv_coef, 0.0)
        self.assertEqual(cfg.median_value_coef, 0.5)
        self.assertEqual(cfg.entropy_coef, 0.003)
        self.assertEqual(cfg.max_grad_norm, 0.5)
        self.assertTrue(cfg.compile_rollout)
        self.assertTrue(cfg.compile_policy_update)
        self.assertEqual(cfg.compile_policy_chunk_steps, 16)
        self.assertEqual(cfg.target_kl, 0.025)
        self.assertEqual(cfg.target_kl_stop_multiplier, 2.0)
        self.assertTrue(cfg.adaptive_lr)
        self.assertEqual(cfg.max_learning_rate, 3.0e-4)
        self.assertEqual(cfg.recurrent_hidden_dim, 128)
        self.assertFalse(cfg.use_moe)
        self.assertEqual(cfg.num_experts, 4)
        self.assertEqual(cfg.residual_scale, 0.35)
        self.assertGreater(cfg.gate_change_coef, 0.0)
        self.assertGreater(cfg.expert_load_balance_coef, 0.0)
        self.assertGreater(cfg.residual_l2_coef, 0.0)
        self.assertGreater(cfg.privileged_context_loss_coef, 0.0)


if __name__ == "__main__":
    unittest.main()
