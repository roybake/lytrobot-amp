from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from lytrobot_amp.networks import ActorCritic
from lytrobot_amp.ppo_amp import PPOAMPConfig, PPOAMPTrainer


class TrainingNumericsTests(unittest.TestCase):
    def test_train_iteration_counts_real_successes_instead_of_timeouts(self) -> None:
        trainer = PPOAMPTrainer.__new__(PPOAMPTrainer)
        trainer.cfg = SimpleNamespace(num_steps=2)
        trainer.env = SimpleNamespace(num_envs=2, cfg=SimpleNamespace(decimation=4))
        trainer.global_step = 4
        trainer.completed_episode_rewards = []
        trainer.completed_episode_lengths = []
        trainer.completed_episode_successes = []
        rollout = {
            "rewards": torch.zeros((2, 2)),
            "task_rewards": torch.zeros((2, 2)),
            "amp_rewards": torch.zeros((2, 2)),
            "dones": torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
            "time_outs": torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
            "successes": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "completed_episode_rewards": [],
            "completed_episode_lengths": [],
            "episode_progress": 0.0,
            "reward_components": {},
            "nonfinite_task_reward_count": torch.tensor(0.0),
        }
        trainer.collect_rollout = lambda: rollout
        trainer.update_policy = lambda _rollout: {}
        trainer.update_discriminator = lambda _rollout: {}

        metrics = trainer.train_iteration()

        self.assertEqual(metrics["completed_episode_count"], 3.0)
        self.assertEqual(metrics["successful_episode_count"], 2.0)
        self.assertAlmostEqual(metrics["rollout_episode_success_rate"], 2.0 / 3.0)

    def test_nonfinite_loss_is_skipped_without_changing_parameters(self) -> None:
        module = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(module.parameters(), lr=1.0e-3)
        before = {name: value.detach().clone() for name, value in module.state_dict().items()}
        loss = module(torch.ones(1, 3)).sum() * torch.tensor(float("inf"))

        stepped = PPOAMPTrainer._optimizer_step_if_finite(
            loss, module, optimizer, max_grad_norm=1.0
        )

        self.assertFalse(stepped)
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, before[name])
            self.assertTrue(torch.isfinite(value).all())

    def test_nonfinite_gradient_is_rejected_before_gradient_clipping(self) -> None:
        class FiniteForwardNanBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value: torch.Tensor) -> torch.Tensor:
                return value.sum()

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
                return (torch.full((2, 3), float("nan")) * grad_output,)

        module = torch.nn.Linear(3, 2, bias=False)
        optimizer = torch.optim.Adam(module.parameters(), lr=1.0e-3)
        before = module.weight.detach().clone()
        loss = FiniteForwardNanBackward.apply(module.weight)

        stepped = PPOAMPTrainer._optimizer_step_if_finite(
            loss, module, optimizer, max_grad_norm=1.0
        )

        self.assertFalse(stepped)
        torch.testing.assert_close(module.weight, before)
        self.assertTrue(torch.isfinite(module.weight).all())

    def test_extreme_importance_ratio_does_not_poison_policy(self) -> None:
        torch.manual_seed(4)
        trainer = PPOAMPTrainer.__new__(PPOAMPTrainer)
        trainer.device = torch.device("cpu")
        trainer.env = SimpleNamespace(obs_dim=7, action_dim=3)
        trainer.cfg = PPOAMPConfig(
            update_epochs=1,
            num_minibatches=2,
            importance_ratio_log_clip=20.0,
        )
        trainer.actor_critic = ActorCritic(obs_dim=7, action_dim=3)
        trainer.optimizer = torch.optim.Adam(
            trainer.actor_critic.parameters(), lr=trainer.cfg.learning_rate
        )

        obs = torch.randn(2, 8, 7)
        actions = torch.randn(2, 8, 3)
        with torch.no_grad():
            _, log_probs, _, values = trainer.actor_critic.get_action_and_value(
                obs.reshape(-1, 7), actions.reshape(-1, 3)
            )
        rollout = {
            "obs": obs,
            "actions": actions,
            "log_probs": (log_probs - 1.0e4).reshape(2, 8),
            "returns": (values + 1.0).reshape(2, 8),
            "values": values.reshape(2, 8),
        }

        metrics = trainer.update_policy(rollout)

        self.assertGreater(metrics["importance_ratio_log_clip_fraction"], 0.99)
        self.assertEqual(metrics["policy_updates_skipped_nonfinite"], 0.0)
        for value in trainer.actor_critic.state_dict().values():
            self.assertTrue(torch.isfinite(value).all())

    def test_one_bad_return_is_dropped_without_skipping_good_minibatches(self) -> None:
        torch.manual_seed(7)
        trainer = PPOAMPTrainer.__new__(PPOAMPTrainer)
        trainer.device = torch.device("cpu")
        trainer.env = SimpleNamespace(obs_dim=7, action_dim=3)
        trainer.cfg = PPOAMPConfig(update_epochs=1, num_minibatches=2)
        trainer.actor_critic = ActorCritic(obs_dim=7, action_dim=3)
        trainer.optimizer = torch.optim.Adam(
            trainer.actor_critic.parameters(), lr=trainer.cfg.learning_rate
        )

        obs = torch.randn(2, 8, 7)
        actions = torch.randn(2, 8, 3)
        with torch.no_grad():
            _, log_probs, _, values = trainer.actor_critic.get_action_and_value(
                obs.reshape(-1, 7), actions.reshape(-1, 3)
            )
        returns = (values + 1.0).reshape(2, 8)
        returns[0, 0] = float("nan")
        rollout = {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs.reshape(2, 8),
            "returns": returns,
            "values": values.reshape(2, 8),
        }

        metrics = trainer.update_policy(rollout)

        self.assertEqual(metrics["policy_samples_dropped_nonfinite"], 1.0)
        self.assertEqual(metrics["policy_updates_skipped_nonfinite"], 0.0)
        self.assertGreater(metrics["value_loss"], 0.0)
        for value in trainer.actor_critic.state_dict().values():
            self.assertTrue(torch.isfinite(value).all())

    def test_checkpoint_guard_rejects_nan_before_replacing_existing_file(self) -> None:
        trainer = PPOAMPTrainer.__new__(PPOAMPTrainer)
        trainer.device = torch.device("cpu")
        trainer.env = SimpleNamespace(obs_dim=7)
        trainer.actor_critic = ActorCritic(obs_dim=7, action_dim=3)
        checkpoint = {
            "iteration": 1,
            "global_step": 100,
            "actor_critic": {"bad": torch.tensor(float("nan"))},
            "obs_dim": 7,
            "action_dim": 3,
            "amp_obs_dim": 5,
            "ppo_amp_config": {},
            "env_config": {},
            "common_joint_names": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model_final.pt"
            target.write_bytes(b"known-good")
            with self.assertRaisesRegex(ValueError, "Refusing non-finite"):
                trainer._save_checkpoint_pair(target, checkpoint)
            self.assertEqual(target.read_bytes(), b"known-good")


if __name__ == "__main__":
    unittest.main()
