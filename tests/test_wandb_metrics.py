from __future__ import annotations

import unittest

from scripts.train import (
    WANDB_EPISODE_LENGTH_KEY,
    WANDB_EPISODE_TOTAL_REWARD_KEY,
    filter_resume_wandb_warmup_metrics,
    prepare_wandb_metrics,
)


class WandbMetricTests(unittest.TestCase):
    def test_prepare_wandb_metrics_adds_episode_curves(self) -> None:
        source = {
            "global_step": 123.0,
            "mean_episode_reward": 42.5,
            "mean_episode_length": 88.0,
        }

        prepared = prepare_wandb_metrics(source)

        self.assertEqual(prepared[WANDB_EPISODE_TOTAL_REWARD_KEY], 42.5)
        self.assertEqual(prepared[WANDB_EPISODE_LENGTH_KEY], 88.0)
        self.assertEqual(prepared["mean_reward"], 42.5)
        self.assertEqual(prepared["mean_length"], 88.0)
        self.assertEqual(
            source,
            {
                "global_step": 123.0,
                "mean_episode_reward": 42.5,
                "mean_episode_length": 88.0,
            },
        )

    def test_prepare_wandb_metrics_supports_legacy_flat_names(self) -> None:
        prepared = prepare_wandb_metrics({"mean_reward": 7.5, "mean_length": 16.0})

        self.assertEqual(prepared["mean_episode_reward"], 7.5)
        self.assertEqual(prepared["mean_episode_length"], 16.0)
        self.assertEqual(prepared[WANDB_EPISODE_TOTAL_REWARD_KEY], 7.5)
        self.assertEqual(prepared[WANDB_EPISODE_LENGTH_KEY], 16.0)

    def test_resume_warmup_filters_episode_curves(self) -> None:
        prepared = prepare_wandb_metrics(
            {
                "global_step": 123.0,
                "mean_episode_reward": 42.5,
                "mean_episode_length": 88.0,
                "episode_success_rate": 0.75,
                "bottom_depth_success_rate": 0.5,
                "policy_loss": 0.25,
            }
        )

        filtered = filter_resume_wandb_warmup_metrics(prepared)

        self.assertNotIn(WANDB_EPISODE_TOTAL_REWARD_KEY, filtered)
        self.assertNotIn(WANDB_EPISODE_LENGTH_KEY, filtered)
        self.assertNotIn("episode_success_rate", filtered)
        self.assertNotIn("bottom_depth_success_rate", filtered)
        self.assertEqual(filtered["policy_loss"], 0.25)
        self.assertEqual(filtered["resume_wandb_warmup"], 1.0)


if __name__ == "__main__":
    unittest.main()
