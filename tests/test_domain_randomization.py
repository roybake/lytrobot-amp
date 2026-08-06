from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
import torch

from lytrobot_amp.env import LytRobotEnvCfg, scaled_randomization_range
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
from scripts.train import (
    adaptive_domain_randomization_step,
    domain_randomization_scale_at_iteration,
)


class DomainRandomizationTests(unittest.TestCase):
    def test_scaled_range_starts_nominal_and_reaches_configured_range(self) -> None:
        self.assertEqual(scaled_randomization_range(1.0, 0.8, 1.2, 0.0), (1.0, 1.0))
        self.assertEqual(scaled_randomization_range(1.0, 0.8, 1.2, 1.0), (0.8, 1.2))
        np.testing.assert_allclose(
            scaled_randomization_range(1.0, 0.8, 1.2, 0.5),
            (0.9, 1.1),
        )

    def test_curriculum_is_linear_and_clamped(self) -> None:
        self.assertEqual(domain_randomization_scale_at_iteration(0, 10_000), 0.0)
        self.assertEqual(domain_randomization_scale_at_iteration(5_000, 10_000), 0.5)
        self.assertEqual(domain_randomization_scale_at_iteration(20_000, 10_000), 1.0)
        self.assertEqual(domain_randomization_scale_at_iteration(5_000, 0), 1.0)

    def test_warp_partial_reset_does_not_resample_shared_model_parameters(self) -> None:
        env = LytRobotWarpAmpEnv.__new__(LytRobotWarpAmpEnv)
        env.cfg = LytRobotEnvCfg(
            domain_randomization=True,
            warp_global_domain_randomization=False,
        )
        env.domain_randomization = True
        env.domain_randomization_scale = 0.0
        env.device = torch.device("cpu")
        env.position_action_dim = 2
        env.max_action_delay_ticks = 1
        env.rng = np.random.default_rng(7)
        env.env_tendon_length_scale = torch.zeros((4, 2), dtype=torch.float32)
        env.action_delay_ticks = torch.zeros(4, dtype=torch.long)
        env.action_delay_history = torch.zeros((4, 2, 2), dtype=torch.float32)
        env.push_force = torch.zeros((4, 6), dtype=torch.float32)
        env.push_time_left = torch.zeros(4, dtype=torch.long)
        env.push_time_to_next = torch.zeros(4, dtype=torch.long)
        env._apply_global_domain_randomization = Mock()

        env._sample_domain_randomization(np.asarray([1, 3], dtype=np.int64))

        env._apply_global_domain_randomization.assert_not_called()
        torch.testing.assert_close(env.env_tendon_length_scale[[1, 3]], torch.ones((2, 2)))
        torch.testing.assert_close(env.action_delay_ticks[[1, 3]], torch.zeros(2, dtype=torch.long))

        env._sample_domain_randomization(
            np.arange(4, dtype=np.int64),
            resample_global=True,
        )
        env._apply_global_domain_randomization.assert_not_called()

        env.cfg.warp_global_domain_randomization = True
        env._sample_domain_randomization(
            np.arange(4, dtype=np.int64),
            resample_global=True,
        )
        env._apply_global_domain_randomization.assert_called_once_with()

    def test_adaptive_curriculum_promotes_holds_and_backs_off(self) -> None:
        kwargs = {
            "step": 0.025,
            "promote_success_rate": 0.50,
            "promote_depth_success_rate": 0.30,
            "promote_max_termination": 0.001,
            "backoff_success_rate": 0.30,
            "backoff_depth_success_rate": 0.15,
            "backoff_termination": 0.002,
        }
        self.assertEqual(
            adaptive_domain_randomization_step(
                0.10,
                success_rate=0.60,
                depth_success_rate=0.40,
                termination_fraction=0.0005,
                **kwargs,
            ),
            (0.125, 1),
        )
        self.assertEqual(
            adaptive_domain_randomization_step(
                0.10,
                success_rate=0.40,
                depth_success_rate=0.20,
                termination_fraction=0.0015,
                **kwargs,
            ),
            (0.10, 0),
        )
        self.assertEqual(
            adaptive_domain_randomization_step(
                0.10,
                success_rate=0.25,
                depth_success_rate=0.40,
                termination_fraction=0.0005,
                **kwargs,
            ),
            (0.07500000000000001, -1),
        )
        self.assertEqual(
            adaptive_domain_randomization_step(
                0.0,
                success_rate=0.0,
                depth_success_rate=0.0,
                termination_fraction=1.0,
                **kwargs,
            ),
            (0.0, 0),
        )


if __name__ == "__main__":
    unittest.main()
