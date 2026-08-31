from __future__ import annotations

import unittest

import numpy as np
import torch

from lytrobot_amp.env import (
    _sanitize_reward_outputs as sanitize_numpy_reward_outputs,
    _stability_gated_locomotion_terms as gated_numpy_locomotion_terms,
)
from lytrobot_amp.env_warp import (
    _sanitize_reward_outputs as sanitize_torch_reward_outputs,
    _stability_gated_locomotion_terms as gated_torch_locomotion_terms,
)


class RewardNumericsTests(unittest.TestCase):
    def test_warp_sanitizer_quarantines_only_invalid_environment(self) -> None:
        reward = torch.tensor([1.0, 2.0, float("nan")])
        bad_state = torch.tensor([False, False, False])
        fallen = torch.tensor([False, True, False])
        components = {
            "root_vel": torch.tensor([0.5, 0.6, 0.7]),
            "pose_track": torch.tensor([0.8, 0.9, float("nan")]),
        }

        safe_reward, invalid, safe_components = sanitize_torch_reward_outputs(
            reward, bad_state, fallen, components
        )

        torch.testing.assert_close(safe_reward, torch.tensor([1.0, 0.0, -2.0]))
        torch.testing.assert_close(invalid, torch.tensor([False, False, True]))
        torch.testing.assert_close(
            safe_components["pose_track"], torch.tensor([0.8, 0.9, 0.0])
        )
        torch.testing.assert_close(
            safe_components["invalid_state"], torch.tensor([0.0, 0.0, 1.0])
        )

    def test_configured_fall_penalty_makes_falling_transition_negative(self) -> None:
        numpy_reward, numpy_invalid, _ = sanitize_numpy_reward_outputs(
            5.0,
            False,
            True,
            {"root_vel": 0.5},
            fall_penalty=10.0,
        )
        torch_reward, torch_invalid, _ = sanitize_torch_reward_outputs(
            torch.tensor([5.0]),
            torch.tensor([False]),
            torch.tensor([True]),
            {"root_vel": torch.tensor([0.5])},
            fall_penalty=10.0,
        )

        self.assertEqual(numpy_reward, -5.0)
        self.assertFalse(numpy_invalid)
        torch.testing.assert_close(torch_reward, torch.tensor([-5.0]))
        torch.testing.assert_close(torch_invalid, torch.tensor([False]))

    def test_positive_locomotion_reward_is_gated_by_stability(self) -> None:
        numpy_terms = gated_numpy_locomotion_terms(0.8, 0.6, 0.25, 0.25)
        torch_terms = gated_torch_locomotion_terms(
            torch.tensor([0.8, 0.8]),
            torch.tensor([0.6, -0.6]),
            torch.tensor([0.25, 0.25]),
            torch.tensor([0.25, 0.25]),
        )

        np.testing.assert_allclose(numpy_terms, (0.25, 0.20, 0.15))
        torch.testing.assert_close(torch_terms[0], torch.tensor([0.25, 0.25]))
        torch.testing.assert_close(torch_terms[1], torch.tensor([0.20, 0.20]))
        # Instability cannot erase the negative reward for moving backwards.
        torch.testing.assert_close(torch_terms[2], torch.tensor([0.15, -0.60]))

    def test_cpu_sanitizer_returns_finite_penalty_and_components(self) -> None:
        safe_reward, invalid, safe_components = sanitize_numpy_reward_outputs(
            float("nan"),
            False,
            False,
            {"root_vel": float("nan"), "pose_track": 0.8},
        )

        self.assertTrue(invalid)
        self.assertEqual(safe_reward, -2.0)
        self.assertEqual(safe_components["root_vel"], 0.0)
        self.assertEqual(safe_components["pose_track"], 0.0)
        self.assertEqual(safe_components["invalid_state"], 1.0)
        self.assertTrue(all(np.isfinite(value) for value in safe_components.values()))


if __name__ == "__main__":
    unittest.main()
