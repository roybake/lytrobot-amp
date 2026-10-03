"""Training-only privileged-teacher checks."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from lytrobot_amp.networks import ActorCritic
from sensor_training.distillation import PrivilegedTeacher, linear_distillation_coefficient
from sensor_training.distillation_env import remap_teacher_position_action


def _teacher_checkpoint(
    path: Path,
    *,
    reference_ctrl_blend: float = 0.0,
    motion_file: str = "walk1_mjcf_repaired.npz",
) -> None:
    policy = ActorCritic(
        obs_dim=6,
        action_dim=2,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
    )
    torch.save(
        {
            "iteration": 123,
            "global_step": 456,
            "obs_dim": 6,
            "action_dim": 2,
            "actor_critic": policy.state_dict(),
            "ppo_amp_config": {
                "actor_hidden_dims": (8,),
                "critic_hidden_dims": (8,),
                "init_std": 0.35,
                "min_action_std": 0.05,
                "max_action_std": 0.60,
            },
            "env_config": {
                "reference_ctrl_blend": reference_ctrl_blend,
                "action_residual_scale": 0.5,
                "distal_action_scale": 1.0,
                "decimation": 2,
                "motion_file": motion_file,
            },
        },
        path,
    )


class PrivilegedTeacherTests(unittest.TestCase):
    def test_reference_blended_action_is_remapped_to_same_tendon_target(self) -> None:
        mapped = remap_teacher_position_action(
            torch.tensor([[0.2, -0.2]]),
            current_ctrl=torch.tensor([[0.0, 0.0]]),
            reference_ctrl=torch.tensor([[0.5, -0.5]]),
            ctrl_low=torch.tensor([-1.0, -1.0]),
            ctrl_high=torch.tensor([1.0, 1.0]),
            ctrl_half_range=torch.tensor([1.0, 1.0]),
            teacher_reference_blend=1.0,
            teacher_action_scale=torch.ones(2),
            teacher_residual_scale=0.5,
            student_reference_blend=0.0,
            student_action_scale=torch.ones(2),
            student_residual_scale=0.5,
        )
        torch.testing.assert_close(mapped, torch.tensor([[1.0, -1.0]]))

    def test_unblended_equal_control_interfaces_preserve_action(self) -> None:
        action = torch.tensor([[0.2, -0.3]])
        mapped = remap_teacher_position_action(
            action,
            current_ctrl=torch.tensor([[0.1, -0.1]]),
            reference_ctrl=torch.tensor([[0.5, -0.5]]),
            ctrl_low=torch.tensor([-1.0, -1.0]),
            ctrl_high=torch.tensor([1.0, 1.0]),
            ctrl_half_range=torch.tensor([1.0, 1.0]),
            teacher_reference_blend=0.0,
            teacher_action_scale=torch.ones(2),
            teacher_residual_scale=0.5,
            student_reference_blend=0.0,
            student_action_scale=torch.ones(2),
            student_residual_scale=0.5,
        )
        torch.testing.assert_close(mapped, action)

    def test_linear_loss_coefficient(self) -> None:
        self.assertEqual(linear_distillation_coefficient(1.0, 0.2, 0, 100), 1.0)
        self.assertAlmostEqual(linear_distillation_coefficient(1.0, 0.2, 50, 100), 0.6)
        self.assertEqual(linear_distillation_coefficient(1.0, 0.2, 200, 100), 0.2)
        self.assertEqual(linear_distillation_coefficient(1.0, 0.2, 0, 0), 1.0)

    def test_teacher_is_frozen_and_labels_full_state_only(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "teacher.pt"
            _teacher_checkpoint(checkpoint)
            teacher = PrivilegedTeacher(
                checkpoint,
                expected_observation_dim=6,
                expected_action_dim=2,
                student_control_config={
                    "reference_ctrl_blend": 0.0,
                    "action_residual_scale": 0.5,
                    "distal_action_scale": 1.0,
                    "decimation": 2,
                    "motion_file": "walk1_mjcf_repaired.npz",
                },
                device="cpu",
            )
            action = teacher.act(torch.zeros(4, 6))

        self.assertEqual(tuple(action.shape), (4, 2))
        self.assertFalse(action.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.policy.parameters()))
        self.assertEqual(teacher.metadata.iteration, 123)

    def test_reference_blended_teacher_records_control_semantics(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "teacher.pt"
            _teacher_checkpoint(checkpoint, reference_ctrl_blend=1.0)
            teacher = PrivilegedTeacher(
                checkpoint,
                expected_observation_dim=6,
                expected_action_dim=2,
                student_control_config={
                    "reference_ctrl_blend": 0.0,
                    "action_residual_scale": 0.5,
                    "distal_action_scale": 1.0,
                    "decimation": 2,
                    "motion_file": "walk1_mjcf_repaired.npz",
                },
                device="cpu",
            )
            self.assertEqual(teacher.metadata.reference_ctrl_blend, 1.0)

    def test_different_motion_teacher_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "teacher.pt"
            _teacher_checkpoint(checkpoint, motion_file="different_walk.npz")
            with self.assertRaisesRegex(ValueError, "motion reference does not match"):
                PrivilegedTeacher(
                    checkpoint,
                    expected_observation_dim=6,
                    expected_action_dim=2,
                    student_control_config={
                        "reference_ctrl_blend": 0.0,
                        "action_residual_scale": 0.5,
                        "distal_action_scale": 1.0,
                        "decimation": 2,
                        "motion_file": "walk1_mjcf_repaired.npz",
                    },
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
