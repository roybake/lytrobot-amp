"""Training-only privileged teachers for sensor-policy distillation.

The teacher reads the simulator's full observation only while collecting
training rollouts.  It is never attached to the sensor Actor or exported
with the deployable policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lytrobot_amp.networks import ActorCritic
from lytrobot_amp.env import LytRobotEnvCfg


def linear_distillation_coefficient(
    initial: float,
    final: float,
    iteration: int,
    decay_iterations: int,
) -> float:
    """Linearly anneal an action-imitation coefficient by training iteration."""

    initial = float(initial)
    final = float(final)
    if decay_iterations <= 0:
        return initial
    fraction = min(max(float(iteration) / float(decay_iterations), 0.0), 1.0)
    if fraction <= 0.0:
        return initial
    if fraction >= 1.0:
        return final
    return initial + (final - initial) * fraction


@dataclass(frozen=True)
class PrivilegedTeacherMetadata:
    checkpoint: str
    iteration: int
    global_step: int
    observation_dim: int
    action_dim: int
    reference_ctrl_blend: float
    action_residual_scale: float
    distal_action_scale: float
    decimation: int
    startup_action_ramp_steps: int
    motion_file: str


class PrivilegedTeacher:
    """A frozen full-state ActorCritic used to label student-rollout states."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_observation_dim: int,
        expected_action_dim: int,
        student_control_config: dict[str, Any],
        device: torch.device | str,
    ) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint does not exist: {path}")
        checkpoint: dict[str, Any] = torch.load(path, map_location=device)
        if "sensor_policy_architecture" in checkpoint:
            raise ValueError(
                "--teacher-checkpoint must be a privileged stage1/stage2 checkpoint, "
                "not a sensor-policy checkpoint."
            )
        observation_dim = int(checkpoint.get("obs_dim", -1))
        action_dim = int(checkpoint.get("action_dim", -1))
        if observation_dim != int(expected_observation_dim):
            raise ValueError(
                "Teacher observation dimension does not match the base environment: "
                f"teacher={observation_dim}, environment={expected_observation_dim}."
            )
        if action_dim != int(expected_action_dim):
            raise ValueError(
                "Teacher action dimension does not match the sensor student: "
                f"teacher={action_dim}, student={expected_action_dim}."
            )

        teacher_env = checkpoint.get("env_config")
        if not isinstance(teacher_env, dict):
            raise ValueError("Teacher checkpoint has no env_config metadata.")
        teacher_cfg = LytRobotEnvCfg(**teacher_env)
        if teacher_cfg.command_walking != student_control_config["command_walking"]:
            raise ValueError("Teacher and Student must use the same walking command mode.")
        if teacher_cfg.command_walking:
            for name in ("velocity_range", "yaw_rate_range"):
                if tuple(getattr(teacher_cfg, name)) != tuple(student_control_config[name]):
                    raise ValueError(f"Teacher and Student must use the same {name}.")
        if teacher_cfg.symmetric_gait_reference != student_control_config["symmetric_gait_reference"]:
            raise ValueError("Teacher and Student must use the same symmetric walking reference.")
        if (teacher_cfg.symmetric_gait_reference
                and teacher_cfg.target_velocity != student_control_config["target_velocity"]):
            raise ValueError("Teacher and Student must generate the symmetric reference at the same velocity.")
        teacher_blend = float(teacher_env.get("reference_ctrl_blend", 1.0))
        if not 0.0 <= teacher_blend <= 1.0:
            raise ValueError(f"Teacher reference_ctrl_blend is invalid: {teacher_blend}.")
        teacher_decimation = int(teacher_env.get("decimation", -1))
        student_decimation = int(student_control_config.get("decimation", -2))
        if teacher_decimation != student_decimation:
            raise ValueError(
                f"Teacher decimation={teacher_decimation} does not match sensor student "
                f"decimation={student_decimation}."
            )
        teacher_motion_file = str(teacher_env.get("motion_file", ""))
        student_motion_file = str(student_control_config.get("motion_file", ""))
        if student_motion_file and (
            not teacher_motion_file
            or Path(teacher_motion_file).name != Path(student_motion_file).name
        ):
            raise ValueError(
                "Teacher motion reference does not match the sensor student: "
                f"teacher={teacher_motion_file or '<missing>'}, "
                f"student={student_motion_file}. Retrain the teacher with the "
                "same natural gait archive instead of distilling a legacy gait."
            )

        config = checkpoint.get("ppo_amp_config", {})
        if not isinstance(config, dict):
            config = {}

        def dims(name: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
            value = config.get(name, fallback)
            return tuple(int(item) for item in value)

        self.policy = ActorCritic(
            obs_dim=observation_dim,
            action_dim=action_dim,
            actor_hidden_dims=dims("actor_hidden_dims", (256, 128)),
            critic_hidden_dims=dims("critic_hidden_dims", (256, 128)),
            init_std=float(config.get("init_std", 0.35)),
            min_action_std=float(config.get("min_action_std", 0.05)),
            max_action_std=float(config.get("max_action_std", 0.60)),
        ).to(device)
        try:
            self.policy.load_state_dict(checkpoint["actor_critic"], strict=True)
        except KeyError as exc:
            raise ValueError("Teacher checkpoint has no actor_critic weights.") from exc
        self.policy.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

        self.metadata = PrivilegedTeacherMetadata(
            checkpoint=str(path),
            iteration=int(checkpoint.get("iteration", -1)),
            global_step=int(checkpoint.get("global_step", 0)),
            observation_dim=observation_dim,
            action_dim=action_dim,
            reference_ctrl_blend=teacher_blend,
            action_residual_scale=float(teacher_env.get("action_residual_scale", 0.5)),
            distal_action_scale=float(teacher_env.get("distal_action_scale", 1.0)),
            decimation=teacher_decimation,
            startup_action_ramp_steps=int(teacher_env.get("startup_action_ramp_steps", 0)),
            motion_file=teacher_motion_file,
        )

    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.metadata.observation_dim:
            raise ValueError(
                "Teacher received an invalid full-state observation shape: "
                f"{tuple(observation.shape)}."
            )
        return self.policy.act_inference(observation)
