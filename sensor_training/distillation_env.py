"""Environment adapter used exclusively by teacher distillation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from lytrobot_amp.env import LytRobotEnvCfg

from .config import DEFAULT_SENSOR_OBSERVATION_CONFIG, SensorObservationConfig
from .envs import SensorHistoryWarpAmpEnv


def remap_teacher_position_action(
    teacher_action: torch.Tensor,
    *,
    current_ctrl: torch.Tensor,
    reference_ctrl: torch.Tensor,
    ctrl_low: torch.Tensor,
    ctrl_high: torch.Tensor,
    ctrl_half_range: torch.Tensor,
    teacher_reference_blend: float,
    teacher_action_scale: torch.Tensor,
    teacher_residual_scale: float,
    student_reference_blend: float,
    student_action_scale: torch.Tensor,
    student_residual_scale: float,
) -> torch.Tensor:
    """Map a Teacher residual to the Student's physical actuator semantics."""

    teacher_blend = float(np.clip(teacher_reference_blend, 0.0, 1.0))
    student_blend = float(np.clip(student_reference_blend, 0.0, 1.0))
    teacher_base = (1.0 - teacher_blend) * current_ctrl + teacher_blend * reference_ctrl
    teacher_target = teacher_base + (
        torch.clamp(teacher_action, -1.0, 1.0)
        * ctrl_half_range
        * teacher_action_scale
        * float(teacher_residual_scale)
    )
    teacher_target = torch.clamp(teacher_target, min=ctrl_low, max=ctrl_high)
    student_base = (1.0 - student_blend) * current_ctrl + student_blend * reference_ctrl
    student_range = (
        ctrl_half_range * student_action_scale * float(student_residual_scale)
    ).clamp_min(1.0e-6)
    return torch.clamp((teacher_target - student_base) / student_range, -1.0, 1.0)


class SensorDistillationWarpAmpEnv(SensorHistoryWarpAmpEnv):
    """Warp sensor environment that records labels only for distillation runs."""

    def __init__(
        self,
        cfg: LytRobotEnvCfg,
        device: str = "cuda",
        observation_cfg: SensorObservationConfig = DEFAULT_SENSOR_OBSERVATION_CONFIG,
    ) -> None:
        super().__init__(cfg, device=device, observation_cfg=observation_cfg)
        self._teacher_labeler: Callable[[], torch.Tensor] | None = None
        self._teacher_action_labels: list[torch.Tensor] = []

    @property
    def teacher_obs_dim(self) -> int:
        return int(self.base_env.obs_dim)

    def teacher_observe(self) -> torch.Tensor:
        """Observe the real Student-visited state, including its last action."""

        return self.base_env.observe()

    def teacher_action_ramp(self, ramp_steps: int) -> torch.Tensor:
        if ramp_steps <= 0:
            return torch.ones(
                (self.num_envs, 1), dtype=torch.float32, device=self.base_env.device
            )
        ramp = (self.base_env.episode_step.to(torch.float32) + 1.0) / float(ramp_steps)
        return torch.clamp(ramp, 0.0, 1.0)[:, None]

    def _next_teacher_reference_ctrl(self) -> torch.Tensor:
        base = self.base_env
        if base.more_terrain_cfg.serial_course:
            return base._targets_at_current_phase()["tendon_ctrl"]
        phase_scale = base._stand_start_progress()
        frame_counts = base.motion_frame_counts[base.motion_ids].to(torch.float32)
        next_phase = (
            base.motion_phase
            + base.motion_frequencies[base.motion_ids] * base.dt * phase_scale
        )
        looping = base.motion_loop[base.motion_ids]
        next_phase = torch.where(
            looping,
            torch.remainder(next_phase, frame_counts),
            torch.minimum(next_phase, frame_counts - 1.0),
        )
        targets = base._targets_at_phase(next_phase, base.motion_ids)
        return base._blend_start_targets(targets)["tendon_ctrl"]

    def remap_teacher_action(
        self,
        teacher_action: torch.Tensor,
        *,
        teacher_reference_blend: float,
        teacher_action_residual_scale: float,
        teacher_distal_action_scale: float,
    ) -> torch.Tensor:
        base = self.base_env
        current_ctrl = base._current_tendon_ctrl()
        teacher_scale = torch.ones_like(base.action_scale)
        teacher_scale[base.distal_position_action_mask] = float(
            np.clip(teacher_distal_action_scale, 0.05, 1.0)
        )
        return remap_teacher_position_action(
            teacher_action,
            current_ctrl=current_ctrl,
            reference_ctrl=self._next_teacher_reference_ctrl(),
            ctrl_low=base.ctrl_low,
            ctrl_high=base.ctrl_high,
            ctrl_half_range=base.ctrl_half_range,
            teacher_reference_blend=teacher_reference_blend,
            teacher_action_scale=teacher_scale,
            teacher_residual_scale=teacher_action_residual_scale,
            student_reference_blend=float(base.cfg.reference_ctrl_blend),
            student_action_scale=base.action_scale,
            student_residual_scale=float(base.cfg.action_residual_scale),
        )

    def begin_teacher_rollout(self, labeler: Callable[[], torch.Tensor]) -> None:
        if self._teacher_labeler is not None:
            raise RuntimeError("A Teacher rollout is already active.")
        self._teacher_labeler = labeler
        self._teacher_action_labels = []

    def end_teacher_rollout(self) -> torch.Tensor:
        labels = self._teacher_action_labels
        self._teacher_labeler = None
        self._teacher_action_labels = []
        if not labels:
            return torch.empty(
                (0, self.num_envs, self.action_dim),
                dtype=torch.float32,
                device=self.base_env.device,
            )
        return torch.stack(labels)

    def step(self, action: np.ndarray | torch.Tensor):
        # Capture the label before dynamics advance so it corresponds exactly
        # to the observation stored by the ordinary rollout collector.
        if self._teacher_labeler is not None:
            self._teacher_action_labels.append(self._teacher_labeler().detach().clone())
        return super().step(action)
