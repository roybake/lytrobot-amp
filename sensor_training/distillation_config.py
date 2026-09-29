"""Configuration used only by privileged-teacher distillation."""

from __future__ import annotations

from dataclasses import dataclass

from .ppo_config import SensorPPOAMPConfig


@dataclass
class SensorDistillationPPOAMPConfig(SensorPPOAMPConfig):
    """Sensor policy settings that must never leak into direct training."""

    amp_startup_gate: bool = False
    teacher_checkpoint: str | None = None
    teacher_action_loss_coef: float = 0.0
    teacher_action_loss_final_coef: float = 0.0
    teacher_action_loss_decay_iterations: int = 0
    distillation_reset_schedule: bool = False
