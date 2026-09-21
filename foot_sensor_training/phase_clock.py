"""Deployable gait phase clock shared by simulation wrappers and controllers.

The clock deliberately knows nothing about contacts, reference poses, motion
indices, or simulator state.  Its only state is the fixed reset phase and a
controller step counter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhaseClockConfig:
    frame_count: float
    frequency: float
    control_dt: float
    start_phase: float = 0.0
    stand_start_phase_delay_steps: int = 25
    stand_start_transition_steps: int = 120

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")
        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive")
        if self.stand_start_phase_delay_steps < 0:
            raise ValueError("stand_start_phase_delay_steps must be non-negative")
        if self.stand_start_transition_steps <= 0:
            raise ValueError("stand_start_transition_steps must be positive")

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "fields": ["gait_phase_sin", "gait_phase_cos"],
            "deployable": True,
            "source": "controller_internal_clock",
            "frame_count": float(self.frame_count),
            "frequency": float(self.frequency),
            "control_dt": float(self.control_dt),
            "start_phase": float(self.start_phase),
            "stand_start_phase_delay_steps": int(self.stand_start_phase_delay_steps),
            "transition_schedule": {
                "type": "delayed_smoothstep",
                "transition_steps": int(self.stand_start_transition_steps),
            },
            "reference_tendon_input": False,
            "phase_contact_correction_enabled": False,
        }


class DeployablePhaseClock:
    """Fixed-reset frame clock implementing ``phi += frequency*dt*alpha``."""

    def __init__(
        self,
        frame_count: float,
        frequency: float,
        control_dt: float,
        start_phase: float = 0.0,
        *,
        num_envs: int = 1,
        stand_start_phase_delay_steps: int = 25,
        stand_start_transition_steps: int = 120,
        torch_device=None,
    ) -> None:
        self.config = PhaseClockConfig(
            frame_count=frame_count,
            frequency=frequency,
            control_dt=control_dt,
            start_phase=start_phase,
            stand_start_phase_delay_steps=stand_start_phase_delay_steps,
            stand_start_transition_steps=stand_start_transition_steps,
        )
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self._torch = torch_device is not None
        if self._torch:
            import torch

            self.phase = torch.full(
                (self.num_envs,), float(start_phase), dtype=torch.float32, device=torch_device
            )
            self.step_count = torch.zeros(
                self.num_envs, dtype=torch.long, device=torch_device
            )
        else:
            self.phase = np.full(self.num_envs, float(start_phase), dtype=np.float32)
            self.step_count = np.zeros(self.num_envs, dtype=np.int64)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.config.metadata()

    def _ids(self, env_ids=None):
        if env_ids is None:
            return slice(None)
        if self._torch:
            import torch

            return env_ids if torch.is_tensor(env_ids) else torch.as_tensor(
                env_ids, dtype=torch.long, device=self.phase.device
            )
        return np.asarray(env_ids, dtype=np.int64)

    def reset(self, env_ids=None) -> None:
        ids = self._ids(env_ids)
        self.phase[ids] = float(self.config.start_phase)
        self.step_count[ids] = 0

    def startup_progress(self, env_ids=None):
        ids = self._ids(env_ids)
        steps = self.step_count[ids]
        delay = self.config.stand_start_phase_delay_steps
        transition = float(self.config.stand_start_transition_steps)
        if self._torch:
            raw=((steps.to(self.phase.dtype) - delay) / transition).clamp(0.0, 1.0)
            return raw*raw*(3.0-2.0*raw)
        raw=np.clip((steps.astype(np.float32) - delay) / transition, 0.0, 1.0)
        return raw*raw*(3.0-2.0*raw)

    def observe(self):
        angle_scale = 2.0 * np.pi / self.config.frame_count
        if self._torch:
            import torch

            angle = self.phase * angle_scale
            return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)
        angle = self.phase * angle_scale
        return np.stack((np.sin(angle), np.cos(angle)), axis=1).astype(np.float32)

    def advance(self, progress=None, env_ids=None) -> None:
        ids = self._ids(env_ids)
        if progress is None:
            progress = self.startup_progress(env_ids)
        elif self._torch:
            import torch

            progress = torch.as_tensor(progress, dtype=self.phase.dtype, device=self.phase.device)
        else:
            progress = np.asarray(progress, dtype=np.float32)
        increment = self.config.frequency * self.config.control_dt * progress
        if self._torch:
            import torch

            self.phase[ids] = torch.remainder(
                self.phase[ids] + increment, self.config.frame_count
            )
        else:
            self.phase[ids] = np.mod(
                self.phase[ids] + increment, self.config.frame_count
            ).astype(np.float32)
        self.step_count[ids] += 1
