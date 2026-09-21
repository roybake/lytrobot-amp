"""Observation configuration for the sensor-only policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorObservationConfig:
    """Deployable sensor/command frame plus fixed causal history."""

    history_steps: int = 5
    accelerometer_scale: float = 9.81
    gyroscope_scale: float = 10.0
    observation_clip: float = 20.0
    gait_command_enabled: bool = True
    gait_lookahead_seconds: tuple[float, ...] = (0.0, 0.1, 0.2)
    target_velocity_normalizer: float = 3.0

    def __post_init__(self) -> None:
        if self.history_steps < 0:
            raise ValueError("history_steps must be non-negative.")
        if self.accelerometer_scale <= 0.0:
            raise ValueError("accelerometer_scale must be positive.")
        if self.gyroscope_scale <= 0.0:
            raise ValueError("gyroscope_scale must be positive.")
        if self.observation_clip <= 0.0:
            raise ValueError("observation_clip must be positive.")
        if self.gait_command_enabled:
            if not self.gait_lookahead_seconds:
                raise ValueError("gait_lookahead_seconds must not be empty.")
            if any(value < 0.0 for value in self.gait_lookahead_seconds):
                raise ValueError("gait lookahead values must be non-negative.")
            if tuple(sorted(self.gait_lookahead_seconds)) != self.gait_lookahead_seconds:
                raise ValueError("gait lookahead values must be sorted.")
        if self.target_velocity_normalizer <= 0.0:
            raise ValueError("target_velocity_normalizer must be positive.")

    @property
    def frame_count(self) -> int:
        return self.history_steps + 1


DEFAULT_SENSOR_OBSERVATION_CONFIG = SensorObservationConfig()
