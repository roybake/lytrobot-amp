"""Observation configuration for the sensor-only policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorObservationConfig:
    """Current sensor frame plus a fixed number of previous policy frames."""

    history_steps: int = 5
    accelerometer_scale: float = 9.81
    gyroscope_scale: float = 10.0
    observation_clip: float = 20.0

    def __post_init__(self) -> None:
        if self.history_steps < 0:
            raise ValueError("history_steps must be non-negative.")
        if self.accelerometer_scale <= 0.0:
            raise ValueError("accelerometer_scale must be positive.")
        if self.gyroscope_scale <= 0.0:
            raise ValueError("gyroscope_scale must be positive.")
        if self.observation_clip <= 0.0:
            raise ValueError("observation_clip must be positive.")

    @property
    def frame_count(self) -> int:
        return self.history_steps + 1


DEFAULT_SENSOR_OBSERVATION_CONFIG = SensorObservationConfig()

