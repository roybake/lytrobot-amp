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

    include_tendon_velocity: bool = False
    include_tendon_tracking_error: bool = False
    tendon_velocity_scale: float = 1.0  # m/s, before clipping
    tendon_velocity_filter_alpha: float = 1.0  # causal one-pole filter
    tendon_tracking_error_scale: float = 0.05  # m

    # Physical encoder model: metres. Baseline noise applies even without DR;
    # other uncertainties ramp with the environment DR curriculum.
    tendon_measurement_enabled: bool = False
    tendon_measurement_noise_std: float = 0.0
    tendon_measurement_dr_noise_std: float = 0.0
    tendon_measurement_bias_m: float = 0.0
    tendon_measurement_bias_range_m: float = 0.0
    tendon_measurement_scale_error: float = 0.0
    tendon_measurement_quantization_m: float = 0.0
    tendon_measurement_delay_max_ticks: int = 0
    tendon_measurement_clip_margin_m: float = 0.02
    # IMU uncertainties use normalized accelerometer/gyroscope units.
    imu_measurement_noise_std: float = 0.0
    imu_measurement_dr_noise_std: float = 0.0
    imu_measurement_bias_range: float = 0.0
    imu_measurement_delay_max_ticks: int = 0

    def __post_init__(self) -> None:
        for name in ('tendon_measurement_noise_std', 'tendon_measurement_dr_noise_std',
                     'tendon_measurement_bias_range_m', 'tendon_measurement_scale_error',
                     'tendon_measurement_quantization_m', 'tendon_measurement_clip_margin_m',
                     'imu_measurement_noise_std', 'imu_measurement_dr_noise_std', 'imu_measurement_bias_range'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be non-negative.')
        for name in ('tendon_measurement_delay_max_ticks', 'imu_measurement_delay_max_ticks'):
            if not 0 <= getattr(self, name) <= 2:
                raise ValueError(f'{name} must be 0, 1 or 2.')
        if self.tendon_velocity_scale <= 0 or self.tendon_tracking_error_scale <= 0:
            raise ValueError("Tendon feature scales must be positive.")
        if not 0 < self.tendon_velocity_filter_alpha <= 1:
            raise ValueError("Tendon velocity filter alpha must be in (0, 1].")
        if self.include_tendon_tracking_error and (
            not self.gait_command_enabled or not self.gait_lookahead_seconds
            or self.gait_lookahead_seconds[0] != 0.0
        ):
            raise ValueError("Tracking error requires a current onboard tendon command.")
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
