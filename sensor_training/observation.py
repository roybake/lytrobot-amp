"""Sensor layouts and temporally coherent AMP reference histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class ImuSensorLayout:
    names: tuple[str, ...]
    addresses: np.ndarray
    divisors: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.addresses.size)


def resolve_imu_sensor_layout(
    model: mujoco.MjModel,
    *,
    accelerometer_scale: float,
    gyroscope_scale: float,
) -> ImuSensorLayout:
    """Return all accelerometer and gyro scalars in MJCF sensor order."""

    names: list[str] = []
    addresses: list[int] = []
    divisors: list[float] = []
    supported = {
        int(mujoco.mjtSensor.mjSENS_ACCELEROMETER): float(accelerometer_scale),
        int(mujoco.mjtSensor.mjSENS_GYRO): float(gyroscope_scale),
    }
    for sensor_id in range(model.nsensor):
        sensor_type = int(model.sensor_type[sensor_id])
        divisor = supported.get(sensor_type)
        if divisor is None:
            continue
        sensor_dim = int(model.sensor_dim[sensor_id])
        if sensor_dim != 3:
            raise ValueError(f"IMU sensor {sensor_id} must be 3D, got {sensor_dim}D.")
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        names.append(name or f"sensor_{sensor_id}")
        address = int(model.sensor_adr[sensor_id])
        addresses.extend(range(address, address + sensor_dim))
        divisors.extend([divisor] * sensor_dim)

    if not addresses:
        raise RuntimeError("No accelerometer or gyroscope sensors were found in the MJCF model.")
    return ImuSensorLayout(
        names=tuple(names),
        addresses=np.asarray(addresses, dtype=np.int64),
        divisors=np.asarray(divisors, dtype=np.float32),
    )


class HistoryMotionReference:
    """Expose time-stacked AMP reference features without changing MotionReference."""

    def __init__(self, source: Any, *, frame_count: int, policy_dt: float) -> None:
        if frame_count < 1:
            raise ValueError("frame_count must be at least one.")
        self.source = source
        self.history_frame_count = int(frame_count)
        self.policy_dt = float(policy_dt)
        self.feature_dim = int(source.feature_dim * self.history_frame_count)
        self.common_joint_names = source.common_joint_names

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    @staticmethod
    def _interpolate_table(table: np.ndarray, phases: np.ndarray) -> np.ndarray:
        frame_count = table.shape[0]
        # Keep phase arithmetic in float64. A float32 value immediately below
        # frame_count can round up to frame_count (35200 for the current motion),
        # which must wrap to index zero instead of becoming an invalid index.
        wrapped = np.remainder(np.asarray(phases, dtype=np.float64), float(frame_count))
        low_float = np.floor(wrapped)
        low = low_float.astype(np.int64) % frame_count
        high = (low + 1) % frame_count
        blend = (wrapped - low_float).astype(np.float32)[..., None]
        return ((1.0 - blend) * table[low] + blend * table[high]).astype(np.float32)

    def sample_features(
        self,
        batch_size: int,
        rng: np.random.Generator,
        mirror_prob: float = 0.5,
    ) -> np.ndarray:
        current_phase = rng.uniform(0.0, float(self.source.frame_count), size=(batch_size, 1))
        frame_offsets = np.arange(
            1 - self.history_frame_count,
            1,
            dtype=np.float64,
        )[None, :]
        reference_stride = float(self.source.frequency) * self.policy_dt
        phases = current_phase + frame_offsets * reference_stride

        regular = self._interpolate_table(self.source.reference_features, phases)
        probability = float(np.clip(mirror_prob, 0.0, 1.0))
        if probability > 0.0:
            mirrored = self._interpolate_table(self.source.mirrored_reference_features, phases)
            mirror_mask = rng.random(batch_size) < probability
            regular[mirror_mask] = mirrored[mirror_mask]
        return regular.reshape(batch_size, self.feature_dim).astype(np.float32)
