"""Speed/yaw commands and heading coordinates shared by training and playback."""

from __future__ import annotations

import numpy as np
import torch


SPEED_ACCELERATION = 0.30  # m/s²
YAW_ACCELERATION = 0.80  # rad/s²


def heading(quat):
    xp = torch if torch.is_tensor(quat) else np
    w, x, y, z = (quat[..., i] for i in range(4))
    return xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def without_heading(quat):
    xp = torch if torch.is_tensor(quat) else np
    half = heading(quat) * 0.5
    c, s = xp.cos(half), xp.sin(half)
    w, x, y, z = (quat[..., i] for i in range(4))
    result = xp.stack((c * w + s * z, c * x + s * y,
                       c * y - s * x, c * z - s * w), axis=-1)
    return xp.where(result[..., :1] < 0, -result, result)


def heading_velocity(quat, velocity):
    """Free-joint translation is world-frame; angular velocity is body-frame."""
    xp = torch if torch.is_tensor(quat) else np
    angle = heading(quat)
    c, s = xp.cos(angle), xp.sin(angle)
    return xp.stack((c * velocity[..., 0] + s * velocity[..., 1],
                     -s * velocity[..., 0] + c * velocity[..., 1],
                     velocity[..., 2]), axis=-1)


def world_yaw_rate(quat, angular_velocity):
    w, x, y, z = (quat[..., i] for i in range(4))
    return (2 * (x * z - w * y) * angular_velocity[..., 0]
            + 2 * (y * z + w * x) * angular_velocity[..., 1]
            + (1 - 2 * (x * x + y * y)) * angular_velocity[..., 2])


def heading_angular_velocity(quat, angular_velocity):
    """Rotate free-joint body angular velocity into the yaw-aligned world frame."""
    xp = torch if torch.is_tensor(quat) else np
    tilt = without_heading(quat)
    w, x, y, z = (tilt[..., i] for i in range(4))
    vx, vy, vz = (angular_velocity[..., i] for i in range(3))
    return xp.stack((
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz,
        2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz,
        2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
    ), axis=-1)


class WalkingCommands:
    """Per-world continuous commands; sample new goals every 4–8 sim seconds."""

    def __init__(self, cfg, num_envs: int, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.values = torch.zeros((num_envs, 2), device=self.device)
        self.goals = torch.zeros_like(self.values)
        self.time_left = np.zeros(num_envs, dtype=np.float64)
        self.manual = False

    def _sample(self, ids, rng):
        count = len(ids)
        low, high = self.cfg.velocity_range
        speed = rng.uniform(low, high, count)
        # Explicitly train both closed-interval endpoints as well as its interior.
        speed_choice = rng.random(count)
        speed[speed_choice < 0.15] = low
        speed[speed_choice > 0.85] = high
        yaw_low, yaw_high = self.cfg.yaw_rate_range
        yaw = rng.uniform(yaw_low, yaw_high, count)
        yaw_choice = rng.random(count)
        yaw[yaw_choice < 0.35] = 0.0
        yaw[(yaw_choice >= 0.35) & (yaw_choice < 0.45)] = yaw_low
        yaw[yaw_choice > 0.90] = yaw_high
        self.goals[ids] = torch.as_tensor(
            np.stack((speed, yaw), axis=1), dtype=torch.float32, device=self.device)
        self.time_left[ids] = rng.uniform(4.0, 8.0, count)

    def reset(self, ids, rng):
        if self.manual:
            return
        self._sample(ids, rng)
        self.values[ids] = self.goals[ids]
        self.values[ids, 1] = 0.0

    def advance(self, dt: float, rng):
        if self.manual:
            return
        self.time_left -= dt
        ids = np.flatnonzero(self.time_left <= 0)
        if len(ids):
            self._sample(ids, rng)
        limits = self.values.new_tensor((SPEED_ACCELERATION * dt, YAW_ACCELERATION * dt))
        self.values += torch.clamp(self.goals - self.values, min=-limits, max=limits)
        self.values[:, 0].clamp_(*self.cfg.velocity_range)
        self.values[:, 1].clamp_(*self.cfg.yaw_rate_range)

    def set_manual(self, speed: float, yaw_rate: float):
        if not np.isfinite((speed, yaw_rate)).all():
            raise ValueError("Walking commands must be finite.")
        self.manual = True
        self.values[:, 0] = float(np.clip(speed, *self.cfg.velocity_range))
        self.values[:, 1] = float(np.clip(yaw_rate, *self.cfg.yaw_rate_range))
        self.goals.copy_(self.values)
