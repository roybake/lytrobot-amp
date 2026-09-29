"""Deployable clock and reference commands for sensor-only gait policies.

The command contains no simulator-only measurement.  It is generated from a
monotonic control-loop clock and a normalized reference-tendon lookup table,
so the same implementation can run beside the exported ONNX policy on the
robot controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


GAIT_COMMAND_SCALAR_FIELDS = (
    "gait_phase_sin",
    "gait_phase_cos",
    "gait_start_progress",
    "target_velocity",
)


def gait_command_dim(
    action_dim: int, lookahead_seconds: Sequence[float],
    *, trajectory_feedback: bool = False,
) -> int:
    """Return scalar clock fields plus one tendon target per lookahead."""

    return (
        len(GAIT_COMMAND_SCALAR_FIELDS)
        + int(action_dim) * len(tuple(lookahead_seconds))
        + (2 if trajectory_feedback else 0)
    )


def smooth_start_progress(step: int, delay_steps: int, transition_steps: int) -> float:
    """Cubic smoothstep used by simulation and the real-time controller."""

    transition = max(int(transition_steps), 1)
    raw = np.clip((int(step) - max(int(delay_steps), 0)) / transition, 0.0, 1.0)
    return float(raw * raw * (3.0 - 2.0 * raw))


def interpolate_reference_table(
    table: np.ndarray,
    phase: np.ndarray | float,
    *,
    loop: bool = True,
) -> np.ndarray:
    """Linearly sample a [frame, value] table at fractional frame phases."""

    values = np.asarray(table, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("reference table must have shape [frame, value].")
    phases = np.asarray(phase, dtype=np.float64)
    frame_count = values.shape[0]
    bounded = (
        np.remainder(phases, float(frame_count))
        if loop
        else np.clip(phases, 0.0, float(frame_count - 1))
    )
    low_float = np.floor(bounded)
    low = low_float.astype(np.int64)
    high = (low + 1) % frame_count if loop else np.minimum(low + 1, frame_count - 1)
    blend = (bounded - low_float).astype(np.float32)
    while blend.ndim < values[low].ndim:
        blend = blend[..., None]
    return ((1.0 - blend) * values[low] + blend * values[high]).astype(np.float32)


def assemble_gait_command(
    *,
    phase: np.ndarray,
    frame_count: np.ndarray,
    progress: np.ndarray,
    target_velocity: np.ndarray,
    reference_tendon_targets: Sequence[np.ndarray],
    trajectory_error: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble batched commands in the exact Actor input order."""

    phase_arr = np.asarray(phase, dtype=np.float32).reshape(-1)
    frame_count_arr = np.maximum(np.asarray(frame_count, dtype=np.float32).reshape(-1), 1.0)
    progress_arr = np.asarray(progress, dtype=np.float32).reshape(-1)
    velocity_arr = np.asarray(target_velocity, dtype=np.float32).reshape(-1)
    if not (
        phase_arr.shape
        == frame_count_arr.shape
        == progress_arr.shape
        == velocity_arr.shape
    ):
        raise ValueError("phase, frame_count, progress and target_velocity must have equal batch size.")
    angle = 2.0 * np.pi * phase_arr / frame_count_arr
    scalar = np.stack(
        (np.sin(angle), np.cos(angle), progress_arr, velocity_arr),
        axis=1,
    ).astype(np.float32)
    tendon = tuple(np.asarray(value, dtype=np.float32) for value in reference_tendon_targets)
    if any(value.ndim != 2 or value.shape[0] != phase_arr.shape[0] for value in tendon):
        raise ValueError("every reference tendon target must have shape [batch, action].")
    parts = (scalar, *tendon)
    if trajectory_error is not None:
        error = np.asarray(trajectory_error, dtype=np.float32)
        if error.shape != (phase_arr.shape[0], 2):
            raise ValueError("trajectory_error must have shape [batch, 2].")
        parts = (*parts, error)
    return np.concatenate(parts, axis=1).astype(np.float32)


@dataclass
class DeployableGaitCommandGenerator:
    """Single-robot command generator reconstructed from ONNX metadata.

    Call :meth:`command` for the reset observation.  After executing each
    policy action for one control period, call :meth:`advance`, then call
    :meth:`command` again for the next observation.
    """

    reference_tendon_table: np.ndarray
    stand_tendon_target: np.ndarray
    frequency: float
    control_dt: float
    target_velocity: float
    target_velocity_normalizer: float = 3.0
    lookahead_seconds: tuple[float, ...] = (0.0, 0.1, 0.2)
    start_phase: float = 0.0
    delay_steps: int = 50
    transition_steps: int = 325
    action_ramp_steps: int = 50
    ctrl_mid: np.ndarray | None = None
    ctrl_half_range: np.ndarray | None = None
    actuator_action_scale: np.ndarray | None = None
    action_residual_scale: float = 1.0
    reference_ctrl_blend: float = 0.0
    loop: bool = True
    reference_root_xy_table: np.ndarray | None = None
    stand_root_xy: np.ndarray | None = None
    reference_velocity_table: np.ndarray | None = None
    start_in_motion: bool = False

    def __post_init__(self) -> None:
        self.reference_tendon_table = np.asarray(self.reference_tendon_table, dtype=np.float32)
        if self.reference_root_xy_table is not None:
            self.reference_root_xy_table = np.asarray(self.reference_root_xy_table, dtype=np.float32)
            if self.reference_root_xy_table.shape != (self.reference_tendon_table.shape[0], 2):
                raise ValueError("reference_root_xy_table must have shape [frame, 2].")
            if self.stand_root_xy is None:
                raise ValueError("stand_root_xy is required with trajectory feedback.")
            self.stand_root_xy = np.asarray(self.stand_root_xy, dtype=np.float32).reshape(2)
        if self.reference_velocity_table is not None:
            self.reference_velocity_table = np.asarray(self.reference_velocity_table, dtype=np.float32).reshape(-1)
            if len(self.reference_velocity_table) != self.reference_tendon_table.shape[0]:
                raise ValueError("reference_velocity_table must match the reference length.")
        self.stand_tendon_target = np.asarray(self.stand_tendon_target, dtype=np.float32).reshape(-1)
        if self.reference_tendon_table.ndim != 2:
            raise ValueError("reference_tendon_table must have shape [frame, action].")
        if self.reference_tendon_table.shape[1] != self.stand_tendon_target.shape[0]:
            raise ValueError("stand tendon target must match reference action dimension.")
        if self.frequency <= 0.0 or self.control_dt <= 0.0:
            raise ValueError("frequency and control_dt must be positive.")
        if self.target_velocity_normalizer <= 0.0:
            raise ValueError("target_velocity_normalizer must be positive.")
        if not self.lookahead_seconds or any(value < 0.0 for value in self.lookahead_seconds):
            raise ValueError("lookahead_seconds must contain non-negative values.")
        if self.ctrl_mid is not None:
            self.ctrl_mid = np.asarray(self.ctrl_mid, dtype=np.float32).reshape(-1)
            self.ctrl_half_range = np.asarray(self.ctrl_half_range, dtype=np.float32).reshape(-1)
            self.actuator_action_scale = np.asarray(self.actuator_action_scale, dtype=np.float32).reshape(-1)
            for value in (self.ctrl_mid, self.ctrl_half_range, self.actuator_action_scale):
                if value.shape != self.stand_tendon_target.shape:
                    raise ValueError("actuator control metadata must match tendon dimension.")
            if np.any(self.ctrl_half_range <= 0.0):
                raise ValueError("ctrl_half_range must be positive.")
        if not 0.0 <= self.reference_ctrl_blend <= 1.0:
            raise ValueError("reference_ctrl_blend must be in [0, 1].")
        self.reset()

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, Any],
        *,
        transition_steps: int | None = None,
    ) -> "DeployableGaitCommandGenerator":
        sensor = metadata.get("sensor_observation", metadata)
        gait = sensor.get("gait_command", sensor)
        tables = gait.get("normalized_reference_tendon_tables")
        if not isinstance(tables, list) or len(tables) != 1:
            raise ValueError("Deployment helper currently requires exactly one gait reference table.")
        frequencies = gait.get("frequencies")
        loops = gait.get("motion_loop")
        if not isinstance(frequencies, list) or len(frequencies) != 1:
            raise ValueError("gait metadata must contain exactly one reference frequency.")
        configured_transition = transition_steps
        if configured_transition is None:
            low = int(gait["stand_start_transition_min_steps"])
            high = int(gait["stand_start_transition_max_steps"])
            configured_transition = round((low + high) / 2)
        return cls(
            reference_tendon_table=np.asarray(tables[0], dtype=np.float32),
            stand_tendon_target=np.asarray(gait["normalized_stand_tendon_target"], dtype=np.float32),
            frequency=float(frequencies[0]),
            control_dt=float(gait["control_dt"]),
            target_velocity=float(gait["target_velocity"]),
            target_velocity_normalizer=float(gait["target_velocity_normalizer"]),
            lookahead_seconds=tuple(float(value) for value in gait["lookahead_seconds"]),
            start_phase=float(gait["stand_start_motion_phase"]),
            delay_steps=int(gait["stand_start_phase_delay_steps"]),
            transition_steps=int(configured_transition),
            action_ramp_steps=int(gait.get("startup_action_ramp_steps", 0)),
            ctrl_mid=np.asarray(gait["ctrl_mid"], dtype=np.float32),
            ctrl_half_range=np.asarray(gait["ctrl_half_range"], dtype=np.float32),
            actuator_action_scale=np.asarray(gait["actuator_action_scale"], dtype=np.float32),
            action_residual_scale=float(gait["action_residual_scale"]),
            reference_ctrl_blend=float(gait["reference_ctrl_blend"]),
            loop=bool(loops[0]) if isinstance(loops, list) else True,
            reference_root_xy_table=(
                np.asarray(gait["reference_root_xy_tables"][0], dtype=np.float32)
                if gait.get("trajectory_feedback") else None
            ),
            stand_root_xy=(
                np.asarray(gait["stand_root_xy"], dtype=np.float32)
                if gait.get("trajectory_feedback") else None
            ),
            reference_velocity_table=(
                np.asarray(gait["reference_root_velocity_tables"][0], dtype=np.float32)
                if gait.get("use_reference_velocity") else None
            ),
            start_in_motion=float(gait.get("stand_start_prob", 1.0)) <= 0.0,
        )

    @property
    def frame_count(self) -> int:
        return int(self.reference_tendon_table.shape[0])

    @property
    def progress(self) -> float:
        if self.start_in_motion:
            return 1.0
        return smooth_start_progress(self.step, self.delay_steps, self.transition_steps)

    @property
    def action_scale(self) -> float:
        """Scale the next ONNX action exactly like the training environment."""

        if self.action_ramp_steps <= 0:
            return 1.0
        return float(np.clip((self.step + 1) / self.action_ramp_steps, 0.0, 1.0))

    def tendon_control_target(
        self,
        action: np.ndarray,
        *,
        measured_tendon_length: np.ndarray | None = None,
    ) -> np.ndarray:
        """Physical tendon-length target for the upcoming 100 Hz control step.

        Simulation advances phase before applying the action, so this samples
        the reference at the next phase but uses the current startup progress.
        The returned target already includes the startup action ramp; do not
        apply :attr:`action_scale` a second time.
        """

        if self.ctrl_mid is None or self.ctrl_half_range is None or self.actuator_action_scale is None:
            raise ValueError("actuator control metadata is required to produce tendon targets.")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != self.stand_tendon_target.shape:
            raise ValueError("action dimension does not match tendon reference.")
        progress = self.progress
        next_phase = self.phase + self.frequency * self.control_dt * progress
        walk = interpolate_reference_table(
            self.reference_tendon_table, next_phase, loop=self.loop
        ).reshape(-1)
        reference_normalized = (1.0 - progress) * self.stand_tendon_target + progress * walk
        reference = self.ctrl_mid + reference_normalized * self.ctrl_half_range
        if self.reference_ctrl_blend < 1.0:
            if measured_tendon_length is None:
                raise ValueError("measured_tendon_length is required when reference_ctrl_blend < 1.")
            measured = np.asarray(measured_tendon_length, dtype=np.float32).reshape(-1)
            if measured.shape != action.shape:
                raise ValueError("measured tendon dimension does not match action.")
            measured = np.clip(measured, self.ctrl_mid - self.ctrl_half_range, self.ctrl_mid + self.ctrl_half_range)
            reference = (1.0 - self.reference_ctrl_blend) * measured + self.reference_ctrl_blend * reference
        target = reference + (
            np.clip(action, -1.0, 1.0)
            * self.action_scale
            * self.actuator_action_scale
            * self.action_residual_scale
            * self.ctrl_half_range
        )
        return np.clip(
            target, self.ctrl_mid - self.ctrl_half_range, self.ctrl_mid + self.ctrl_half_range
        ).astype(np.float32)

    def reset(self) -> None:
        self.phase = (
            float(np.remainder(self.start_phase, self.reference_tendon_table.shape[0]))
            if self.loop
            else float(np.clip(self.start_phase, 0.0, self.reference_tendon_table.shape[0] - 1.0))
        )
        self.step = 0

    def command(
        self, target_velocity: float | None = None, *, root_xy: np.ndarray | None = None
    ) -> np.ndarray:
        progress = self.progress
        targets: list[np.ndarray] = []
        for lookahead in self.lookahead_seconds:
            future_phase = self.phase + self.frequency * float(lookahead) * progress
            walk = interpolate_reference_table(
                self.reference_tendon_table,
                future_phase,
                loop=self.loop,
            ).reshape(-1)
            targets.append(
                ((1.0 - progress) * self.stand_tendon_target + progress * walk)[None, :]
            )
        velocity = self.target_velocity if target_velocity is None else float(target_velocity)
        if target_velocity is None and self.reference_velocity_table is not None:
            velocity = float(interpolate_reference_table(
                self.reference_velocity_table[:, None], self.phase, loop=self.loop
            ).item())
        trajectory_error = None
        if self.reference_root_xy_table is not None:
            if root_xy is None:
                raise ValueError("root_xy odometry is required for trajectory feedback.")
            target_xy = interpolate_reference_table(
                self.reference_root_xy_table, self.phase, loop=self.loop
            ).reshape(2)
            target_xy = (1.0 - progress) * self.stand_root_xy + progress * target_xy
            trajectory_error = (
                target_xy - np.asarray(root_xy, dtype=np.float32).reshape(2)
            )[None, :]
        return assemble_gait_command(
            phase=np.asarray([self.phase], dtype=np.float32),
            frame_count=np.asarray([self.frame_count], dtype=np.float32),
            progress=np.asarray([progress], dtype=np.float32),
            target_velocity=np.asarray(
                [velocity * progress / self.target_velocity_normalizer],
                dtype=np.float32,
            ),
            reference_tendon_targets=targets,
            trajectory_error=trajectory_error,
        )[0]

    def advance(self) -> None:
        self.phase += self.frequency * self.control_dt * self.progress
        if self.loop:
            self.phase = float(np.remainder(self.phase, self.frame_count))
        else:
            self.phase = float(np.clip(self.phase, 0.0, self.frame_count - 1.0))
        self.step += 1
