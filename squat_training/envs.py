"""Squat environments using posture sensors and a controller-owned phase clock."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
from sensor_training.config import DEFAULT_SENSOR_OBSERVATION_CONFIG, SensorObservationConfig
from sensor_training.envs import SensorHistoryEnv


class SquatRewardMixin:
    """Task-specific reward composition for a planted, symmetric squat.

    The shared environments expose useful tracking features, but their default
    reward is designed for locomotion.  In particular, swing-foot velocity is a
    constant for an all-contact squat reference and the gait phase-symmetry term
    expects the feet to alternate front/back.  This mixin keeps the shared
    feature calculation while composing only squat-relevant terms.
    """

    # A completed cycle is worth 5 by default, while a fall costs 50.  This
    # deliberately makes survival the prerequisite rather than a trade that a
    # high tracking reward can buy back.
    squat_termination_penalty = 50.0

    def __init__(self, *args, **kwargs) -> None:
        # The shared environment performs an initial reset inside its
        # constructor.  Feedback is therefore marked ready only after the
        # complete MuJoCo/Warp state has been allocated.
        self._squat_phase_feedback_ready = False
        super().__init__(*args, **kwargs)
        # Lightweight reward-unit-test harnesses intentionally provide only
        # reward parts after construction and have no simulator state.
        if all(hasattr(self, name) for name in ("cfg", "motion", "ctrl_half_range")):
            self._initialize_squat_phase_feedback()

    def reset(self, env_ids=None):
        observation = super().reset(env_ids)
        if self._squat_phase_feedback_ready:
            self._reset_squat_task_state(env_ids)
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        episode_success = info.get("squat_episode_success")
        if episode_success is not None:
            time_outs = info.get("time_outs")
            if torch.is_tensor(done):
                assert torch.is_tensor(episode_success)
                assert torch.is_tensor(time_outs)
                info["successes"] = (
                    (done > 0.0)
                    & (time_outs > 0.0)
                    & (episode_success > 0.0)
                ).to(torch.float32)
            else:
                info["successes"] = (
                    (np.asarray(done) > 0.0)
                    & (np.asarray(time_outs) > 0.0)
                    & (np.asarray(episode_success) > 0.0)
                ).astype(np.float32)
        return observation, reward, done, info

    def _initialize_squat_phase_feedback(self) -> None:
        """Precompute a tendon-space estimate of physical squat depth.

        The supervisor deliberately avoids simulator root height.  Squat depth
        is projected from the same 14 tendon lengths available on hardware,
        while pelvis uprightness corresponds to an IMU attitude estimate.
        """

        cfg = self.cfg
        lag_tolerance = float(cfg.squat_phase_lag_tolerance)
        min_scale = float(cfg.squat_phase_min_scale)
        bottom_threshold = float(cfg.squat_bottom_depth_threshold)
        stand_threshold = float(cfg.squat_stand_depth_threshold)
        upright_threshold = float(cfg.squat_upright_threshold)
        if lag_tolerance < 0.0:
            raise ValueError("squat_phase_lag_tolerance must be non-negative.")
        if not 0.0 <= min_scale <= 1.0:
            raise ValueError("squat_phase_min_scale must be in [0, 1].")
        if not 0.0 <= stand_threshold < bottom_threshold <= 1.0:
            raise ValueError(
                "squat phase thresholds require 0 <= stand < bottom <= 1."
            )
        if not 0.0 <= upright_threshold <= 1.0:
            raise ValueError("squat_upright_threshold must be in [0, 1].")
        stall_threshold = float(cfg.squat_phase_stall_threshold)
        max_stall_steps = int(cfg.squat_phase_max_stall_steps)
        stall_penalty = float(cfg.squat_phase_stall_penalty)
        slowdown_penalty = float(cfg.squat_phase_slowdown_penalty_coef)
        completion_bonus = float(cfg.squat_cycle_completion_bonus)
        if not 0.0 <= stall_threshold <= 1.0:
            raise ValueError("squat_phase_stall_threshold must be in [0, 1].")
        if max_stall_steps <= 0:
            raise ValueError("squat_phase_max_stall_steps must be positive.")
        if stall_penalty < 0.0:
            raise ValueError("squat_phase_stall_penalty must be non-negative.")
        if slowdown_penalty < 0.0:
            raise ValueError(
                "squat_phase_slowdown_penalty_coef must be non-negative."
            )
        if completion_bonus < 0.0:
            raise ValueError("squat_cycle_completion_bonus must be non-negative.")

        reference_ctrl = np.asarray(
            self.motion.reference_tendon_ctrl,
            dtype=np.float32,
        )
        bottom_index = int(np.argmin(self.motion.reference_root_height))
        stand_ctrl = reference_ctrl[0]
        half_range = np.asarray(
            self.ctrl_half_range.detach().cpu().numpy()
            if torch.is_tensor(self.ctrl_half_range)
            else self.ctrl_half_range,
            dtype=np.float32,
        )
        direction = (reference_ctrl[bottom_index] - stand_ctrl) / np.maximum(
            half_range, 1.0e-6
        )
        denominator = float(np.dot(direction, direction))
        if denominator <= 1.0e-8:
            raise ValueError(
                "Squat reference has no measurable stand-to-bottom tendon motion."
            )
        normalized_reference = (
            reference_ctrl - stand_ctrl[None, :]
        ) / np.maximum(half_range[None, :], 1.0e-6)
        depth_curve = np.clip(
            normalized_reference @ direction / denominator,
            0.0,
            1.0,
        ).astype(np.float32)

        self._squat_bottom_phase_fraction = bottom_index / float(
            max(self.motion.frame_count, 1)
        )
        self._squat_feedback_stand_ctrl_np = stand_ctrl.astype(np.float32)
        self._squat_feedback_half_range_np = half_range
        self._squat_feedback_direction_np = direction.astype(np.float32)
        self._squat_feedback_denominator = denominator
        self._squat_feedback_depth_curve_np = depth_curve
        if bool(getattr(self, "is_torch_env", False)):
            self._squat_feedback_stand_ctrl = torch.as_tensor(
                stand_ctrl, dtype=torch.float32, device=self.device
            )
            self._squat_feedback_half_range = torch.as_tensor(
                half_range, dtype=torch.float32, device=self.device
            )
            self._squat_feedback_direction = torch.as_tensor(
                direction, dtype=torch.float32, device=self.device
            )
            self._squat_feedback_depth_curve = torch.as_tensor(
                depth_curve, dtype=torch.float32, device=self.device
            )
            self.squat_phase_feedback_scale = torch.ones(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.squat_depth_sensor_progress = torch.zeros_like(
                self.squat_phase_feedback_scale
            )
            self.squat_phase_stalled = torch.zeros_like(
                self.squat_phase_feedback_scale, dtype=torch.bool
            )
            self.squat_phase_stall_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.squat_cycle_stage = torch.zeros_like(
                self.squat_phase_stall_steps
            )
            self.squat_cycle_count = torch.zeros_like(
                self.squat_phase_stall_steps
            )
        else:
            self.squat_phase_feedback_scale = np.ones(
                self.num_envs, dtype=np.float32
            )
            self.squat_depth_sensor_progress = np.zeros(
                self.num_envs, dtype=np.float32
            )
            self.squat_phase_stalled = np.zeros(self.num_envs, dtype=bool)
            self.squat_phase_stall_steps = np.zeros(
                self.num_envs, dtype=np.int32
            )
            self.squat_cycle_stage = np.zeros(self.num_envs, dtype=np.int32)
            self.squat_cycle_count = np.zeros(self.num_envs, dtype=np.int32)
        self._squat_phase_feedback_ready = True

    def _reset_squat_task_state(self, env_ids=None) -> None:
        if env_ids is None:
            if torch.is_tensor(self.squat_phase_stall_steps):
                self.squat_phase_stalled.zero_()
                self.squat_phase_stall_steps.zero_()
                self.squat_cycle_stage.zero_()
                self.squat_cycle_count.zero_()
            else:
                self.squat_phase_stalled.fill(False)
                self.squat_phase_stall_steps.fill(0)
                self.squat_cycle_stage.fill(0)
                self.squat_cycle_count.fill(0)
            return

        ids_np = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if torch.is_tensor(self.squat_phase_stall_steps):
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=self.device)
            self.squat_phase_stalled[ids] = False
            self.squat_phase_stall_steps[ids] = 0
            self.squat_cycle_stage[ids] = 0
            self.squat_cycle_count[ids] = 0
        else:
            self.squat_phase_stalled[ids_np] = False
            self.squat_phase_stall_steps[ids_np] = 0
            self.squat_cycle_stage[ids_np] = 0
            self.squat_cycle_count[ids_np] = 0

    def _record_squat_phase_stall(self, scale) -> None:
        threshold = float(self.cfg.squat_phase_stall_threshold)
        if torch.is_tensor(scale):
            stalled = scale < threshold
            self.squat_phase_stalled = stalled
            self.squat_phase_stall_steps += stalled.to(torch.long)
        else:
            stalled = np.asarray(scale) < threshold
            self.squat_phase_stalled = stalled
            self.squat_phase_stall_steps += stalled.astype(np.int32)

    def _update_squat_cycle_state(self, depth, upright, env_id: int = 0):
        """Track a physical stand -> bottom -> stand sequence with hysteresis."""

        stand_threshold = float(self.cfg.squat_stand_depth_threshold)
        bottom_threshold = float(self.cfg.squat_bottom_depth_threshold)
        upright_threshold = float(self.cfg.squat_upright_threshold)
        if torch.is_tensor(depth):
            standing = (depth <= stand_threshold) & (upright >= upright_threshold)
            at_bottom = (depth >= bottom_threshold) & (upright >= upright_threshold)
            stage = self.squat_cycle_stage
            completed = (stage == 2) & standing
            next_stage = torch.where((stage == 0) & standing, 1, stage)
            next_stage = torch.where((stage == 1) & at_bottom, 2, next_stage)
            next_stage = torch.where(completed, 1, next_stage)
            self.squat_cycle_stage = next_stage
            self.squat_cycle_count += completed.to(torch.long)
            self.squat_phase_stall_steps = torch.where(
                completed,
                torch.zeros_like(self.squat_phase_stall_steps),
                self.squat_phase_stall_steps,
            )
            return completed, self.squat_cycle_count > 0

        depth_values = np.asarray(depth).reshape(-1)
        upright_values = np.asarray(upright).reshape(-1)
        index = min(env_id, len(depth_values) - 1)
        standing = bool(
            depth_values[index] <= stand_threshold
            and upright_values[index] >= upright_threshold
        )
        at_bottom = bool(
            depth_values[index] >= bottom_threshold
            and upright_values[index] >= upright_threshold
        )
        stage = int(self.squat_cycle_stage[index])
        completed = stage == 2 and standing
        if completed:
            self.squat_cycle_stage[index] = 1
            self.squat_cycle_count[index] += 1
            self.squat_phase_stall_steps[index] = 0
        elif stage == 0 and standing:
            self.squat_cycle_stage[index] = 1
        elif stage == 1 and at_bottom:
            self.squat_cycle_stage[index] = 2
        return float(completed), float(self.squat_cycle_count[index] > 0)

    def _squat_feedback_measurements(self):
        """Return tendon-derived depth and IMU-equivalent pelvis uprightness."""

        if bool(getattr(self, "is_torch_env", False)):
            current = self._current_tendon_ctrl()
            normalized = (
                current - self._squat_feedback_stand_ctrl
            ) / self._squat_feedback_half_range
            depth = torch.clamp(
                torch.sum(normalized * self._squat_feedback_direction, dim=1)
                / self._squat_feedback_denominator,
                0.0,
                1.0,
            )
            # This rotation-matrix term is the simulated equivalent of the
            # gravity direction produced by the pelvis IMU attitude filter.
            upright = self.xmat[:, self.pelvis_body_id, 2, 2]
            return depth, upright

        current = np.stack(
            [self._current_tendon_ctrl(data) for data in self.datas],
            axis=0,
        )
        normalized = (
            current - self._squat_feedback_stand_ctrl_np[None, :]
        ) / self._squat_feedback_half_range_np[None, :]
        depth = np.clip(
            np.sum(
                normalized * self._squat_feedback_direction_np[None, :],
                axis=1,
            )
            / self._squat_feedback_denominator,
            0.0,
            1.0,
        ).astype(np.float32)
        upright = np.asarray(
            [data.xmat[self.pelvis_body_id, 8] for data in self.datas],
            dtype=np.float32,
        )
        return depth, upright

    def _squat_reference_depth_at_phase(self):
        frame_count = max(int(self.motion.frame_count), 1)
        if bool(getattr(self, "is_torch_env", False)):
            wrapped = torch.remainder(self.motion_phase, float(frame_count))
            low = torch.floor(wrapped).to(torch.long)
            high = torch.remainder(low + 1, frame_count)
            blend = wrapped - low.to(torch.float32)
            return (
                (1.0 - blend) * self._squat_feedback_depth_curve[low]
                + blend * self._squat_feedback_depth_curve[high]
            )

        wrapped = np.remainder(self.motion_phase, float(frame_count))
        low = np.floor(wrapped).astype(np.int64)
        high = (low + 1) % frame_count
        blend = wrapped - low.astype(np.float32)
        return (
            (1.0 - blend) * self._squat_feedback_depth_curve_np[low]
            + blend * self._squat_feedback_depth_curve_np[high]
        ).astype(np.float32)

    def _squat_feedback_phase_scale(self):
        """Slow/freeze phase when sensor motion lags its reference clock."""

        depth, upright = self._squat_feedback_measurements()
        self.squat_depth_sensor_progress = depth
        if not bool(self.cfg.squat_phase_feedback):
            if torch.is_tensor(depth):
                scale = torch.ones_like(depth)
            else:
                scale = np.ones_like(depth, dtype=np.float32)
            self.squat_phase_feedback_scale = scale
            self._record_squat_phase_stall(scale)
            return scale

        desired_depth = self._squat_reference_depth_at_phase()
        frame_count = max(int(self.motion.frame_count), 1)
        phase_fraction = self.motion_phase / float(frame_count)
        descending = phase_fraction < self._squat_bottom_phase_fraction
        lag = (
            torch.where(descending, desired_depth - depth, depth - desired_depth)
            if torch.is_tensor(depth)
            else np.where(descending, desired_depth - depth, depth - desired_depth)
        )
        tolerance = float(self.cfg.squat_phase_lag_tolerance)
        correction_band = max(tolerance, 0.02)
        min_scale = float(self.cfg.squat_phase_min_scale)
        boundary_margin = max(
            float(self.motion.frequency) * float(self.dt) / float(frame_count),
            1.0 / float(frame_count),
        ) * 1.05

        if torch.is_tensor(depth):
            scale = torch.clamp(
                1.0 - torch.clamp_min(lag - tolerance, 0.0) / correction_band,
                min=min_scale,
                max=1.0,
            )
            bottom_blocked = (
                phase_fraction
                >= self._squat_bottom_phase_fraction - boundary_margin
            ) & (
                phase_fraction
                <= self._squat_bottom_phase_fraction + boundary_margin
            ) & (depth < float(self.cfg.squat_bottom_depth_threshold))
            stand_blocked = (
                phase_fraction >= 1.0 - boundary_margin
            ) & (depth > float(self.cfg.squat_stand_depth_threshold))
            unstable = upright < float(self.cfg.squat_upright_threshold)
            scale = torch.where(
                bottom_blocked | stand_blocked | unstable,
                torch.zeros_like(scale),
                scale,
            )
        else:
            scale = np.clip(
                1.0 - np.maximum(lag - tolerance, 0.0) / correction_band,
                min_scale,
                1.0,
            ).astype(np.float32)
            bottom_blocked = (
                phase_fraction
                >= self._squat_bottom_phase_fraction - boundary_margin
            ) & (
                phase_fraction
                <= self._squat_bottom_phase_fraction + boundary_margin
            ) & (depth < float(self.cfg.squat_bottom_depth_threshold))
            stand_blocked = (
                phase_fraction >= 1.0 - boundary_margin
            ) & (depth > float(self.cfg.squat_stand_depth_threshold))
            unstable = upright < float(self.cfg.squat_upright_threshold)
            scale[bottom_blocked | stand_blocked | unstable] = 0.0

        self.squat_phase_feedback_scale = scale
        self._record_squat_phase_stall(scale)
        return scale

    def _motion_phase_progress(self):
        base_progress = super()._motion_phase_progress()
        if not self._squat_phase_feedback_ready:
            return base_progress
        return base_progress * self._squat_feedback_phase_scale()

    def _squat_phase_fraction(self, reward_value: float | torch.Tensor, env_id: int = 0):
        """Return the controller-owned cycle phase in [0, 1).

        The phase is generated by the controller rather than estimated from a
        robot sensor.  Keeping it explicit removes the stand/squat ambiguity at
        the zero-velocity turning points of the motion.
        """

        frame_count = max(int(self.motion.frame_count), 1)
        if torch.is_tensor(reward_value):
            return torch.remainder(self.motion_phase, frame_count) / float(frame_count)
        phase = float(np.asarray(self.motion_phase)[env_id])
        return float(np.remainder(phase, frame_count) / float(frame_count))

    def _symmetry_track(self, *args, **kwargs):
        (
            _locomotion_symmetry,
            _locomotion_foot_symmetry,
            stride_symmetry,
            phase_symmetry,
            lateral_symmetry,
            contact_balance,
            action_balance,
        ) = super()._symmetry_track(*args, **kwargs)

        # During a squat both planted feet should travel equally relative to the
        # pelvis and remain laterally mirrored.  Front/back gait phase symmetry
        # is deliberately retained only as a diagnostic and is not rewarded.
        foot_symmetry = 0.65 * stride_symmetry + 0.35 * lateral_symmetry
        foot_weight = max(float(self.cfg.symmetry_foot_weight), 0.0)
        contact_weight = max(float(self.cfg.symmetry_contact_weight), 0.0)
        action_weight = max(float(self.cfg.symmetry_action_weight), 0.0)
        weight_sum = max(foot_weight + contact_weight + action_weight, 1.0e-6)
        symmetry = (
            foot_weight * foot_symmetry
            + contact_weight * contact_balance
            + action_weight * action_balance
        ) / weight_sum
        return (
            symmetry,
            foot_symmetry,
            stride_symmetry,
            phase_symmetry,
            lateral_symmetry,
            contact_balance,
            action_balance,
        )

    def _compute_reward_done(self, *args, **kwargs):
        _locomotion_reward, done, timed_out, parts = super()._compute_reward_done(
            *args,
            **kwargs,
        )

        # Track the complete stand -> squat -> stand reference.  Pose and pelvis
        # height carry most of the motion score; velocity terms further separate
        # the descending and ascending dynamics.
        squat_track = (
            0.42 * parts["pose_track"]
            + 0.30 * parts["height"]
            + 0.08 * parts["root_orientation"]
            + 0.08 * parts["root_vel"]
            + 0.07 * parts["vel_track"]
            + 0.05 * parts["distal_vel"]
        )

        # A valid squat is double support throughout: both feet stay at their
        # reference locations, remain planted, and preserve ground contact.
        support_track = (
            0.45 * parts["foot_pos"]
            + 0.35 * parts["stance_foot"]
            + 0.20 * parts["contact"]
        )

        # A planted, symmetric crouch is safe but must not outscore completing
        # the reference.  Gate auxiliary rewards by motion tracking so they
        # cannot dominate when pose and pelvis height are wrong.
        motion_gate = 0.15 + 0.85 * squat_track

        # During the second half of the cycle the reference is rising.  Make a
        # missed stand-up explicitly costly instead of merely losing the height
        # bonus.  The smooth gate avoids a discontinuity at the bottom turn.
        cpu_env_id = args[3] if len(args) > 3 and not torch.is_tensor(parts["height"]) else 0
        env_id = int(kwargs.get("env_id", cpu_env_id))
        phase_fraction = self._squat_phase_fraction(parts["height"], env_id)
        if torch.is_tensor(parts["height"]):
            ascent_gate = torch.clamp((phase_fraction - 0.50) / 0.35, 0.0, 1.0)
        else:
            ascent_gate = float(np.clip((phase_fraction - 0.50) / 0.35, 0.0, 1.0))
        stand_track = ascent_gate * parts["height"]
        missed_stand = ascent_gate * (1.0 - parts["height"])

        # Gaussian tracking is precise near the target but saturated when a
        # robot remains far below the rising reference.  Normalized pelvis
        # progress gives PPO a dense recovery signal throughout the ascent.
        root_height = parts["root_height_value"]
        reference_root_height = parts["reference_root_height_value"]
        stand_height = float(np.max(self.motion.reference_root_height))
        bottom_height = float(np.min(self.motion.reference_root_height))
        squat_depth_range = max(stand_height - bottom_height, 1.0e-6)
        if torch.is_tensor(root_height):
            squat_depth_progress = torch.clamp(
                (stand_height - root_height) / squat_depth_range, 0.0, 1.0
            )
            reference_depth_progress = torch.clamp(
                (stand_height - reference_root_height) / squat_depth_range, 0.0, 1.0
            )
            bottom_gate = torch.clamp(
                1.0 - torch.abs(phase_fraction - 0.50) / 0.30, 0.0, 1.0
            )
            bottom_depth_track = bottom_gate * torch.clamp(
                1.0 - torch.abs(squat_depth_progress - reference_depth_progress), 0.0, 1.0
            )
            bottom_depth_deficit = bottom_gate * torch.clamp(
                reference_depth_progress - squat_depth_progress, 0.0, 1.0
            )
            bottom_depth_success = bottom_gate * (squat_depth_progress >= 0.85).to(torch.float32)
        else:
            squat_depth_progress = float(np.clip(
                (stand_height - root_height) / squat_depth_range, 0.0, 1.0
            ))
            reference_depth_progress = float(np.clip(
                (stand_height - reference_root_height) / squat_depth_range, 0.0, 1.0
            ))
            bottom_gate = float(np.clip(
                1.0 - abs(phase_fraction - 0.50) / 0.30, 0.0, 1.0
            ))
            bottom_depth_track = bottom_gate * float(np.clip(
                1.0 - abs(squat_depth_progress - reference_depth_progress), 0.0, 1.0
            ))
            bottom_depth_deficit = bottom_gate * float(np.clip(
                reference_depth_progress - squat_depth_progress, 0.0, 1.0
            ))
            bottom_depth_success = bottom_gate * float(squat_depth_progress >= 0.85)

        progress_floor = float(self.cfg.fall_height)
        progress_range = max(stand_height - progress_floor, 1.0e-6)
        if torch.is_tensor(root_height):
            height_progress = torch.clamp(
                (root_height - progress_floor) / progress_range, 0.0, 1.0
            )
            reference_height_progress = torch.clamp(
                (reference_root_height - progress_floor) / progress_range, 0.0, 1.0
            )
            ascent_progress_track = ascent_gate * torch.clamp(
                1.0 - torch.abs(height_progress - reference_height_progress), 0.0, 1.0
            )
            ascent_height_lag = ascent_gate * torch.clamp(
                reference_height_progress - height_progress, 0.0, 1.0
            )
        else:
            height_progress = float(np.clip(
                (root_height - progress_floor) / progress_range, 0.0, 1.0
            ))
            reference_height_progress = float(np.clip(
                (reference_root_height - progress_floor) / progress_range, 0.0, 1.0
            ))
            ascent_progress_track = ascent_gate * float(np.clip(
                1.0 - abs(height_progress - reference_height_progress), 0.0, 1.0
            ))
            ascent_height_lag = ascent_gate * float(np.clip(
                reference_height_progress - height_progress, 0.0, 1.0
            ))

        if torch.is_tensor(parts["height"]):
            feedback_scale = getattr(
                self,
                "squat_phase_feedback_scale",
                torch.ones_like(parts["height"]),
            )
            stalled = getattr(
                self,
                "squat_phase_stalled",
                torch.zeros_like(parts["height"], dtype=torch.bool),
            )
            stall_steps = getattr(
                self,
                "squat_phase_stall_steps",
                torch.zeros_like(parts["height"], dtype=torch.long),
            )
            sensor_depth = getattr(
                self,
                "squat_depth_sensor_progress",
                torch.zeros_like(parts["height"]),
            )
            cycle_completed = torch.zeros_like(parts["height"], dtype=torch.bool)
            episode_success = torch.zeros_like(parts["height"], dtype=torch.bool)
            cycle_count = torch.zeros_like(parts["height"], dtype=torch.long)
            cycle_stage = torch.zeros_like(parts["height"], dtype=torch.long)
            if self._squat_phase_feedback_ready:
                sensor_depth, upright = self._squat_feedback_measurements()
                self.squat_depth_sensor_progress = sensor_depth
                cycle_completed, episode_success = self._update_squat_cycle_state(
                    sensor_depth,
                    upright,
                )
                stall_steps = self.squat_phase_stall_steps
                cycle_count = self.squat_cycle_count
                cycle_stage = self.squat_cycle_stage
            stall_timeout = stall_steps >= int(
                self.cfg.squat_phase_max_stall_steps
            )
            incomplete_cycle_timeout = (timed_out > 0.0) & ~episode_success
            parts["termination"] = torch.maximum(
                parts["termination"],
                (stall_timeout | incomplete_cycle_timeout).to(torch.float32),
            )
            if done.dtype == torch.bool:
                done = done | stall_timeout
            else:
                done = torch.maximum(done, stall_timeout.to(done.dtype))
            if timed_out.dtype == torch.bool:
                timed_out = timed_out & ~incomplete_cycle_timeout
            else:
                timed_out = timed_out * (~incomplete_cycle_timeout).to(
                    timed_out.dtype
                )
            if bool(self.cfg.squat_terminate_on_cycle_completion):
                if done.dtype == torch.bool:
                    done = done | cycle_completed
                else:
                    done = torch.maximum(
                        done,
                        cycle_completed.to(done.dtype),
                    )
                # The shared trainer treats successful motion completion as a
                # bootstrapped truncation.  This avoids making completion less
                # valuable merely because it resets the episode.
                if timed_out.dtype == torch.bool:
                    timed_out = timed_out | cycle_completed
                else:
                    timed_out = torch.maximum(
                        timed_out,
                        cycle_completed.to(timed_out.dtype),
                    )
            phase_slowdown = torch.clamp(1.0 - feedback_scale, 0.0, 1.0)
        else:
            feedback_scale_values = np.asarray(
                getattr(self, "squat_phase_feedback_scale", np.ones(1))
            )
            stalled_values = np.asarray(
                getattr(self, "squat_phase_stalled", np.zeros(1, dtype=bool))
            )
            stall_step_values = np.asarray(
                getattr(self, "squat_phase_stall_steps", np.zeros(1, dtype=np.int32))
            )
            sensor_depth_values = np.asarray(
                getattr(self, "squat_depth_sensor_progress", np.zeros(1))
            )
            diagnostic_env_id = min(env_id, len(feedback_scale_values) - 1)
            feedback_scale = float(feedback_scale_values[diagnostic_env_id])
            stalled = bool(stalled_values[min(env_id, len(stalled_values) - 1)])
            stall_steps = int(
                stall_step_values[min(env_id, len(stall_step_values) - 1)]
            )
            sensor_depth = float(
                sensor_depth_values[min(env_id, len(sensor_depth_values) - 1)]
            )
            cycle_completed = 0.0
            episode_success = 0.0
            cycle_count = 0
            cycle_stage = 0
            if self._squat_phase_feedback_ready:
                measured_depth, upright = self._squat_feedback_measurements()
                self.squat_depth_sensor_progress = measured_depth
                cycle_completed, episode_success = self._update_squat_cycle_state(
                    measured_depth,
                    upright,
                    env_id,
                )
                sensor_depth = float(
                    measured_depth[min(env_id, len(measured_depth) - 1)]
                )
                stall_steps = int(
                    self.squat_phase_stall_steps[
                        min(env_id, len(self.squat_phase_stall_steps) - 1)
                    ]
                )
                cycle_count = int(
                    self.squat_cycle_count[
                        min(env_id, len(self.squat_cycle_count) - 1)
                    ]
                )
                cycle_stage = int(
                    self.squat_cycle_stage[
                        min(env_id, len(self.squat_cycle_stage) - 1)
                    ]
                )
            stall_timeout = bool(
                stall_steps >= int(self.cfg.squat_phase_max_stall_steps)
            )
            incomplete_cycle_timeout = bool(timed_out and not episode_success)
            parts["termination"] = max(
                float(parts["termination"]),
                float(stall_timeout or incomplete_cycle_timeout),
            )
            done = bool(done or stall_timeout)
            if incomplete_cycle_timeout:
                timed_out = False
            if (
                bool(self.cfg.squat_terminate_on_cycle_completion)
                and bool(cycle_completed)
            ):
                done = True
                timed_out = True
            phase_slowdown = float(np.clip(1.0 - feedback_scale, 0.0, 1.0))

        reward = (
            0.10
            + 5.50 * squat_track
            + 2.20 * support_track * motion_gate
            + self.cfg.symmetry_reward_coef * parts["symmetry"] * motion_gate
            + 0.35 * parts["ctrl_track"]
            + 2.50 * stand_track
            - 2.50 * missed_stand
            + 4.00 * ascent_progress_track
            - 6.00 * ascent_height_lag
            + 3.00 * bottom_depth_track
            - 5.00 * bottom_depth_deficit
            + self.cfg.squat_cycle_completion_bonus * cycle_completed
            - self.cfg.squat_phase_slowdown_penalty_coef * phase_slowdown
            - self.cfg.ctrl_penalty_coef * parts["ctrl_penalty"]
            - self.cfg.action_rate_penalty_coef * parts["action_rate"]
            - self.cfg.distal_action_penalty_coef * parts["distal_action"]
            - self.cfg.distal_action_rate_penalty_coef * parts["distal_action_rate"]
            - self.cfg.missed_contact_penalty_coef * parts["missed_contact"]
            - self.squat_termination_penalty * parts["termination"]
        )
        # A low-progress phase grants extra time at an easier, nearly static
        # target.  Cap every such step below zero so waiting can never outscore
        # completing the physical motion.  The cumulative per-cycle budget
        # above turns persistent stalling into a terminal failure.
        stall_penalty = float(self.cfg.squat_phase_stall_penalty)
        if torch.is_tensor(reward):
            reward = torch.where(
                stalled,
                torch.minimum(reward, torch.zeros_like(reward)) - stall_penalty,
                reward,
            )
            terminal_floor = torch.full_like(
                reward,
                -(self.squat_termination_penalty + stall_penalty),
            )
            terminal_failure = stall_timeout | incomplete_cycle_timeout
            reward = torch.where(
                terminal_failure,
                torch.minimum(reward, terminal_floor),
                reward,
            )
        elif stalled:
            reward = min(float(reward), 0.0) - stall_penalty
        if not torch.is_tensor(reward) and (
            stall_timeout or incomplete_cycle_timeout
        ):
            reward = min(
                float(reward),
                -(self.squat_termination_penalty + stall_penalty),
            )

        parts["squat_track"] = squat_track
        parts["support_track"] = support_track
        parts["motion_gate"] = motion_gate
        parts["stand_track"] = stand_track
        parts["missed_stand"] = missed_stand
        parts["height_progress"] = height_progress
        parts["reference_height_progress"] = reference_height_progress
        parts["ascent_progress_track"] = ascent_progress_track
        parts["ascent_height_lag"] = ascent_height_lag
        parts["squat_depth_progress"] = squat_depth_progress
        parts["reference_depth_progress"] = reference_depth_progress
        parts["bottom_gate"] = bottom_gate
        parts["bottom_depth_track"] = bottom_depth_track
        parts["bottom_depth_deficit"] = bottom_depth_deficit
        parts["bottom_depth_success"] = bottom_depth_success
        if torch.is_tensor(parts["height"]):
            parts["phase_feedback_scale"] = feedback_scale
            parts["phase_feedback_frozen"] = (
                feedback_scale <= 1.0e-6
            ).to(torch.float32)
            parts["phase_stalled"] = stalled.to(torch.float32)
            parts["phase_slowdown"] = phase_slowdown
            parts["phase_stall_steps"] = stall_steps.to(torch.float32)
            parts["phase_stall_timeout"] = stall_timeout.to(torch.float32)
            parts["incomplete_squat_cycle"] = incomplete_cycle_timeout.to(
                torch.float32
            )
            parts["sensor_depth_progress"] = sensor_depth
            parts["squat_cycle_stage"] = cycle_stage.to(torch.float32)
            parts["squat_cycle_completed"] = cycle_completed.to(torch.float32)
            parts["squat_cycle_count"] = cycle_count.to(torch.float32)
            parts["squat_episode_success"] = episode_success.to(torch.float32)
        else:
            parts["phase_feedback_scale"] = feedback_scale
            parts["phase_feedback_frozen"] = float(
                feedback_scale <= 1.0e-6
            )
            parts["phase_stalled"] = float(stalled)
            parts["phase_slowdown"] = float(phase_slowdown)
            parts["phase_stall_steps"] = float(stall_steps)
            parts["phase_stall_timeout"] = float(stall_timeout)
            parts["incomplete_squat_cycle"] = float(incomplete_cycle_timeout)
            parts["sensor_depth_progress"] = sensor_depth
            parts["squat_cycle_stage"] = float(cycle_stage)
            parts["squat_cycle_completed"] = float(cycle_completed)
            parts["squat_cycle_count"] = float(cycle_count)
            parts["squat_episode_success"] = float(episode_success)
        return reward, done, timed_out, parts


class SquatWarpAmpEnv(SquatRewardMixin, LytRobotWarpAmpEnv):
    """Warp dynamics with the squat-specific reward profile."""


class SquatCpuAmpEnv(SquatRewardMixin, LytRobotAmpEnv):
    """CPU dynamics with the squat-specific reward profile."""


class SquatSensorHistoryEnv(SensorHistoryEnv):
    """Squat task with a strictly IMU+tendon Actor and privileged phase Critic."""

    actor_history_enabled = False

    @property
    def _task_privileged_fields(self) -> tuple[str, ...]:
        return (
            "squat_sin_phase",
            "squat_cos_phase",
            "squat_phase_feedback_scale",
            "squat_sensor_depth",
            "squat_phase_stalled",
            "squat_phase_stall_progress",
            "squat_cycle_stage",
            "squat_cycle_count",
        )

    def _phase_features(self) -> np.ndarray | torch.Tensor:
        frame_count = max(int(self.base_motion.frame_count), 1)
        if self.is_torch_env:
            angle = 2.0 * torch.pi * self.base_env.motion_phase / float(frame_count)
            return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)
        angle = 2.0 * np.pi * self.base_env.motion_phase / float(frame_count)
        return np.stack((np.sin(angle), np.cos(angle)), axis=1).astype(np.float32)

    def _task_privileged_observe(self) -> np.ndarray | torch.Tensor:
        phase = self._phase_features()
        env = self.base_env
        stall_normalizer = max(int(env.cfg.squat_phase_max_stall_steps), 1)
        if self.is_torch_env:
            return torch.cat(
                (
                    phase,
                    env.squat_phase_feedback_scale[:, None],
                    env.squat_depth_sensor_progress[:, None],
                    env.squat_phase_stalled.to(torch.float32)[:, None],
                    (
                        env.squat_phase_stall_steps.to(torch.float32)
                        / float(stall_normalizer)
                    )[:, None],
                    (env.squat_cycle_stage.to(torch.float32) / 2.0)[:, None],
                    torch.clamp(env.squat_cycle_count.to(torch.float32), 0.0, 4.0)[:, None]
                    / 4.0,
                ),
                dim=1,
            )
        return np.concatenate(
            (
                phase,
                env.squat_phase_feedback_scale[:, None],
                env.squat_depth_sensor_progress[:, None],
                env.squat_phase_stalled.astype(np.float32)[:, None],
                (
                    env.squat_phase_stall_steps.astype(np.float32)
                    / float(stall_normalizer)
                )[:, None],
                (env.squat_cycle_stage.astype(np.float32) / 2.0)[:, None],
                (np.clip(env.squat_cycle_count, 0, 4).astype(np.float32) / 4.0)[:, None],
            ),
            axis=1,
        ).astype(np.float32)

    @property
    def observation_metadata(self) -> dict[str, Any]:
        metadata = super().observation_metadata
        metadata.update(
            {
                "frame_order": "current_only",
                "command_fields": (),
                "command_dim": 0,
                "actor_obs_dim": self.obs_dim,
                "sensor_history_obs_dim": self.sensor_history_obs_dim,
                "actor_frame_dim": self.sensor_frame_dim,
                "actor_input_fields": ("imu", "tendon_length"),
                "task": "squat",
                "policy_inputs": ("imu", "tendon_length"),
                "phase_source": "training_only_asymmetric_critic",
                "phase_feedback_inputs": (
                    "tendon_length",
                    "imu_gravity_direction",
                ),
                "phase_feedback": {
                    "enabled": bool(self.cfg.squat_phase_feedback),
                    "reference_frequency_hz": float(self.base_motion.frequency),
                    "reference_frame_count": int(self.base_motion.frame_count),
                    "policy_dt_seconds": float(self.base_env.dt),
                    "bottom_phase_fraction": float(
                        self.base_env._squat_bottom_phase_fraction
                    ),
                    "lag_tolerance": float(self.cfg.squat_phase_lag_tolerance),
                    "minimum_scale": float(self.cfg.squat_phase_min_scale),
                    "bottom_depth_threshold": float(
                        self.cfg.squat_bottom_depth_threshold
                    ),
                    "stand_depth_threshold": float(
                        self.cfg.squat_stand_depth_threshold
                    ),
                    "upright_threshold": float(self.cfg.squat_upright_threshold),
                    "stall_threshold": float(
                        self.cfg.squat_phase_stall_threshold
                    ),
                    "max_stall_steps": int(
                        self.cfg.squat_phase_max_stall_steps
                    ),
                    "stall_penalty": float(
                        self.cfg.squat_phase_stall_penalty
                    ),
                    "slowdown_penalty_coefficient": float(
                        self.cfg.squat_phase_slowdown_penalty_coef
                    ),
                    "cycle_completion_bonus": float(
                        self.cfg.squat_cycle_completion_bonus
                    ),
                    "terminate_on_cycle_completion": bool(
                        self.cfg.squat_terminate_on_cycle_completion
                    ),
                    # Deployment calibration required to reproduce the exact
                    # tendon-space depth estimate without simulator state.
                    "stand_tendon_length": self.base_env._squat_feedback_stand_ctrl_np.tolist(),
                    "tendon_half_range": self.base_env._squat_feedback_half_range_np.tolist(),
                    "depth_direction": self.base_env._squat_feedback_direction_np.tolist(),
                    "depth_denominator": float(
                        self.base_env._squat_feedback_denominator
                    ),
                    "reference_depth_curve": self.base_env._squat_feedback_depth_curve_np.tolist(),
                },
                "success_definition": (
                    "completed_stand_squat_stand_cycle"
                    if self.cfg.squat_terminate_on_cycle_completion
                    else "survived_episode_with_complete_stand_squat_stand_cycle"
                ),
                "posture_sensor_inputs": ("imu", "tendon_length"),
                "strict_sensor_only": True,
            }
        )
        return metadata


class SquatSensorHistoryWarpAmpEnv(SquatSensorHistoryEnv):
    """GPU/Warp squat environment used by both training stages."""

    def __init__(
        self,
        cfg: LytRobotEnvCfg,
        device: str = "cuda",
        observation_cfg: SensorObservationConfig = DEFAULT_SENSOR_OBSERVATION_CONFIG,
    ) -> None:
        super().__init__(SquatWarpAmpEnv(cfg, device=device), observation_cfg)


class SquatSensorHistoryCpuAmpEnv(SquatSensorHistoryEnv):
    """CPU squat environment used for tests and playback."""

    def __init__(
        self,
        cfg: LytRobotEnvCfg,
        observation_cfg: SensorObservationConfig = DEFAULT_SENSOR_OBSERVATION_CONFIG,
    ) -> None:
        super().__init__(SquatCpuAmpEnv(cfg), observation_cfg)
