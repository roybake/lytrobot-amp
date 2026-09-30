"""Sensor-policy wrappers with IMU, tendon-length and applied-action history."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import torch

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.env_mjlab import LytRobotMjlabAmpEnv
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
from lytrobot_amp.motion import MotionLibrary

from .config import DEFAULT_SENSOR_OBSERVATION_CONFIG, SensorObservationConfig
from .gait_command import (
    GAIT_COMMAND_SCALAR_FIELDS,
    assemble_gait_command,
    gait_command_dim,
)
from .observation import HistoryMotionReference, ImuSensorLayout, resolve_imu_sensor_layout


ROOT_CONTEXT_FIELDS = (
    "root_velocity_x",
    "root_velocity_y",
    "root_velocity_z",
    "right_foot_contact",
    "left_foot_contact",
)
TERRAIN_CONTEXT_FIELDS = (
    "terrain_grade",
    "terrain_roughness",
    "terrain_step_height",
    "terrain_step_depth",
    "terrain_gap_width",
    "terrain_pit_depth",
)
ROOT_VELOCITY_NORMALIZER = 3.0
JOINT_VELOCITY_NORMALIZER = 10.0


def sensor_stability_task_reward(
    info: dict[str, np.ndarray | torch.Tensor],
    *,
    fall_penalty: float,
    stall_penalty: float,
    knee_gait_reward_coef: float = 0.0,
    natural_gait_reward_coef: float = 0.0,
    mirror_reward_coef: float = 0.0,
    stride_reward_coef: float = 0.0,
    foot_roll_reward_coef: float = 0.0,
    alternation_reward_coef: float = 0.0,
    hop_penalty_coef: float = 0.0,
    tiptoe_penalty_coef: float = 0.0,
    stance_slip_penalty_coef: float = 0.0,
    sync_forward_penalty_coef: float = 0.0,
    velocity_excess_penalty_coef: float = 0.0,
) -> np.ndarray | torch.Tensor:
    """Return a sensor-deployable reward without hidden reference phase terms.

    The full-state reward includes joint, foot and tendon targets sampled from
    ``motion_phase``.  Those targets are intentionally absent from the
    IMU+tendon actor input, so using them as the dominant student objective
    creates contradictory labels after random resets.  This reward keeps only
    stability and world-frame locomotion terms that the GRU can infer from its
    causal feedback history.  A fall is an unambiguously bad terminal event,
    not a small one-step deduction after many positive tracking rewards.
    """

    sensor_upright = info["sensor_upright"]
    is_torch = torch.is_tensor(sensor_upright)
    clip = torch.clamp if is_torch else np.clip
    minimum = torch.minimum if is_torch else np.minimum
    where = torch.where if is_torch else np.where
    def gait_score(name: str, default: float, maximum: float = 1.0):
        value = info.get(name)
        if value is None:
            return (
                torch.full_like(sensor_upright, float(default))
                if is_torch
                else np.full_like(sensor_upright, float(default), dtype=np.float32)
            )
        return clip(value, 0.0, maximum)

    posture_track = clip(info.get("sensor_posture_track", info["sensor_upright"]), 0.0, 1.0)
    height_track = clip(info.get("sensor_gait_height_track", info["sensor_height"]), 0.0, 1.0)
    stability = clip(
        minimum(posture_track, height_track),
        0.0,
        1.0,
    )
    speed_track = clip(info.get("forward_velocity_track", info["stable_forward_velocity_track"]), 0.0, 1.0)
    forward_progress = clip(info.get("forward_progress", info["stable_forward_progress"]), 0.0, 1.0)
    velocity_deficit = clip(info["forward_velocity_deficit"], 0.0, 2.0)
    velocity_excess = gait_score("forward_velocity_excess", 0.0, 2.0)
    gait_progress = clip(info.get("gait_progress", 1.0), 0.0, 1.0)
    knee_flexion_track = clip(info.get("knee_flexion_track", 1.0), 0.0, 1.0)
    quality_terms = (
        gait_score("sensor_stride_length", 0.5),
        gait_score("sensor_step_cadence", 0.5),
        gait_score("sensor_swing_clearance", 0.5),
        gait_score("sensor_foot_roll", 0.5),
        gait_score("sensor_alternation", 0.5),
        gait_score("sensor_mirror", 0.5),
    )
    gait_quality = 6.0 / sum(1.0 / clip(term, 0.05, 1.0) for term in quality_terms)
    control_cost = (
        0.04 * info["ctrl_penalty"]
        + 0.10 * info["action_rate"]
        + 0.03 * info["distal_action"]
        + 0.12 * info["distal_action_rate"]
    )
    # Natural walking is a joint condition rather than a sum of independently
    # exploitable bonuses.  The harmonic gait score is dominated by its worst
    # component, then speed and upright posture gate the whole walking reward.
    reward = 0.50 * stability + stability * (
        1.50 * speed_track
        + 0.50 * forward_progress
        - 2.50 * velocity_deficit
        - float(velocity_excess_penalty_coef) * velocity_excess
        + float(natural_gait_reward_coef)
        * gait_progress
        * speed_track
        * gait_quality
        + float(knee_gait_reward_coef)
        * gait_progress
        * (knee_flexion_track - 1.0)
        + gait_progress * (
            - float(hop_penalty_coef) * gait_score("sensor_hop", 0.0)
            - float(tiptoe_penalty_coef) * (
                gait_score("sensor_tiptoe", 0.0)
                + gait_score("sensor_roll_violation", 0.0)
            )
            - float(stance_slip_penalty_coef) * gait_score("sensor_stance_slip", 0.0, 2.0)
            - float(sync_forward_penalty_coef) * gait_score("sensor_synchronous_forward", 0.0, 2.0)
        )
    ) - control_cost
    insufficient_progress = info.get("insufficient_progress")
    if insufficient_progress is not None:
        stall_value = (
            torch.full_like(reward, -float(stall_penalty))
            if is_torch
            else np.full_like(reward, -float(stall_penalty), dtype=np.float32)
        )
        reward = where(insufficient_progress > 0.5, stall_value, reward)

    termination = info.get("termination")
    if termination is None:
        return reward
    fall_value = torch.full_like(reward, -float(fall_penalty)) if is_torch else np.full_like(
        reward, -float(fall_penalty), dtype=np.float32
    )
    return where(termination > 0.5, fall_value, reward)


def _normalized_terrain_context(
    scenario: str,
    cfg: LytRobotEnvCfg,
) -> np.ndarray:
    """Encode training-only terrain properties as bounded continuous values."""

    context = np.zeros(6, dtype=np.float32)
    scenario = str(scenario).lower()
    if scenario == "rough":
        context[1] = float(cfg.rough_terrain_height) / 0.20
    elif scenario == "slope_up":
        context[0] = float(cfg.slope_angle_deg) / 30.0
    elif scenario == "slope_down":
        context[0] = -float(cfg.slope_angle_deg) / 30.0
    elif scenario in {"stair_up", "stairs"}:
        # A single continuous ascent/platform/descent clip uses the generic
        # "stairs" label.  Direction is already encoded by its gait command;
        # the privileged table records the shared physical geometry.
        context[2] = float(cfg.stair_step_height) / 0.25
        context[3] = float(cfg.stair_step_depth) / 0.60
    elif scenario == "stair_down":
        context[2] = -float(cfg.stair_step_height) / 0.25
        context[3] = float(cfg.stair_step_depth) / 0.60
    elif scenario == "gap":
        context[4] = float(cfg.gap_terrain_width) / 0.50
    elif scenario == "pit":
        context[5] = float(cfg.terrain_course_base_height) / 0.50
    elif scenario != "flat":
        raise ValueError(f"Unsupported privileged terrain scenario: {scenario!r}.")
    return np.clip(context, -1.0, 1.0)


class SensorHistoryEnv:
    """Wrap an existing environment while leaving its dynamics and rewards intact."""

    def __init__(
        self,
        base_env: LytRobotAmpEnv | LytRobotWarpAmpEnv,
        observation_cfg: SensorObservationConfig = DEFAULT_SENSOR_OBSERVATION_CONFIG,
    ) -> None:
        self.base_env = base_env
        self.sensor_observation_cfg = (
            observation_cfg if base_env.cfg.sensor_gait_command_enabled
            else replace(observation_cfg, gait_command_enabled=False)
        )
        self.is_torch_env = bool(getattr(base_env, "is_torch_env", False))
        if self.is_torch_env:
            # The wrapper supplies the actor observation; avoid assembling the
            # much larger privileged base observation on every environment step.
            self.base_env.actor_observation_enabled = False
        self.imu_layout = resolve_imu_sensor_layout(
            base_env.model,
            accelerometer_scale=observation_cfg.accelerometer_scale,
            gyroscope_scale=observation_cfg.gyroscope_scale,
            sensor_names=("rootacc", "rootgyro") if base_env.cfg.sensor_root_imu_only else None,
        )
        self.base_motion = base_env.motion
        self.motion = HistoryMotionReference(
            self.base_motion,
            frame_count=observation_cfg.frame_count,
            policy_dt=base_env.dt,
        )
        self._bind_sensor_tensors()
        self._allocate_histories()
        self._build_privileged_context_tables()
        self._fill_history(self._all_env_ids())
        self.base_env.terminal_observer = self._terminal_observation

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    @property
    def sensor_frame_dim(self) -> int:
        return int(
            self.imu_layout.dim
            + self.base_env.position_action_dim
            + self.base_env.action_dim
            + int(self.base_env.cfg.sensor_phase_observation)
            + self.gait_command_dim
        )

    @property
    def gait_command_dim(self) -> int:
        if not self.sensor_observation_cfg.gait_command_enabled:
            return 0
        return gait_command_dim(
            self.base_env.action_dim,
            self.sensor_observation_cfg.gait_lookahead_seconds,
            trajectory_feedback=bool(self.base_env.cfg.trajectory_feedback),
        )

    @property
    def obs_dim(self) -> int:
        return int(self.sensor_frame_dim * self.sensor_observation_cfg.frame_count)

    @property
    def amp_obs_dim(self) -> int:
        return int(self.base_motion.feature_dim * self.sensor_observation_cfg.frame_count)

    @property
    def privileged_context_dim(self) -> int:
        return len(self.privileged_context_fields)

    @property
    def privileged_context_fields(self) -> tuple[str, ...]:
        joint_names = tuple(self.base_motion.common_joint_names)
        joint_velocity_fields = tuple(
            f"joint_velocity_{name}" for name in joint_names
        )
        return ROOT_CONTEXT_FIELDS + joint_velocity_fields + TERRAIN_CONTEXT_FIELDS

    @property
    def privileged_context_metadata(self) -> dict[str, Any]:
        return {
            "fields": self.privileged_context_fields,
            "dim": self.privileged_context_dim,
            "actor_input": False,
            "training_only": True,
            "root_velocity_normalizer": ROOT_VELOCITY_NORMALIZER,
            "joint_velocity_normalizer": JOINT_VELOCITY_NORMALIZER,
            "loss_weights": (
                (2.0,) * len(ROOT_CONTEXT_FIELDS)
                + (1.0,) * len(self.base_motion.common_joint_names)
                + (0.5,) * len(TERRAIN_CONTEXT_FIELDS)
            ),
            "terrain_encoding": "continuous_bounded_parameters",
        }

    @property
    def observation_metadata(self) -> dict[str, Any]:
        command_fields = (
            GAIT_COMMAND_SCALAR_FIELDS
            + tuple(
                f"reference_tendon_{seconds:.3f}s"
                for seconds in self.sensor_observation_cfg.gait_lookahead_seconds
            )
            + (("trajectory_error_x", "trajectory_error_y") if self.base_env.cfg.trajectory_feedback else ())
            if self.sensor_observation_cfg.gait_command_enabled
            else ()
        )
        return {
            "config": asdict(self.sensor_observation_cfg),
            "frame_order": "oldest_to_current",
            "imu_sensor_names": self.imu_layout.names,
            "imu_dim_per_frame": self.imu_layout.dim,
            "tendon_length_dim_per_frame": self.base_env.position_action_dim,
            "previous_action_dim_per_frame": self.base_env.action_dim,
            "actor_frame_dim": self.sensor_frame_dim,
            "actor_history_frames": self.sensor_observation_cfg.frame_count,
            "actor_history_span_seconds": (
                (self.sensor_observation_cfg.frame_count - 1) * self.base_env.dt
            ),
            "actor_input_fields": (
                (
                    "imu", "tendon_length", "previous_action",
                    *(("phase",) if self.base_env.cfg.sensor_phase_observation else ()),
                    *(("gait_command",) if self.sensor_observation_cfg.gait_command_enabled else ()),
                )
            ),
            "hardware_sensor_fields": (
                ("imu", "tendon_length", "odometry_xy")
                if self.sensor_observation_cfg.gait_command_enabled
                and self.base_env.cfg.trajectory_feedback
                else ("imu", "tendon_length")
            ),
            "command_fields": command_fields,
            "command_dim": self.gait_command_dim,
            "strict_sensor_only": not self.sensor_observation_cfg.gait_command_enabled,
            "hardware_sensor_only": (
                not self.sensor_observation_cfg.gait_command_enabled
                or not self.base_env.cfg.trajectory_feedback
            ),
            "deployable_command_only": self.sensor_observation_cfg.gait_command_enabled,
            "gait_command": self.gait_command_metadata,
            "actor_obs_dim": self.obs_dim,
            "amp_frame_dim": self.base_motion.feature_dim,
            "amp_obs_dim": self.amp_obs_dim,
            "privileged_training_context": self.privileged_context_metadata,
        }

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @property
    def gait_command_metadata(self) -> dict[str, Any]:
        """Self-contained lookup data required beside the deployed ONNX Actor."""

        motions = (
            tuple(self.base_motion.motions)
            if isinstance(self.base_motion, MotionLibrary)
            else (self.base_motion,)
        )
        ctrl_mid = self._numpy(self.base_env.ctrl_mid).astype(np.float32)
        ctrl_half_range = self._numpy(self.base_env.ctrl_half_range).astype(np.float32)
        tables = [
            np.clip(
                (np.asarray(motion.reference_tendon_ctrl, dtype=np.float32) - ctrl_mid[None, :])
                / ctrl_half_range[None, :],
                -1.0,
                1.0,
            ).tolist()
            for motion in motions
        ]
        root_xy_tables = [
            np.asarray(motion.reference_root_xy, dtype=np.float32).tolist()
            for motion in motions
        ]
        root_velocity_tables = [
            np.asarray(motion.reference_root_lin_vel[:, 0], dtype=np.float32).tolist()
            for motion in motions
        ]
        stand = np.clip(
            (self._numpy(self.base_env.stand_tendon_ctrl)[0] - ctrl_mid) / ctrl_half_range,
            -1.0,
            1.0,
        ).astype(np.float32)
        if isinstance(self.base_motion, MotionLibrary):
            frequencies = np.asarray(self.base_motion.frequencies, dtype=np.float32)
            frame_counts = np.asarray(self.base_motion.frame_counts, dtype=np.int32)
            motion_loop = np.asarray(self.base_motion.motion_loop, dtype=bool)
            motion_names = tuple(str(value) for value in self.base_motion.motion_names)
        else:
            frequencies = np.asarray([self.base_motion.frequency], dtype=np.float32)
            frame_counts = np.asarray([self.base_motion.frame_count], dtype=np.int32)
            motion_loop = np.asarray([self.base_motion.source_loop], dtype=bool)
            motion_names = ("walk",)
        return {
            "source": "onboard_clock_and_reference_tendon_table",
            "actor_input": bool(self.sensor_observation_cfg.gait_command_enabled),
            "external_hardware_required": bool(self.base_env.cfg.trajectory_feedback),
            "hardware_observations": (
                ["imu", "tendon_length", "odometry_xy"]
                if self.base_env.cfg.trajectory_feedback else ["imu", "tendon_length"]
            ),
            "trajectory_feedback": bool(self.base_env.cfg.trajectory_feedback),
            "use_reference_velocity": bool(self.base_env.cfg.use_reference_velocity),
            "reference_root_xy_tables": root_xy_tables if self.base_env.cfg.trajectory_feedback else [],
            "stand_root_xy": self._numpy(self.base_env.stand_root_xy)[0].tolist(),
            "reference_root_velocity_tables": root_velocity_tables if self.base_env.cfg.use_reference_velocity else [],
            "lookahead_seconds": list(self.sensor_observation_cfg.gait_lookahead_seconds),
            "command_dim": self.gait_command_dim,
            "control_dt": float(self.base_env.dt),
            "target_velocity": float(self.base_env.cfg.target_velocity),
            "target_velocity_normalizer": float(
                self.sensor_observation_cfg.target_velocity_normalizer
            ),
            "target_velocity_normalized": float(
                self.base_env.cfg.target_velocity
                / self.sensor_observation_cfg.target_velocity_normalizer
            ),
            "motion_names": list(motion_names),
            "frequencies": frequencies.tolist(),
            "frame_counts": frame_counts.tolist(),
            "motion_loop": motion_loop.tolist(),
            "stand_start_motion_phase": float(self.base_env.cfg.stand_start_motion_phase),
            "stand_start_prob": float(self.base_env.cfg.stand_start_prob),
            "stand_start_phase_delay_steps": int(
                self.base_env.cfg.stand_start_phase_delay_steps
            ),
            "startup_action_ramp_steps": int(
                self.base_env.cfg.startup_action_ramp_steps
            ),
            "stand_start_transition_min_steps": int(
                self.base_env.cfg.stand_start_transition_min_steps
            ),
            "stand_start_transition_max_steps": int(
                self.base_env.cfg.stand_start_transition_max_steps
            ),
            "normalized_stand_tendon_target": stand.tolist(),
            "normalized_reference_tendon_tables": tables,
            "ctrl_mid": ctrl_mid.tolist(),
            "ctrl_half_range": ctrl_half_range.tolist(),
            "actuator_action_scale": self._numpy(self.base_env.action_scale).tolist(),
            "action_residual_scale": float(self.base_env.cfg.action_residual_scale),
            "reference_ctrl_blend": float(self.base_env.cfg.reference_ctrl_blend),
            "frame_layout": [
                *GAIT_COMMAND_SCALAR_FIELDS,
                *(
                    f"reference_tendon_{seconds:.3f}s[{self.base_env.action_dim}]"
                    for seconds in self.sensor_observation_cfg.gait_lookahead_seconds
                ),
                *(("trajectory_error_xy[2]",) if self.base_env.cfg.trajectory_feedback else ()),
            ],
        }

    def _bind_sensor_tensors(self) -> None:
        if self.is_torch_env:
            import warp as wp

            self._sensordata = wp.to_torch(self.base_env.wp_data.sensordata)
            self._imu_addresses = torch.as_tensor(
                self.imu_layout.addresses,
                dtype=torch.long,
                device=self.base_env.device,
            )
            self._imu_divisors = torch.as_tensor(
                self.imu_layout.divisors,
                dtype=torch.float32,
                device=self.base_env.device,
            )

    def _allocate_histories(self) -> None:
        shape = (self.num_envs, self.sensor_observation_cfg.frame_count, self.sensor_frame_dim)
        amp_shape = (self.num_envs, self.sensor_observation_cfg.frame_count, self.base_motion.feature_dim)
        if self.is_torch_env:
            self._actor_history = torch.zeros(shape, dtype=torch.float32, device=self.base_env.device)
            self._amp_history = torch.zeros(amp_shape, dtype=torch.float32, device=self.base_env.device)
        else:
            self._actor_history = np.zeros(shape, dtype=np.float32)
            self._amp_history = np.zeros(amp_shape, dtype=np.float32)

    def _build_privileged_context_tables(self) -> None:
        cfg = self.base_env.cfg
        course_rows = tuple(
            tuple(_normalized_terrain_context(scenario, cfg) for scenario in row)
            for row in self.base_env.course_stage_scenario_names
        )
        motion_names = tuple(
            getattr(self.base_motion, "motion_names", ("walk",))
        )
        motion_scenarios: list[str] = []
        for motion_id, name in enumerate(motion_names):
            normalized_name = str(name).lower()
            if normalized_name in {"stair_up", "stair_down"}:
                motion_scenarios.append(normalized_name)
            else:
                motion_scenarios.append(
                    str(self.base_env.motion_terrain_kinds[motion_id])
                )
        motion_rows = tuple(
            _normalized_terrain_context(scenario, cfg)
            for scenario in motion_scenarios
        )

        if self.is_torch_env:
            device = self.base_env.device
            self._course_terrain_context = (
                torch.as_tensor(
                    np.asarray(course_rows, dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                if course_rows
                else torch.empty((0, 0, 6), dtype=torch.float32, device=device)
            )
            self._motion_terrain_context = torch.as_tensor(
                np.asarray(motion_rows, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
        else:
            self._course_terrain_context = (
                np.asarray(course_rows, dtype=np.float32)
                if course_rows
                else np.empty((0, 0, 6), dtype=np.float32)
            )
            self._motion_terrain_context = np.asarray(
                motion_rows,
                dtype=np.float32,
            )

    def _terrain_context(self) -> np.ndarray | torch.Tensor:
        if self.base_env.more_terrain_cfg.serial_course:
            return self._course_terrain_context[
                self.base_env.course_variant_ids,
                self.base_env.course_stage_ids,
            ]
        return self._motion_terrain_context[self.base_env.motion_ids]

    def privileged_observe(self) -> np.ndarray | torch.Tensor:
        """Return simulator-only targets for estimator and asymmetric critic.

        These values never enter ``observe`` or the exported actor.  The GRU
        must reconstruct them from causal IMU, tendon-length and action history.
        """

        if self.is_torch_env:
            velocity = torch.clamp(
                self.base_env.qvel[:, :3] / ROOT_VELOCITY_NORMALIZER,
                -1.0,
                1.0,
            )
            contacts = self.base_env._foot_contacts()
            joint_velocity = torch.clamp(
                self.base_env.qvel[:, self.base_env.robot_qvel_adrs]
                / JOINT_VELOCITY_NORMALIZER,
                -1.0,
                1.0,
            )
            return torch.cat(
                (velocity, contacts, joint_velocity, self._terrain_context()),
                dim=1,
            )

        velocity = np.stack(
            [np.asarray(data.qvel[:3], dtype=np.float32) for data in self.base_env.datas],
            axis=0,
        )
        velocity = np.clip(
            velocity / ROOT_VELOCITY_NORMALIZER,
            -1.0,
            1.0,
        ).astype(np.float32)
        contacts = np.stack(
            [self.base_env._foot_contacts(data) for data in self.base_env.datas],
            axis=0,
        )
        joint_velocity = np.stack(
            [
                np.asarray(
                    data.qvel[self.base_env.motion.robot_qvel_adrs],
                    dtype=np.float32,
                )
                for data in self.base_env.datas
            ],
            axis=0,
        )
        joint_velocity = np.clip(
            joint_velocity / JOINT_VELOCITY_NORMALIZER,
            -1.0,
            1.0,
        ).astype(np.float32)
        return np.concatenate(
            (velocity, contacts, joint_velocity, self._terrain_context()),
            axis=1,
        ).astype(np.float32)

    def _all_env_ids(self) -> np.ndarray | torch.Tensor:
        if self.is_torch_env:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.base_env.device)
        return np.arange(self.num_envs, dtype=np.int64)

    def _normalized_sensor_frame(self) -> np.ndarray | torch.Tensor:
        if self.is_torch_env:
            imu = self._sensordata[:, self._imu_addresses] / self._imu_divisors
            tendon = self.base_env.ten_length[:, self.base_env.actuator_tendon_ids]
            tendon = (tendon - self.base_env.ctrl_mid) / self.base_env.ctrl_half_range
            sensor = torch.cat((imu, tendon), dim=1)
            sensor = torch.nan_to_num(sensor, nan=0.0, posinf=20.0, neginf=-20.0)
            if self.base_env.domain_randomization and self.base_env.cfg.obs_noise_std > 0.0:
                noise_std = (
                    self.base_env.domain_randomization_scale * float(self.base_env.cfg.obs_noise_std)
                )
                sensor = sensor + noise_std * torch.randn_like(sensor)
            sensor = torch.clamp(
                sensor,
                -self.sensor_observation_cfg.observation_clip,
                self.sensor_observation_cfg.observation_clip,
            )
            previous_action = torch.clamp(self.base_env.last_action, -1.0, 1.0)
            return torch.cat(
                (
                    sensor,
                    previous_action,
                    self._deepmimic_phase_frame(),
                    self._gait_command_frame(),
                ),
                dim=1,
            )

        imu = np.stack(
            [np.asarray(data.sensordata[self.imu_layout.addresses], dtype=np.float32) for data in self.base_env.datas],
            axis=0,
        )
        imu = imu / self.imu_layout.divisors[None, :]
        tendon = np.stack(
            [
                np.asarray(data.ten_length[self.base_env.actuator_tendon_ids], dtype=np.float32)
                for data in self.base_env.datas
            ],
            axis=0,
        )
        tendon = (tendon - self.base_env.ctrl_mid[None, :]) / self.base_env.ctrl_half_range[None, :]
        sensor = np.concatenate((imu, tendon), axis=1).astype(np.float32)
        sensor = np.nan_to_num(sensor, nan=0.0, posinf=20.0, neginf=-20.0)
        if self.base_env.domain_randomization and self.base_env.cfg.obs_noise_std > 0.0:
            sensor += self.base_env.rng.normal(
                0.0,
                self.base_env.domain_randomization_scale * float(self.base_env.cfg.obs_noise_std),
                size=sensor.shape,
            ).astype(np.float32)
        sensor = np.clip(
            sensor,
            -self.sensor_observation_cfg.observation_clip,
            self.sensor_observation_cfg.observation_clip,
        ).astype(np.float32)
        previous_action = np.clip(
            np.asarray(self.base_env.last_action, dtype=np.float32),
            -1.0,
            1.0,
        )
        return np.concatenate(
            (
                sensor,
                previous_action,
                self._deepmimic_phase_frame(),
                self._gait_command_frame(),
            ),
            axis=1,
        ).astype(np.float32)

    def _deepmimic_phase_frame(self) -> np.ndarray | torch.Tensor:
        """Return DeepMimic's scalar phase in [0, 1], or an empty frame."""

        enabled = bool(self.base_env.cfg.sensor_phase_observation)
        if self.is_torch_env:
            if not enabled:
                return torch.empty(
                    (self.num_envs, 0), dtype=torch.float32,
                    device=self.base_env.device,
                )
            frame_count = self.base_env.motion_frame_counts[
                self.base_env.motion_ids
            ].to(torch.float32)
            phase = self.base_env.motion_phase / (frame_count - 1.0).clamp_min(1.0)
            return torch.clamp(phase, 0.0, 1.0)[:, None]

        if not enabled:
            return np.empty((self.num_envs, 0), dtype=np.float32)
        if isinstance(self.base_motion, MotionLibrary):
            frame_count = np.asarray(
                self.base_motion.frame_counts, dtype=np.float32
            )[np.asarray(self.base_env.motion_ids, dtype=np.int32)]
        else:
            frame_count = np.full(
                self.num_envs, self.base_motion.frame_count, dtype=np.float32
            )
        phase = np.asarray(self.base_env.motion_phase, dtype=np.float32)
        return np.clip(phase / np.maximum(frame_count - 1.0, 1.0), 0.0, 1.0)[:, None]

    def _gait_command_frame(self) -> np.ndarray | torch.Tensor:
        """Return clock, smooth-start and reference-tendon commands."""

        if not self.sensor_observation_cfg.gait_command_enabled:
            if self.is_torch_env:
                return torch.empty(
                    (self.num_envs, 0),
                    dtype=torch.float32,
                    device=self.base_env.device,
                )
            return np.empty((self.num_envs, 0), dtype=np.float32)

        lookaheads = self.sensor_observation_cfg.gait_lookahead_seconds
        velocity_normalizer = float(
            self.sensor_observation_cfg.target_velocity_normalizer
        )
        if self.is_torch_env:
            phase = self.base_env.motion_phase
            motion_ids = self.base_env.motion_ids
            frame_count = self.base_env.motion_frame_counts[motion_ids].to(torch.float32)
            frequency = self.base_env.motion_frequencies[motion_ids]
            progress = self.base_env._stand_start_progress()
            angle = 2.0 * torch.pi * phase / frame_count.clamp_min(1.0)
            if self.base_env.cfg.use_reference_velocity:
                velocity = self.base_env._targets_at_phase(
                    phase, motion_ids
                )["root_lin_vel"][:, 0] * progress
            else:
                velocity = float(self.base_env.cfg.target_velocity) * progress
            scalar = torch.stack(
                (
                    torch.sin(angle),
                    torch.cos(angle),
                    progress,
                    velocity / velocity_normalizer,
                ),
                dim=1,
            )
            stand = self.base_env.stand_tendon_ctrl.expand(self.num_envs, -1)
            tendon_targets = []
            for seconds in lookaheads:
                future_phase = phase + frequency * float(seconds) * progress
                walk = self.base_env._targets_at_phase(
                    future_phase, motion_ids
                )["tendon_ctrl"]
                blended = (1.0 - progress[:, None]) * stand + progress[:, None] * walk
                tendon_targets.append(self.base_env.ctrl_to_action(blended))
            parts = (scalar, *tendon_targets)
            if self.base_env.cfg.trajectory_feedback:
                target_xy = self.base_env._targets_at_current_phase()["root_xy"]
                error = target_xy - self.base_env.qpos[:, :2]
                parts = (*parts, error)
            return torch.cat(parts, dim=1)

        phase = np.asarray(self.base_env.motion_phase, dtype=np.float32)
        motion_ids = np.asarray(self.base_env.motion_ids, dtype=np.int32)
        progress = np.asarray(self.base_env._stand_start_progress(), dtype=np.float32)
        if isinstance(self.base_motion, MotionLibrary):
            frame_count = np.asarray(self.base_motion.frame_counts, dtype=np.float32)[motion_ids]
            frequency = np.asarray(self.base_motion.frequencies, dtype=np.float32)[motion_ids]

            def targets_at(value: np.ndarray):
                return self.base_motion.targets_at_phase(value, motion_ids)
        else:
            frame_count = np.full(self.num_envs, self.base_motion.frame_count, dtype=np.float32)
            frequency = np.full(self.num_envs, self.base_motion.frequency, dtype=np.float32)

            def targets_at(value: np.ndarray):
                return self.base_motion.targets_at_phase(value)

        current = targets_at(phase)
        velocity = (
            current.root_lin_vel[:, 0] * progress
            if self.base_env.cfg.use_reference_velocity
            else float(self.base_env.cfg.target_velocity) * progress
        )
        stand = np.repeat(
            np.asarray(self.base_env.stand_tendon_ctrl, dtype=np.float32),
            self.num_envs,
            axis=0,
        )
        tendon_targets = []
        for seconds in lookaheads:
            future_phase = phase + frequency * float(seconds) * progress
            walk = targets_at(future_phase).tendon_ctrl
            blended = (1.0 - progress[:, None]) * stand + progress[:, None] * walk
            tendon_targets.append(self.base_env.ctrl_to_action(blended))
        return assemble_gait_command(
            phase=phase,
            frame_count=frame_count,
            progress=progress,
            target_velocity=velocity / velocity_normalizer,
            reference_tendon_targets=tendon_targets,
            trajectory_error=(
                self.base_env._targets_at_current_phase().root_xy
                - np.stack([data.qpos[:2] for data in self.base_env.datas]).astype(np.float32)
                if self.base_env.cfg.trajectory_feedback else None
            ),
        )

    def _current_amp_frame(self) -> np.ndarray | torch.Tensor:
        return self.base_env.amp_observe()

    def _fill_history(self, env_ids: np.ndarray | torch.Tensor) -> None:
        actor_frame = self._normalized_sensor_frame()
        amp_frame = self._current_amp_frame()
        if self.is_torch_env:
            ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self.base_env.device,
            )
            self._actor_history[ids] = actor_frame[ids, None, :].expand(-1, self.sensor_observation_cfg.frame_count, -1)
            self._amp_history[ids] = amp_frame[ids, None, :].expand(-1, self.sensor_observation_cfg.frame_count, -1)
            return

        ids = np.asarray(env_ids, dtype=np.int64)
        self._actor_history[ids] = np.repeat(
            actor_frame[ids, None, :],
            self.sensor_observation_cfg.frame_count,
            axis=1,
        )
        self._amp_history[ids] = np.repeat(
            amp_frame[ids, None, :],
            self.sensor_observation_cfg.frame_count,
            axis=1,
        )

    def _append_history(self) -> None:
        actor_frame = self._normalized_sensor_frame()
        amp_frame = self._current_amp_frame()
        if self.is_torch_env:
            self._actor_history = torch.roll(self._actor_history, shifts=-1, dims=1)
            self._amp_history = torch.roll(self._amp_history, shifts=-1, dims=1)
        else:
            self._actor_history = np.roll(self._actor_history, shift=-1, axis=1)
            self._amp_history = np.roll(self._amp_history, shift=-1, axis=1)
        self._actor_history[:, -1] = actor_frame
        self._amp_history[:, -1] = amp_frame

    def observe(self) -> np.ndarray | torch.Tensor:
        return self._actor_history.reshape(self.num_envs, self.obs_dim)

    def amp_observe(self) -> np.ndarray | torch.Tensor:
        return self._amp_history.reshape(self.num_envs, self.amp_obs_dim)

    def reset(
        self,
        env_ids: np.ndarray | list[int] | torch.Tensor | None = None,
    ) -> np.ndarray | torch.Tensor:
        self.base_env.reset(env_ids)
        if env_ids is None:
            ids = self._all_env_ids()
        elif self.is_torch_env:
            ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self.base_env.device,
            )
        else:
            ids = np.asarray(env_ids, dtype=np.int64)
        self._fill_history(ids)
        return self.observe()

    def _terminal_observation(self, ids):
        """Build the next sensor history before autoreset, including squat phase."""
        actor_history, amp_history = self._actor_history, self._amp_history
        try:
            self._append_history()
            observation = self.observe()[ids]
            context = self.privileged_observe()[ids]
            return {
                "terminal_observation": observation.clone() if self.is_torch_env else observation.copy(),
                "terminal_privileged_context": context.clone() if self.is_torch_env else context.copy(),
            }
        finally:
            self._actor_history, self._amp_history = actor_history, amp_history

    def step(self, action: np.ndarray | torch.Tensor):
        _base_obs, reward, done, info = self.base_env.step(action)
        if bool(getattr(self.base_env.cfg, "sensor_stability_reward", False)):
            reward = sensor_stability_task_reward(
                info,
                fall_penalty=float(self.base_env.cfg.sensor_fall_penalty),
                stall_penalty=float(self.base_env.cfg.forward_progress_termination_penalty),
                knee_gait_reward_coef=float(
                    self.base_env.cfg.sensor_knee_gait_reward_coef
                ),
                natural_gait_reward_coef=float(
                    self.base_env.cfg.sensor_natural_gait_reward_coef
                ),
                mirror_reward_coef=float(self.base_env.cfg.sensor_mirror_reward_coef),
                stride_reward_coef=float(self.base_env.cfg.sensor_stride_reward_coef),
                foot_roll_reward_coef=float(self.base_env.cfg.sensor_foot_roll_reward_coef),
                alternation_reward_coef=float(self.base_env.cfg.sensor_alternation_reward_coef),
                hop_penalty_coef=float(self.base_env.cfg.sensor_hop_penalty_coef),
                tiptoe_penalty_coef=float(self.base_env.cfg.sensor_tiptoe_penalty_coef),
                stance_slip_penalty_coef=float(self.base_env.cfg.sensor_stance_slip_penalty_coef),
                sync_forward_penalty_coef=float(self.base_env.cfg.sensor_sync_forward_penalty_coef),
                velocity_excess_penalty_coef=float(self.base_env.cfg.sensor_velocity_excess_penalty_coef),
            )
            info["sensor_stability_reward"] = reward
        self._append_history()
        if self.is_torch_env:
            # Boolean indexing stays on device and avoids a second CUDA sync
            # after the base environment has already handled resets.
            self._fill_history(done > 0.0)
        else:
            done_ids = np.nonzero(np.asarray(done) > 0.0)[0].astype(np.int64)
            if len(done_ids) > 0:
                self._fill_history(done_ids)
        return self.observe(), reward, done, info

    def close(self) -> None:
        self.base_env.close()


class SensorHistoryWarpAmpEnv(SensorHistoryEnv):
    """Drop-in replacement for the existing Warp training environment."""

    def __init__(
        self,
        cfg: LytRobotEnvCfg,
        device: str = "cuda",
        observation_cfg: SensorObservationConfig = DEFAULT_SENSOR_OBSERVATION_CONFIG,
    ) -> None:
        super().__init__(LytRobotWarpAmpEnv(cfg, device=device), observation_cfg)


class SensorHistoryMjlabAmpEnv(SensorHistoryEnv):
    """Sensor-policy wrapper for the mjlab Simulation backend."""

    def __init__(
        self,
        cfg: LytRobotEnvCfg,
        device: str = "cuda",
        observation_cfg: SensorObservationConfig = DEFAULT_SENSOR_OBSERVATION_CONFIG,
    ) -> None:
        super().__init__(LytRobotMjlabAmpEnv(cfg, device=device), observation_cfg)


class SensorHistoryCpuAmpEnv(SensorHistoryEnv):
    """CPU environment used by the isolated playback entry point."""

    def __init__(
        self,
        cfg: LytRobotEnvCfg,
        observation_cfg: SensorObservationConfig = DEFAULT_SENSOR_OBSERVATION_CONFIG,
    ) -> None:
        super().__init__(LytRobotAmpEnv(cfg), observation_cfg)
