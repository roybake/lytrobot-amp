"""Environment wrappers that expose only IMU and tendon-length policy inputs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv

from .config import DEFAULT_SENSOR_OBSERVATION_CONFIG, SensorObservationConfig
from .observation import HistoryMotionReference, ImuSensorLayout, resolve_imu_sensor_layout


PRIVILEGED_CONTEXT_VERSION = 2
TERRAIN_CONTEXT_FIELDS = (
    "terrain_grade",
    "terrain_roughness",
    "terrain_step_height",
    "terrain_step_depth",
    "terrain_gap_width",
    "terrain_pit_depth",
)
TASK_CONTEXT_FIELDS = (
    "target_forward_velocity",
    "motion_phase_progress",
    "episode_progress",
    "course_variant",
    "course_stage",
    "course_start_stage",
    "course_stage_progress",
    "stand_start_progress",
)
RANDOMIZATION_CONTEXT_FIELDS = (
    "ground_friction_ratio",
    "gravity_ratio",
    "action_delay_ratio",
    "external_force_x",
    "external_force_y",
    "external_force_z",
    "external_torque_x",
    "external_torque_y",
    "external_torque_z",
    "push_time_remaining",
    "push_time_to_next",
    "domain_randomization_scale",
)
ESTIMATOR_CONTEXT_FIELDS = (
    "root_velocity_x",
    "root_velocity_y",
    "root_velocity_z",
    "right_foot_contact",
    "left_foot_contact",
    *TERRAIN_CONTEXT_FIELDS,
)
ROOT_VELOCITY_NORMALIZER = 3.0


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
    elif scenario == "stair_up":
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

    # The recurrent locomotion actor has always consumed only the newest sensor
    # frame; its GRU owns the causal history. Keep a stacked actor history only
    # for task wrappers (currently squat) that explicitly expose it.
    actor_history_enabled = False

    def __init__(
        self,
        base_env: LytRobotAmpEnv | LytRobotWarpAmpEnv,
        observation_cfg: SensorObservationConfig = DEFAULT_SENSOR_OBSERVATION_CONFIG,
    ) -> None:
        self.base_env = base_env
        self.sensor_observation_cfg = observation_cfg
        self.is_torch_env = bool(getattr(base_env, "is_torch_env", False))
        if self.is_torch_env:
            # The wrapper supplies the actor observation; avoid assembling the
            # much larger privileged base observation on every environment step.
            self.base_env.actor_observation_enabled = False
        self.imu_layout = resolve_imu_sensor_layout(
            base_env.model,
            accelerometer_scale=observation_cfg.accelerometer_scale,
            gyroscope_scale=observation_cfg.gyroscope_scale,
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    @property
    def sensor_frame_dim(self) -> int:
        return int(self.imu_layout.dim + self.base_env.position_action_dim)

    @property
    def obs_dim(self) -> int:
        return int(self.sensor_frame_dim)

    @property
    def sensor_history_obs_dim(self) -> int:
        return int(self.sensor_frame_dim * self.sensor_observation_cfg.frame_count)

    @property
    def amp_obs_dim(self) -> int:
        return int(self.base_motion.feature_dim * self.sensor_observation_cfg.frame_count)

    @property
    def privileged_context_dim(self) -> int:
        return len(self._privileged_context_fields)

    @property
    def privileged_context_metadata(self) -> dict[str, Any]:
        return {
            "version": PRIVILEGED_CONTEXT_VERSION,
            "fields": self._privileged_context_fields,
            "groups": self._privileged_context_groups,
            "dim": self.privileged_context_dim,
            "actor_input": False,
            "training_only": True,
            "base_observation_noise": False,
            "terrain_encoding": "continuous_bounded_parameters",
            "description": (
                "Clean simulator/reference state, actuator dynamics, contacts, "
                "terrain/course state, motion identity and domain-randomization "
                "parameters used only by the asymmetric critic."
            ),
        }

    @property
    def estimator_context_dim(self) -> int:
        return len(ESTIMATOR_CONTEXT_FIELDS)

    @property
    def estimator_context_metadata(self) -> dict[str, Any]:
        return {
            "fields": ESTIMATOR_CONTEXT_FIELDS,
            "dim": self.estimator_context_dim,
            "actor_input": False,
            "training_target_only": True,
            "root_velocity_normalizer": ROOT_VELOCITY_NORMALIZER,
        }

    @property
    def _task_privileged_fields(self) -> tuple[str, ...]:
        return ()

    def _task_privileged_observe(self) -> np.ndarray | torch.Tensor:
        shape = (self.num_envs, 0)
        if self.is_torch_env:
            return torch.empty(shape, dtype=torch.float32, device=self.base_env.device)
        return np.empty(shape, dtype=np.float32)

    @property
    def observation_metadata(self) -> dict[str, Any]:
        motion_count = int(getattr(self.base_env, "motion_count", 1))
        return {
            "config": asdict(self.sensor_observation_cfg),
            "frame_order": "current_only",
            "imu_sensor_names": self.imu_layout.names,
            "imu_dim_per_frame": self.imu_layout.dim,
            "tendon_length_dim_per_frame": self.base_env.position_action_dim,
            "actor_frame_dim": self.sensor_frame_dim,
            "actor_input_fields": ("imu", "tendon_length"),
            "command_fields": (),
            "command_dim": 0,
            "strict_sensor_only": True,
            "actor_obs_dim": self.obs_dim,
            "sensor_history_obs_dim": self.sensor_history_obs_dim,
            "actor_uses_stacked_history": bool(self.actor_history_enabled),
            "amp_frame_dim": self.base_motion.feature_dim,
            "amp_obs_dim": self.amp_obs_dim,
            "conditional_amp": {
                "version": 1 if motion_count > 1 else 0,
                "condition": "motion_id" if motion_count > 1 else None,
                "condition_dim": motion_count if motion_count > 1 else 0,
                "actor_input": False,
                "training_only": True,
            },
            "privileged_training_context": self.privileged_context_metadata,
            "estimator_training_target": self.estimator_context_metadata,
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
        amp_shape = (self.num_envs, self.sensor_observation_cfg.frame_count, self.base_motion.feature_dim)
        if self.is_torch_env:
            if self.actor_history_enabled:
                actor_shape = (
                    self.num_envs,
                    self.sensor_observation_cfg.frame_count,
                    self.sensor_frame_dim,
                )
                self._actor_history = torch.zeros(
                    actor_shape,
                    dtype=torch.float32,
                    device=self.base_env.device,
                )
            else:
                self._actor_frame = torch.zeros(
                    (self.num_envs, self.sensor_frame_dim),
                    dtype=torch.float32,
                    device=self.base_env.device,
                )
            self._amp_history = torch.zeros(amp_shape, dtype=torch.float32, device=self.base_env.device)
        else:
            if self.actor_history_enabled:
                actor_shape = (
                    self.num_envs,
                    self.sensor_observation_cfg.frame_count,
                    self.sensor_frame_dim,
                )
                self._actor_history = np.zeros(actor_shape, dtype=np.float32)
            else:
                self._actor_frame = np.zeros(
                    (self.num_envs, self.sensor_frame_dim),
                    dtype=np.float32,
                )
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

        force_range = np.asarray(
            self.base_env.model.actuator_forcerange[
                np.asarray(self.base_env.position_actuator_ids_np, dtype=np.int64)
                if hasattr(self.base_env, "position_actuator_ids_np")
                else np.asarray(self.base_env.position_actuator_ids, dtype=np.int64)
            ],
            dtype=np.float32,
        )
        force_scale = np.maximum(np.max(np.abs(force_range), axis=1), 1.0)
        self._privileged_friction_geom_id = (
            int(self.base_env.terrain_geom_ids[0].item())
            if self.is_torch_env
            else int(self.base_env.floor_geom_id)
        )
        frame_counts = np.asarray(
            getattr(
                self.base_motion,
                "frame_counts",
                (int(self.base_motion.frame_count),),
            ),
            dtype=np.float32,
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
            self._actuator_force_scale = torch.as_tensor(
                force_scale, dtype=torch.float32, device=device
            )
            self._motion_frame_counts = torch.as_tensor(
                frame_counts, dtype=torch.float32, device=device
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
            self._actuator_force_scale = force_scale
            self._motion_frame_counts = frame_counts

        fields: list[str] = []
        groups: dict[str, dict[str, int]] = {}

        def add_group(name: str, names: tuple[str, ...]) -> None:
            start = len(fields)
            fields.extend(names)
            groups[name] = {"start": start, "stop": len(fields)}

        action_dim = int(self.base_env.action_dim)
        add_group(
            "clean_simulator_reference_observation",
            tuple(
                f"simulator_reference_observation_{index:03d}"
                for index in range(int(self.base_env.obs_dim))
            ),
        )
        add_group(
            "actuator_control",
            tuple(f"actuator_control_{index:02d}" for index in range(action_dim)),
        )
        add_group(
            "actuator_force",
            tuple(f"actuator_force_{index:02d}" for index in range(action_dim)),
        )
        add_group(
            "actuator_velocity",
            tuple(f"actuator_velocity_{index:02d}" for index in range(action_dim)),
        )
        add_group(
            "detailed_foot_contact",
            tuple(
                f"foot_geom_contact_{index}"
                for index in range(len(self.base_env.contact_geom_ids))
            ),
        )
        add_group("terrain", TERRAIN_CONTEXT_FIELDS)
        add_group(
            "motion_identity",
            tuple(f"motion_id_{index}" for index in range(int(self.base_env.motion_count))),
        )
        add_group("task_and_course", TASK_CONTEXT_FIELDS)
        add_group(
            "body_mass_randomization",
            tuple(
                f"body_mass_ratio_{index:02d}"
                for index in range(int(self.base_env.model.nbody))
            ),
        )
        add_group(
            "tendon_length_randomization",
            tuple(f"tendon_length_scale_{index:02d}" for index in range(action_dim)),
        )
        add_group("randomization_and_external_force", RANDOMIZATION_CONTEXT_FIELDS)
        add_group("task_specific", tuple(self._task_privileged_fields))
        self._privileged_context_fields = tuple(fields)
        self._privileged_context_groups = groups

    def _terrain_context(self) -> np.ndarray | torch.Tensor:
        if self.base_env.more_terrain_cfg.serial_course:
            return self._course_terrain_context[
                self.base_env.course_variant_ids,
                self.base_env.course_stage_ids,
            ]
        return self._motion_terrain_context[self.base_env.motion_ids]

    def _task_course_context(self) -> np.ndarray | torch.Tensor:
        env = self.base_env
        serial = bool(env.more_terrain_cfg.serial_course)
        if self.is_torch_env:
            dtype = torch.float32
            frame_counts = self._motion_frame_counts[env.motion_ids].clamp_min(1.0)
            motion_progress = torch.clamp(env.motion_phase / frame_counts, 0.0, 1.0)
            episode_progress = torch.clamp(
                env.episode_step.to(dtype) / env.episode_timeout_step.to(dtype).clamp_min(1.0),
                0.0,
                1.0,
            )
            if serial:
                ends = env.course_stage_end_x
                variants = env.course_variant_ids
                stages = env.course_stage_ids
                previous = torch.clamp(stages - 1, min=0)
                stage_end = ends[variants, stages]
                stage_start = torch.where(
                    stages > 0,
                    ends[variants, previous],
                    torch.zeros_like(stage_end),
                )
                stage_progress = torch.clamp(
                    (env.qpos[:, 0] - stage_start)
                    / (stage_end - stage_start).clamp_min(1.0e-6),
                    0.0,
                    1.0,
                )
                variant_scale = max(int(ends.shape[0]) - 1, 1)
                stage_scale = max(int(ends.shape[1]) - 1, 1)
                variant = variants.to(dtype) / float(variant_scale)
                stage = stages.to(dtype) / float(stage_scale)
                start_stage = env.course_start_stage_ids.to(dtype) / float(stage_scale)
            else:
                variant = torch.zeros_like(motion_progress)
                stage = torch.zeros_like(motion_progress)
                start_stage = torch.zeros_like(motion_progress)
                stage_progress = torch.zeros_like(motion_progress)
            target_velocity = torch.full_like(
                motion_progress,
                float(np.clip(env.cfg.target_velocity / 3.0, -1.0, 1.0)),
            )
            stand_progress = env._stand_start_progress().to(dtype)
            return torch.stack(
                (
                    target_velocity,
                    motion_progress,
                    episode_progress,
                    variant,
                    stage,
                    start_stage,
                    stage_progress,
                    stand_progress,
                ),
                dim=1,
            )

        frame_counts = self._motion_frame_counts[env.motion_ids]
        motion_progress = np.clip(
            env.motion_phase / np.maximum(frame_counts, 1.0), 0.0, 1.0
        ).astype(np.float32)
        episode_progress = np.clip(
            env.episode_step.astype(np.float32)
            / np.maximum(env.episode_timeout_step.astype(np.float32), 1.0),
            0.0,
            1.0,
        ).astype(np.float32)
        if serial:
            ends = env.course_stage_end_x
            variants = env.course_variant_ids
            stages = env.course_stage_ids
            previous = np.maximum(stages - 1, 0)
            stage_end = ends[variants, stages]
            stage_start = np.where(
                stages > 0,
                ends[variants, previous],
                np.zeros_like(stage_end),
            )
            root_x = np.asarray([data.qpos[0] for data in env.datas], dtype=np.float32)
            stage_progress = np.clip(
                (root_x - stage_start) / np.maximum(stage_end - stage_start, 1.0e-6),
                0.0,
                1.0,
            ).astype(np.float32)
            variant = variants.astype(np.float32) / float(max(ends.shape[0] - 1, 1))
            stage = stages.astype(np.float32) / float(max(ends.shape[1] - 1, 1))
            start_stage = env.course_start_stage_ids.astype(np.float32) / float(
                max(ends.shape[1] - 1, 1)
            )
        else:
            variant = np.zeros_like(motion_progress)
            stage = np.zeros_like(motion_progress)
            start_stage = np.zeros_like(motion_progress)
            stage_progress = np.zeros_like(motion_progress)
        target_velocity = np.full_like(
            motion_progress,
            np.clip(env.cfg.target_velocity / 3.0, -1.0, 1.0),
        )
        return np.stack(
            (
                target_velocity,
                motion_progress,
                episode_progress,
                variant,
                stage,
                start_stage,
                stage_progress,
                env._stand_start_progress().astype(np.float32),
            ),
            axis=1,
        ).astype(np.float32)

    def _randomization_context(self) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        env = self.base_env
        force_normalizer = max(float(env.cfg.push_force_max), 1.0)
        duration_normalizer = max(int(env.cfg.push_duration_max_steps), 1)
        interval_normalizer = max(int(env.cfg.push_interval_max_steps), 1)
        delay_normalizer = max(int(env.max_action_delay_ticks), 1)
        base_friction = (
            env.base_geom_friction_np
            if self.is_torch_env
            else env.base_geom_friction
        )
        base_gravity = env.base_gravity_np if self.is_torch_env else env.base_gravity
        nominal_friction = max(
            float(base_friction[self._privileged_friction_geom_id, 0]),
            1.0e-6,
        )
        nominal_gravity = max(abs(float(base_gravity[2])), 1.0e-6)

        if self.is_torch_env:
            body_mass_ratio = (env.wp_body_mass[0] / env.base_body_mass.clamp_min(1.0e-6)).expand(
                self.num_envs, -1
            )
            friction = (
                env.wp_geom_friction[0, self._privileged_friction_geom_id, 0]
                / nominal_friction
            )
            gravity = torch.abs(env.wp_gravity[0, 2]) / nominal_gravity
            scalars = torch.stack(
                (
                    friction.expand(self.num_envs),
                    gravity.expand(self.num_envs),
                    env.action_delay_ticks.to(torch.float32) / float(delay_normalizer),
                    *(env.push_force[:, index] / force_normalizer for index in range(6)),
                    env.push_time_left.to(torch.float32) / float(duration_normalizer),
                    env.push_time_to_next.to(torch.float32) / float(interval_normalizer),
                    torch.full(
                        (self.num_envs,),
                        float(env.domain_randomization_scale),
                        dtype=torch.float32,
                        device=env.device,
                    ),
                ),
                dim=1,
            )
            return body_mass_ratio, scalars

        body_mass_ratio = env.env_body_mass_scale.copy()
        body_mass_ratio[:, env.pelvis_body_id] += (
            env.env_payload_mass / max(float(env.base_body_mass[env.pelvis_body_id]), 1.0e-6)
        )
        scalars = np.stack(
            (
                env.env_ground_friction / nominal_friction,
                env.env_gravity / nominal_gravity,
                env.action_delay_ticks.astype(np.float32) / float(delay_normalizer),
                *(env.push_force[:, index] / force_normalizer for index in range(6)),
                env.push_time_left.astype(np.float32) / float(duration_normalizer),
                env.push_time_to_next.astype(np.float32) / float(interval_normalizer),
                np.full(self.num_envs, env.domain_randomization_scale, dtype=np.float32),
            ),
            axis=1,
        ).astype(np.float32)
        return body_mass_ratio.astype(np.float32), scalars

    def privileged_observe(self) -> np.ndarray | torch.Tensor:
        """Return rich simulator-only input for the asymmetric critic.

        These values never enter ``observe``, the exported Actor, or the GRU
        estimator target. Only the asymmetric Critic reads the full context.
        """

        if self.is_torch_env:
            env = self.base_env
            base_observation = env.observe(add_noise=False)
            controls = torch.clamp(
                (env.ctrl[:, env.position_actuator_ids] - env.ctrl_mid)
                / env.ctrl_half_range,
                -2.0,
                2.0,
            )
            forces = torch.clamp(
                env.actuator_force[:, env.position_actuator_ids]
                / self._actuator_force_scale,
                -2.0,
                2.0,
            )
            velocities = torch.clamp(
                env.actuator_velocity[:, env.position_actuator_ids] / 20.0,
                -2.0,
                2.0,
            )
            contacts = env.last_foot_geom_contact
            motion_identity = torch.nn.functional.one_hot(
                env.motion_ids, num_classes=env.motion_count
            ).to(torch.float32)
            body_mass_ratio, randomization = self._randomization_context()
            context = torch.cat(
                (
                    base_observation,
                    controls,
                    forces,
                    velocities,
                    contacts,
                    self._terrain_context(),
                    motion_identity,
                    self._task_course_context(),
                    body_mass_ratio,
                    env.env_tendon_length_scale,
                    randomization,
                    self._task_privileged_observe(),
                ),
                dim=1,
            )
            if context.shape[1] != self.privileged_context_dim:
                raise RuntimeError(
                    f"Privileged context layout mismatch: expected {self.privileged_context_dim}, "
                    f"got {context.shape[1]}."
                )
            return torch.clamp(
                torch.nan_to_num(context, nan=0.0, posinf=20.0, neginf=-20.0),
                -20.0,
                20.0,
            )

        env = self.base_env
        base_observation = env.observe(add_noise=False)
        controls = np.stack(
            [data.ctrl[env.position_actuator_ids] for data in env.datas], axis=0
        ).astype(np.float32)
        controls = np.clip(
            (controls - env.ctrl_mid[None, :]) / env.ctrl_half_range[None, :],
            -2.0,
            2.0,
        )
        forces = np.stack(
            [data.actuator_force[env.position_actuator_ids] for data in env.datas], axis=0
        ).astype(np.float32)
        forces = np.clip(forces / self._actuator_force_scale[None, :], -2.0, 2.0)
        velocities = np.stack(
            [data.actuator_velocity[env.position_actuator_ids] for data in env.datas], axis=0
        ).astype(np.float32)
        velocities = np.clip(velocities / 20.0, -2.0, 2.0)
        contacts = np.stack(
            [env._foot_geom_contacts(data) for data in env.datas], axis=0
        )
        motion_identity = np.eye(env.motion_count, dtype=np.float32)[env.motion_ids]
        body_mass_ratio, randomization = self._randomization_context()
        context = np.concatenate(
            (
                base_observation,
                controls,
                forces,
                velocities,
                contacts,
                self._terrain_context(),
                motion_identity,
                self._task_course_context(),
                body_mass_ratio,
                env.env_tendon_length_scale,
                randomization,
                self._task_privileged_observe(),
            ),
            axis=1,
        ).astype(np.float32)
        if context.shape[1] != self.privileged_context_dim:
            raise RuntimeError(
                f"Privileged context layout mismatch: expected {self.privileged_context_dim}, "
                f"got {context.shape[1]}."
            )
        return np.clip(
            np.nan_to_num(context, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        ).astype(np.float32)

    def estimator_observe(self) -> np.ndarray | torch.Tensor:
        """Return only sensor-inferable targets for the GRU auxiliary head."""

        if self.is_torch_env:
            velocity = torch.clamp(
                self.base_env.qvel[:, :3] / ROOT_VELOCITY_NORMALIZER,
                -1.0,
                1.0,
            )
            contacts = self.base_env._foot_contacts()
            return torch.cat((velocity, contacts, self._terrain_context()), dim=1)

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
        return np.concatenate(
            (velocity, contacts, self._terrain_context()),
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
            frame = torch.cat((imu, tendon), dim=1)
            frame = torch.nan_to_num(frame, nan=0.0, posinf=20.0, neginf=-20.0)
            if self.base_env.domain_randomization and self.base_env.cfg.obs_noise_std > 0.0:
                noise_std = (
                    self.base_env.domain_randomization_scale * float(self.base_env.cfg.obs_noise_std)
                )
                frame = frame + noise_std * torch.randn_like(frame)
            return torch.clamp(
                frame,
                -self.sensor_observation_cfg.observation_clip,
                self.sensor_observation_cfg.observation_clip,
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
        frame = np.concatenate((imu, tendon), axis=1).astype(np.float32)
        frame = np.nan_to_num(frame, nan=0.0, posinf=20.0, neginf=-20.0)
        if self.base_env.domain_randomization and self.base_env.cfg.obs_noise_std > 0.0:
            frame += self.base_env.rng.normal(
                0.0,
                self.base_env.domain_randomization_scale * float(self.base_env.cfg.obs_noise_std),
                size=frame.shape,
            ).astype(np.float32)
        return np.clip(
            frame,
            -self.sensor_observation_cfg.observation_clip,
            self.sensor_observation_cfg.observation_clip,
        ).astype(np.float32)

    def _current_amp_frame(self) -> np.ndarray | torch.Tensor:
        return self.base_env.amp_observe()

    def _fill_history(
        self,
        env_ids: np.ndarray | torch.Tensor,
        *,
        actor_frame: np.ndarray | torch.Tensor | None = None,
        amp_frame: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        # ``step`` has already assembled both full-batch frames before it knows
        # which histories need refilling. Reusing them avoids recomputing all
        # sensor and AMP features for every world after each partial reset.
        actor_frame = (
            self._normalized_sensor_frame() if actor_frame is None else actor_frame
        )
        amp_frame = self._current_amp_frame() if amp_frame is None else amp_frame
        if self.is_torch_env:
            ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self.base_env.device,
            )
            if self.actor_history_enabled:
                self._actor_history[ids] = actor_frame[ids, None, :].expand(
                    -1,
                    self.sensor_observation_cfg.frame_count,
                    -1,
                )
            else:
                self._actor_frame[ids] = actor_frame[ids]
            self._amp_history[ids] = amp_frame[ids, None, :].expand(-1, self.sensor_observation_cfg.frame_count, -1)
            return

        ids = np.asarray(env_ids, dtype=np.int64)
        if self.actor_history_enabled:
            self._actor_history[ids] = np.repeat(
                actor_frame[ids, None, :],
                self.sensor_observation_cfg.frame_count,
                axis=1,
            )
        else:
            self._actor_frame[ids] = actor_frame[ids]
        self._amp_history[ids] = np.repeat(
            amp_frame[ids, None, :],
            self.sensor_observation_cfg.frame_count,
            axis=1,
        )

    def _append_history(
        self,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        actor_frame = self._normalized_sensor_frame()
        amp_frame = self._current_amp_frame()
        if self.is_torch_env:
            if self.actor_history_enabled:
                self._actor_history = torch.roll(self._actor_history, shifts=-1, dims=1)
                self._actor_history[:, -1] = actor_frame
            else:
                self._actor_frame = actor_frame
            self._amp_history = torch.roll(self._amp_history, shifts=-1, dims=1)
        else:
            if self.actor_history_enabled:
                self._actor_history = np.roll(self._actor_history, shift=-1, axis=1)
                self._actor_history[:, -1] = actor_frame
            else:
                self._actor_frame = actor_frame
            self._amp_history = np.roll(self._amp_history, shift=-1, axis=1)
        self._amp_history[:, -1] = amp_frame
        return actor_frame, amp_frame

    def observe(self) -> np.ndarray | torch.Tensor:
        if self.actor_history_enabled:
            return self._actor_history[:, -1]
        return self._actor_frame

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

    def step(self, action: np.ndarray | torch.Tensor):
        _base_obs, reward, done, info = self.base_env.step(action)
        actor_frame, amp_frame = self._append_history()
        if self.is_torch_env:
            # Boolean indexing stays on device and avoids a second CUDA sync
            # after the base environment has already handled resets.
            self._fill_history(
                done > 0.0,
                actor_frame=actor_frame,
                amp_frame=amp_frame,
            )
        else:
            done_ids = np.nonzero(np.asarray(done) > 0.0)[0].astype(np.int64)
            if len(done_ids) > 0:
                self._fill_history(
                    done_ids,
                    actor_frame=actor_frame,
                    amp_frame=amp_frame,
                )
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


class SensorHistoryCpuAmpEnv(SensorHistoryEnv):
    """CPU environment used by the isolated playback entry point."""

    def __init__(
        self,
        cfg: LytRobotEnvCfg,
        observation_cfg: SensorObservationConfig = DEFAULT_SENSOR_OBSERVATION_CONFIG,
    ) -> None:
        super().__init__(LytRobotAmpEnv(cfg), observation_cfg)
