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


PRIVILEGED_CONTEXT_FIELDS = (
    "root_velocity_x",
    "root_velocity_y",
    "root_velocity_z",
    "right_foot_contact",
    "left_foot_contact",
    "terrain_grade",
    "terrain_roughness",
    "terrain_step_height",
    "terrain_step_depth",
    "terrain_gap_width",
    "terrain_pit_depth",
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
        self.base_env.terminal_observer = self._terminal_observation

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    @property
    def sensor_frame_dim(self) -> int:
        return int(self.imu_layout.dim + self.base_env.position_action_dim)

    @property
    def obs_dim(self) -> int:
        return int(self.sensor_frame_dim * self.sensor_observation_cfg.frame_count)

    @property
    def amp_obs_dim(self) -> int:
        return int(self.base_motion.feature_dim * self.sensor_observation_cfg.frame_count)

    @property
    def privileged_context_dim(self) -> int:
        return len(PRIVILEGED_CONTEXT_FIELDS)

    @property
    def privileged_context_metadata(self) -> dict[str, Any]:
        return {
            "fields": PRIVILEGED_CONTEXT_FIELDS,
            "dim": self.privileged_context_dim,
            "actor_input": False,
            "training_only": True,
            "root_velocity_normalizer": ROOT_VELOCITY_NORMALIZER,
            "terrain_encoding": "continuous_bounded_parameters",
        }

    @property
    def observation_metadata(self) -> dict[str, Any]:
        return {
            "config": asdict(self.sensor_observation_cfg),
            "frame_order": "oldest_to_current",
            "imu_sensor_names": self.imu_layout.names,
            "imu_dim_per_frame": self.imu_layout.dim,
            "tendon_length_dim_per_frame": self.base_env.position_action_dim,
            "actor_frame_dim": self.sensor_frame_dim,
            "actor_input_fields": ("imu", "tendon_length"),
            "command_fields": (),
            "command_dim": 0,
            "strict_sensor_only": True,
            "actor_obs_dim": self.obs_dim,
            "amp_frame_dim": self.base_motion.feature_dim,
            "amp_obs_dim": self.amp_obs_dim,
            "privileged_training_context": self.privileged_context_metadata,
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
        must reconstruct them from causal IMU and tendon-length feedback.
        """

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


class SensorHistoryCpuAmpEnv(SensorHistoryEnv):
    """CPU environment used by the isolated playback entry point."""

    def __init__(
        self,
        cfg: LytRobotEnvCfg,
        observation_cfg: SensorObservationConfig = DEFAULT_SENSOR_OBSERVATION_CONFIG,
    ) -> None:
        super().__init__(LytRobotAmpEnv(cfg), observation_cfg)
