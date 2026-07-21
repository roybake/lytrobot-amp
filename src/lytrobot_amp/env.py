"""A small vectorized MuJoCo environment for the LYT biped."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import secrets

import mujoco
import numpy as np

from .motion import CONTACT_GEOM_NAMES, MotionReference, MotionTargets, resolve_contact_geom_ids


def _canonical_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat[0] < 0.0:
        quat *= -1.0
    norm = np.linalg.norm(quat)
    return quat / max(float(norm), 1.0e-6)


@dataclass
class LytRobotEnvCfg:
    model_xml: str = "robot/MJCFmodelposition.xml"
    motion_file: str = "robot/reference_data/walk1_mjcf.npz"
    num_envs: int = 4096
    seed: int = 42
    randomize_episode_seed: bool = False
    decimation: int = 4
    timestep: float = 0.005
    solver_iterations: int = 10
    solver_ls_iterations: int = 20
    warp_nconmax: int = 32
    warp_njmax: int = 128
    warp_cuda_graph: bool = True
    episode_steps: int = 1000
    episode_timeout_jitter_steps: int = 240
    target_velocity: float = 1.0
    forward_velocity_tracking_sigma: float = 0.25
    forward_velocity_reward_coef: float = 0.0
    forward_progress_reward_coef: float = 0.0
    forward_velocity_deficit_penalty_coef: float = 0.0
    forward_velocity_excess_penalty_coef: float = 0.0
    reset_noise_scale: float = 0.01
    motion_reset_prob: float = 0.5
    action_residual_scale: float = 0.5
    distal_action_scale: float = 1.0
    reference_ctrl_blend: float = 1.0
    fall_height: float = 0.45
    max_height: float = 1.55
    min_upright_z: float = 0.45
    foot_vel_tracking_sigma: float = 1.0
    stance_foot_vel_sigma: float = 0.15
    foot_vel_reward_coef: float = 0.35
    stance_foot_reward_coef: float = 1.00
    contact_reward_coef: float = 0.70
    missed_contact_penalty_coef: float = 0.80
    side_foot_pos_sigma: float = 0.08
    step_gap_sigma: float = 0.12
    swing_advance_sigma: float = 0.10
    side_foot_pos_coef: float = 0.0
    step_gap_coef: float = 0.0
    swing_advance_coef: float = 0.0
    distal_joint_vel_sigma: float = 2.0
    symmetry_reward_coef: float = 1.20
    symmetry_foot_pos_sigma: float = 0.08
    symmetry_balance_sigma: float = 0.25
    symmetry_ema_decay: float = 0.99
    symmetry_foot_weight: float = 0.60
    symmetry_contact_weight: float = 0.25
    symmetry_action_weight: float = 0.15
    startup_action_ramp_steps: int = 20
    stand_start_prob: float = 0.5
    stand_start_phase_delay_steps: int = 25
    stand_start_transition_min_steps: int = 80
    stand_start_transition_max_steps: int = 160
    stand_start_push_delay_steps: int = 100
    ctrl_penalty_coef: float = 0.040
    action_rate_penalty_coef: float = 0.100
    distal_action_penalty_coef: float = 0.030
    distal_action_rate_penalty_coef: float = 0.120
    domain_randomization: bool = False
    payload_mass_min: float = 0.0
    payload_mass_max: float = 3.0
    body_mass_scale_min: float = 0.85
    body_mass_scale_max: float = 1.15
    tendon_length_scale_min: float = 0.95
    tendon_length_scale_max: float = 1.05
    ground_friction_min: float = 0.55
    ground_friction_max: float = 1.40
    gravity_min: float = 9.75
    gravity_max: float = 9.85
    push_force_min: float = 0.0
    push_force_max: float = 80.0
    push_interval_min_steps: int = 25
    push_interval_max_steps: int = 100
    push_duration_min_steps: int = 2
    push_duration_max_steps: int = 8
    obs_noise_std: float = 0.02
    action_noise_std: float = 0.01
    action_delay_min_ms: float = 2.0
    action_delay_max_ms: float = 5.0


class LytRobotAmpEnv:
    """Serial MuJoCo vector env.

    The environment keeps one ``MjData`` per simulated robot.  It is deliberately
    lightweight so smoke tests can run on CPU without the full mjlab/G1 stack.
    """

    def __init__(self, cfg: LytRobotEnvCfg) -> None:
        self.cfg = cfg
        self.model_path = Path(cfg.model_xml).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model XML not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.model.opt.timestep = float(cfg.timestep)
        self.model.opt.iterations = max(1, int(cfg.solver_iterations))
        self.model.opt.ls_iterations = max(1, int(cfg.solver_ls_iterations))
        self.position_actuator_ids = np.arange(self.model.nu, dtype=np.int32)
        self.position_action_dim = int(len(self.position_actuator_ids))
        self.action_dim = self.position_action_dim
        self.ctrl_dim = int(self.model.nu)
        self.motion = MotionReference(cfg.motion_file, self.model, actuator_ids=self.position_actuator_ids)
        self.rng = np.random.default_rng(cfg.seed)
        self.base_body_mass = np.asarray(self.model.body_mass, dtype=np.float32).copy()
        self.base_body_inertia = np.asarray(self.model.body_inertia, dtype=np.float32).copy()
        self.base_geom_friction = np.asarray(self.model.geom_friction, dtype=np.float32).copy()
        self.base_gravity = np.asarray(self.model.opt.gravity, dtype=np.float32).copy()

        self.num_envs = int(cfg.num_envs)
        self.ctrl_low = np.asarray(self.model.actuator_ctrlrange[self.position_actuator_ids, 0], dtype=np.float32)
        self.ctrl_high = np.asarray(self.model.actuator_ctrlrange[self.position_actuator_ids, 1], dtype=np.float32)
        self.ctrl_mid = 0.5 * (self.ctrl_low + self.ctrl_high)
        self.ctrl_half_range = np.maximum(0.5 * (self.ctrl_high - self.ctrl_low), 1.0e-6)
        self.actuator_tendon_ids = np.asarray(self.model.actuator_trnid[self.position_actuator_ids, 0], dtype=np.int32)
        self._init_stand_reference()
        self.distal_action_mask = self._build_distal_action_mask()
        self.has_distal_action = bool(np.any(self.distal_action_mask))
        self.action_scale = self._build_action_scale()

        self.nonroot_qpos_adrs, self.nonroot_qvel_adrs = self._get_nonroot_joint_adrs()
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_body_id < 0:
            raise RuntimeError("Body 'pelvis' not found in robot model.")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.contact_geom_ids = resolve_contact_geom_ids(self.model)
        if self.floor_geom_id < 0:
            raise RuntimeError("Expected floor and foot contact geoms were not found in robot model.")
        self.right_contact_geom_ids = set(int(geom_id) for geom_id in self.contact_geom_ids[:2])
        self.left_contact_geom_ids = set(int(geom_id) for geom_id in self.contact_geom_ids[2:])
        self.contact_geom_id_to_index = {
            int(geom_id): contact_index for contact_index, geom_id in enumerate(self.contact_geom_ids)
        }

        self.joint_reward_weights = self._build_joint_reward_weights()
        self.distal_joint_indices = np.asarray(
            [
                joint_id
                for joint_id, name in enumerate(self.motion.common_joint_names)
                if "ankle" in name or "subtalar" in name or "mtp" in name
            ],
            dtype=np.int32,
        )
        self.right_action_indices, self.left_action_indices = self._build_side_action_indices()

        self.datas = [mujoco.MjData(self.model) for _ in range(self.num_envs)]
        self.episode_step = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_timeout_step = np.full(self.num_envs, int(cfg.episode_steps), dtype=np.int32)
        self.motion_phase = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_seed = np.full(self.num_envs, int(cfg.seed), dtype=np.uint64)
        self.last_action = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.last_foot_pos = np.zeros((self.num_envs, len(CONTACT_GEOM_NAMES), 3), dtype=np.float32)
        self.last_foot_world_pos = np.zeros_like(self.last_foot_pos)
        self.foot_stride_ema = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.foot_x_min = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.foot_x_max = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.contact_balance_ema = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.action_energy_ema = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.last_obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self.stand_start = np.zeros(self.num_envs, dtype=bool)
        self.stand_start_transition_steps = np.ones(self.num_envs, dtype=np.int32)
        self._init_domain_randomization()
        self.reset()

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.cfg.decimation)

    @property
    def obs_dim(self) -> int:
        current_state_dim = 1 + 4 + 6 + 2 * len(self.nonroot_qpos_adrs) + self.action_dim + 3
        tracking_dim = (
            1
            + 4
            + 6
            + 2 * len(self.motion.common_joint_names)
            + self.action_dim
            + self.motion.foot_pos_dim
            + self.motion.contact_dim
        )
        return int(current_state_dim + tracking_dim)

    @property
    def amp_obs_dim(self) -> int:
        return self.motion.feature_dim

    def _get_nonroot_joint_adrs(self) -> tuple[np.ndarray, np.ndarray]:
        qpos_adrs: list[int] = []
        qvel_adrs: list[int] = []
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            qpos_adrs.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_adrs.append(int(self.model.jnt_dofadr[joint_id]))
        return np.asarray(qpos_adrs, dtype=np.int32), np.asarray(qvel_adrs, dtype=np.int32)

    def _build_joint_reward_weights(self) -> np.ndarray:
        weights = []
        for name in self.motion.common_joint_names:
            if "knee" in name:
                weights.append(1.35)
            elif "ankle" in name:
                weights.append(1.60)
            elif "subtalar" in name:
                weights.append(0.80)
            elif "mtp" in name:
                weights.append(0.60)
            else:
                weights.append(1.0)
        return np.asarray(weights, dtype=np.float32)

    def _build_action_scale(self) -> np.ndarray:
        scale = np.ones(self.position_action_dim, dtype=np.float32)
        distal_scale = float(np.clip(self.cfg.distal_action_scale, 0.05, 1.0))
        scale[self.distal_position_action_mask] = distal_scale
        return scale

    def _build_distal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=bool)
        self.distal_position_action_mask = np.zeros(self.position_action_dim, dtype=bool)
        for action_id, actuator_id in enumerate(self.position_actuator_ids):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
            tendon_id = int(self.model.actuator_trnid[actuator_id, 0])
            tendon_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id) or ""
            if "ankle" in actuator_name or "ankle" in tendon_name:
                mask[action_id] = True
                self.distal_position_action_mask[action_id] = True
        return mask

    def _build_side_action_indices(self) -> tuple[np.ndarray, np.ndarray]:
        right: list[int] = []
        left: list[int] = []
        for action_id, actuator_id in enumerate(self.position_actuator_ids):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
            tendon_id = int(self.model.actuator_trnid[actuator_id, 0])
            tendon_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id) or ""
            name = f"{actuator_name} {tendon_name}"
            if "_r" in name:
                right.append(action_id)
            elif "_l" in name:
                left.append(action_id)
        return np.asarray(right, dtype=np.int32), np.asarray(left, dtype=np.int32)

    def _init_stand_reference(self) -> None:
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.motion.model_qpos0
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        tendon_lengths = np.asarray(data.ten_length, dtype=np.float32)
        tendon_ctrl = np.clip(tendon_lengths[self.actuator_tendon_ids], self.ctrl_low, self.ctrl_high)
        self.stand_root_height = np.asarray([data.qpos[2]], dtype=np.float32)
        self.stand_root_quat = _canonical_quat(data.qpos[3:7])[None, :]
        self.stand_root_lin_vel = np.zeros((1, 3), dtype=np.float32)
        self.stand_root_ang_vel = np.zeros((1, 3), dtype=np.float32)
        self.stand_joint_pos = np.asarray(data.qpos[self.motion.robot_qpos_adrs], dtype=np.float32)[None, :]
        self.stand_joint_vel = np.zeros((1, len(self.motion.robot_qvel_adrs)), dtype=np.float32)
        self.stand_tendon_ctrl = tendon_ctrl[None, :]
        self.stand_foot_pos = self.motion.foot_positions_local_from_data(data)[None, :]
        self.stand_foot_vel = np.zeros((1, len(CONTACT_GEOM_NAMES), 3), dtype=np.float32)
        self.stand_foot_contact = np.ones((1, 2), dtype=np.float32)

    def _sample_stand_transition_steps(self, size: int) -> np.ndarray:
        low = max(1, int(self.cfg.stand_start_transition_min_steps))
        high = max(low, int(self.cfg.stand_start_transition_max_steps))
        if high <= low:
            return np.full(size, low, dtype=np.int32)
        return self.rng.integers(low, high + 1, size=size, dtype=np.int32)

    def _refresh_episode_rng(self, ids: np.ndarray) -> None:
        if not self.cfg.randomize_episode_seed or len(ids) == 0:
            return
        seeds = np.asarray([secrets.randbits(64) for _ in range(len(ids))], dtype=np.uint64)
        self.episode_seed[ids] = seeds
        seed_words = [int(seed & np.uint64(0xFFFFFFFF)) for seed in seeds]
        seed_words += [int(seed >> np.uint64(32)) for seed in seeds]
        self.rng = np.random.default_rng(np.random.SeedSequence(seed_words))

    def _stand_start_progress(self) -> np.ndarray:
        delay = max(0, int(self.cfg.stand_start_phase_delay_steps))
        denom = np.maximum(self.stand_start_transition_steps.astype(np.float32), 1.0)
        raw = (self.episode_step.astype(np.float32) - float(delay)) / denom
        progress = np.clip(raw, 0.0, 1.0)
        progress = progress * progress * (3.0 - 2.0 * progress)
        return np.where(self.stand_start, progress, 1.0).astype(np.float32)

    def _blend_start_targets(self, targets: MotionTargets) -> MotionTargets:
        progress = self._stand_start_progress()
        if not np.any(progress < 1.0):
            return targets

        def blend(stand: np.ndarray, walk: np.ndarray) -> np.ndarray:
            weight_shape = (self.num_envs,) + (1,) * (walk.ndim - 1)
            weight = progress.reshape(weight_shape)
            return ((1.0 - weight) * stand + weight * walk).astype(np.float32)

        root_quat = blend(self.stand_root_quat, targets.root_quat)
        root_quat /= np.maximum(np.linalg.norm(root_quat, axis=1, keepdims=True), 1.0e-6)
        return MotionTargets(
            root_height=blend(self.stand_root_height, targets.root_height),
            root_quat=root_quat.astype(np.float32),
            root_lin_vel=blend(self.stand_root_lin_vel, targets.root_lin_vel),
            root_ang_vel=blend(self.stand_root_ang_vel, targets.root_ang_vel),
            joint_pos=blend(self.stand_joint_pos, targets.joint_pos),
            joint_vel=blend(self.stand_joint_vel, targets.joint_vel),
            tendon_ctrl=blend(self.stand_tendon_ctrl, targets.tendon_ctrl),
            foot_pos=blend(self.stand_foot_pos, targets.foot_pos),
            foot_vel=blend(self.stand_foot_vel, targets.foot_vel),
            foot_contact=blend(self.stand_foot_contact, targets.foot_contact),
        )

    def _targets_at_current_phase(self) -> MotionTargets:
        return self._blend_start_targets(self.motion.targets_at_phase(self.motion_phase))

    def _init_domain_randomization(self) -> None:
        self.domain_randomization = bool(self.cfg.domain_randomization)
        max_delay_s = max(float(self.cfg.action_delay_min_ms), float(self.cfg.action_delay_max_ms), 0.0) / 1000.0
        self.max_action_delay_ticks = int(np.ceil(max_delay_s / max(float(self.cfg.timestep), 1.0e-9)))
        history_len = max(1, self.max_action_delay_ticks + 1)
        self.action_delay_history = np.zeros((self.num_envs, history_len, self.action_dim), dtype=np.float32)
        self.action_delay_cursor = np.zeros(self.num_envs, dtype=np.int32)
        self.action_delay_ticks = np.zeros(self.num_envs, dtype=np.int32)
        self.env_tendon_length_scale = np.ones((self.num_envs, self.position_action_dim), dtype=np.float32)
        self.env_body_mass_scale = np.ones((self.num_envs, self.model.nbody), dtype=np.float32)
        self.env_payload_mass = np.zeros(self.num_envs, dtype=np.float32)
        self.env_ground_friction = np.full(self.num_envs, float(self.base_geom_friction[self.floor_geom_id, 0]) if hasattr(self, "floor_geom_id") else 1.0, dtype=np.float32)
        self.env_gravity = np.full(self.num_envs, abs(float(self.base_gravity[2])), dtype=np.float32)
        self.push_force = np.zeros((self.num_envs, 6), dtype=np.float32)
        self.push_time_left = np.zeros(self.num_envs, dtype=np.int32)
        self.push_time_to_next = np.zeros(self.num_envs, dtype=np.int32)

    def _uniform(self, low: float, high: float, size: int | tuple[int, ...] | None = None) -> np.ndarray:
        low_f = float(low)
        high_f = float(high)
        if high_f < low_f:
            low_f, high_f = high_f, low_f
        if high_f == low_f:
            return np.full(size or (), low_f, dtype=np.float32)
        return np.asarray(self.rng.uniform(low_f, high_f, size=size), dtype=np.float32)

    def _sample_domain_randomization(self, ids: np.ndarray) -> None:
        if not self.domain_randomization or len(ids) == 0:
            return
        ids = np.asarray(ids, dtype=np.int32)
        self.env_payload_mass[ids] = self._uniform(self.cfg.payload_mass_min, self.cfg.payload_mass_max, len(ids))
        self.env_body_mass_scale[ids] = self._uniform(
            self.cfg.body_mass_scale_min,
            self.cfg.body_mass_scale_max,
            (len(ids), self.model.nbody),
        )
        self.env_body_mass_scale[ids, 0] = 1.0
        self.env_tendon_length_scale[ids] = self._uniform(
            self.cfg.tendon_length_scale_min,
            self.cfg.tendon_length_scale_max,
            (len(ids), self.position_action_dim),
        )
        self.env_ground_friction[ids] = self._uniform(
            self.cfg.ground_friction_min,
            self.cfg.ground_friction_max,
            len(ids),
        )
        self.env_gravity[ids] = self._uniform(self.cfg.gravity_min, self.cfg.gravity_max, len(ids))

        min_delay = max(0.0, float(self.cfg.action_delay_min_ms)) / 1000.0
        max_delay = max(min_delay, float(self.cfg.action_delay_max_ms) / 1000.0)
        delay_seconds = self.rng.uniform(min_delay, max_delay, size=len(ids))
        delay_ticks = np.ceil(delay_seconds / max(float(self.cfg.timestep), 1.0e-9)).astype(np.int32)
        self.action_delay_ticks[ids] = np.clip(delay_ticks, 0, self.max_action_delay_ticks)
        self.action_delay_history[ids] = 0.0
        self.action_delay_cursor[ids] = 0
        self.push_force[ids] = 0.0
        self.push_time_left[ids] = 0
        self.push_time_to_next[ids] = self._sample_push_interval(len(ids))

    def _sample_push_interval(self, size: int) -> np.ndarray:
        low = max(1, int(self.cfg.push_interval_min_steps))
        high = max(low + 1, int(self.cfg.push_interval_max_steps) + 1)
        return self.rng.integers(low, high, size=size, dtype=np.int32)

    def _sample_push_duration(self) -> int:
        low = max(1, int(self.cfg.push_duration_min_steps))
        high = max(low + 1, int(self.cfg.push_duration_max_steps) + 1)
        return int(self.rng.integers(low, high))

    def _apply_model_randomization(self, env_id: int) -> None:
        if not self.domain_randomization:
            return
        body_mass = self.base_body_mass * self.env_body_mass_scale[env_id]
        if self.pelvis_body_id >= 0:
            body_mass[self.pelvis_body_id] += self.env_payload_mass[env_id]
        self.model.body_mass[:] = body_mass
        self.model.body_inertia[:] = self.base_body_inertia * self.env_body_mass_scale[env_id, :, None]
        self.model.body_inertia[0] = self.base_body_inertia[0]
        self.model.geom_friction[:] = self.base_geom_friction
        self.model.geom_friction[self.floor_geom_id, 0] = self.env_ground_friction[env_id]
        self.model.opt.gravity[:] = self.base_gravity
        self.model.opt.gravity[2] = -float(self.env_gravity[env_id])

    def _apply_tendon_length_randomization(self, env_id: int, ctrl: np.ndarray) -> np.ndarray:
        if not self.domain_randomization:
            return ctrl
        scaled = self.ctrl_mid + (np.asarray(ctrl, dtype=np.float32) - self.ctrl_mid) * self.env_tendon_length_scale[env_id]
        return np.clip(scaled, self.ctrl_low, self.ctrl_high)

    def _apply_action_noise(self, action: np.ndarray) -> np.ndarray:
        if not self.domain_randomization or self.cfg.action_noise_std <= 0.0:
            return action
        return np.clip(
            action + self.rng.normal(0.0, float(self.cfg.action_noise_std), size=action.shape).astype(np.float32),
            -1.0,
            1.0,
        )

    def _startup_action_ramp(self, env_id: int) -> float:
        ramp_steps = max(0, int(self.cfg.startup_action_ramp_steps))
        if ramp_steps <= 0:
            return 1.0
        return float(np.clip((int(self.episode_step[env_id]) + 1) / ramp_steps, 0.0, 1.0))

    def _delayed_action(self, env_id: int, action: np.ndarray) -> np.ndarray:
        if not self.domain_randomization or self.max_action_delay_ticks <= 0:
            return action
        cursor = int(self.action_delay_cursor[env_id])
        self.action_delay_history[env_id, cursor] = action
        delay = int(self.action_delay_ticks[env_id])
        delayed = self.action_delay_history[env_id, (cursor - delay) % self.action_delay_history.shape[1]]
        self.action_delay_cursor[env_id] = (cursor + 1) % self.action_delay_history.shape[1]
        return delayed.copy()

    def _update_push_force(self, env_id: int) -> None:
        if not self.domain_randomization:
            return
        if self.push_time_left[env_id] > 0:
            self.push_time_left[env_id] -= 1
            if self.push_time_left[env_id] <= 0:
                self.push_force[env_id] = 0.0
                self.push_time_to_next[env_id] = self._sample_push_interval(1)[0]
            return
        self.push_time_to_next[env_id] -= 1
        if self.push_time_to_next[env_id] > 0:
            return
        direction = self.rng.normal(0.0, 1.0, size=3).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
        magnitude = float(self._uniform(self.cfg.push_force_min, self.cfg.push_force_max))
        self.push_force[env_id, :3] = direction * magnitude
        self.push_force[env_id, 3:] = 0.0
        self.push_time_left[env_id] = self._sample_push_duration()

    def action_to_ctrl(
        self,
        action: np.ndarray,
        base_ctrl: np.ndarray | None = None,
        env_id: int | None = None,
    ) -> np.ndarray:
        position_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if base_ctrl is None:
            base = self.ctrl_mid
        else:
            base = np.asarray(base_ctrl, dtype=np.float32)
        position_ctrl = base + (
            position_action * self.ctrl_half_range * self.action_scale * float(self.cfg.action_residual_scale)
        )
        position_ctrl = np.clip(position_ctrl, self.ctrl_low, self.ctrl_high)
        if env_id is not None:
            position_ctrl = self._apply_tendon_length_randomization(env_id, position_ctrl)
        return position_ctrl

    def ctrl_to_action(self, ctrl: np.ndarray) -> np.ndarray:
        ctrl_arr = np.asarray(ctrl, dtype=np.float32)
        return np.clip((ctrl_arr - self.ctrl_mid) / self.ctrl_half_range, -1.0, 1.0)

    def _current_tendon_ctrl(self, data: mujoco.MjData) -> np.ndarray:
        tendon_lengths = np.asarray(data.ten_length, dtype=np.float32)
        ctrl = tendon_lengths[self.actuator_tendon_ids]
        return np.clip(ctrl, self.ctrl_low, self.ctrl_high)

    def _reference_base_ctrl(self, reference_ctrl: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        blend = float(np.clip(self.cfg.reference_ctrl_blend, 0.0, 1.0))
        if blend <= 0.0:
            return self._current_tendon_ctrl(data)
        current = self._current_tendon_ctrl(data)
        ref = np.asarray(reference_ctrl, dtype=np.float32)
        return np.clip((1.0 - blend) * current + blend * ref, self.ctrl_low, self.ctrl_high)

    def _foot_positions_local(self, data: mujoco.MjData) -> np.ndarray:
        return self.motion.foot_positions_local_from_data(data)

    def _foot_positions_world(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.geom_xpos[self.contact_geom_ids], dtype=np.float32).copy()

    @staticmethod
    def _side_contacts_from_geom(geom_contact: np.ndarray) -> np.ndarray:
        geom_contact_arr = np.asarray(geom_contact, dtype=np.float32)
        return np.asarray(
            [np.max(geom_contact_arr[0:2]), np.max(geom_contact_arr[2:4])],
            dtype=np.float32,
        )

    def _foot_geom_contacts(self, data: mujoco.MjData) -> np.ndarray:
        contact = np.zeros(len(CONTACT_GEOM_NAMES), dtype=np.float32)
        for contact_id in range(data.ncon):
            geom1 = int(data.contact[contact_id].geom1)
            geom2 = int(data.contact[contact_id].geom2)
            other = None
            if geom1 == self.floor_geom_id:
                other = geom2
            elif geom2 == self.floor_geom_id:
                other = geom1
            if other is None:
                continue
            contact_index = self.contact_geom_id_to_index.get(other)
            if contact_index is not None:
                contact[contact_index] = 1.0
        return contact

    def _foot_contacts(self, data: mujoco.MjData) -> np.ndarray:
        return self._side_contacts_from_geom(self._foot_geom_contacts(data))

    def reset(self, env_ids: np.ndarray | list[int] | None = None) -> np.ndarray:
        if env_ids is None:
            ids = np.arange(self.num_envs, dtype=np.int32)
        else:
            ids = np.asarray(env_ids, dtype=np.int32)
        self._refresh_episode_rng(ids)

        states = self.motion.sample_states(len(ids), self.rng)
        use_stand_start = self.rng.random(len(ids)) < float(np.clip(self.cfg.stand_start_prob, 0.0, 1.0))
        use_motion_reset = (
            ~use_stand_start
            & (self.rng.random(len(ids)) < float(np.clip(self.cfg.motion_reset_prob, 0.0, 1.0)))
        )
        self._sample_domain_randomization(ids)
        self.stand_start[ids] = use_stand_start
        self.stand_start_transition_steps[ids] = self._sample_stand_transition_steps(len(ids))
        for row, env_id in enumerate(ids):
            self._apply_model_randomization(int(env_id))
            data = self.datas[int(env_id)]
            mujoco.mj_resetData(self.model, data)

            if use_stand_start[row]:
                qpos = self.motion.model_qpos0.copy()
                qvel = np.zeros(self.model.nv, dtype=np.float32)
                phase = 0.0
            elif use_motion_reset[row]:
                qpos = states.qpos[row].copy()
                qvel = states.qvel[row].copy()
                phase = float(states.indices[row])
            else:
                qpos = self.motion.model_qpos0.copy()
                qvel = np.zeros(self.model.nv, dtype=np.float32)
                phase = 0.0

            if self.cfg.reset_noise_scale > 0.0:
                qpos[self.motion.robot_qpos_adrs] += self.rng.normal(
                    0.0, self.cfg.reset_noise_scale, size=len(self.motion.robot_qpos_adrs)
                ).astype(np.float32)
                qvel[self.motion.robot_qvel_adrs] += self.rng.normal(
                    0.0, 2.0 * self.cfg.reset_noise_scale, size=len(self.motion.robot_qvel_adrs)
                ).astype(np.float32)
                self._clip_joint_ranges(qpos)

            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(self.model, data)
            reference_ctrl = self.motion.targets_at_phase(np.asarray([phase], dtype=np.float32)).tendon_ctrl[0]
            if use_stand_start[row]:
                reference_ctrl = self.stand_tendon_ctrl[0]
            data.ctrl[:] = self.action_to_ctrl(
                np.zeros(self.action_dim, dtype=np.float32),
                base_ctrl=self._reference_base_ctrl(reference_ctrl, data),
                env_id=int(env_id),
            )
            self.last_action[int(env_id)] = 0.0
            self.action_delay_history[int(env_id)] = 0.0
            self.action_delay_cursor[int(env_id)] = 0
            foot_pos = self._foot_positions_local(data)
            self.last_foot_pos[int(env_id)] = foot_pos
            self.last_foot_world_pos[int(env_id)] = self._foot_positions_world(data)
            side_x = np.asarray(
                [
                    np.mean(foot_pos[0:2, 0]),
                    np.mean(foot_pos[2:4, 0]),
                ],
                dtype=np.float32,
            )
            self.foot_x_min[int(env_id)] = side_x
            self.foot_x_max[int(env_id)] = side_x
            self.foot_stride_ema[int(env_id)] = 0.0
            self.contact_balance_ema[int(env_id)] = 0.0
            self.action_energy_ema[int(env_id)] = 0.0
            self.episode_step[int(env_id)] = 0
            self.episode_timeout_step[int(env_id)] = self._sample_timeout_step()
            self.motion_phase[int(env_id)] = phase

        self.last_obs = self.observe()
        return self.last_obs.copy()

    def _sample_timeout_step(self) -> int:
        base_steps = int(self.cfg.episode_steps)
        jitter = int(max(self.cfg.episode_timeout_jitter_steps, 0))
        if jitter <= 0:
            return base_steps
        low = max(1, base_steps - jitter)
        high = max(low + 1, base_steps + jitter + 1)
        return int(self.rng.integers(low, high))

    def _clip_joint_ranges(self, qpos: np.ndarray) -> None:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if not self.model.jnt_limited[joint_id]:
                continue
            adr = int(self.model.jnt_qposadr[joint_id])
            low, high = self.model.jnt_range[joint_id]
            qpos[adr] = np.clip(qpos[adr], low, high)

    def observe(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        reference_targets = self._targets_at_current_phase()
        stand_start_progress = self._stand_start_progress()
        for env_id, data in enumerate(self.datas):
            quat = _canonical_quat(data.qpos[3:7])
            up_z = float(data.xmat[self.pelvis_body_id, 8])
            phase = 2.0 * np.pi * float(self.motion_phase[env_id]) / float(self.motion.frame_count)
            joint_pos = np.asarray(data.qpos[self.motion.robot_qpos_adrs], dtype=np.float32)
            joint_vel = np.asarray(data.qvel[self.motion.robot_qvel_adrs], dtype=np.float32)
            foot_pos = self._foot_positions_local(data).reshape(-1)
            ref_foot_pos = reference_targets.foot_pos[env_id].reshape(-1)
            ref_ctrl_action = self.ctrl_to_action(reference_targets.tendon_ctrl[env_id])
            root_vel_target = np.concatenate(
                (reference_targets.root_lin_vel[env_id], reference_targets.root_ang_vel[env_id])
            ).astype(np.float32)
            root_vel_target[0] = float(self.cfg.target_velocity) * float(stand_start_progress[env_id])
            obs[env_id] = np.concatenate(
                (
                    np.asarray([data.qpos[2]], dtype=np.float32),
                    quat,
                    np.asarray(data.qvel[:6], dtype=np.float32),
                    np.asarray(data.qpos[self.nonroot_qpos_adrs], dtype=np.float32),
                    np.asarray(data.qvel[self.nonroot_qvel_adrs], dtype=np.float32),
                    self.last_action[env_id],
                    np.asarray([up_z, np.sin(phase), np.cos(phase)], dtype=np.float32),
                    np.asarray([data.qpos[2] - reference_targets.root_height[env_id]], dtype=np.float32),
                    reference_targets.root_quat[env_id],
                    np.asarray(data.qvel[:6], dtype=np.float32) - root_vel_target,
                    joint_pos - reference_targets.joint_pos[env_id],
                    joint_vel - reference_targets.joint_vel[env_id],
                    ref_ctrl_action,
                    foot_pos - ref_foot_pos,
                    reference_targets.foot_contact[env_id],
                )
            )
        obs = np.nan_to_num(obs, nan=0.0, posinf=20.0, neginf=-20.0)
        if self.domain_randomization and self.cfg.obs_noise_std > 0.0:
            obs += self.rng.normal(0.0, float(self.cfg.obs_noise_std), size=obs.shape).astype(np.float32)
        return np.clip(obs, -20.0, 20.0).astype(np.float32)

    def amp_observe(self) -> np.ndarray:
        qpos = np.stack([np.asarray(data.qpos, dtype=np.float32) for data in self.datas], axis=0)
        qvel = np.stack([np.asarray(data.qvel, dtype=np.float32) for data in self.datas], axis=0)
        foot_pos = np.stack([self._foot_positions_local(data) for data in self.datas], axis=0)
        return self.motion.features_from_state(qpos, qvel, foot_pos=foot_pos)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        actions = np.asarray(action, dtype=np.float32)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape {(self.num_envs, self.action_dim)}, got {actions.shape}")

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        time_outs = np.zeros(self.num_envs, dtype=bool)
        episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        components: dict[str, list[float]] = {
            "root_vel": [],
            "forward_velocity": [],
            "forward_velocity_error": [],
            "forward_velocity_track": [],
            "forward_progress": [],
            "forward_velocity_deficit": [],
            "forward_velocity_excess": [],
            "termination": [],
            "root_orientation": [],
            "height": [],
            "pose_track": [],
            "vel_track": [],
            "foot_pos": [],
            "side_foot_pos": [],
            "right_foot_pos": [],
            "left_foot_pos": [],
            "step_gap": [],
            "step_gap_error": [],
            "swing_advance": [],
            "right_swing_advance": [],
            "left_swing_advance": [],
            "foot_vel": [],
            "stance_foot": [],
            "stance_foot_speed": [],
            "contact": [],
            "missed_contact": [],
            "distal_vel": [],
            "symmetry": [],
            "foot_symmetry": [],
            "stride_symmetry": [],
            "phase_symmetry": [],
            "lateral_symmetry": [],
            "contact_balance": [],
            "action_balance": [],
            "ctrl_track": [],
            "ctrl_penalty": [],
            "action_rate": [],
            "distal_action": [],
            "distal_action_rate": [],
        }
        phase_scale = self._stand_start_progress()
        self.motion_phase = (
            self.motion_phase + self.motion.frequency * self.dt * phase_scale
        ) % self.motion.frame_count
        reference_targets = self._targets_at_current_phase()

        for env_id, data in enumerate(self.datas):
            self._apply_model_randomization(env_id)
            if (not self.stand_start[env_id]) or self.episode_step[env_id] >= int(self.cfg.stand_start_push_delay_steps):
                self._update_push_force(env_id)
            else:
                self.push_force[env_id] = 0.0
                self.push_time_left[env_id] = 0
                self.push_time_to_next[env_id] = max(
                    int(self.push_time_to_next[env_id]),
                    max(1, int(self.cfg.push_interval_min_steps)),
                )
            prev_action = self.last_action[env_id].copy()
            commanded_action = self._apply_action_noise(np.clip(actions[env_id], -1.0, 1.0))
            commanded_action = commanded_action * self._startup_action_ramp(env_id)
            base_ctrl = self._reference_base_ctrl(reference_targets.tendon_ctrl[env_id], data)
            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[self.pelvis_body_id] = self.push_force[env_id]
            applied_action = commanded_action
            for _ in range(self.cfg.decimation):
                applied_action = self._delayed_action(env_id, commanded_action)
                data.ctrl[:] = self.action_to_ctrl(applied_action, base_ctrl=base_ctrl, env_id=env_id)
                mujoco.mj_step(self.model, data)

            self.episode_step[env_id] += 1
            foot_pos = self._foot_positions_local(data)
            foot_vel = (foot_pos - self.last_foot_pos[env_id]) / max(self.dt, 1.0e-6)
            foot_world_pos = self._foot_positions_world(data)
            foot_world_vel = (foot_world_pos - self.last_foot_world_pos[env_id]) / max(self.dt, 1.0e-6)
            foot_geom_contact = self._foot_geom_contacts(data)
            foot_contact = self._side_contacts_from_geom(foot_geom_contact)
            reward, done, timed_out, reward_parts = self._compute_reward_done(
                data,
                commanded_action,
                prev_action,
                env_id,
                float(self.cfg.target_velocity) * float(phase_scale[env_id]),
                reference_targets,
                foot_pos,
                foot_vel,
                foot_world_vel,
                foot_geom_contact,
                foot_contact,
            )
            rewards[env_id] = reward
            dones[env_id] = done
            time_outs[env_id] = timed_out
            episode_steps[env_id] = self.episode_step[env_id]
            self.last_action[env_id] = commanded_action
            self.last_foot_pos[env_id] = foot_pos
            self.last_foot_world_pos[env_id] = foot_world_pos
            for key, value in reward_parts.items():
                components.setdefault(key, []).append(float(value))

        done_ids = np.nonzero(dones)[0].astype(np.int32)
        if len(done_ids) > 0:
            self.reset(done_ids)
        else:
            self.last_obs = self.observe()

        info = {key: np.asarray(value, dtype=np.float32) for key, value in components.items()}
        info["episode_step"] = episode_steps
        info["time_outs"] = time_outs.astype(np.float32)
        return self.last_obs.copy(), rewards, dones.astype(np.float32), info

    def _compute_reward_done(
        self,
        data: mujoco.MjData,
        action: np.ndarray,
        prev_action: np.ndarray,
        env_id: int,
        target_forward_velocity: float,
        reference_targets: MotionTargets,
        foot_pos: np.ndarray,
        foot_vel: np.ndarray,
        foot_world_vel: np.ndarray,
        foot_geom_contact: np.ndarray,
        foot_contact: np.ndarray,
    ) -> tuple[float, bool, bool, dict[str, float]]:
        root_height = float(data.qpos[2])
        up_z = float(data.xmat[self.pelvis_body_id, 8])
        root_quat = _canonical_quat(data.qpos[3:7])
        ref_root_quat = _canonical_quat(reference_targets.root_quat[env_id])
        ref_root_height = float(reference_targets.root_height[env_id])

        quat_dot = float(np.clip(abs(np.dot(root_quat, ref_root_quat)), 0.0, 1.0))
        orient_error = 2.0 * np.arccos(quat_dot)
        root_orientation = float(np.exp(-(orient_error**2) / (0.35**2)))

        root_lin_vel_target = reference_targets.root_lin_vel[env_id].copy()
        root_lin_vel_target[0] = target_forward_velocity
        lin_vel_error = np.asarray(data.qvel[:3], dtype=np.float32) - root_lin_vel_target
        ang_vel_error = np.asarray(data.qvel[3:6], dtype=np.float32) - reference_targets.root_ang_vel[env_id]
        lin_vel_track = np.exp(-float(np.mean(np.square(lin_vel_error))) / (0.75**2))
        ang_vel_track = np.exp(-float(np.mean(np.square(ang_vel_error))) / (1.50**2))
        root_vel = float(0.65 * lin_vel_track + 0.35 * ang_vel_track)
        forward_velocity = float(data.qvel[0])
        forward_velocity_error = forward_velocity - target_forward_velocity
        forward_velocity_track = float(
            np.exp(
                -(forward_velocity_error**2)
                / (max(float(self.cfg.forward_velocity_tracking_sigma), 1.0e-6) ** 2)
            )
        )
        target_velocity_scale = max(abs(float(self.cfg.target_velocity)), 1.0e-6)
        forward_progress = float(np.clip(forward_velocity / target_velocity_scale, -1.0, 1.0))
        forward_velocity_deficit = float(
            np.clip((target_forward_velocity - forward_velocity) / target_velocity_scale, 0.0, 2.0)
        )
        forward_velocity_excess = float(
            np.clip((forward_velocity - target_forward_velocity) / target_velocity_scale, 0.0, 2.0)
        )

        height = float(np.exp(-((root_height - ref_root_height) ** 2) / (0.09**2)))
        joint_pos = np.asarray(data.qpos[self.motion.robot_qpos_adrs], dtype=np.float32)
        joint_vel = np.asarray(data.qvel[self.motion.robot_qvel_adrs], dtype=np.float32)
        pose_error = float(
            np.average(np.square(joint_pos - reference_targets.joint_pos[env_id]), weights=self.joint_reward_weights)
        )
        vel_error = float(
            np.average(np.square(joint_vel - reference_targets.joint_vel[env_id]), weights=self.joint_reward_weights)
        )
        pose_track = float(0.70 * np.exp(-pose_error / (0.28**2)) + 0.30 * np.exp(-pose_error / (0.10**2)))
        vel_track = float(np.exp(-vel_error / (4.0**2)))

        foot_pos_error = float(np.mean(np.square(foot_pos - reference_targets.foot_pos[env_id])))
        foot_pos_track = float(np.exp(-foot_pos_error / (0.075**2)))
        side_metrics = self._side_foot_tracks(
            foot_pos,
            reference_targets.foot_pos[env_id],
            reference_targets.foot_contact[env_id],
        )
        foot_vel_track = self._swing_foot_vel_track(
            foot_vel,
            reference_targets.foot_vel[env_id],
            reference_targets.foot_contact[env_id],
        )
        stance_foot_track, stance_foot_speed = self._stance_foot_track(
            foot_world_vel,
            foot_geom_contact,
            reference_targets.foot_contact[env_id],
        )
        contact_track = float(
            np.clip(1.0 - np.mean(np.abs(foot_contact - reference_targets.foot_contact[env_id])), 0.0, 1.0)
        )
        missed_contact = float(
            np.mean(reference_targets.foot_contact[env_id] * (1.0 - foot_contact))
        )
        position_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_position_action = np.clip(np.asarray(prev_action, dtype=np.float32), -1.0, 1.0)
        ctrl_delta = (
            np.asarray(data.ctrl[self.position_actuator_ids], dtype=np.float32)
            - reference_targets.tendon_ctrl[env_id]
        ) / self.ctrl_half_range
        ctrl_error = float(np.mean(np.square(ctrl_delta)))
        ctrl_track = float(np.exp(-ctrl_error / (0.45**2)))
        ctrl_penalty = float(np.mean(np.square(position_action)))
        action_rate = float(np.mean(np.square(position_action - prev_position_action)))
        if len(self.distal_joint_indices) > 0:
            distal_vel_error = float(
                np.mean(
                    np.square(
                        joint_vel[self.distal_joint_indices]
                        - reference_targets.joint_vel[env_id, self.distal_joint_indices]
                    )
                )
            )
        else:
            distal_vel_error = vel_error
        distal_vel_track = float(
            np.exp(-distal_vel_error / (max(self.cfg.distal_joint_vel_sigma, 1.0e-6) ** 2))
        )
        if self.has_distal_action:
            distal_action = float(np.mean(np.square(position_action[self.distal_position_action_mask])))
            distal_action_rate = float(
                np.mean(
                    np.square(
                        position_action[self.distal_position_action_mask]
                        - prev_position_action[self.distal_position_action_mask]
                    )
                )
            )
        else:
            distal_action = ctrl_penalty
            distal_action_rate = action_rate
        (
            symmetry_track,
            foot_symmetry_track,
            stride_symmetry_track,
            phase_symmetry_track,
            lateral_symmetry_track,
            contact_balance_track,
            action_balance_track,
        ) = self._symmetry_track(
            env_id,
            foot_pos,
            foot_contact,
            position_action,
        )

        reward = (
            0.05
            + 1.40 * pose_track
            + 0.45 * vel_track
            + 0.75 * height
            + 0.90 * root_orientation
            + 0.90 * root_vel
            + self.cfg.forward_velocity_reward_coef * forward_velocity_track
            + self.cfg.forward_progress_reward_coef * forward_progress
            - self.cfg.forward_velocity_deficit_penalty_coef * forward_velocity_deficit
            - self.cfg.forward_velocity_excess_penalty_coef * forward_velocity_excess
            + 1.25 * foot_pos_track
            + self.cfg.side_foot_pos_coef * side_metrics["side_foot_pos"]
            + self.cfg.step_gap_coef * side_metrics["step_gap"]
            + self.cfg.swing_advance_coef * side_metrics["swing_advance"]
            + self.cfg.foot_vel_reward_coef * foot_vel_track
            + self.cfg.stance_foot_reward_coef * stance_foot_track
            + self.cfg.contact_reward_coef * contact_track
            + 0.45 * distal_vel_track
            + self.cfg.symmetry_reward_coef * symmetry_track
            + 0.35 * ctrl_track
            - self.cfg.ctrl_penalty_coef * ctrl_penalty
            - self.cfg.action_rate_penalty_coef * action_rate
            - self.cfg.distal_action_penalty_coef * distal_action
            - self.cfg.distal_action_rate_penalty_coef * distal_action_rate
            - self.cfg.missed_contact_penalty_coef * missed_contact
        )

        bad_state = not (
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.ctrl).all()
        )
        fallen = (
            root_height < self.cfg.fall_height
            or root_height > self.cfg.max_height
            or up_z < self.cfg.min_upright_z
        )
        timed_out = self.episode_step[env_id] >= self.episode_timeout_step[env_id]
        done = bool(bad_state or fallen or timed_out)
        termination = bool(bad_state or fallen)
        if bad_state or fallen:
            reward -= 2.0

        return float(reward), done, bool(timed_out), {
            "root_height_value": root_height,
            "reference_root_height_value": ref_root_height,
            "root_vel": root_vel,
            "forward_velocity": forward_velocity,
            "forward_velocity_error": forward_velocity_error,
            "forward_velocity_track": forward_velocity_track,
            "forward_progress": forward_progress,
            "forward_velocity_deficit": forward_velocity_deficit,
            "forward_velocity_excess": forward_velocity_excess,
            "termination": float(termination),
            "root_orientation": root_orientation,
            "height": height,
            "pose_track": pose_track,
            "vel_track": vel_track,
            "foot_pos": foot_pos_track,
            "side_foot_pos": side_metrics["side_foot_pos"],
            "right_foot_pos": side_metrics["right_foot_pos"],
            "left_foot_pos": side_metrics["left_foot_pos"],
            "step_gap": side_metrics["step_gap"],
            "step_gap_error": side_metrics["step_gap_error"],
            "swing_advance": side_metrics["swing_advance"],
            "right_swing_advance": side_metrics["right_swing_advance"],
            "left_swing_advance": side_metrics["left_swing_advance"],
            "foot_vel": foot_vel_track,
            "stance_foot": stance_foot_track,
            "stance_foot_speed": stance_foot_speed,
            "contact": contact_track,
            "missed_contact": missed_contact,
            "distal_vel": distal_vel_track,
            "symmetry": symmetry_track,
            "foot_symmetry": foot_symmetry_track,
            "stride_symmetry": stride_symmetry_track,
            "phase_symmetry": phase_symmetry_track,
            "lateral_symmetry": lateral_symmetry_track,
            "contact_balance": contact_balance_track,
            "action_balance": action_balance_track,
            "ctrl_track": ctrl_track,
            "ctrl_penalty": ctrl_penalty,
            "action_rate": action_rate,
            "distal_action": distal_action,
            "distal_action_rate": distal_action_rate,
        }

    def _side_foot_tracks(
        self,
        foot_pos: np.ndarray,
        ref_foot_pos: np.ndarray,
        ref_foot_contact: np.ndarray,
    ) -> dict[str, float]:
        side_center = np.asarray(
            [
                np.mean(foot_pos[0:2], axis=0),
                np.mean(foot_pos[2:4], axis=0),
            ],
            dtype=np.float32,
        )
        ref_side_center = np.asarray(
            [
                np.mean(ref_foot_pos[0:2], axis=0),
                np.mean(ref_foot_pos[2:4], axis=0),
            ],
            dtype=np.float32,
        )
        sigma = max(float(self.cfg.side_foot_pos_sigma), 1.0e-6)
        side_error = np.mean(np.square(side_center - ref_side_center), axis=1)
        right_foot_pos = float(np.exp(-float(side_error[0]) / (sigma**2)))
        left_foot_pos = float(np.exp(-float(side_error[1]) / (sigma**2)))
        side_foot_pos = min(right_foot_pos, left_foot_pos)

        actual_gap = float(side_center[1, 0] - side_center[0, 0])
        ref_gap = float(ref_side_center[1, 0] - ref_side_center[0, 0])
        step_gap_error = actual_gap - ref_gap
        step_gap = float(np.exp(-(step_gap_error**2) / (max(float(self.cfg.step_gap_sigma), 1.0e-6) ** 2)))

        swing_weight = 1.0 - np.asarray(ref_foot_contact, dtype=np.float32)
        swing_sigma = max(float(self.cfg.swing_advance_sigma), 1.0e-6)
        x_error = side_center[:, 0] - ref_side_center[:, 0]
        side_swing = np.exp(-np.square(x_error) / (swing_sigma**2)).astype(np.float32)
        if float(np.sum(swing_weight)) > 1.0e-6:
            swing_advance = float(np.sum(side_swing * swing_weight) / np.sum(swing_weight))
        else:
            swing_advance = 1.0
        return {
            "side_foot_pos": float(side_foot_pos),
            "right_foot_pos": right_foot_pos,
            "left_foot_pos": left_foot_pos,
            "step_gap": step_gap,
            "step_gap_error": float(step_gap_error),
            "swing_advance": swing_advance,
            "right_swing_advance": float(side_swing[0]),
            "left_swing_advance": float(side_swing[1]),
        }

    def _swing_foot_vel_track(
        self,
        foot_vel: np.ndarray,
        ref_foot_vel: np.ndarray,
        ref_foot_contact: np.ndarray,
    ) -> float:
        swing_weight = np.repeat(
            1.0 - np.clip(np.asarray(ref_foot_contact, dtype=np.float32), 0.0, 1.0),
            2,
        )
        weight_sum = float(np.sum(swing_weight))
        if weight_sum <= 1.0e-6:
            return 1.0
        error = np.mean(np.square(foot_vel - ref_foot_vel), axis=1)
        score = np.exp(
            -error / (max(self.cfg.foot_vel_tracking_sigma, 1.0e-6) ** 2)
        )
        return float(np.sum(score * swing_weight) / weight_sum)

    def _stance_foot_track(
        self,
        foot_world_vel: np.ndarray,
        foot_geom_contact: np.ndarray,
        ref_foot_contact: np.ndarray,
    ) -> tuple[float, float]:
        geom_contact = np.asarray(foot_geom_contact, dtype=np.float32)
        side_contact = self._side_contacts_from_geom(geom_contact)
        side_weight = np.maximum(
            side_contact,
            np.clip(np.asarray(ref_foot_contact, dtype=np.float32), 0.0, 1.0),
        )
        horizontal_speed_sq = np.sum(np.square(foot_world_vel[:, 0:2]), axis=1)
        geom_score = np.exp(
            -horizontal_speed_sq / (max(self.cfg.stance_foot_vel_sigma, 1.0e-6) ** 2)
        )

        side_score = np.zeros(2, dtype=np.float32)
        for side_index, contact_slice in enumerate((slice(0, 2), slice(2, 4))):
            contact_count = float(np.sum(geom_contact[contact_slice]))
            if contact_count > 1.0e-6:
                side_score[side_index] = float(
                    np.sum(geom_score[contact_slice] * geom_contact[contact_slice]) / contact_count
                )

        side_weight_sum = float(np.sum(side_weight))
        stance_track = (
            float(np.sum(side_score * side_weight) / side_weight_sum)
            if side_weight_sum > 1.0e-6
            else 1.0
        )
        actual_contact_count = float(np.sum(geom_contact))
        stance_speed = (
            float(np.sum(np.sqrt(horizontal_speed_sq) * geom_contact) / actual_contact_count)
            if actual_contact_count > 1.0e-6
            else 0.0
        )
        return stance_track, stance_speed

    def _symmetry_track(
        self,
        env_id: int,
        foot_pos: np.ndarray,
        foot_contact: np.ndarray,
        action: np.ndarray,
    ) -> tuple[float, float, float, float, float, float, float]:
        decay = float(np.clip(self.cfg.symmetry_ema_decay, 0.0, 0.9999))
        side_x = np.asarray(
            [
                np.mean(foot_pos[0:2, 0]),
                np.mean(foot_pos[2:4, 0]),
            ],
            dtype=np.float32,
        )
        self.foot_x_min[env_id] = np.minimum(self.foot_x_min[env_id], side_x)
        self.foot_x_max[env_id] = np.maximum(self.foot_x_max[env_id], side_x)
        side_stride = self.foot_x_max[env_id] - self.foot_x_min[env_id]
        self.foot_stride_ema[env_id] = decay * self.foot_stride_ema[env_id] + (1.0 - decay) * side_stride
        stride_diff = float(self.foot_stride_ema[env_id, 0] - self.foot_stride_ema[env_id, 1])
        sigma = max(self.cfg.symmetry_foot_pos_sigma, 1.0e-6)
        stride_symmetry = float(np.exp(-(stride_diff**2) / (sigma**2)))
        # With foot positions expressed in the pelvis frame, a symmetric gait
        # should keep the two side centers roughly mirrored front/back.
        phase_diff = float(side_x[0] + side_x[1])
        phase_symmetry = float(np.exp(-(phase_diff**2) / (sigma**2)))
        right_y = float(np.mean(foot_pos[0:2, 1]))
        left_y = float(np.mean(foot_pos[2:4, 1]))
        lateral_symmetry = float(
            np.exp(-((right_y + left_y) ** 2) / (max(self.cfg.symmetry_balance_sigma, 1.0e-6) ** 2))
        )
        foot_symmetry = 0.85 * min(stride_symmetry, phase_symmetry) + 0.15 * lateral_symmetry

        self.contact_balance_ema[env_id] = (
            decay * self.contact_balance_ema[env_id] + (1.0 - decay) * np.asarray(foot_contact, dtype=np.float32)
        )
        contact_diff = float(self.contact_balance_ema[env_id, 0] - self.contact_balance_ema[env_id, 1])
        contact_balance = float(
            np.exp(-(contact_diff**2) / (max(self.cfg.symmetry_balance_sigma, 1.0e-6) ** 2))
        )

        if len(self.right_action_indices) > 0 and len(self.left_action_indices) > 0:
            side_energy = np.asarray(
                [
                    np.mean(np.square(action[self.right_action_indices])),
                    np.mean(np.square(action[self.left_action_indices])),
                ],
                dtype=np.float32,
            )
            self.action_energy_ema[env_id] = decay * self.action_energy_ema[env_id] + (1.0 - decay) * side_energy
            action_diff = float(self.action_energy_ema[env_id, 0] - self.action_energy_ema[env_id, 1])
            action_balance = float(
                np.exp(-(action_diff**2) / (max(self.cfg.symmetry_balance_sigma, 1.0e-6) ** 2))
            )
        else:
            action_balance = 1.0

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
            float(symmetry),
            float(foot_symmetry),
            float(stride_symmetry),
            float(phase_symmetry),
            float(lateral_symmetry),
            float(contact_balance),
            float(action_balance),
        )

    def close(self) -> None:
        self.datas.clear()
