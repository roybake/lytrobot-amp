"""GPU batched MuJoCo environment powered by MuJoCo Warp."""

from __future__ import annotations

import importlib
from pathlib import Path
import secrets

import mujoco
import numpy as np
import torch

from .env import (
    LytRobotEnvCfg,
    _stair_root_height_transform,
    forward_velocity_bounds,
    scaled_randomization_range,
    sample_action_delay,
    sensor_stability_reward_enabled,
)
from .gait_constraints import natural_gait_bonus, sensor_gait_scores
from .step_symmetry import StepLengthTracker, step_symmetry_reward
from .walking_commands import (
    WalkingCommands, heading_angular_velocity, heading_velocity, without_heading, world_yaw_rate,
)
from .mimickit_deepmimic import deepmimic_reward
from .motion import (
    CONTACT_GEOM_NAMES,
    MotionLibrary,
    load_motion_library,
    resolve_contact_geom_ids,
)
from .terrain import (
    MoreTerrainCfg,
    StairTerrainCfg,
    course_end_x,
    course_motion_sequence,
    course_stage_end_positions,
    course_stage_scenarios,
    load_model,
    randomized_course_orders,
    sample_course_start_stages,
    terrain_geom_ids,
    terrain_kind_from_motion,
    terrain_spawn_offset,
    validate_terrain_types,
)

try:
    import mujoco_warp as mjwarp
    import warp as wp
except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
    mjwarp = None
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


_WARP_DYNAMIC_SOURCE_ERROR = "Directly evaluating Warp code defined as a string using `exec()` is not supported"


def _reload_mujoco_warp_reset_data():
    """Reload reset_data from its source file after a stale-code failure.

    ``mujoco_warp.reset_data`` creates a few Warp kernels lazily.  Warp uses
    ``inspect`` when those kernels are created, so a long-running process can
    fail after its environment or source tree has been replaced underneath
    it.  Reloading the owning module gives the kernels code objects backed by
    the current on-disk source instead of the stale loader state.
    """

    if mjwarp is None:  # pragma: no cover - guarded by environment construction.
        raise RuntimeError("mujoco-warp is unavailable")
    reset_module = importlib.import_module("mujoco_warp._src.io")
    reset_module = importlib.reload(reset_module)
    mjwarp.reset_data = reset_module.reset_data
    return reset_module.reset_data


def _canonical_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-6)


def _sanitize_reward_outputs(
    reward: torch.Tensor,
    bad_state: torch.Tensor,
    fallen: torch.Tensor,
    components: dict[str, torch.Tensor],
    *,
    fall_penalty: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Quarantine one invalid simulation without contaminating the whole batch."""

    invalid = bad_state | ~torch.isfinite(reward)
    for value in components.values():
        invalid = invalid | ~torch.isfinite(value)

    finite_reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    finite_reward = finite_reward - torch.where(
        fallen,
        torch.full_like(finite_reward, max(float(fall_penalty), 0.0)),
        torch.zeros_like(finite_reward),
    )
    finite_reward = torch.where(
        invalid,
        torch.full_like(finite_reward, -2.0),
        finite_reward,
    )

    safe_components: dict[str, torch.Tensor] = {}
    for key, value in components.items():
        safe_value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        safe_value = torch.clamp(safe_value, -1.0e6, 1.0e6)
        safe_components[key] = torch.where(invalid, torch.zeros_like(safe_value), safe_value)
    safe_components["invalid_state"] = invalid.to(torch.float32)
    return finite_reward, invalid, safe_components


def _stability_gated_locomotion_terms(
    forward_velocity_track: torch.Tensor,
    forward_progress: torch.Tensor,
    root_orientation: torch.Tensor,
    height: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grant positive locomotion reward only while the torso remains stable."""

    stability = torch.sqrt(
        torch.clamp(root_orientation, 0.0, 1.0)
        * torch.clamp(height, 0.0, 1.0)
    )
    gated_track = forward_velocity_track * stability
    gated_progress = torch.clamp_max(forward_progress, 0.0) + (
        torch.clamp_min(forward_progress, 0.0) * stability
    )
    return stability, gated_track, gated_progress


def _forward_progress_guard(
    position_x: torch.Tensor,
    episode_step: torch.Tensor,
    anchor_x: torch.Tensor,
    anchor_step: torch.Tensor,
    *,
    dt: float,
    min_speed: float,
    grace_steps: int,
    window_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check net world-X progress over fixed windows and advance the anchors."""

    next_anchor_x = anchor_x.clone()
    next_anchor_step = anchor_step.clone()
    speed = torch.zeros_like(position_x)
    stalled = torch.zeros_like(position_x, dtype=torch.bool)
    if min_speed <= 0.0 or window_steps <= 0:
        return speed, stalled, next_anchor_x, next_anchor_step

    grace_steps = max(int(grace_steps), 0)
    start_tracking = episode_step == grace_steps
    next_anchor_x = torch.where(start_tracking, position_x, next_anchor_x)
    next_anchor_step = torch.where(start_tracking, episode_step, next_anchor_step)

    elapsed = episode_step - next_anchor_step
    active = episode_step > grace_steps
    finite = torch.isfinite(position_x) & torch.isfinite(next_anchor_x)
    raw_speed = (position_x - next_anchor_x) / (
        elapsed.clamp_min(1).to(position_x.dtype) * max(float(dt), 1.0e-6)
    )
    speed = torch.where(active & finite, raw_speed, speed)
    check = active & (elapsed >= int(window_steps))
    stalled = check & (~finite | (speed < float(min_speed)))
    next_anchor_x = torch.where(check, position_x, next_anchor_x)
    next_anchor_step = torch.where(check, episode_step, next_anchor_step)
    return speed, stalled, next_anchor_x, next_anchor_step


class LytRobotWarpAmpEnv:
    """Batched CUDA MuJoCo environment.

    This keeps the current task/AMP observation design, but moves simulation,
    observation assembly, and reward computation to tensors on ``device``.
    """

    is_torch_env = True

    def __init__(self, cfg: LytRobotEnvCfg, device: str = "cuda") -> None:
        if _WARP_IMPORT_ERROR is not None:
            raise RuntimeError("mujoco-warp and warp-lang are required for the warp backend.") from _WARP_IMPORT_ERROR
        if device == "cuda":
            device = "cuda:0"
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("The warp backend requires a CUDA device.")

        self.cfg = cfg
        self.model_path = Path(cfg.model_xml).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model XML not found: {self.model_path}")

        terrain_types = validate_terrain_types(cfg.terrain_types)
        if cfg.serial_terrain_course and not terrain_types:
            raise ValueError("serial terrain course requires at least one terrain type.")
        self.stair_cfg = StairTerrainCfg(
            enabled=bool(cfg.stair_terrain or "stairs" in terrain_types),
            step_height=cfg.stair_step_height,
            step_depth=cfg.stair_step_depth,
            step_count=cfg.stair_step_count,
            width=cfg.stair_width,
            run_in=cfg.stair_run_in,
            top_length=cfg.stair_top_length,
            run_out=cfg.stair_run_out,
            start_x=cfg.stair_start_x,
        )
        course_orders = (
            randomized_course_orders(
                terrain_types,
                cfg.terrain_course_variants,
                cfg.seed,
            )
            if cfg.serial_terrain_course
            else ()
        )
        self.more_terrain_cfg = MoreTerrainCfg(
            enabled=bool(terrain_types),
            terrain_types=terrain_types,
            asset_dir=cfg.terrain_asset_dir,
            serial_course=bool(cfg.serial_terrain_course),
            course_orders=course_orders,
            course_lane_spacing=cfg.terrain_course_lane_spacing,
            course_base_height=cfg.terrain_course_base_height,
            flat_lane_y=cfg.flat_lane_y,
            rough_lane_y=cfg.rough_lane_y,
            pit_lane_y=cfg.pit_lane_y,
            gap_lane_y=cfg.gap_lane_y,
            lane_width=cfg.terrain_lane_width,
            rough_height=cfg.rough_terrain_height,
            slope_angle_deg=cfg.slope_angle_deg,
            gap_width=cfg.gap_terrain_width,
        )
        self.complex_terrain_enabled = bool(
            self.stair_cfg.enabled or self.more_terrain_cfg.enabled
        )
        self.model = load_model(
            self.model_path,
            None if cfg.terrain_embedded_in_model else self.stair_cfg,
            None if cfg.terrain_embedded_in_model else self.more_terrain_cfg,
        )
        self.model.opt.timestep = float(cfg.timestep)
        self.model.opt.iterations = max(1, int(cfg.solver_iterations))
        self.model.opt.ls_iterations = max(1, int(cfg.solver_ls_iterations))
        # MuJoCo Warp rejects non-zero contact margins with CCD enabled. The LYT
        # MJCF defaults geom margin to 0.001, so disable CCD for this backend.
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD.value
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD.value
        self.base_actuator_gainprm_np = np.asarray(self.model.actuator_gainprm, dtype=np.float32).copy()
        self.base_actuator_biasprm_np = np.asarray(self.model.actuator_biasprm, dtype=np.float32).copy()
        self.base_body_mass_np = np.asarray(self.model.body_mass, dtype=np.float32).copy()
        self.base_body_inertia_np = np.asarray(self.model.body_inertia, dtype=np.float32).copy()
        self.base_geom_friction_np = np.asarray(self.model.geom_friction, dtype=np.float32).copy()
        self.base_gravity_np = np.asarray(self.model.opt.gravity, dtype=np.float32).copy()

        self.position_actuator_ids_np = np.arange(self.model.nu, dtype=np.int64)
        self.position_action_dim = int(len(self.position_actuator_ids_np))
        self.action_dim = self.position_action_dim
        self.ctrl_dim = int(self.model.nu)
        motion_files = cfg.motion_files or (cfg.motion_file,)
        preserve_stair_world_xy = bool(
            cfg.preserve_motion_world_xy
            or (
                cfg.stair_terrain
                and len(motion_files) == 1
                and Path(motion_files[0]).stem == "stair_up_down_omniretarget"
            )
        )
        self.continuous_stair_phase_curriculum = bool(
            preserve_stair_world_xy and not cfg.stair_fixed_start
        )
        self.motion = load_motion_library(
            motion_files,
            self.model,
            actuator_ids=self.position_actuator_ids_np,
            motion_weights=cfg.motion_weights or None,
            motion_names=cfg.motion_names or None,
            motion_loop=cfg.motion_loop or None,
            zero_root_xy=not preserve_stair_world_xy,
            root_x_offset=cfg.reference_root_x_offset,
            symmetric_gait_velocity=cfg.target_velocity if cfg.symmetric_gait_reference else None,
            heading_invariant=cfg.command_walking,
        )
        self.motion_count = (
            self.motion.motion_count if isinstance(self.motion, MotionLibrary) else 1
        )
        motion_names = (
            self.motion.motion_names
            if isinstance(self.motion, MotionLibrary)
            # A one-element ``--motion-files`` list is represented by a
            # MotionReference, so task detection must use that selected path.
            else (Path(motion_files[0]).stem,)
        )
        self.stair_motion_mask_np = np.asarray(
            ["stair" in name.lower() for name in motion_names], dtype=bool
        )
        self.stair_down_motion_mask_np = np.asarray(
            ["stair_down" in name.lower() for name in motion_names], dtype=bool
        )
        motion_references = (
            self.motion.motions
            if isinstance(self.motion, MotionLibrary)
            else (self.motion,)
        )
        self.motion_stair_height_anchor_np = np.zeros(
            self.motion_count, dtype=np.float32
        )
        self.motion_stair_height_scale_np = np.ones(
            self.motion_count, dtype=np.float32
        )
        self.motion_stair_distance_np = np.zeros(
            self.motion_count, dtype=np.float32
        )
        for motion_id, motion_reference in enumerate(motion_references):
            if not self.stair_motion_mask_np[motion_id]:
                continue
            if preserve_stair_world_xy and not self.more_terrain_cfg.serial_course:
                # Continuous ascent/descent returns to its initial height and
                # is already contact-solved in the embedded MJCF world.
                anchor, scale = 0.0, 1.0
            else:
                anchor, scale = _stair_root_height_transform(
                    motion_reference,
                    step_height=self.stair_cfg.step_height,
                    step_count=self.stair_cfg.step_count,
                    descending=bool(self.stair_down_motion_mask_np[motion_id]),
                )
                if (
                    self.cfg.stair_fixed_start
                    and not self.more_terrain_cfg.serial_course
                ):
                    scale = 1.0
            self.motion_stair_height_anchor_np[motion_id] = anchor
            self.motion_stair_height_scale_np[motion_id] = scale
            self.motion_stair_distance_np[motion_id] = float(
                motion_reference.reference_qpos[-1, 0]
                - motion_reference.reference_qpos[0, 0]
            )
        self.nominal_root_height = float(self.motion.model_qpos0[2])
        self.motion_terrain_kinds = tuple(
            terrain_kind_from_motion(name) for name in motion_names
        )
        missing_terrains = sorted(
            {
                kind
                for kind in self.motion_terrain_kinds
                if kind not in {"flat", "stairs"} and kind not in terrain_types
            }
        )
        if missing_terrains:
            raise ValueError(
                "Motion library contains terrain-conditioned clips without matching "
                f"terrain geometry: {missing_terrains}. Enable them with terrain_types."
            )
        self.course_motion_sequences_np = self._build_course_motion_sequences(motion_names)
        self.course_stage_scenario_names = tuple(
            course_stage_scenarios(order)
            for order in self.more_terrain_cfg.normalized_course_orders
        )
        self.course_stage_end_x_np = self._build_course_stage_end_x()
        self.course_finish_x = (
            course_end_x(self.more_terrain_cfg, 0)
            if self.more_terrain_cfg.serial_course
            else float("inf")
        )
        rng_seed = secrets.randbits(128) if cfg.randomize_episode_seed else int(cfg.seed)
        self.rng = np.random.default_rng(rng_seed)
        self.num_envs = int(cfg.num_envs)
        self.actor_observation_enabled = True

        self.ctrl_low_np = np.asarray(self.model.actuator_ctrlrange[self.position_actuator_ids_np, 0], dtype=np.float32)
        self.ctrl_high_np = np.asarray(self.model.actuator_ctrlrange[self.position_actuator_ids_np, 1], dtype=np.float32)
        self.ctrl_low = torch.as_tensor(self.ctrl_low_np, dtype=torch.float32, device=self.device)
        self.ctrl_high = torch.as_tensor(self.ctrl_high_np, dtype=torch.float32, device=self.device)
        self.ctrl_mid = 0.5 * (self.ctrl_low + self.ctrl_high)
        self.ctrl_half_range = torch.clamp(0.5 * (self.ctrl_high - self.ctrl_low), min=1.0e-6)
        self.robot_mass = float(np.sum(self.model.body_mass))
        self.position_actuator_ids = torch.as_tensor(
            self.position_actuator_ids_np, dtype=torch.long, device=self.device
        )
        self.actuator_tendon_ids = torch.as_tensor(
            np.asarray(self.model.actuator_trnid[self.position_actuator_ids_np, 0], dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )
        distal_action_mask_np = self._build_distal_action_mask()
        action_scale_np = self._build_action_scale(distal_action_mask_np)
        self.action_scale = torch.as_tensor(action_scale_np, dtype=torch.float32, device=self.device)
        self.distal_action_mask = torch.as_tensor(distal_action_mask_np, dtype=torch.bool, device=self.device)
        self.distal_position_action_mask = torch.as_tensor(
            self.distal_position_action_mask_np, dtype=torch.bool, device=self.device
        )
        self.has_distal_action = bool(np.any(distal_action_mask_np))
        right_action_indices_np, left_action_indices_np = self._build_side_action_indices()
        self.right_action_indices = torch.as_tensor(right_action_indices_np, dtype=torch.long, device=self.device)
        self.left_action_indices = torch.as_tensor(left_action_indices_np, dtype=torch.long, device=self.device)

        nonroot_qpos_adrs, nonroot_qvel_adrs = self._get_nonroot_joint_adrs()
        self.nonroot_qpos_adrs = torch.as_tensor(nonroot_qpos_adrs, dtype=torch.long, device=self.device)
        self.nonroot_qvel_adrs = torch.as_tensor(nonroot_qvel_adrs, dtype=torch.long, device=self.device)
        self.robot_qpos_adrs = torch.as_tensor(self.motion.robot_qpos_adrs, dtype=torch.long, device=self.device)
        self.robot_qvel_adrs = torch.as_tensor(self.motion.robot_qvel_adrs, dtype=torch.long, device=self.device)

        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_body_id < 0:
            raise RuntimeError("Body 'pelvis' not found in robot model.")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        terrain_ids_np = terrain_geom_ids(self.model).astype(np.int64)
        contact_geom_ids_np = resolve_contact_geom_ids(self.model).astype(np.int64)
        if self.floor_geom_id < 0:
            raise RuntimeError("Expected floor and foot contact geoms were not found in robot model.")
        self.contact_geom_ids = torch.as_tensor(contact_geom_ids_np, dtype=torch.long, device=self.device)
        self.terrain_geom_ids = torch.as_tensor(
            terrain_ids_np, dtype=torch.long, device=self.device
        )
        self.contact_geom_id_values = tuple(int(geom_id) for geom_id in contact_geom_ids_np)
        self.right_contact_geom_ids = self.contact_geom_ids[:2]
        self.left_contact_geom_ids = self.contact_geom_ids[2:]
        self.sensor_foot_body_ids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("foot_r", "foot_l")
        )
        self._build_stand_reference()

        self.joint_reward_weights = torch.as_tensor(
            self._build_joint_reward_weights(), dtype=torch.float32, device=self.device
        )
        self.distal_joint_indices = torch.as_tensor(
            np.asarray(
                [
                    joint_id
                    for joint_id, name in enumerate(self.motion.common_joint_names)
                    if "ankle" in name or "subtalar" in name or "mtp" in name
                ],
                dtype=np.int64,
            ),
            dtype=torch.long,
            device=self.device,
        )
        self.knee_joint_indices = torch.as_tensor(
            np.asarray(
                [
                    joint_id
                    for joint_id, name in enumerate(self.motion.common_joint_names)
                    if "knee" in name
                ],
                dtype=np.int64,
            ),
            dtype=torch.long,
            device=self.device,
        )
        self._build_reference_tensors()
        self._build_reset_helpers()
        self._init_warp_data()

        self.episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_seed = np.full(self.num_envs, int(cfg.seed), dtype=np.uint64)
        self.episode_timeout_step = torch.full(
            (self.num_envs,),
            int(cfg.episode_steps),
            dtype=torch.long,
            device=self.device,
        )
        self.forward_progress_anchor_x = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.forward_progress_anchor_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.stair_start_x = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.stair_start_y = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.stair_remaining_distance = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.stair_remaining_lateral = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.stair_full_start = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.stair_terminal_hold_elapsed = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.course_variant_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.course_stage_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.course_start_stage_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.course_motion_sequences = torch.as_tensor(
            self.course_motion_sequences_np,
            dtype=torch.long,
            device=self.device,
        )
        self.course_stage_end_x = torch.as_tensor(
            self.course_stage_end_x_np,
            dtype=torch.float32,
            device=self.device,
        )
        self.last_action = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float32, device=self.device)
        self.last_reference_ctrl = torch.zeros_like(self.last_action)
        self.motion_transition_from_ctrl = torch.zeros_like(self.last_action)
        self.motion_transition_steps_left = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.last_foot_pos = torch.zeros(
            (self.num_envs, len(CONTACT_GEOM_NAMES), 3), dtype=torch.float32, device=self.device
        )
        self.last_foot_world_pos = torch.zeros_like(self.last_foot_pos)
        self.step_lengths = StepLengthTracker(self.num_envs, self.dt, self.device)
        self.sensor_last_foot_contact = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.sensor_foot_roll_stage = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self.sensor_toe_only_steps = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self.sensor_last_touchdown_side = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.sensor_liftoff_x = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.sensor_had_liftoff = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.sensor_stride_ema = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.sensor_stride_seen = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.sensor_contact_duty_ema = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.sensor_single_support_ema = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sensor_time_since_touchdown = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sensor_step_interval_ema = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sensor_touchdown_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.sensor_cadence_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.sensor_swing_steps = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self.sensor_prev_foot_world = torch.zeros((self.num_envs, 2, 3), dtype=torch.float32, device=self.device)
        self.last_obs = torch.zeros((self.num_envs, self.obs_dim), dtype=torch.float32, device=self.device)
        self.stand_start = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stand_start_transition_steps = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.walking_commands = WalkingCommands(cfg, self.num_envs, self.device) if cfg.command_walking else None
        self._init_domain_randomization()
        self._capture_warp_graphs()
        self.reset()

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.cfg.decimation)

    @property
    def obs_dim(self) -> int:
        current_state_dim = 1 + 4 + 6 + 2 * len(self.nonroot_qpos_adrs) + self.action_dim + 3
        if self.cfg.observe_tendon_velocity:
            current_state_dim += self.position_action_dim
        tracking_dim = (
            1
            + 4
            + 6
            + 2 * len(self.motion.common_joint_names)
            + self.action_dim
            + self.motion.foot_pos_dim
            + self.motion.contact_dim
        )
        skill_dim = self.motion_count if self.cfg.skill_conditioning and self.motion_count > 1 else 0
        return int(current_state_dim + tracking_dim + skill_dim + (2 if self.cfg.command_walking else 0))

    @property
    def amp_obs_dim(self) -> int:
        return self.motion.feature_dim

    def _build_course_motion_sequences(
        self,
        motion_names: tuple[str, ...],
    ) -> np.ndarray:
        if not self.more_terrain_cfg.serial_course:
            return np.empty((0, 0), dtype=np.int64)
        if not isinstance(self.motion, MotionLibrary):
            raise ValueError("serial terrain course requires a multi-motion library.")
        name_to_id = {name.lower(): index for index, name in enumerate(motion_names)}
        # Two-reference compatibility: when no slope expert exists, both
        # slope stages use the original rough-walk reference.  Four-reference
        # layouts contain the requested clip names and bypass this fallback.
        fallback_names = {"slope_up": "rough_walk", "slope_down": "rough_walk"}
        rows: list[list[int]] = []
        for order in self.more_terrain_cfg.normalized_course_orders:
            requested_names = course_motion_sequence(order)
            names = [
                name
                if name in name_to_id
                else fallback_names.get(name, name)
                for name in requested_names
            ]
            missing = [name for name in names if name not in name_to_id]
            if missing:
                raise ValueError(
                    "Serial terrain course is missing required motion clips: "
                    f"{sorted(set(missing))}."
                )
            rows.append([name_to_id[name] for name in names])
        return np.asarray(rows, dtype=np.int64)

    def _build_course_stage_end_x(self) -> np.ndarray:
        """Physical root-X boundary for each action in every serial course."""

        if not self.more_terrain_cfg.serial_course:
            return np.empty((0, 0), dtype=np.float32)
        rows = [
            course_stage_end_positions(self.more_terrain_cfg, variant)
            for variant in range(len(self.more_terrain_cfg.normalized_course_orders))
        ]
        result = np.asarray(rows, dtype=np.float32)
        if result.shape != self.course_motion_sequences_np.shape:
            raise ValueError(
                "Serial course motion stages and physical boundaries do not align: "
                f"motions={self.course_motion_sequences_np.shape}, boundaries={result.shape}."
            )
        return result

    def _get_nonroot_joint_adrs(self) -> tuple[np.ndarray, np.ndarray]:
        qpos_adrs: list[int] = []
        qvel_adrs: list[int] = []
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            qpos_adrs.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_adrs.append(int(self.model.jnt_dofadr[joint_id]))
        return np.asarray(qpos_adrs, dtype=np.int64), np.asarray(qvel_adrs, dtype=np.int64)

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

    def _build_action_scale(self, distal_action_mask: np.ndarray) -> np.ndarray:
        scale = np.ones(self.position_action_dim, dtype=np.float32)
        distal_scale = float(np.clip(self.cfg.distal_action_scale, 0.05, 1.0))
        scale[self.distal_position_action_mask_np] = distal_scale
        return scale

    def _build_distal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=bool)
        self.distal_position_action_mask_np = np.zeros(self.position_action_dim, dtype=bool)
        for action_id, actuator_id in enumerate(self.position_actuator_ids_np):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
            tendon_id = int(self.model.actuator_trnid[actuator_id, 0])
            tendon_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id) or ""
            if "ankle" in actuator_name or "ankle" in tendon_name:
                mask[action_id] = True
                self.distal_position_action_mask_np[action_id] = True
        return mask

    def _build_side_action_indices(self) -> tuple[np.ndarray, np.ndarray]:
        right: list[int] = []
        left: list[int] = []
        for action_id, actuator_id in enumerate(self.position_actuator_ids_np):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
            tendon_id = int(self.model.actuator_trnid[actuator_id, 0])
            tendon_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id) or ""
            name = f"{actuator_name} {tendon_name}"
            if "_r" in name:
                right.append(action_id)
            elif "_l" in name:
                left.append(action_id)
        return np.asarray(right, dtype=np.int64), np.asarray(left, dtype=np.int64)

    def _build_reference_tensors(self) -> None:
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        def packed(attribute: str) -> np.ndarray:
            if isinstance(self.motion, MotionLibrary):
                return self.motion.packed(attribute)
            return np.asarray(getattr(self.motion, attribute))[None, ...]

        self.ref_root_height = tensor(packed("reference_root_height"))
        self.ref_root_xy = tensor(packed("reference_root_xy"))
        self.ref_root_quat = tensor(packed("reference_root_quat"))
        self.ref_root_lin_vel = tensor(packed("reference_root_lin_vel"))
        self.ref_root_ang_vel = tensor(packed("reference_root_ang_vel"))
        self.ref_joint_pos = tensor(packed("reference_joint_pos"))
        self.ref_joint_vel = tensor(packed("reference_joint_vel"))
        self.ref_tendon_ctrl = tensor(packed("reference_tendon_ctrl"))
        self.ref_foot_pos = tensor(packed("reference_foot_pos"))
        self.ref_foot_vel = tensor(packed("reference_foot_vel"))
        self.ref_foot_contact = tensor(packed("reference_foot_contact"))
        self.motion_frame_counts = torch.as_tensor(
            (
                self.motion.frame_counts
                if isinstance(self.motion, MotionLibrary)
                else np.asarray([self.motion.frame_count], dtype=np.int32)
            ),
            dtype=torch.long,
            device=self.device,
        )
        self.motion_frequencies = torch.as_tensor(
            (
                self.motion.frequencies
                if isinstance(self.motion, MotionLibrary)
                else np.asarray([self.motion.frequency], dtype=np.float32)
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.motion_loop = torch.as_tensor(
            (
                self.motion.motion_loop
                if isinstance(self.motion, MotionLibrary)
                else np.asarray([self.motion.source_loop])
            ),
            dtype=torch.bool,
            device=self.device,
        )
        self.feature_mean = tensor(self.motion.feature_mean)
        self.feature_std = tensor(self.motion.feature_std)
        self.motion_stair_height_anchor = tensor(
            self.motion_stair_height_anchor_np
        )
        self.motion_stair_height_scale = tensor(
            self.motion_stair_height_scale_np
        )
        self.motion_stair_distance = tensor(self.motion_stair_distance_np)
        self.motion_stair_mask = torch.as_tensor(
            self.stair_motion_mask_np, dtype=torch.bool, device=self.device
        )

    def _build_stand_reference(self) -> None:
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.motion.model_qpos0
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        tendon_lengths = np.asarray(data.ten_length, dtype=np.float32)
        actuator_tendon_ids = np.asarray(self.model.actuator_trnid[self.position_actuator_ids_np, 0], dtype=np.int64)
        tendon_ctrl = np.clip(tendon_lengths[actuator_tendon_ids], self.ctrl_low_np, self.ctrl_high_np)
        root_mat = np.asarray(data.xmat[self.pelvis_body_id], dtype=np.float32).reshape(3, 3)
        root_pos = np.asarray(data.xpos[self.pelvis_body_id], dtype=np.float32)
        contact_geom_ids = self.contact_geom_ids.detach().cpu().numpy().astype(np.int64)
        foot_world = np.asarray(data.geom_xpos[contact_geom_ids], dtype=np.float32)
        foot_pos = (foot_world - root_pos) @ root_mat

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        quat = np.asarray(data.qpos[3:7], dtype=np.float32)
        quat = quat / max(float(np.linalg.norm(quat)), 1.0e-6)
        self.stand_root_height = tensor(np.asarray([float(data.qpos[2])], dtype=np.float32))
        self.stand_root_xy = tensor(np.asarray(data.qpos[:2], dtype=np.float32)[None, :])
        self.stand_root_quat = tensor(quat[None, :])
        self.stand_root_lin_vel = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.stand_root_ang_vel = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.stand_joint_pos = tensor(np.asarray(data.qpos[self.motion.robot_qpos_adrs], dtype=np.float32)[None, :])
        self.stand_joint_vel = torch.zeros((1, len(self.motion.robot_qvel_adrs)), dtype=torch.float32, device=self.device)
        self.stand_tendon_ctrl = tensor(tendon_ctrl[None, :])
        self.stand_foot_pos = tensor(foot_pos[None, :])
        self.stand_foot_vel = torch.zeros((1, len(CONTACT_GEOM_NAMES), 3), dtype=torch.float32, device=self.device)
        self.stand_foot_contact = torch.ones((1, 2), dtype=torch.float32, device=self.device)

    def _build_reset_helpers(self) -> None:
        limited_qpos_adrs: list[int] = []
        limited_qvel_adrs: list[int] = []
        joint_ranges: list[tuple[float, float]] = []
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if not self.model.jnt_limited[joint_id]:
                continue
            limited_qpos_adrs.append(int(self.model.jnt_qposadr[joint_id]))
            limited_qvel_adrs.append(int(self.model.jnt_dofadr[joint_id]))
            low, high = self.model.jnt_range[joint_id]
            joint_ranges.append((float(low), float(high)))
        self._limited_qpos_adrs_np = np.asarray(limited_qpos_adrs, dtype=np.int64)
        self._limited_qvel_adrs_np = np.asarray(limited_qvel_adrs, dtype=np.int64)
        self._joint_ranges_np = np.asarray(joint_ranges, dtype=np.float32)

    def _init_warp_data(self) -> None:
        assert wp is not None and mjwarp is not None
        cpu_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, cpu_data)

        wp.init()
        self.wp_device = wp.get_device(f"cuda:{self.device.index or 0}")
        with wp.ScopedDevice(self.wp_device):
            self.wp_model = mjwarp.put_model(self.model)
            self.wp_model.opt.timestep = wp.full(
                self.num_envs, float(self.cfg.timestep), dtype=wp.float32, device=self.wp_device,
            )
            self.wp_data = mjwarp.put_data(
                self.model,
                cpu_data,
                nworld=self.num_envs,
                nconmax=max(1, int(self.cfg.warp_nconmax)),
                njmax=max(1, int(self.cfg.warp_njmax)),
            )
            mjwarp.forward(self.wp_model, self.wp_data)

        self._init_warp_views()

    def _init_warp_views(self) -> None:
        """Expose the selected simulation's Warp buffers to the LYT task."""
        self.qpos = wp.to_torch(self.wp_data.qpos)
        self.qvel = wp.to_torch(self.wp_data.qvel)
        self.ctrl = wp.to_torch(self.wp_data.ctrl)
        self.wp_timestep = wp.to_torch(self.wp_model.opt.timestep)
        self.xpos = wp.to_torch(self.wp_data.xpos)
        self.xmat = wp.to_torch(self.wp_data.xmat)
        self.geom_xpos = wp.to_torch(self.wp_data.geom_xpos)
        self.ten_length = wp.to_torch(self.wp_data.ten_length)
        self.actuator_force = wp.to_torch(self.wp_data.actuator_force)
        self.qfrc_actuator = wp.to_torch(self.wp_data.qfrc_actuator)
        self.actuator_velocity = wp.to_torch(self.wp_data.actuator_velocity)
        self.xfrc_applied = wp.to_torch(self.wp_data.xfrc_applied)
        self.wp_body_mass = wp.to_torch(self.wp_model.body_mass)
        self.wp_body_inertia = wp.to_torch(self.wp_model.body_inertia)
        self.wp_body_subtreemass = wp.to_torch(self.wp_model.body_subtreemass)
        self.wp_geom_friction = wp.to_torch(self.wp_model.geom_friction)
        self.wp_gravity = wp.to_torch(self.wp_model.opt.gravity)
        self.wp_actuator_gainprm = wp.to_torch(self.wp_model.actuator_gainprm)
        self.wp_actuator_biasprm = wp.to_torch(self.wp_model.actuator_biasprm)
        # Keep live Warp views.  Casting to int64 here copies the int32
        # buffers once, freezing contact identities/counts for every step.
        self.contact_geom = wp.to_torch(self.wp_data.contact.geom)
        self.contact_pos = wp.to_torch(self.wp_data.contact.pos)
        self.contact_worldid = wp.to_torch(self.wp_data.contact.worldid)
        self.contact_dim = wp.to_torch(self.wp_data.contact.dim)
        self.contact_nacon = wp.to_torch(self.wp_data.nacon)
        self.contact_slot_ids = torch.arange(
            self.contact_dim.numel(), dtype=torch.long, device=self.device
        )
        self._warp_reset_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

    def _capture_warp_graphs(self) -> None:
        """Capture fixed-shape MuJoCo Warp launches to remove Python launch overhead."""
        self._step_graph = None
        self._forward_graph = None
        if not bool(self.cfg.warp_cuda_graph):
            return
        assert wp is not None and mjwarp is not None
        try:
            with wp.ScopedDevice(self.wp_device):
                with wp.ScopedCapture() as step_capture:
                    mjwarp.step(self.wp_model, self.wp_data)
            self._step_graph = step_capture.graph
        except Exception as exc:
            self._step_graph = None
            print(f"[WARN] Warp step CUDA graph capture failed; using direct launches: {exc}")
        try:
            with wp.ScopedDevice(self.wp_device):
                with wp.ScopedCapture() as forward_capture:
                    mjwarp.forward(self.wp_model, self.wp_data)
            self._forward_graph = forward_capture.graph
        except Exception as exc:
            self._forward_graph = None
            print(f"[WARN] Warp forward CUDA graph capture failed; using direct launches: {exc}")

    def _warp_step(self) -> None:
        assert wp is not None and mjwarp is not None
        if self._step_graph is None:
            mjwarp.step(self.wp_model, self.wp_data)
        else:
            wp.capture_launch(self._step_graph)

    def _warp_forward(self) -> None:
        assert wp is not None and mjwarp is not None
        if self._forward_graph is None:
            mjwarp.forward(self.wp_model, self.wp_data)
        else:
            wp.capture_launch(self._forward_graph)

    def _warp_decimated_step(self) -> None:
        for _ in range(int(self.cfg.decimation)):
            self._warp_step()

    def _init_domain_randomization(self) -> None:
        self.domain_randomization = bool(self.cfg.domain_randomization)
        self.domain_randomization_scale = float(np.clip(self.cfg.domain_randomization_scale, 0.0, 1.0))
        max_delay_s = max(float(self.cfg.action_delay_min_ms), float(self.cfg.action_delay_max_ms), 0.0) / 1000.0
        self.max_action_delay_ticks = int(np.ceil(max_delay_s / max(float(self.cfg.timestep), 1.0e-9)))
        history_len = max(2, self.max_action_delay_ticks + 2)
        self.delay_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.action_delay_history = torch.zeros(
            (self.num_envs, history_len, self.action_dim), dtype=torch.float32, device=self.device
        )
        self.action_delay_ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.action_delay_remainder = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.action_delay_cursor = 0
        self.env_tendon_length_scale = torch.ones(
            (self.num_envs, self.position_action_dim), dtype=torch.float32, device=self.device
        )
        self.push_force = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.push_time_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.push_time_to_next = torch.as_tensor(
            self._sample_push_interval(self.num_envs), dtype=torch.long, device=self.device
        )
        self.base_body_mass = torch.as_tensor(self.base_body_mass_np, dtype=torch.float32, device=self.device)
        self.base_body_inertia = torch.as_tensor(self.base_body_inertia_np, dtype=torch.float32, device=self.device)
        self.base_body_subtreemass = torch.as_tensor(
            np.asarray(self.model.body_subtreemass, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        self.base_geom_friction = torch.as_tensor(self.base_geom_friction_np, dtype=torch.float32, device=self.device)
        self.base_gravity = torch.as_tensor(self.base_gravity_np, dtype=torch.float32, device=self.device)
        self.base_actuator_gainprm = torch.as_tensor(self.base_actuator_gainprm_np, device=self.device)
        self.base_actuator_biasprm = torch.as_tensor(self.base_actuator_biasprm_np, device=self.device)

    def set_domain_randomization_scale(self, scale: float) -> None:
        """Set curriculum strength without changing ongoing global dynamics."""

        self.domain_randomization_scale = float(np.clip(scale, 0.0, 1.0))
        self.cfg.domain_randomization_scale = self.domain_randomization_scale

    def set_terrain_course_random_start_prob(self, probability: float) -> None:
        """Set the random-stage curriculum probability used by future resets."""

        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("terrain course random-start probability must be in [0, 1].")
        self.cfg.terrain_course_random_start_prob = probability

    def _reset_warp_worlds(self, ids: torch.Tensor) -> None:
        """Fully clear MuJoCo-Warp state for selected worlds before respawning.

        Assigning only qpos/qvel leaves solver warm-start, acceleration and
        contact buffers untouched.  Once one of those buffers becomes
        non-finite, that world remains poisoned across every later episode.
        MuJoCo-Warp's per-world reset is the authoritative equivalent of
        ``mj_resetData`` and preserves all worlds not selected by ``ids``.
        """

        self._warp_reset_mask.zero_()
        self._warp_reset_mask[ids] = True
        reset_mask = wp.from_torch(self._warp_reset_mask)
        try:
            with wp.ScopedDevice(self.wp_device):
                mjwarp.reset_data(self.wp_model, self.wp_data, reset=reset_mask)
        except RuntimeError as exc:
            if _WARP_DYNAMIC_SOURCE_ERROR not in str(exc):
                raise
            print(
                "[WARN] Warp reset kernel source became stale; "
                "reloading mujoco_warp._src.io and retrying once."
            )
            reset_data = _reload_mujoco_warp_reset_data()
            with wp.ScopedDevice(self.wp_device):
                reset_data(self.wp_model, self.wp_data, reset=reset_mask)

    def _uniform_np(self, low: float, high: float, size: int | tuple[int, ...] | None = None) -> np.ndarray:
        low_f = float(low)
        high_f = float(high)
        if high_f < low_f:
            low_f, high_f = high_f, low_f
        if high_f == low_f:
            return np.full(size or (), low_f, dtype=np.float32)
        return np.asarray(self.rng.uniform(low_f, high_f, size=size), dtype=np.float32)

    def _sample_push_interval(self, size: int) -> np.ndarray:
        low = max(1, int(self.cfg.push_interval_min_steps))
        high = max(low + 1, int(self.cfg.push_interval_max_steps) + 1)
        return self.rng.integers(low, high, size=size, dtype=np.int32)

    def _sample_push_duration(self, size: int) -> np.ndarray:
        low = max(1, int(self.cfg.push_duration_min_steps))
        high = max(low + 1, int(self.cfg.push_duration_max_steps) + 1)
        return self.rng.integers(low, high, size=size, dtype=np.int32)

    def _sample_stand_transition_steps(self, size: int) -> torch.Tensor:
        low = max(1, int(self.cfg.stand_start_transition_min_steps))
        high = max(low, int(self.cfg.stand_start_transition_max_steps))
        if high <= low:
            values = np.full(size, low, dtype=np.int64)
        else:
            values = self.rng.integers(low, high + 1, size=size, dtype=np.int64)
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    def _refresh_episode_rng(self, ids_np: np.ndarray) -> None:
        if not self.cfg.randomize_episode_seed or ids_np.size == 0:
            return
        seeds = self.rng.integers(0, np.iinfo(np.uint64).max, size=len(ids_np), dtype=np.uint64)
        self.episode_seed[ids_np] = seeds

    def _sample_domain_randomization(
        self,
        ids_np: np.ndarray,
        *,
        resample_global: bool = False,
    ) -> None:
        if not self.domain_randomization or ids_np.size == 0:
            return
        scale = self.domain_randomization_scale
        tendon_low, tendon_high = scaled_randomization_range(
            1.0, self.cfg.tendon_length_scale_min, self.cfg.tendon_length_scale_max, scale
        )
        tendon_scale = self._uniform_np(
            tendon_low,
            tendon_high,
            (len(ids_np), self.position_action_dim),
        )
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=self.device)
        self.env_tendon_length_scale[ids] = torch.as_tensor(tendon_scale, dtype=torch.float32, device=self.device)
        delay_ticks, remainder = sample_action_delay(self.rng, self.cfg, scale, len(ids_np))
        self.action_delay_ticks[ids] = torch.as_tensor(
            delay_ticks,
            dtype=torch.long,
            device=self.device,
        )
        self.action_delay_remainder[ids] = torch.as_tensor(remainder, device=self.device)
        self.action_delay_history[ids] = 0.0
        self.push_force[ids] = 0.0
        self.push_time_left[ids] = 0
        self.push_time_to_next[ids] = torch.as_tensor(
            self._sample_push_interval(len(ids_np)), dtype=torch.long, device=self.device
        )
        if resample_global and bool(self.cfg.warp_global_domain_randomization):
            self._apply_global_domain_randomization()

    def _apply_global_domain_randomization(self) -> None:
        if not self.domain_randomization:
            return
        scale = self.domain_randomization_scale
        body_low, body_high = scaled_randomization_range(
            1.0, self.cfg.body_mass_scale_min, self.cfg.body_mass_scale_max, scale
        )
        payload_low, payload_high = scaled_randomization_range(
            0.0, self.cfg.payload_mass_min, self.cfg.payload_mass_max, scale
        )
        nominal_friction = float(self.base_geom_friction_np[self.floor_geom_id, 0])
        friction_low, friction_high = scaled_randomization_range(
            nominal_friction, self.cfg.ground_friction_min, self.cfg.ground_friction_max, scale
        )
        nominal_gravity = abs(float(self.base_gravity_np[2]))
        gravity_low, gravity_high = scaled_randomization_range(
            nominal_gravity, self.cfg.gravity_min, self.cfg.gravity_max, scale
        )
        body_scale = torch.as_tensor(
            self._uniform_np(body_low, body_high, self.model.nbody),
            dtype=torch.float32,
            device=self.device,
        )
        body_scale[0] = 1.0
        body_mass = self.base_body_mass * body_scale
        if self.pelvis_body_id >= 0:
            payload_mass = float(self._uniform_np(payload_low, payload_high))
            body_mass[self.pelvis_body_id] += payload_mass
        strength_low, strength_high = scaled_randomization_range(
            1.0, self.cfg.actuator_strength_scale_min, self.cfg.actuator_strength_scale_max, scale)
        strength = torch.as_tensor(self._uniform_np(strength_low, strength_high, (self.model.nu,)),
                                   dtype=torch.float32, device=self.device)
        self.wp_actuator_gainprm[0] = self.base_actuator_gainprm * strength[:, None]
        self.wp_actuator_biasprm[0] = self.base_actuator_biasprm * strength[:, None]
        self.wp_body_mass[0] = body_mass
        self.wp_body_inertia[0] = self.base_body_inertia * body_scale[:, None]
        self.wp_body_inertia[0, 0] = self.base_body_inertia[0]
        self.wp_geom_friction[0] = self.base_geom_friction
        self.wp_geom_friction[0, self.terrain_geom_ids, 0] = float(
            self._uniform_np(friction_low, friction_high)
        )
        self.wp_gravity[0] = self.base_gravity
        self.wp_gravity[0, 2] = -float(self._uniform_np(gravity_low, gravity_high))
        # Refresh subtree masses, inverse inertias, and actuator constants
        # after changing mass and inertia on the device model.
        with wp.ScopedDevice(self.wp_device):
            mjwarp.set_const(self.wp_model, self.wp_data)

    def _apply_tendon_length_randomization(
        self,
        ctrl: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.domain_randomization:
            return ctrl
        scale = self.env_tendon_length_scale if env_ids is None else self.env_tendon_length_scale[env_ids]
        scaled = self.ctrl_mid + (ctrl - self.ctrl_mid) * scale
        return torch.clamp(scaled, min=self.ctrl_low, max=self.ctrl_high)

    def _apply_action_noise(self, action: torch.Tensor) -> torch.Tensor:
        if not self.domain_randomization or self.cfg.action_noise_std <= 0.0:
            return action
        noise_std = self.domain_randomization_scale * float(self.cfg.action_noise_std)
        return torch.clamp(action + noise_std * torch.randn_like(action), -1.0, 1.0)

    def _startup_action_ramp(self) -> torch.Tensor:
        ramp_steps = max(0, int(self.cfg.startup_action_ramp_steps))
        if ramp_steps <= 0:
            return torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ramp = (self.episode_step.to(torch.float32) + 1.0) / float(ramp_steps)
        return torch.clamp(ramp, 0.0, 1.0)[:, None]

    def _step_delayed_ctrl(self, commanded_ctrl: torch.Tensor) -> None:
        """Integrate each world on both sides of its control arrival time."""
        history_len = self.action_delay_history.shape[1]
        self.action_delay_history[:, self.action_delay_cursor] = commanded_ctrl
        read_index = (self.action_delay_cursor - self.action_delay_ticks) % history_len
        before = self.action_delay_history[self.delay_env_ids, (read_index - 1) % history_len]
        after = self.action_delay_history[self.delay_env_ids, read_index]
        has_remainder = self.action_delay_remainder > 0.0
        # At an exact tick boundary the new control is active throughout.
        # Two positive half-steps keep the batched integrator free of zero dt.
        first_step = torch.where(
            has_remainder, self.action_delay_remainder, float(self.cfg.timestep) * 0.5,
        )
        try:
            self.ctrl[:] = torch.where(has_remainder[:, None], before, after)
            self.wp_timestep[:] = first_step
            self._warp_step()
            self.ctrl[:] = after
            self.wp_timestep[:] = float(self.cfg.timestep) - first_step
            self._warp_step()
        finally:
            self.wp_timestep.fill_(float(self.cfg.timestep))
        self.action_delay_cursor = (self.action_delay_cursor + 1) % history_len

    def _update_push_forces(self) -> None:
        self.xfrc_applied.zero_()
        if not self.domain_randomization:
            return
        active = self.push_time_left > 0
        self.push_time_left.sub_(active.to(self.push_time_left.dtype))
        ended = active & (self.push_time_left <= 0)
        interval_low = max(1, int(self.cfg.push_interval_min_steps))
        interval_high = max(interval_low + 1, int(self.cfg.push_interval_max_steps) + 1)
        sampled_intervals = torch.randint(
            interval_low,
            interval_high,
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self.push_time_to_next[:] = torch.where(ended, sampled_intervals, self.push_time_to_next)
        self.push_force[:] = torch.where(ended[:, None], torch.zeros_like(self.push_force), self.push_force)

        idle = self.push_time_left <= 0
        self.push_time_to_next.sub_(idle.to(self.push_time_to_next.dtype))
        start = idle & (self.push_time_to_next <= 0)
        directions = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1.0e-6)
        low_force = self.domain_randomization_scale * min(
            float(self.cfg.push_force_min), float(self.cfg.push_force_max)
        )
        high_force = self.domain_randomization_scale * max(
            float(self.cfg.push_force_min), float(self.cfg.push_force_max)
        )
        magnitudes = low_force + (high_force - low_force) * torch.rand(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )
        sampled_force = torch.cat((directions * magnitudes, torch.zeros_like(directions)), dim=1)
        self.push_force[:] = torch.where(start[:, None], sampled_force, self.push_force)
        duration_low = max(1, int(self.cfg.push_duration_min_steps))
        duration_high = max(duration_low + 1, int(self.cfg.push_duration_max_steps) + 1)
        sampled_durations = torch.randint(
            duration_low,
            duration_high,
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self.push_time_left[:] = torch.where(start, sampled_durations, self.push_time_left)

        self.xfrc_applied[:, self.pelvis_body_id] = self.push_force

    def _targets_at_phase(
        self,
        phase: torch.Tensor,
        motion_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ids = (
            torch.zeros_like(phase, dtype=torch.long)
            if motion_ids is None
            else motion_ids
        )
        frame_counts = self.motion_frame_counts[ids]
        looping = self.motion_loop[ids]
        floor_phase = torch.floor(phase).long()
        low = torch.where(
            looping,
            torch.remainder(floor_phase, frame_counts),
            torch.minimum(torch.clamp_min(floor_phase, 0), frame_counts - 1),
        )
        high = torch.where(looping, torch.remainder(low + 1, frame_counts), torch.minimum(low + 1, frame_counts - 1))
        blend = (phase - torch.floor(phase)).to(torch.float32)

        def interp(values: torch.Tensor) -> torch.Tensor:
            low_values = values[ids, low]
            high_values = values[ids, high]
            shape = (blend.shape[0],) + (1,) * (low_values.ndim - 1)
            b = blend.reshape(shape)
            return (1.0 - b) * low_values + b * high_values

        return {
            "root_xy": interp(self.ref_root_xy),
            "root_height": interp(self.ref_root_height),
            "root_quat": _canonical_quat(interp(self.ref_root_quat)),
            "root_lin_vel": interp(self.ref_root_lin_vel),
            "root_ang_vel": interp(self.ref_root_ang_vel),
            "joint_pos": interp(self.ref_joint_pos),
            "joint_vel": interp(self.ref_joint_vel),
            "tendon_ctrl": interp(self.ref_tendon_ctrl),
            "foot_pos": interp(self.ref_foot_pos),
            "foot_vel": interp(self.ref_foot_vel),
            "foot_contact": interp(self.ref_foot_contact),
        }

    def _stand_start_progress(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        episode_step = self.episode_step if env_ids is None else self.episode_step[env_ids]
        transition_steps = (
            self.stand_start_transition_steps
            if env_ids is None
            else self.stand_start_transition_steps[env_ids]
        )
        stand_start = self.stand_start if env_ids is None else self.stand_start[env_ids]
        delay = max(0, int(self.cfg.stand_start_phase_delay_steps))
        raw = (episode_step.to(torch.float32) - float(delay)) / transition_steps.to(torch.float32).clamp_min(1.0)
        progress = torch.clamp(raw, 0.0, 1.0)
        progress = progress * progress * (3.0 - 2.0 * progress)
        return torch.where(stand_start, progress, torch.ones_like(progress))

    def _blend_start_targets(
        self,
        targets: dict[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        progress = self._stand_start_progress(env_ids)

        def blend(stand: torch.Tensor, walk: torch.Tensor) -> torch.Tensor:
            shape = (walk.shape[0],) + (1,) * (walk.ndim - 1)
            weight = progress.reshape(shape)
            return (1.0 - weight) * stand + weight * walk

        root_quat = _canonical_quat(blend(self.stand_root_quat, targets["root_quat"]))
        return {
            "root_xy": blend(self.stand_root_xy, targets["root_xy"]),
            "root_height": blend(self.stand_root_height, targets["root_height"]),
            "root_quat": root_quat,
            "root_lin_vel": blend(self.stand_root_lin_vel, targets["root_lin_vel"]),
            "root_ang_vel": blend(self.stand_root_ang_vel, targets["root_ang_vel"]),
            "joint_pos": blend(self.stand_joint_pos, targets["joint_pos"]),
            "joint_vel": blend(self.stand_joint_vel, targets["joint_vel"]),
            "tendon_ctrl": blend(self.stand_tendon_ctrl, targets["tendon_ctrl"]),
            "foot_pos": blend(self.stand_foot_pos, targets["foot_pos"]),
            "foot_vel": blend(self.stand_foot_vel, targets["foot_vel"]),
            "foot_contact": blend(self.stand_foot_contact, targets["foot_contact"]),
        }

    def _targets_at_current_phase(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        phase = self.motion_phase if env_ids is None else self.motion_phase[env_ids]
        motion_ids = self.motion_ids if env_ids is None else self.motion_ids[env_ids]
        targets = self._targets_at_phase(phase, motion_ids)
        stair_anchor = self.motion_stair_height_anchor[motion_ids]
        stair_scale = self.motion_stair_height_scale[motion_ids]
        targets["root_height"] = stair_anchor + (
            targets["root_height"] - stair_anchor
        ) * stair_scale
        targets["root_lin_vel"] = targets["root_lin_vel"].clone()
        targets["root_lin_vel"][:, 2] *= stair_scale
        targets = self._blend_start_targets(targets, env_ids=env_ids)
        if self.cfg.command_walking:
            values = self.walking_commands.values if env_ids is None else self.walking_commands.values[env_ids]
            ratio = values[:, 0] / self.cfg.target_velocity
            progress = self._stand_start_progress(env_ids)
            targets["joint_vel"] = targets["joint_vel"] * ratio[:, None]
            targets["foot_vel"] = targets["foot_vel"] * ratio[:, None, None]
            targets["root_ang_vel"] = heading_angular_velocity(targets["root_quat"], targets["root_ang_vel"])
            targets["root_quat"] = without_heading(targets["root_quat"])
            targets["root_lin_vel"] = targets["root_lin_vel"] * ratio[:, None]
            targets["root_lin_vel"][:, 0] = values[:, 0] * progress
            targets["root_lin_vel"][:, 1] = 0.0
            targets["root_ang_vel"] = targets["root_ang_vel"] * ratio[:, None]
            targets["root_ang_vel"][:, 2] = values[:, 1] * progress
        if self.more_terrain_cfg.serial_course:
            terrain_offset = torch.full_like(
                targets["root_height"],
                float(self.more_terrain_cfg.course_base_height),
            )
            no_offset = torch.zeros_like(terrain_offset)
            pit_gap = torch.zeros_like(motion_ids, dtype=torch.bool)
            for motion_id, kind in enumerate(self.motion_terrain_kinds):
                if kind in {"gap", "pit"}:
                    pit_gap = pit_gap | (motion_ids == motion_id)
            targets["root_height"] = targets["root_height"] + torch.where(
                pit_gap, no_offset, terrain_offset
            )
        return targets

    def _clip_joint_ranges_batch(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        if len(self._limited_qpos_adrs_np) == 0:
            return
        qadr = self._limited_qpos_adrs_np
        dadr = self._limited_qvel_adrs_np
        low = self._joint_ranges_np[:, 0]
        high = self._joint_ranges_np[:, 1]
        values = qpos[:, qadr].copy()
        clipped = np.clip(values, low, high)
        qpos[:, qadr] = clipped
        crossed_low = (values < low) & (qvel[:, dadr] < 0.0)
        crossed_high = (values > high) & (qvel[:, dadr] > 0.0)
        qvel[:, dadr] = np.where(crossed_low | crossed_high, 0.0, qvel[:, dadr])

    def action_to_ctrl(
        self,
        action: torch.Tensor,
        base_ctrl: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        position_action = torch.clamp(action.to(dtype=torch.float32, device=self.device), -1.0, 1.0)
        base = self.ctrl_mid if base_ctrl is None else base_ctrl
        position_ctrl = base + (
            position_action * self.ctrl_half_range * self.action_scale * float(self.cfg.action_residual_scale)
        )
        position_ctrl = torch.clamp(position_ctrl, min=self.ctrl_low, max=self.ctrl_high)
        position_ctrl = self._apply_tendon_length_randomization(position_ctrl, env_ids=env_ids)
        return position_ctrl

    def ctrl_to_action(self, ctrl: torch.Tensor) -> torch.Tensor:
        ctrl_tensor = ctrl.to(dtype=torch.float32, device=self.device)
        return torch.clamp((ctrl_tensor - self.ctrl_mid) / self.ctrl_half_range, -1.0, 1.0)

    def _current_tendon_ctrl(self) -> torch.Tensor:
        ctrl = self.ten_length[:, self.actuator_tendon_ids]
        return torch.clamp(ctrl, min=self.ctrl_low, max=self.ctrl_high)

    def _reference_base_ctrl(
        self,
        reference_ctrl: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        blend = float(np.clip(self.cfg.reference_ctrl_blend, 0.0, 1.0))
        current = self._current_tendon_ctrl()
        if env_ids is not None:
            current = current[env_ids]
        if blend <= 0.0:
            return current
        return torch.clamp((1.0 - blend) * current + blend * reference_ctrl, min=self.ctrl_low, max=self.ctrl_high)

    def _transition_reference_ctrl(
        self, reference_ctrl: torch.Tensor
    ) -> torch.Tensor:
        """Blend only a newly selected motion onto the previous reference."""
        target = reference_ctrl.to(dtype=torch.float32, device=self.device)
        total_steps = max(0, int(self.cfg.motion_transition_blend_steps))
        if total_steps <= 0:
            self.motion_transition_steps_left.zero_()
            self.last_reference_ctrl.copy_(target)
            return target

        active = self.motion_transition_steps_left > 0
        alpha = (
            total_steps - self.motion_transition_steps_left + 1
        ).to(torch.float32) / float(total_steps)
        alpha = torch.clamp(alpha, 0.0, 1.0)[:, None]
        transitioned = torch.where(
            active[:, None],
            (1.0 - alpha) * self.motion_transition_from_ctrl + alpha * target,
            target,
        )
        self.motion_transition_steps_left.copy_(
            torch.where(
                active,
                self.motion_transition_steps_left - 1,
                torch.zeros_like(self.motion_transition_steps_left),
            )
        )
        self.last_reference_ctrl.copy_(transitioned)
        return transitioned

    def _foot_positions_local(self) -> torch.Tensor:
        root_pos = self.xpos[:, self.pelvis_body_id]
        root_mat = self.xmat[:, self.pelvis_body_id]
        foot_world = self.geom_xpos[:, self.contact_geom_ids]
        return torch.matmul(foot_world - root_pos[:, None, :], root_mat)

    def _foot_positions_world(self) -> torch.Tensor:
        return self.geom_xpos[:, self.contact_geom_ids]

    @staticmethod
    def _side_contacts_from_geom(geom_contact: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (torch.amax(geom_contact[:, 0:2], dim=1), torch.amax(geom_contact[:, 2:4], dim=1)),
            dim=1,
        )

    def _foot_geom_contacts(self) -> torch.Tensor:
        contact = torch.zeros(
            (self.num_envs, len(CONTACT_GEOM_NAMES)),
            dtype=torch.float32,
            device=self.device,
        )
        g1 = self.contact_geom[:, 0]
        g2 = self.contact_geom[:, 1]
        world = self.contact_worldid
        valid = self._valid_contact_slots()
        safe_world = torch.clamp(world, 0, self.num_envs - 1).long()

        g1_terrain = torch.isin(g1, self.terrain_geom_ids)
        g2_terrain = torch.isin(g2, self.terrain_geom_ids)
        for contact_index, geom in enumerate(self.contact_geom_id_values):
            matched = valid & (
                (g1_terrain & (g2 == geom))
                | (g2_terrain & (g1 == geom))
            )
            contact[:, contact_index].scatter_reduce_(
                0,
                safe_world,
                matched.to(torch.float32),
                reduce="amax",
                include_self=True,
            )
        return contact

    def _foot_contacts(self) -> torch.Tensor:
        return self._side_contacts_from_geom(self._foot_geom_contacts())

    def _valid_contact_slots(self) -> torch.Tensor:
        """Ignore stale contacts beyond Warp's current active prefix."""

        return (
            (self.contact_slot_ids < self.contact_nacon[0])
            & (self.contact_dim > 0)
            & (self.contact_worldid >= 0)
            & (self.contact_worldid < self.num_envs)
        )

    def _sensor_foot_roll_contacts(self) -> torch.Tensor:
        """Classify real floor contacts by heel, sole and toe position."""

        zones = torch.zeros((self.num_envs, 2, 3), dtype=torch.float32, device=self.device)
        g1, g2 = self.contact_geom[:, 0], self.contact_geom[:, 1]
        world = self.contact_worldid
        valid = self._valid_contact_slots()
        safe_world = world.clamp(0, self.num_envs - 1).long()
        terrain1 = torch.isin(g1, self.terrain_geom_ids)
        terrain2 = torch.isin(g2, self.terrain_geom_ids)
        for side, (foot_geom, toe_geom) in enumerate(
            ((self.contact_geom_id_values[0], self.contact_geom_id_values[1]),
             (self.contact_geom_id_values[2], self.contact_geom_id_values[3]))
        ):
            foot_contact = valid & ((terrain1 & (g2 == foot_geom)) | (terrain2 & (g1 == foot_geom)))
            toe_contact = valid & ((terrain1 & (g2 == toe_geom)) | (terrain2 & (g1 == toe_geom)))
            body_id = self.sensor_foot_body_ids[side]
            # Foot-local X separates the posterior heel from the central sole.
            # The toe is its own articulated geom.  These boundaries are
            # inside the observed foot mesh contact span (-0.026..0.12 m).
            local_x = torch.sum(
                (self.contact_pos - self.xpos[safe_world, body_id])
                * self.xmat[safe_world, body_id, :, 0],
                dim=1,
            )
            toe_from_box = foot_contact & (local_x > 0.075)
            for zone, matched in enumerate(
                (foot_contact & (local_x <= 0.0),
                 foot_contact & (local_x > 0.0) & (local_x <= 0.075),
                 toe_contact | toe_from_box)
            ):
                zones[:, side, zone].scatter_reduce_(
                    0, safe_world, matched.float(), reduce="amax", include_self=True
                )
        # Warp's convex mesh narrowphase often retains only the posterior
        # foot contact while the articulated toe also contacts the floor.
        # Two loaded endpoints imply support through the intervening sole;
        # this recovers the reference roll without treating a toe-only strike
        # as a complete heel/sole/toe sequence.
        zones[:, :, 1] = torch.maximum(
            zones[:, :, 1], torch.minimum(zones[:, :, 0], zones[:, :, 2])
        )
        return zones > 0.5

    def reset(self, env_ids: np.ndarray | list[int] | torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            ids_np = np.arange(self.num_envs, dtype=np.int64)
        elif torch.is_tensor(env_ids):
            ids_np = env_ids.detach().cpu().numpy().astype(np.int64)
        else:
            ids_np = np.asarray(env_ids, dtype=np.int64)
        if ids_np.size == 0:
            return self.last_obs
        self._refresh_episode_rng(ids_np)
        if self.walking_commands is not None:
            self.walking_commands.reset(ids_np, self.rng)

        serial_course = self.more_terrain_cfg.serial_course
        if serial_course:
            variant_count = len(self.more_terrain_cfg.normalized_course_orders)
            variants = (
                self.rng.integers(0, variant_count, size=len(ids_np), dtype=np.int64)
                if self.cfg.terrain_course_random_assignment
                else np.mod(ids_np, variant_count).astype(np.int64)
            )
            start_stages = sample_course_start_stages(
                self.rng,
                len(ids_np),
                self.course_motion_sequences_np.shape[1],
                self.cfg.terrain_course_random_start_prob,
            ).astype(np.int64)
            sampled_motion_ids = self.course_motion_sequences_np[
                variants, start_stages
            ].astype(np.int32)
            assert isinstance(self.motion, MotionLibrary)
            states = self.motion.sample_states(
                len(ids_np), self.rng, motion_ids=sampled_motion_ids
            )
            for row, motion_id in enumerate(sampled_motion_ids):
                states.qpos[row] = self.motion.motions[int(motion_id)].reference_qpos[0]
                # Match the CPU serial-course reset: enter every route at rest
                # instead of applying a reference seam velocity as an impulse.
                states.qvel[row] = 0.0
                states.indices[row] = 0
        elif isinstance(self.motion, MotionLibrary):
            sampled_motion_ids = self.motion.sample_motion_ids(len(ids_np), self.rng)
            states = self.motion.sample_states(len(ids_np), self.rng, motion_ids=sampled_motion_ids)
        else:
            sampled_motion_ids = np.zeros(len(ids_np), dtype=np.int32)
            states = self.motion.sample_states(len(ids_np), self.rng)
        if self.continuous_stair_phase_curriculum:
            reference = (
                self.motion.motions[0]
                if isinstance(self.motion, MotionLibrary)
                else self.motion
            )
            landing_frames = np.arange(
                0, max(reference.frame_count - 1, 1), 32, dtype=np.int32
            )
            sampled_frames = self.rng.choice(landing_frames, size=len(ids_np))
            states.qpos[:] = reference.reference_qpos[sampled_frames]
            states.qvel[:] = 0.0
            states.indices[:] = sampled_frames
        use_stand_start = (
            np.zeros(len(ids_np), dtype=bool)
            if serial_course
            else self.rng.random(len(ids_np))
            < float(np.clip(self.cfg.stand_start_prob, 0.0, 1.0))
        )
        use_motion_reset = (
            np.ones(len(ids_np), dtype=bool)
            if serial_course
            else ~use_stand_start
            & (
                self.rng.random(len(ids_np))
                < float(np.clip(self.cfg.motion_reset_prob, 0.0, 1.0))
            )
        )
        stair_training_start = (
            np.zeros(len(ids_np), dtype=bool)
            if serial_course
            else self.stair_motion_mask_np[sampled_motion_ids]
            & bool(self.cfg.stair_fixed_start)
            & ~use_stand_start
        )
        stair_random_phase = stair_training_start & (
            self.rng.random(len(ids_np)) < float(np.clip(self.cfg.stair_phase_reset_prob, 0.0, 1.0))
        )
        use_motion_reset |= stair_training_start
        if np.any(stair_training_start):
            for row in np.flatnonzero(stair_training_start):
                motion_id = int(sampled_motion_ids[row])
                reference = (
                    self.motion.motions[motion_id]
                    if isinstance(self.motion, MotionLibrary)
                    else self.motion
                )
                if stair_random_phase[row]:
                    # Original DeepMimic reference-state initialization uses
                    # the complete motion distribution, not sparse landings.
                    last_start = max(reference.frame_count - 1, 1)
                    frame = int(self.rng.integers(0, last_start))
                else:
                    frame = 0
                states.qpos[row] = reference.reference_qpos[frame]
                states.qvel[row] = reference.reference_qvel[frame]
                states.indices[row] = frame
        # Mass, inertia, friction, and gravity live on the single shared Warp
        # model.  They are opt-in because one draw would otherwise change all
        # worlds at once and destroy the within-batch dynamics distribution.
        full_reset = ids_np.size == self.num_envs
        self._sample_domain_randomization(ids_np, resample_global=full_reset)
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=self.device)
        self._reset_warp_worlds(ids)
        if serial_course:
            self.course_variant_ids[ids] = torch.as_tensor(
                variants, dtype=torch.long, device=self.device
            )
            start_stage_tensor = torch.as_tensor(
                start_stages, dtype=torch.long, device=self.device
            )
            self.course_stage_ids[ids] = start_stage_tensor
            self.course_start_stage_ids[ids] = start_stage_tensor
        self.motion_ids[ids] = torch.as_tensor(
            sampled_motion_ids, dtype=torch.long, device=self.device
        )
        self.stand_start[ids] = torch.as_tensor(use_stand_start, dtype=torch.bool, device=self.device)
        self.stand_start_transition_steps[ids] = self._sample_stand_transition_steps(len(ids_np))
        qpos = np.repeat(self.motion.model_qpos0[None, :], len(ids_np), axis=0).astype(np.float32)
        qvel = np.zeros((len(ids_np), self.model.nv), dtype=np.float32)
        phase = np.zeros(len(ids_np), dtype=np.float32)
        phase[use_stand_start] = float(self.cfg.stand_start_motion_phase)
        qpos[use_motion_reset] = states.qpos[use_motion_reset]
        qvel[use_motion_reset] = states.qvel[use_motion_reset]
        phase[use_motion_reset] = states.indices[use_motion_reset].astype(np.float32)
        reset_frame_counts = np.asarray(
            self.motion.frame_counts
            if isinstance(self.motion, MotionLibrary)
            else [self.motion.frame_count],
            dtype=np.int32,
        )[sampled_motion_ids]
        reset_looping = np.asarray(
            self.motion.motion_loop
            if isinstance(self.motion, MotionLibrary)
            else [self.motion.source_loop],
            dtype=bool,
        )[sampled_motion_ids]
        phase = np.where(
            reset_looping,
            np.mod(phase, reset_frame_counts),
            np.clip(phase, 0.0, reset_frame_counts - 1.0),
        ).astype(np.float32)
        reset_motion_ids = sampled_motion_ids[use_motion_reset]
        reset_height_anchor = self.motion_stair_height_anchor_np[reset_motion_ids]
        reset_height_scale = self.motion_stair_height_scale_np[reset_motion_ids]
        qpos[use_motion_reset, 2] = reset_height_anchor + (
            qpos[use_motion_reset, 2] - reset_height_anchor
        ) * reset_height_scale
        qvel[use_motion_reset, 2] *= reset_height_scale
        if self.complex_terrain_enabled:
            for row, motion_id in enumerate(sampled_motion_ids):
                if serial_course:
                    scenario = self.course_stage_scenario_names[
                        int(variants[row])
                    ][int(start_stages[row])]
                    terrain_kind = (
                        "stairs" if scenario in {"stair_up", "stair_down"} else scenario
                    )
                    stair_down = scenario == "stair_down"
                else:
                    terrain_kind = self.motion_terrain_kinds[int(motion_id)]
                    stair_down = bool(
                        self.stair_down_motion_mask_np[int(motion_id)]
                    )
                offset = terrain_spawn_offset(
                    terrain_kind,
                    self.more_terrain_cfg,
                    self.stair_cfg,
                    stair_down=stair_down,
                    stair_down_offset=self.cfg.stair_down_offset,
                    course_variant=(int(variants[row]) if serial_course else 0),
                )
                qpos[row, :3] += offset

        if self.cfg.reset_noise_scale > 0.0:
            qpos[:, self.motion.robot_qpos_adrs] += self.rng.normal(
                0.0, self.cfg.reset_noise_scale, size=(len(ids_np), len(self.motion.robot_qpos_adrs))
            ).astype(np.float32)
            qvel[:, self.motion.robot_qvel_adrs] += self.rng.normal(
                0.0, 2.0 * self.cfg.reset_noise_scale, size=(len(ids_np), len(self.motion.robot_qvel_adrs))
            ).astype(np.float32)
            self._clip_joint_ranges_batch(qpos, qvel)

        self.qpos[ids] = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        self.qvel[ids] = torch.as_tensor(qvel, dtype=torch.float32, device=self.device)
        self.motion_phase[ids] = torch.as_tensor(phase, dtype=torch.float32, device=self.device)
        self.ctrl[ids] = 0.0
        with wp.ScopedDevice(self.wp_device):
            self._warp_forward()

        self.episode_step[ids] = 0
        self.episode_timeout_step[ids] = self._sample_timeout_steps(len(ids_np))
        self.forward_progress_anchor_x[ids] = self.qpos[ids, 0]
        self.forward_progress_anchor_step[ids] = 0
        self.stair_start_x[ids] = self.qpos[ids, 0]
        self.stair_start_y[ids] = self.qpos[ids, 1]
        remaining = np.empty(len(ids_np), dtype=np.float32)
        remaining_lateral = np.empty(len(ids_np), dtype=np.float32)
        for row, motion_id in enumerate(sampled_motion_ids):
            reference = (
                self.motion.motions[int(motion_id)]
                if isinstance(self.motion, MotionLibrary)
                else self.motion
            )
            remaining[row] = (
                reference.reference_qpos[-1, 0]
                - reference.reference_qpos[int(phase[row]), 0]
            )
            remaining_lateral[row] = (
                reference.reference_qpos[-1, 1]
                - reference.reference_qpos[int(phase[row]), 1]
            )
        self.stair_remaining_distance[ids] = torch.as_tensor(
            remaining, dtype=torch.float32, device=self.device
        )
        self.stair_remaining_lateral[ids] = torch.as_tensor(
            remaining_lateral, dtype=torch.float32, device=self.device
        )
        self.stair_full_start[ids] = torch.as_tensor(
            phase < 1.0, dtype=torch.bool, device=self.device
        )
        self.stair_terminal_hold_elapsed[ids] = 0
        targets = self._targets_at_current_phase() if self.actor_observation_enabled else None
        reset_targets = (
            {key: value[ids] for key, value in targets.items()}
            if targets is not None
            else self._targets_at_current_phase(ids)
        )
        self.last_reference_ctrl[ids] = reset_targets["tendon_ctrl"]
        self.motion_transition_from_ctrl[ids] = reset_targets["tendon_ctrl"]
        self.motion_transition_steps_left[ids] = 0
        zero_action = torch.zeros((len(ids_np), self.action_dim), dtype=torch.float32, device=self.device)
        reset_base_ctrl = self._reference_base_ctrl(reset_targets["tendon_ctrl"], env_ids=ids)
        self.ctrl[ids] = self.action_to_ctrl(zero_action, base_ctrl=reset_base_ctrl, env_ids=ids)
        self.action_delay_history[ids] = self.ctrl[ids, None, :]
        self.last_action[ids] = 0.0
        foot_pos = self._foot_positions_local()[ids]
        self.last_foot_pos[ids] = foot_pos
        self.last_foot_world_pos[ids] = self._foot_positions_world()[ids]
        self.step_lengths.reset(ids)
        self.sensor_last_foot_contact[ids] = False
        self.sensor_foot_roll_stage[ids] = 0
        self.sensor_toe_only_steps[ids] = 0
        self.sensor_last_touchdown_side[ids] = -1
        self.sensor_liftoff_x[ids] = self.geom_xpos[ids][:, self.contact_geom_ids[[0, 2]], 0]
        self.sensor_had_liftoff[ids] = False
        self.sensor_stride_ema[ids] = 0.0
        self.sensor_stride_seen[ids] = False
        self.sensor_contact_duty_ema[ids] = 0.0
        self.sensor_single_support_ema[ids] = 0.0
        self.sensor_time_since_touchdown[ids] = 0.0
        self.sensor_step_interval_ema[ids] = 0.0
        self.sensor_touchdown_seen[ids] = False
        self.sensor_cadence_seen[ids] = False
        self.sensor_swing_steps[ids] = 0
        self.sensor_prev_foot_world[ids] = self.geom_xpos[ids][:, self.contact_geom_ids[[0, 2]]]
        if self.actor_observation_enabled:
            assert targets is not None
            self.last_obs = self.observe(targets=targets)
        return self.last_obs

    def _sample_timeout_steps(self, batch_size: int) -> torch.Tensor:
        base_steps = int(self.cfg.episode_steps)
        if base_steps == 0:
            return torch.zeros(batch_size, dtype=torch.long, device=self.device)
        jitter = int(max(self.cfg.episode_timeout_jitter_steps, 0))
        if jitter <= 0:
            values = np.full(batch_size, base_steps, dtype=np.int64)
        else:
            low = max(1, base_steps - jitter)
            high = max(low + 1, base_steps + jitter + 1)
            values = self.rng.integers(low, high, size=batch_size, dtype=np.int64)
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    def observe(self, targets: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        targets = self._targets_at_current_phase() if targets is None else targets
        quat = _canonical_quat(self.qpos[:, 3:7])
        up_z = self.xmat[:, self.pelvis_body_id, 2, 2]
        frame_counts = self.motion_frame_counts[self.motion_ids].to(torch.float32)
        phase = 2.0 * torch.pi * self.motion_phase / frame_counts
        joint_pos = self.qpos[:, self.robot_qpos_adrs]
        joint_vel = self.qvel[:, self.robot_qvel_adrs]
        foot_pos = self._foot_positions_local().reshape(self.num_envs, -1)
        ref_foot_pos = targets["foot_pos"].reshape(self.num_envs, -1)
        ref_ctrl_action = self.ctrl_to_action(targets["tendon_ctrl"])
        root_lin_vel_target = targets["root_lin_vel"].clone()
        root_velocity = self.qvel[:, :6]
        if self.cfg.command_walking:
            quat = without_heading(quat)
            root_velocity = root_velocity.clone()
            root_velocity[:, :3] = heading_velocity(self.qpos[:, 3:7], root_velocity[:, :3])
            root_velocity[:, 3:6] = heading_angular_velocity(self.qpos[:, 3:7], root_velocity[:, 3:6])
        elif not self.cfg.use_reference_velocity:
            root_lin_vel_target[:, 0] = (
                float(self.cfg.target_velocity) * self._stand_start_progress()
            )
        root_vel_error = root_velocity - torch.cat(
            (root_lin_vel_target, targets["root_ang_vel"]),
            dim=1,
        )
        parts = [
                self.qpos[:, 2:3],
                quat,
                root_velocity,
                self.qpos[:, self.nonroot_qpos_adrs],
                self.qvel[:, self.nonroot_qvel_adrs],
                self.last_action,
                torch.stack((up_z, torch.sin(phase), torch.cos(phase)), dim=1),
                self.qpos[:, 2:3] - targets["root_height"][:, None],
                targets["root_quat"],
                root_vel_error,
                joint_pos - targets["joint_pos"],
                joint_vel - targets["joint_vel"],
                ref_ctrl_action,
                foot_pos - ref_foot_pos,
                targets["foot_contact"],
        ]
        if self.cfg.observe_tendon_velocity:
            parts.append(self.actuator_velocity[:, self.position_actuator_ids])
        if self.cfg.command_walking:
            parts.append(self.walking_commands.values)
        if self.cfg.skill_conditioning and self.motion_count > 1:
            parts.append(
                torch.nn.functional.one_hot(
                    self.motion_ids, num_classes=self.motion_count
                ).to(torch.float32)
            )
        obs = torch.cat(parts, dim=1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=20.0, neginf=-20.0)
        if self.domain_randomization and self.cfg.obs_noise_std > 0.0:
            noise_std = self.domain_randomization_scale * float(self.cfg.obs_noise_std)
            obs = obs + noise_std * torch.randn_like(obs)
        return torch.clamp(obs, -20.0, 20.0)

    def amp_observe(self) -> torch.Tensor:
        foot_pos = self._foot_positions_local().reshape(self.num_envs, -1)
        root_height = self.qpos[:, 2]
        if self.more_terrain_cfg.serial_course:
            terrain_offset = torch.full_like(
                root_height, float(self.more_terrain_cfg.course_base_height)
            )
            no_offset = torch.zeros_like(terrain_offset)
            pit_gap = torch.zeros_like(self.motion_ids, dtype=torch.bool)
            for motion_id, kind in enumerate(self.motion_terrain_kinds):
                if kind in {"gap", "pit"}:
                    pit_gap = pit_gap | (self.motion_ids == motion_id)
            root_height = root_height - torch.where(
                pit_gap, no_offset, terrain_offset
            )
        stair_anchor = self.motion_stair_height_anchor[self.motion_ids]
        stair_scale = self.motion_stair_height_scale[self.motion_ids]
        root_height = stair_anchor + (root_height - stair_anchor) / stair_scale
        root_quat = _canonical_quat(self.qpos[:, 3:7])
        joint_vel = self.qvel[:, self.robot_qvel_adrs]
        if self.cfg.command_walking:
            root_quat = without_heading(root_quat)
            joint_vel = joint_vel / (self.walking_commands.values[:, :1] / self.cfg.target_velocity)
        raw_features = torch.cat(
            (
                root_height[:, None],
                root_quat,
                self.qpos[:, self.robot_qpos_adrs],
                joint_vel,
                foot_pos,
            ),
            dim=1,
        )
        features = (raw_features - self.feature_mean) / self.feature_std
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        return torch.clamp(features, -10.0, 10.0)

    def step(self, action: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if torch.is_tensor(action):
            actions = action.to(device=self.device, dtype=torch.float32)
        else:
            actions = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape {(self.num_envs, self.action_dim)}, got {tuple(actions.shape)}")

        prev_action = self.last_action.clone()
        commanded_action = self._apply_action_noise(torch.clamp(actions, -1.0, 1.0))
        commanded_action = commanded_action * self._startup_action_ramp()
        phase_scale = self._stand_start_progress()
        if self.cfg.command_walking:
            phase_scale = phase_scale * self.walking_commands.values[:, 0] / self.cfg.target_velocity
        frame_counts = self.motion_frame_counts[self.motion_ids].to(torch.float32)
        next_phase = (
            self.motion_phase
            + self.motion_frequencies[self.motion_ids] * self.dt * phase_scale
        )
        course_complete = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        course_suffix_complete = torch.zeros_like(course_complete)
        course_stage_advance = torch.zeros_like(course_complete)
        incomplete_motion_end_candidate = torch.zeros_like(course_complete)
        incomplete_motion_required_x = torch.full_like(
            self.qpos[:, 0], float("inf")
        )
        stair_motion = self.motion_stair_mask[self.motion_ids]
        stair_distance_error = torch.zeros_like(self.qpos[:, 0])
        stair_distance_success = torch.zeros_like(course_complete)
        stair_distance_failure = torch.zeros_like(course_complete)
        stair_lateral_error = torch.zeros_like(self.qpos[:, 0])
        stair_lateral_failure = torch.zeros_like(course_complete)
        if self.more_terrain_cfg.serial_course:
            motion_complete = next_phase >= frame_counts - 1.0
            active_looping = self.motion_loop[self.motion_ids].clone()
            last_stage = self.course_motion_sequences.shape[1] - 1
            at_last_stage = self.course_stage_ids >= last_stage
            reached_finish = self.qpos[:, 0] >= (
                self.course_finish_x
                - self.more_terrain_cfg.course_completion_tolerance
            )
            course_suffix_complete = (
                motion_complete
                & at_last_stage
                & reached_finish
            )
            course_complete = course_suffix_complete & (
                self.course_start_stage_ids == 0
            )
            stage_end_x = self.course_stage_end_x[
                self.course_variant_ids, self.course_stage_ids
            ]
            reached_stage_end = self.qpos[:, 0] >= (
                stage_end_x - float(self.cfg.terrain_course_stage_tolerance)
            )
            # Keep the current action active until the robot has physically
            # reached its expected course boundary.
            advance = motion_complete & reached_stage_end & ~at_last_stage
            course_stage_advance = advance
            transition_steps = max(0, int(self.cfg.motion_transition_blend_steps))
            if transition_steps > 0:
                self.motion_transition_from_ctrl.copy_(
                    torch.where(
                        advance[:, None],
                        self.last_reference_ctrl,
                        self.motion_transition_from_ctrl,
                    )
                )
                self.motion_transition_steps_left.copy_(
                    torch.where(
                        advance,
                        torch.full_like(
                            self.motion_transition_steps_left, transition_steps
                        ),
                        self.motion_transition_steps_left,
                    )
                )
            incomplete_motion_required_x = torch.where(
                at_last_stage,
                torch.full_like(
                    stage_end_x,
                    self.course_finish_x
                    - self.more_terrain_cfg.course_completion_tolerance,
                ),
                stage_end_x - float(self.cfg.terrain_course_stage_tolerance),
            )
            incomplete_motion_end_candidate = (
                bool(self.cfg.terminate_incomplete_motion_end)
                & motion_complete
                & ~active_looping
                & ~advance
                & ~course_suffix_complete
            )
            self.course_stage_ids[:] = self.course_stage_ids + advance.to(torch.long)
            next_motion_ids = self.course_motion_sequences[
                self.course_variant_ids, self.course_stage_ids
            ]
            self.motion_ids[:] = torch.where(advance, next_motion_ids, self.motion_ids)
            # Keep replaying an unfinished terrain clip.  Otherwise a
            # non-looping reference is clamped to its final (standing) frame
            # while the robot is still before the physical stage boundary.
            # Preserve terminate_incomplete_motion_end as the opt-in reset
            # path used for diagnostics/curriculum enforcement.
            retry_loop = (
                motion_complete
                & ~advance
                & ~course_suffix_complete
                & ~incomplete_motion_end_candidate
            )
            self.motion_phase[:] = torch.where(
                advance,
                torch.zeros_like(next_phase),
                torch.where(
                    retry_loop,
                    torch.remainder(next_phase, frame_counts),
                    torch.minimum(next_phase, frame_counts - 1.0),
                ),
            )
        else:
            looping = self.motion_loop[self.motion_ids]
            self.motion_phase = torch.where(
                looping,
                torch.remainder(next_phase, frame_counts),
                torch.minimum(next_phase, frame_counts - 1.0),
            )
            motion_complete = (~looping) & (next_phase >= frame_counts - 1.0)
        targets = self._targets_at_current_phase()
        transitioned_reference_ctrl = self._transition_reference_ctrl(
            targets["tendon_ctrl"]
        )
        base_ctrl = self._reference_base_ctrl(transitioned_reference_ctrl)
        self._update_push_forces()
        push_delay = int(self.cfg.stand_start_push_delay_steps)
        if push_delay > 0:
            starting = self.stand_start & (self.episode_step < push_delay)
            self.push_force[:] = torch.where(starting[:, None], torch.zeros_like(self.push_force), self.push_force)
            self.xfrc_applied[:, self.pelvis_body_id] = torch.where(
                starting[:, None],
                torch.zeros_like(self.push_force),
                self.xfrc_applied[:, self.pelvis_body_id],
            )
            self.push_time_left[:] = torch.where(starting, torch.zeros_like(self.push_time_left), self.push_time_left)
            self.push_time_to_next[:] = torch.where(
                starting,
                torch.clamp_min(self.push_time_to_next, max(1, int(self.cfg.push_interval_min_steps))),
                self.push_time_to_next,
            )

        with wp.ScopedDevice(self.wp_device):
            if not self.domain_randomization or self.max_action_delay_ticks <= 0:
                self.ctrl[:] = self.action_to_ctrl(commanded_action, base_ctrl=base_ctrl)
                self._warp_decimated_step()
            else:
                commanded_ctrl = self.action_to_ctrl(commanded_action, base_ctrl=base_ctrl)
                for _ in range(self.cfg.decimation):
                    self._step_delayed_ctrl(commanded_ctrl)
            # mj_step leaves derived body/geom transforms at the state before
            # its final integration. Rewards and observations below read those
            # transforms, so refresh them at the new qpos/qvel first.
            self._warp_forward()

        self.episode_step += 1
        target_forward_velocity = (
            targets["root_lin_vel"][:, 0]
            if self.cfg.use_reference_velocity or self.cfg.command_walking
            else float(self.cfg.target_velocity) * phase_scale
        )
        sensor_reward_fast_path = sensor_stability_reward_enabled(self.cfg)
        if sensor_reward_fast_path:
            # SensorHistoryEnv replaces the reference-tracking reward.  Skip
            # pose, foot, contact and symmetry terms that it would discard.
            rewards, dones, time_outs, components = self._compute_sensor_reward_done(
                commanded_action,
                prev_action,
                target_forward_velocity,
                targets,
            )
            foot_pos = None
            foot_world_pos = None
        else:
            foot_pos = self._foot_positions_local()
            foot_vel = (foot_pos - self.last_foot_pos) / max(self.dt, 1.0e-6)
            foot_world_pos = self._foot_positions_world()
            foot_world_vel = (foot_world_pos - self.last_foot_world_pos) / max(self.dt, 1.0e-6)
            foot_geom_contact = self._foot_geom_contacts()
            foot_contact = self._side_contacts_from_geom(foot_geom_contact)
            rewards, dones, time_outs, components = self._compute_reward_done(
                commanded_action,
                prev_action,
                target_forward_velocity,
                targets,
                foot_pos,
                foot_vel,
                foot_world_vel,
                foot_geom_contact,
                foot_contact,
            )
        if self.more_terrain_cfg.serial_course:
            rewards = rewards + (
                float(self.cfg.terrain_course_completion_reward)
                * course_suffix_complete.to(torch.float32)
                + float(self.cfg.terrain_course_stage_reward)
                * course_stage_advance.to(torch.float32)
            )
            components["course_completion"] = course_suffix_complete.to(torch.float32)
            components["course_stage_completion"] = course_stage_advance.to(torch.float32)
        incomplete_motion_end = incomplete_motion_end_candidate & (
            self.qpos[:, 0] < incomplete_motion_required_x
        )
        (
            progress_speed,
            insufficient_progress,
            next_anchor_x,
            next_anchor_step,
        ) = _forward_progress_guard(
            self.qpos[:, 0],
            self.episode_step,
            self.forward_progress_anchor_x,
            self.forward_progress_anchor_step,
            dt=self.dt,
            min_speed=float(self.cfg.min_forward_progress_speed),
            grace_steps=int(self.cfg.forward_progress_grace_steps),
            window_steps=int(self.cfg.forward_progress_window_steps),
        )
        self.forward_progress_anchor_x[:] = next_anchor_x
        self.forward_progress_anchor_step[:] = next_anchor_step
        forward_progress_failure = incomplete_motion_end | insufficient_progress
        rewards = rewards - (
            float(self.cfg.forward_progress_termination_penalty)
            * forward_progress_failure.to(torch.float32)
        )
        components["forward_progress_window_speed"] = progress_speed
        components["insufficient_progress"] = insufficient_progress.to(torch.float32)
        components["incomplete_motion_end"] = incomplete_motion_end.to(torch.float32)
        # Non-looping stair references define a fixed root-X displacement.  A
        # terminal frame that stops short of that displacement is a failed
        # stair traversal, even if the robot remains upright.
        if self.more_terrain_cfg.serial_course:
            stair_active = torch.zeros_like(stair_motion)
            stair_terminal = torch.zeros_like(motion_complete)
            self.stair_terminal_hold_elapsed.zero_()
        else:
            stair_active = stair_motion
            hold_steps = (
                max(int(self.cfg.stair_terminal_hold_steps), 0)
                if self.cfg.stair_fixed_start
                else 0
            )
            completed_stair = stair_active & motion_complete
            if hold_steps > 0:
                self.stair_terminal_hold_elapsed[:] = torch.where(
                    completed_stair,
                    torch.clamp(
                        self.stair_terminal_hold_elapsed + 1,
                        max=hold_steps,
                    ),
                    torch.zeros_like(self.stair_terminal_hold_elapsed),
                )
                stair_terminal = completed_stair & (
                    self.stair_terminal_hold_elapsed >= hold_steps
                )
            else:
                self.stair_terminal_hold_elapsed.zero_()
                stair_terminal = completed_stair
        expected_stair_distance = self.stair_remaining_distance
        stair_distance_error = torch.where(
            stair_terminal,
            torch.linalg.vector_norm(torch.stack((
                (self.qpos[:, 0] - self.stair_start_x) - expected_stair_distance,
                (self.qpos[:, 1] - self.stair_start_y) - self.stair_remaining_lateral,
            ), dim=1), dim=1),
            torch.zeros_like(self.qpos[:, 0]),
        )
        stair_distance_success = stair_terminal & (
            stair_distance_error <= max(float(self.cfg.stair_distance_tolerance), 0.0)
        )
        stair_distance_failure = stair_terminal & ~stair_distance_success
        rewards = rewards - (
            max(float(self.cfg.stair_distance_failure_penalty), 0.0)
            * stair_distance_failure.to(torch.float32)
        )
        stair_lateral_error = torch.where(
            stair_active,
            torch.abs(self.qpos[:, 1] - self.stair_start_y),
            torch.zeros_like(self.qpos[:, 0]),
        )
        stair_lateral_failure = stair_active & (
            stair_lateral_error > max(float(self.cfg.stair_lateral_tolerance), 0.0)
        )
        rewards = rewards - (
            max(float(self.cfg.stair_lateral_failure_penalty), 0.0)
            * stair_lateral_failure.to(torch.float32)
        )
        successful_motion_end = (
            course_suffix_complete
            if self.more_terrain_cfg.serial_course
            else (
                (motion_complete & ~stair_active)
                | (stair_terminal & stair_distance_success
                   & ~stair_lateral_failure & ~dones)
            )
        )
        full_route_success = (
            stair_active & stair_terminal & stair_distance_success
            & ~stair_lateral_failure & self.stair_full_start & ~dones
        )
        dones = (
            dones
            | successful_motion_end
            | stair_distance_failure
            | stair_lateral_failure
            | forward_progress_failure
        )
        time_outs = time_outs | successful_motion_end
        episode_steps = self.episode_step.clone()
        reported_stair_terminal_hold_elapsed = (
            self.stair_terminal_hold_elapsed.clone()
        )
        self.last_action[:] = commanded_action
        if foot_pos is not None and foot_world_pos is not None:
            self.last_foot_pos[:] = foot_pos
            self.last_foot_world_pos[:] = foot_world_pos

        if self.cfg.termination_mode == "fall":
            dones = components["fallen"] > 0.5
            time_outs = torch.zeros_like(dones)
            components["termination"] = dones.to(torch.float32)

        reported_course_stage = self.course_stage_ids.clone()
        reported_course_start_stage = self.course_start_stage_ids.clone()
        # Only time-limit transitions need a pre-reset value bootstrap.  Falls
        # are true terminals, so constructing their terminal sensor history
        # and evaluating the critic is wasted work.  Reuse the single host
        # transfer needed by the current CPU reset path for both masks.
        terminal_masks_np = torch.stack((dones, time_outs)).detach().cpu().numpy()
        done_ids_np = np.flatnonzero(terminal_masks_np[0])
        bootstrap_ids_np = np.flatnonzero(terminal_masks_np[1])
        terminal_payload = None
        terminal_observer = getattr(self, "terminal_observer", None)
        if bootstrap_ids_np.size > 0 and callable(terminal_observer):
            # The terminal history has to be read before reset; otherwise a
            # time-limit bootstrap incorrectly evaluates the next episode.
            terminal_ids = torch.as_tensor(
                bootstrap_ids_np, dtype=torch.long, device=self.device
            )
            terminal_payload = terminal_observer(terminal_ids)
        if self.walking_commands is not None:
            self.walking_commands.advance(self.dt, self.rng)
        if done_ids_np.size > 0:
            self.reset(done_ids_np)
        else:
            if self.actor_observation_enabled:
                self.last_obs = self.observe()
        info = {key: value for key, value in components.items()}
        info["episode_step"] = episode_steps
        info["time_outs"] = time_outs.to(torch.float32)
        if terminal_payload is not None:
            info.update(terminal_payload)
            info["terminal_env_ids"] = terminal_ids
        info["stair_distance_error"] = stair_distance_error
        info["stair_distance_success"] = stair_distance_success.to(torch.float32)
        info["stair_distance_failure"] = stair_distance_failure.to(torch.float32)
        info["stair_lateral_error"] = stair_lateral_error
        info["stair_lateral_failure"] = stair_lateral_failure.to(torch.float32)
        info["stair_terminal_hold_elapsed"] = reported_stair_terminal_hold_elapsed.to(
            torch.float32
        )
        if self.more_terrain_cfg.serial_course:
            info["course_complete"] = course_complete.to(torch.float32)
            info["course_suffix_complete"] = course_suffix_complete.to(torch.float32)
            info["course_stage"] = reported_course_stage.to(torch.float32)
            info["course_start_stage"] = reported_course_start_stage.to(torch.float32)
            info["successes"] = course_complete.to(torch.float32)
        elif self.cfg.stair_fixed_start:
            info["successes"] = torch.where(
                stair_active, full_route_success, time_outs
            ).to(torch.float32)
        return self.last_obs, rewards, dones.to(torch.float32), info

    def _compute_sensor_reward_done(
        self,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        target_forward_velocity: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute only terms consumed by the sensor-stability objective."""

        root_height = self.qpos[:, 2]
        up_z = self.xmat[:, self.pelvis_body_id, 2, 2]
        # These stability terms must not depend on the unobserved reference
        # phase.  The pelvis up axis is available from IMU history, while root
        # clearance is used only by the training reward/termination path.
        upright_span = max(1.0 - float(self.cfg.min_upright_z), 1.0e-6)
        sensor_upright = torch.clamp(
            (up_z - float(self.cfg.min_upright_z)) / upright_span,
            0.0,
            1.0,
        )
        terrain_height = (
            float(self.more_terrain_cfg.course_base_height)
            if self.more_terrain_cfg.serial_course
            else 0.0
        )
        root_clearance = root_height - terrain_height
        clearance_span = max(
            self.nominal_root_height - float(self.cfg.fall_height),
            1.0e-6,
        )
        sensor_height = torch.clamp(
            (root_clearance - float(self.cfg.fall_height)) / clearance_span,
            0.0,
            1.0,
        )
        pelvis_tilt = torch.acos(torch.clamp(up_z, -1.0, 1.0))
        sensor_posture_track = torch.exp(
            -(pelvis_tilt / max(float(getattr(self.cfg, "sensor_upright_sigma", 0.12)), 1.0e-6)).square()
        )
        sensor_gait_height_track = torch.exp(
            -((root_clearance - float(getattr(self.cfg, "sensor_gait_height_target", 0.845)))
              / max(float(getattr(self.cfg, "sensor_gait_height_sigma", 0.07)), 1.0e-6)).square()
        )
        # Preserve the existing metric keys for log/CSV compatibility; in the
        # sensor fast path they now carry phase-independent stability scores.
        root_orientation = sensor_upright
        height = sensor_height

        target_velocity_scale = max(abs(float(self.cfg.target_velocity)), 1.0e-6)
        gait_progress = torch.clamp(
            torch.abs(target_forward_velocity) / target_velocity_scale,
            0.0,
            1.0,
        )
        velocity_lower, velocity_upper = forward_velocity_bounds(
            self.cfg, target_forward_velocity, gait_progress,
        )
        forward_velocity = self.qvel[:, 0]
        forward_velocity_error = forward_velocity - torch.clamp(
            forward_velocity, min=velocity_lower, max=velocity_upper,
        )
        forward_velocity_track = torch.exp(
            -forward_velocity_error.square()
            / (max(float(self.cfg.forward_velocity_tracking_sigma), 1.0e-6) ** 2)
        )
        progress_velocity_scale = (
            self.cfg.velocity_range[0] if self.cfg.velocity_range is not None else target_velocity_scale
        )
        forward_progress = torch.clamp(forward_velocity / progress_velocity_scale, -1.0, 1.0)
        forward_velocity_deficit = torch.clamp(
            (velocity_lower - forward_velocity) / target_velocity_scale,
            0.0,
            2.0,
        )
        forward_velocity_excess = torch.clamp(
            (forward_velocity - velocity_upper) / target_velocity_scale,
            0.0,
            2.0,
        )
        knee_target = max(float(self.cfg.sensor_knee_flexion_target), 1.0e-6)
        if len(self.knee_joint_indices) > 0:
            knee_flexion = torch.clamp(
                -self.qpos[:, self.robot_qpos_adrs][:, self.knee_joint_indices],
                min=0.0,
            )
            knee_flexion_track = torch.clamp(
                torch.amin(knee_flexion, dim=1) / knee_target,
                0.0,
                1.0,
            )
        else:
            knee_flexion_track = torch.ones_like(root_height)
        gait_scores = {}
        if any(
            float(getattr(self.cfg, name, 0.0)) > 0.0
            for name in (
                "sensor_mirror_reward_coef",
                "sensor_natural_gait_reward_coef",
                "sensor_stride_reward_coef",
                "sensor_foot_roll_reward_coef",
                "sensor_alternation_reward_coef",
                "sensor_hop_penalty_coef",
                "sensor_tiptoe_penalty_coef",
                "sensor_stance_slip_penalty_coef",
                "sensor_sync_forward_penalty_coef",
            )
        ):
            zones = self._sensor_foot_roll_contacts()
            all_foot_world = self.geom_xpos[:, self.contact_geom_ids]
            foot_world = all_foot_world[:, [0, 2]]
            foot_x = foot_world[:, :, 0]
            foot_height = torch.stack(
                (torch.amin(all_foot_world[:, 0:2, 2], dim=1),
                 torch.amin(all_foot_world[:, 2:4, 2], dim=1)),
                dim=1,
            )
            stance_speed = torch.linalg.vector_norm(
                (foot_world[:, :, :2] - self.sensor_prev_foot_world[:, :, :2])
                / max(self.dt, 1.0e-6),
                dim=2,
            )
            foot_forward_speed = (
                foot_world[:, :, 0] - self.sensor_prev_foot_world[:, :, 0]
            ) / max(self.dt, 1.0e-6)
            self.sensor_prev_foot_world.copy_(foot_world)
            foot_local = self._foot_positions_local()
            gait_scores = sensor_gait_scores(
                self, zones, foot_x, foot_local[:, 0, 1] + foot_local[:, 2, 1],
                stance_speed, foot_forward_speed, foot_height
            )
        (
            locomotion_stability,
            stable_forward_velocity_track,
            stable_forward_progress,
        ) = _stability_gated_locomotion_terms(
            forward_velocity_track,
            forward_progress,
            sensor_upright,
            sensor_height,
        )

        position_action = torch.clamp(action, -1.0, 1.0)
        prev_position_action = torch.clamp(prev_action, -1.0, 1.0)
        # Actuator forces projected onto the tracked hinge joints, in N m.
        joint_torque = self.qfrc_actuator[:, self.robot_qvel_adrs]
        torque_penalty = torch.mean(joint_torque.square(), dim=1)
        action_rate = torch.mean((position_action - prev_position_action).square(), dim=1)
        if self.has_distal_action:
            distal_action_rate = torch.mean(
                (
                    position_action[:, self.distal_position_action_mask]
                    - prev_position_action[:, self.distal_position_action_mask]
                ).square(),
                dim=1,
            )
        else:
            distal_action_rate = action_rate

        bad_state = ~(
            torch.isfinite(self.qpos).all(dim=1)
            & torch.isfinite(self.qvel).all(dim=1)
            & torch.isfinite(self.ctrl).all(dim=1)
        )
        fallen = (
            (root_clearance < float(self.cfg.fall_height))
            | (root_clearance > float(self.cfg.max_height))
            | (up_z < self.cfg.min_upright_z)
        )

        step_metrics = self.step_lengths.update(
            self._foot_contacts(),
            self._foot_positions_world()[:, :, 0].reshape(self.num_envs, 2, 2).mean(dim=2),
            active=self._stand_start_progress() >= 1.,
        )
        symmetry_track = step_symmetry_reward(
            step_metrics, gait_progress, forward_velocity_track, locomotion_stability,
        )
        # Match sensor_stability_task_reward before its fall/stall overrides.
        stability = torch.clamp(
            torch.minimum(sensor_posture_track, sensor_gait_height_track),
            0.0,
            1.0,
        )
        quality_terms = torch.stack(
            tuple(torch.clamp(gait_scores.get(name, torch.full_like(root_height, 0.5)), 0.05, 1.0)
                  for name in (
                      "sensor_stride_length", "sensor_step_cadence",
                      "sensor_swing_clearance", "sensor_foot_roll",
                      "sensor_alternation", "sensor_mirror",
                  )),
            dim=1,
        )
        gait_quality = 6.0 / torch.sum(1.0 / quality_terms, dim=1)
        speed_track = torch.clamp(forward_velocity_track, 0.0, 1.0)
        reward = 0.50 * stability + stability * (
            1.50 * speed_track
            + 0.50 * torch.clamp(forward_progress, 0.0, 1.0)
            - 2.50 * torch.clamp(forward_velocity_deficit, 0.0, 2.0)
            - float(getattr(self.cfg, "sensor_velocity_excess_penalty_coef", 0.0))
            * torch.clamp(forward_velocity_excess, 0.0, 2.0)
            + float(getattr(self.cfg, "sensor_natural_gait_reward_coef", 0.0))
            * gait_progress
            * speed_track
            * gait_quality
            + float(self.cfg.sensor_knee_gait_reward_coef)
            * gait_progress
            * (knee_flexion_track - 1.0)
            + gait_progress * (
                - float(getattr(self.cfg, "sensor_hop_penalty_coef", 0.0))
                * gait_scores.get("sensor_hop", 0.0)
                - float(getattr(self.cfg, "sensor_tiptoe_penalty_coef", 0.0))
                * (gait_scores.get("sensor_tiptoe", 0.0)
                   + gait_scores.get("sensor_roll_violation", 0.0))
                - float(getattr(self.cfg, "sensor_stance_slip_penalty_coef", 0.0))
                * gait_scores.get("sensor_stance_slip", 0.0)
                - float(getattr(self.cfg, "sensor_sync_forward_penalty_coef", 0.0))
                * gait_scores.get("sensor_synchronous_forward", 0.0)
            )
        ) - self.cfg.torque_penalty_coef * torque_penalty - 0.10 * action_rate - 0.12 * distal_action_rate
        reward = reward + self.cfg.symmetry_reward_coef * symmetry_track
        components = {
            "root_height_value": root_height,
            "reference_root_height_value": targets["root_height"],
            "sensor_root_clearance": root_clearance,
            "forward_velocity": forward_velocity,
            "forward_velocity_error": forward_velocity_error,
            "forward_velocity_track": forward_velocity_track,
            "forward_progress": forward_progress,
            "locomotion_stability": locomotion_stability,
            "stable_forward_velocity_track": stable_forward_velocity_track,
            "stable_forward_progress": stable_forward_progress,
            "forward_velocity_deficit": forward_velocity_deficit,
            "forward_velocity_excess": forward_velocity_excess,
            "root_orientation": root_orientation,
            "height": height,
            "sensor_upright": sensor_upright,
            "sensor_height": sensor_height,
            "sensor_posture_track": sensor_posture_track,
            "sensor_gait_height_track": sensor_gait_height_track,
            "sensor_gait_quality": gait_quality,
            "symmetry": symmetry_track,
            **step_metrics,
            "gait_progress": gait_progress,
            "knee_flexion_track": knee_flexion_track,
            "knee_extension": 1.0 - knee_flexion_track,
            **gait_scores,
            "torque_penalty": torque_penalty,
            "action_rate": action_rate,
            "distal_action_rate": distal_action_rate,
        }
        if bool(getattr(self.cfg, "deepmimic_reward", False)):
            components.update({
                f"deepmimic_{name}": value
                for name, value in _dm_terms.items()
            })
        reward, bad_state, components = _sanitize_reward_outputs(
            reward,
            bad_state,
            fallen,
            components,
            fall_penalty=self.cfg.fall_penalty,
        )
        components["fallen"] = fallen.to(torch.float32)
        timed_out = (
            self.episode_step >= self.episode_timeout_step
            if self.cfg.episode_steps > 0 else torch.zeros_like(fallen)
        )
        dones = bad_state | fallen | timed_out
        components["termination"] = (bad_state | fallen).to(torch.float32)
        return reward, dones, timed_out, components

    def _compute_reward_done(
        self,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        target_forward_velocity: torch.Tensor,
        targets: dict[str, torch.Tensor],
        foot_pos: torch.Tensor,
        foot_vel: torch.Tensor,
        foot_world_vel: torch.Tensor,
        foot_geom_contact: torch.Tensor,
        foot_contact: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        root_height = self.qpos[:, 2]
        up_z = self.xmat[:, self.pelvis_body_id, 2, 2]
        root_quat = _canonical_quat(self.qpos[:, 3:7])
        ref_root_quat = _canonical_quat(targets["root_quat"])
        if self.cfg.command_walking:
            root_quat = without_heading(root_quat)

        quat_dot = torch.abs(torch.sum(root_quat * ref_root_quat, dim=1)).clamp(0.0, 1.0)
        orient_error = 2.0 * torch.acos(quat_dot)
        root_orientation = torch.exp(-(orient_error.square()) / (0.35**2))

        target_velocity_scale = max(abs(float(self.cfg.target_velocity)), 1.0e-6)
        gait_progress = torch.clamp(
            torch.abs(target_forward_velocity) / target_velocity_scale, 0.0, 1.0,
        )
        if self.cfg.command_walking:
            gait_progress = self._stand_start_progress()
        velocity_lower, velocity_upper = forward_velocity_bounds(
            self.cfg, target_forward_velocity, gait_progress,
        )
        root_velocity = self.qvel[:, :3]
        if self.cfg.command_walking:
            root_velocity = heading_velocity(self.qpos[:, 3:7], root_velocity)
        forward_velocity = root_velocity[:, 0]
        tracked_forward_velocity = torch.clamp(forward_velocity, min=velocity_lower, max=velocity_upper)
        root_lin_vel_target = targets["root_lin_vel"].clone()
        root_lin_vel_target[:, 0] = tracked_forward_velocity
        lin_vel_error = root_velocity - root_lin_vel_target
        angular_velocity = self.qvel[:, 3:6]
        if self.cfg.command_walking:
            angular_velocity = heading_angular_velocity(self.qpos[:, 3:7], angular_velocity)
        ang_vel_error = angular_velocity - targets["root_ang_vel"]
        lin_vel_track = torch.exp(-torch.mean(lin_vel_error.square(), dim=1) / (0.75**2))
        ang_vel_track = torch.exp(-torch.mean(ang_vel_error.square(), dim=1) / (1.50**2))
        root_vel = 0.65 * lin_vel_track + 0.35 * ang_vel_track
        forward_velocity_error = forward_velocity - tracked_forward_velocity
        forward_velocity_track = torch.exp(
            -forward_velocity_error.square()
            / (max(float(self.cfg.forward_velocity_tracking_sigma), 1.0e-6) ** 2)
        )
        yaw_rate_track = torch.zeros_like(forward_velocity_track)
        if self.cfg.command_walking:
            forward_velocity_track = forward_velocity_track * torch.exp(
                -(root_velocity[:, 1] / self.cfg.forward_velocity_tracking_sigma).square())
            yaw_error = world_yaw_rate(self.qpos[:, 3:7], self.qvel[:, 3:6]) - targets["root_ang_vel"][:, 2]
            yaw_rate_track = torch.exp(-(yaw_error / 0.35).square())
        progress_velocity_scale = (
            self.cfg.velocity_range[0] if self.cfg.velocity_range is not None else target_velocity_scale
        )
        forward_progress = torch.clamp(forward_velocity / progress_velocity_scale, -1.0, 1.0)
        forward_velocity_deficit = torch.clamp(
            (velocity_lower - forward_velocity) / target_velocity_scale,
            0.0,
            2.0,
        )
        forward_velocity_excess = torch.clamp(
            (forward_velocity - velocity_upper) / target_velocity_scale,
            0.0,
            2.0,
        )

        height = torch.exp(-((root_height - targets["root_height"]).square()) / (0.09**2))
        upright_span = max(1.0 - float(self.cfg.min_upright_z), 1.0e-6)
        sensor_upright = torch.clamp(
            (up_z - float(self.cfg.min_upright_z)) / upright_span,
            0.0,
            1.0,
        )
        terrain_height = (
            float(self.more_terrain_cfg.course_base_height)
            if self.more_terrain_cfg.serial_course
            else 0.0
        )
        root_clearance = root_height - terrain_height
        clearance_span = max(
            self.nominal_root_height - float(self.cfg.fall_height),
            1.0e-6,
        )
        sensor_height = torch.clamp(
            (root_clearance - float(self.cfg.fall_height)) / clearance_span,
            0.0,
            1.0,
        )
        (
            locomotion_stability,
            stable_forward_velocity_track,
            stable_forward_progress,
        ) = _stability_gated_locomotion_terms(
            forward_velocity_track,
            forward_progress,
            root_orientation,
            height,
        )
        joint_pos = self.qpos[:, self.robot_qpos_adrs]
        joint_vel = self.qvel[:, self.robot_qvel_adrs]
        knee_target = max(float(self.cfg.sensor_knee_flexion_target), 1.0e-6)
        if len(self.knee_joint_indices) > 0:
            knee_flexion_track = torch.clamp(
                torch.amin(
                    torch.clamp(-joint_pos[:, self.knee_joint_indices], min=0.0),
                    dim=1,
                )
                / knee_target,
                0.0,
                1.0,
            )
        else:
            knee_flexion_track = torch.ones_like(root_height)
        gait_scores = {}
        if any(
            float(getattr(self.cfg, name, 0.0)) > 0.0
            for name in (
                "sensor_natural_gait_reward_coef", "sensor_knee_gait_reward_coef",
                "sensor_hop_penalty_coef", "sensor_tiptoe_penalty_coef",
                "sensor_stance_slip_penalty_coef", "sensor_sync_forward_penalty_coef",
            )
        ):
            zones = self._sensor_foot_roll_contacts()
            all_foot_world = self.geom_xpos[:, self.contact_geom_ids]
            foot_world = all_foot_world[:, [0, 2]]
            foot_height = torch.stack((
                torch.amin(all_foot_world[:, 0:2, 2], dim=1),
                torch.amin(all_foot_world[:, 2:4, 2], dim=1),
            ), dim=1)
            foot_velocity = (foot_world - self.sensor_prev_foot_world) / max(self.dt, 1.0e-6)
            self.sensor_prev_foot_world.copy_(foot_world)
            foot_local = self._foot_positions_local()
            gait_scores = sensor_gait_scores(
                self, zones, foot_world[:, :, 0],
                foot_local[:, 0, 1] + foot_local[:, 2, 1],
                torch.linalg.vector_norm(foot_velocity[:, :, :2], dim=2),
                foot_velocity[:, :, 0], foot_height,
            )
            quality_names = (
                "sensor_stride_length", "sensor_step_cadence", "sensor_swing_clearance",
                "sensor_foot_roll", "sensor_alternation", "sensor_mirror",
            )
            gait_scores["sensor_gait_quality"] = len(quality_names) / sum(
                1.0 / torch.clamp(gait_scores[name], 0.05, 1.0)
                for name in quality_names
            )
        weight_sum = torch.sum(self.joint_reward_weights).clamp_min(1.0e-6)
        pose_error = torch.sum((joint_pos - targets["joint_pos"]).square() * self.joint_reward_weights, dim=1) / weight_sum
        vel_error = torch.sum((joint_vel - targets["joint_vel"]).square() * self.joint_reward_weights, dim=1) / weight_sum
        pose_track = 0.70 * torch.exp(-pose_error / (0.28**2)) + 0.30 * torch.exp(-pose_error / (0.10**2))
        vel_track = torch.exp(-vel_error / (4.0**2))

        foot_pos_error = torch.mean((foot_pos - targets["foot_pos"]).square(), dim=(1, 2))
        foot_pos_track = torch.exp(-foot_pos_error / (0.075**2))
        side_metrics = self._side_foot_tracks(foot_pos, targets["foot_pos"], targets["foot_contact"])
        foot_vel_track = self._swing_foot_vel_track(
            foot_vel,
            targets["foot_vel"],
            targets["foot_contact"],
        )
        stance_foot_track, stance_foot_speed = self._stance_foot_track(
            foot_world_vel,
            foot_geom_contact,
            targets["foot_contact"],
        )
        contact_track = torch.clamp(
            1.0 - torch.mean(torch.abs(foot_contact - targets["foot_contact"]), dim=1),
            0.0,
            1.0,
        )
        missed_contact = torch.mean(targets["foot_contact"] * (1.0 - foot_contact), dim=1)
        position_action = torch.clamp(action.to(dtype=torch.float32, device=self.device), -1.0, 1.0)
        prev_position_action = torch.clamp(prev_action.to(dtype=torch.float32, device=self.device), -1.0, 1.0)
        # Tendon command offsets are needed to produce supporting forces;
        # penalize joint torque, not deviation from geometric tendon targets.
        joint_torque = self.qfrc_actuator[:, self.robot_qvel_adrs]
        torque_penalty = torch.mean(joint_torque.square(), dim=1)
        action_rate = torch.mean((position_action - prev_position_action).square(), dim=1)
        if len(self.distal_joint_indices) > 0:
            distal_vel_error = torch.mean(
                (
                    joint_vel[:, self.distal_joint_indices]
                    - targets["joint_vel"][:, self.distal_joint_indices]
                ).square(),
                dim=1,
            )
        else:
            distal_vel_error = vel_error
        distal_vel_track = torch.exp(
            -distal_vel_error / (max(self.cfg.distal_joint_vel_sigma, 1.0e-6) ** 2)
        )
        if self.has_distal_action:
            distal_action_rate = torch.mean(
                (
                    position_action[:, self.distal_position_action_mask]
                    - prev_position_action[:, self.distal_position_action_mask]
                ).square(),
                dim=1,
            )
        else:
            distal_action_rate = action_rate
        step_metrics = self.step_lengths.update(
            foot_contact, (foot_pos if self.cfg.command_walking else self._foot_positions_world())[:, :, 0].reshape(self.num_envs, 2, 2).mean(dim=2),
            active=self._stand_start_progress() >= 1.,
        )
        symmetry_track = step_symmetry_reward(
            step_metrics, gait_progress, forward_velocity_track, locomotion_stability,
        )
        if self.cfg.command_walking:
            yaw_limit = max(abs(value) for value in self.cfg.yaw_rate_range)
            symmetry_track = symmetry_track * torch.clamp(
                1.0 - targets["root_ang_vel"][:, 2].abs() / yaw_limit, min=0.0)

        reward = (
            0.05
            + self.cfg.reference_joint_position_reward_coef * pose_track
            + self.cfg.reference_joint_velocity_reward_coef * vel_track
            + 0.75 * height
            + 0.90 * root_orientation
            + self.cfg.reference_root_velocity_reward_coef * root_vel
            + self.cfg.forward_velocity_reward_coef * stable_forward_velocity_track
            + self.cfg.forward_progress_reward_coef * stable_forward_progress
            - self.cfg.forward_velocity_deficit_penalty_coef * forward_velocity_deficit
            - self.cfg.forward_velocity_excess_penalty_coef * forward_velocity_excess
            + self.cfg.reference_foot_position_reward_coef * foot_pos_track
            + self.cfg.side_foot_pos_coef * side_metrics["side_foot_pos"]
            + self.cfg.step_gap_coef * side_metrics["step_gap"]
            + self.cfg.swing_advance_coef * side_metrics["swing_advance"]
            + self.cfg.foot_vel_reward_coef * foot_vel_track
            + self.cfg.stance_foot_reward_coef * stance_foot_track
            + self.cfg.contact_reward_coef * contact_track
            + 0.45 * distal_vel_track
            - self.cfg.torque_penalty_coef * torque_penalty
            - self.cfg.action_rate_penalty_coef * action_rate
            - self.cfg.distal_action_rate_penalty_coef * distal_action_rate
            - self.cfg.missed_contact_penalty_coef * missed_contact
        )
        if self.cfg.deepmimic_reward:
            reward, _dm_terms = deepmimic_reward(
                root_height=root_height,
                root_quat=root_quat,
                root_lin_vel=self.qvel[:, :3],
                root_ang_vel=self.qvel[:, 3:6],
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                key_pos_root_local=foot_pos,
                target_root_height=targets["root_height"],
                target_root_quat=ref_root_quat,
                target_root_lin_vel=(
                    root_lin_vel_target if self.cfg.velocity_range is not None else targets["root_lin_vel"]
                ),
                target_root_ang_vel=targets["root_ang_vel"],
                target_joint_pos=targets["joint_pos"],
                target_joint_vel=targets["joint_vel"],
                target_key_pos_root_local=targets["foot_pos"],
                root_xy=(self.qpos[:, :2] if self.cfg.deepmimic_track_root_xy else None),
                target_root_xy=(targets["root_xy"] if self.cfg.deepmimic_track_root_xy else None),
                root_xy_penalty=self.cfg.deepmimic_root_xy_penalty,
                route_tracking=self.cfg.deepmimic_route_tracking,
                route_xy_sigma=self.cfg.deepmimic_route_xy_sigma,
                route_velocity_sigma=self.cfg.deepmimic_route_velocity_sigma,
            )
        reward = reward + self.cfg.symmetry_reward_coef * symmetry_track
        if self.cfg.command_walking:
            reward = reward + 1.5 * yaw_rate_track * locomotion_stability
        if gait_scores:
            reward = reward + natural_gait_bonus(
                gait_scores, gait_progress, forward_velocity_track,
                locomotion_stability, knee_flexion_track, self.cfg,
            )
        reference_gait_style = torch.exp(
            -torch.mean((joint_pos-targets["joint_pos"]).square(), dim=1) / .20**2
            -torch.mean((foot_pos-targets["foot_pos"]).square(), dim=(1, 2)) / .07**2
        )
        if self.cfg.reference_gait_style_reward_coef > 0:
            reward = reward + self.cfg.reference_gait_style_reward_coef * gait_progress * (
                reference_gait_style * forward_velocity_track * locomotion_stability - 1.0)

        bad_state = ~(
            torch.isfinite(self.qpos).all(dim=1)
            & torch.isfinite(self.qvel).all(dim=1)
            & torch.isfinite(self.ctrl).all(dim=1)
        )
        min_root_height = targets["root_height"] + (
            float(self.cfg.fall_height) - self.nominal_root_height
        )
        max_root_height = targets["root_height"] + (
            float(self.cfg.max_height) - self.nominal_root_height
        )
        fallen = (
            (root_height < min_root_height)
            | (root_height > max_root_height)
            | (up_z < self.cfg.min_upright_z)
        )
        components = {
            "root_height_value": root_height,
            "reference_root_height_value": targets["root_height"],
            **({"command_speed": target_forward_velocity,
                "command_yaw_rate": targets["root_ang_vel"][:, 2],
                "yaw_rate_track": yaw_rate_track} if self.cfg.command_walking else {}),
            "sensor_root_clearance": root_clearance,
            "root_vel": root_vel,
            "forward_velocity": forward_velocity,
            "forward_velocity_error": forward_velocity_error,
            "forward_velocity_track": forward_velocity_track,
            "forward_progress": forward_progress,
            "locomotion_stability": locomotion_stability,
            "stable_forward_velocity_track": stable_forward_velocity_track,
            "stable_forward_progress": stable_forward_progress,
            "forward_velocity_deficit": forward_velocity_deficit,
            "forward_velocity_excess": forward_velocity_excess,
            "root_orientation": root_orientation,
            "height": height,
            "sensor_upright": sensor_upright,
            "sensor_height": sensor_height,
            "gait_progress": gait_progress,
            "knee_flexion_track": knee_flexion_track,
            "knee_extension": 1.0 - knee_flexion_track,
            "reference_gait_style": reference_gait_style,
            **gait_scores,
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
            **step_metrics,
            "torque_penalty": torque_penalty,
            "action_rate": action_rate,
            "distal_action_rate": distal_action_rate,
        }
        if self.cfg.deepmimic_reward:
            components.update({
                f"deepmimic_{name}": value
                for name, value in _dm_terms.items()
            })
        reward, bad_state, components = _sanitize_reward_outputs(
            reward,
            bad_state,
            fallen,
            components,
            fall_penalty=self.cfg.fall_penalty,
        )
        components["fallen"] = fallen.to(torch.float32)
        timed_out = (
            self.episode_step >= self.episode_timeout_step
            if self.cfg.episode_steps > 0 else torch.zeros_like(fallen)
        )
        route_failure = torch.zeros_like(fallen)
        if self.cfg.deepmimic_route_tracking:
            route_failure = (
                components["deepmimic_root_xy_error"]
                > self.cfg.deepmimic_route_max_xy_error
            )
            components["deepmimic_route_failure"] = route_failure.to(torch.float32)
        timed_out = timed_out & ~route_failure
        dones = bad_state | fallen | route_failure | timed_out
        termination = bad_state | fallen | route_failure
        components["termination"] = termination.to(torch.float32)
        return reward, dones, timed_out, components

    def _side_foot_tracks(
        self,
        foot_pos: torch.Tensor,
        ref_foot_pos: torch.Tensor,
        ref_foot_contact: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        right_center = torch.mean(foot_pos[:, 0:2], dim=1)
        left_center = torch.mean(foot_pos[:, 2:4], dim=1)
        side_center = torch.stack((right_center, left_center), dim=1)
        ref_right_center = torch.mean(ref_foot_pos[:, 0:2], dim=1)
        ref_left_center = torch.mean(ref_foot_pos[:, 2:4], dim=1)
        ref_side_center = torch.stack((ref_right_center, ref_left_center), dim=1)

        sigma = max(float(self.cfg.side_foot_pos_sigma), 1.0e-6)
        side_error = torch.mean((side_center - ref_side_center).square(), dim=2)
        right_foot_pos = torch.exp(-side_error[:, 0] / (sigma**2))
        left_foot_pos = torch.exp(-side_error[:, 1] / (sigma**2))
        side_foot_pos = torch.minimum(right_foot_pos, left_foot_pos)

        actual_gap = side_center[:, 1, 0] - side_center[:, 0, 0]
        ref_gap = ref_side_center[:, 1, 0] - ref_side_center[:, 0, 0]
        step_gap_error = actual_gap - ref_gap
        step_gap = torch.exp(
            -step_gap_error.square() / (max(float(self.cfg.step_gap_sigma), 1.0e-6) ** 2)
        )

        swing_weight = 1.0 - ref_foot_contact
        swing_sigma = max(float(self.cfg.swing_advance_sigma), 1.0e-6)
        x_error = side_center[:, :, 0] - ref_side_center[:, :, 0]
        side_swing = torch.exp(-x_error.square() / (swing_sigma**2))
        swing_sum = torch.sum(swing_weight, dim=1)
        swing_advance = torch.sum(side_swing * swing_weight, dim=1) / swing_sum.clamp_min(1.0e-6)
        swing_advance = torch.where(swing_sum > 1.0e-6, swing_advance, torch.ones_like(swing_advance))

        return {
            "side_foot_pos": side_foot_pos,
            "right_foot_pos": right_foot_pos,
            "left_foot_pos": left_foot_pos,
            "step_gap": step_gap,
            "step_gap_error": step_gap_error,
            "swing_advance": swing_advance,
            "right_swing_advance": side_swing[:, 0],
            "left_swing_advance": side_swing[:, 1],
        }

    def _swing_foot_vel_track(
        self,
        foot_vel: torch.Tensor,
        ref_foot_vel: torch.Tensor,
        ref_foot_contact: torch.Tensor,
    ) -> torch.Tensor:
        swing_weight = torch.repeat_interleave(
            1.0 - torch.clamp(ref_foot_contact, 0.0, 1.0),
            repeats=2,
            dim=1,
        )
        weight_sum = torch.sum(swing_weight, dim=1)
        error = torch.mean((foot_vel - ref_foot_vel).square(), dim=2)
        score = torch.exp(
            -error / (max(self.cfg.foot_vel_tracking_sigma, 1.0e-6) ** 2)
        )
        track = torch.sum(score * swing_weight, dim=1) / weight_sum.clamp_min(1.0e-6)
        return torch.where(weight_sum > 1.0e-6, track, torch.ones_like(track))

    def _stance_foot_track(
        self,
        foot_world_vel: torch.Tensor,
        foot_geom_contact: torch.Tensor,
        ref_foot_contact: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        side_contact = self._side_contacts_from_geom(foot_geom_contact)
        side_weight = torch.maximum(side_contact, torch.clamp(ref_foot_contact, 0.0, 1.0))
        horizontal_speed_sq = torch.sum(foot_world_vel[:, :, 0:2].square(), dim=2)
        geom_score = torch.exp(
            -horizontal_speed_sq / (max(self.cfg.stance_foot_vel_sigma, 1.0e-6) ** 2)
        )

        right_count = torch.sum(foot_geom_contact[:, 0:2], dim=1)
        left_count = torch.sum(foot_geom_contact[:, 2:4], dim=1)
        right_score = torch.sum(
            geom_score[:, 0:2] * foot_geom_contact[:, 0:2], dim=1
        ) / right_count.clamp_min(1.0e-6)
        left_score = torch.sum(
            geom_score[:, 2:4] * foot_geom_contact[:, 2:4], dim=1
        ) / left_count.clamp_min(1.0e-6)
        side_score = torch.stack(
            (
                torch.where(right_count > 1.0e-6, right_score, torch.zeros_like(right_score)),
                torch.where(left_count > 1.0e-6, left_score, torch.zeros_like(left_score)),
            ),
            dim=1,
        )
        side_weight_sum = torch.sum(side_weight, dim=1)
        stance_track = torch.sum(side_score * side_weight, dim=1) / side_weight_sum.clamp_min(1.0e-6)
        stance_track = torch.where(
            side_weight_sum > 1.0e-6,
            stance_track,
            torch.ones_like(stance_track),
        )

        horizontal_speed = torch.sqrt(horizontal_speed_sq)
        contact_count = torch.sum(foot_geom_contact, dim=1)
        stance_speed = torch.sum(horizontal_speed * foot_geom_contact, dim=1) / contact_count.clamp_min(1.0e-6)
        stance_speed = torch.where(
            contact_count > 1.0e-6,
            stance_speed,
            torch.zeros_like(stance_speed),
        )
        return stance_track, stance_speed

    def close(self) -> None:
        pass
