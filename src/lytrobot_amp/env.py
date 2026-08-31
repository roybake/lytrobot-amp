"""A small vectorized MuJoCo environment for the LYT biped."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import secrets

import mujoco
import numpy as np

from .motion import (
    CONTACT_GEOM_NAMES,
    MotionLibrary,
    MotionReference,
    MotionTargets,
    load_motion_library,
    resolve_contact_geom_ids,
)
from .training_defaults import CONTROL_DECIMATION, SIMULATION_TIMESTEP_S
from .terrain import (
    MoreTerrainCfg,
    STRICT_RESET_CLEARANCE,
    StairTerrainCfg,
    course_end_x,
    course_motion_sequence,
    course_stage_end_positions,
    course_stage_scenarios,
    load_model,
    randomized_course_orders,
    sample_course_start_stages,
    serial_course_stair_cfg,
    terrain_geom_ids,
    terrain_height_at,
    terrain_kind_from_motion,
    terrain_spawn_offset,
    validate_terrain_types,
)

ROBOT_COLLISION_PROFILES = ("full_body", "feet")


def configure_robot_collision_profile(
    model: mujoco.MjModel,
    profile: str,
) -> str:
    """Select terrain colliders without replacing the detailed foot meshes."""

    normalized = str(profile).strip().lower()
    if normalized not in ROBOT_COLLISION_PROFILES:
        raise ValueError(
            f"robot_collision_profile must be one of {ROBOT_COLLISION_PROFILES}, "
            f"got {profile!r}."
        )
    foot_names = set(CONTACT_GEOM_NAMES)
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = int(
            normalized == "full_body" or name in foot_names
        )
    return normalized


def _canonical_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat[0] < 0.0:
        quat *= -1.0
    norm = np.linalg.norm(quat)
    return quat / max(float(norm), 1.0e-6)


def _stair_root_height_transform(
    motion: MotionReference,
    *,
    step_height: float,
    step_count: int,
    descending: bool,
) -> tuple[float, float]:
    """Return anchor/scale mapping a stair clip to the compiled stair rise."""

    start = float(motion.reference_root_height[0])
    end = float(motion.reference_root_height[-1])
    reference_rise = abs(end - start)
    desired_rise = max(float(step_height), 0.0) * max(int(step_count), 0)
    if desired_rise > 0.0 and reference_rise < 1.0e-4:
        raise ValueError(
            "Stair reference has no measurable root-height change; it cannot be "
            "matched to non-zero stair geometry."
        )
    scale = desired_rise / reference_rise if desired_rise > 0.0 else 1.0
    # Scaling about the low end preserves the robot's pelvis clearance at the
    # course datum while moving the high end onto the actual stair top.
    anchor = end if descending else start
    return anchor, scale


def _sanitize_reward_outputs(
    reward: float,
    bad_state: bool,
    fallen: bool,
    components: dict[str, float],
    *,
    fall_penalty: float = 2.0,
) -> tuple[float, bool, dict[str, float]]:
    """Return finite reward diagnostics and mark derived numerical failures."""

    invalid = bool(bad_state or not np.isfinite(reward))
    invalid = invalid or any(not np.isfinite(value) for value in components.values())
    if invalid:
        safe_reward = -2.0
    else:
        safe_reward = float(reward) - (
            max(float(fall_penalty), 0.0) if fallen else 0.0
        )

    safe_components = {
        key: 0.0 if invalid else float(np.clip(np.nan_to_num(value), -1.0e6, 1.0e6))
        for key, value in components.items()
    }
    safe_components["invalid_state"] = float(invalid)
    return safe_reward, invalid, safe_components


def _stability_gated_locomotion_terms(
    forward_velocity_track: float,
    forward_progress: float,
    root_orientation: float,
    height: float,
) -> tuple[float, float, float]:
    """Grant positive locomotion reward only while the torso remains stable."""

    stability = float(
        np.sqrt(
            np.clip(float(root_orientation), 0.0, 1.0)
            * np.clip(float(height), 0.0, 1.0)
        )
    )
    gated_track = float(forward_velocity_track) * stability
    progress = float(forward_progress)
    gated_progress = min(progress, 0.0) + max(progress, 0.0) * stability
    return stability, gated_track, gated_progress


def _forward_progress_guard(
    position_x: np.ndarray,
    episode_step: np.ndarray,
    anchor_x: np.ndarray,
    anchor_step: np.ndarray,
    *,
    dt: float,
    min_speed: float,
    grace_steps: int,
    window_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check net world-X progress over fixed windows and advance the anchors."""

    position_x = np.asarray(position_x, dtype=np.float32)
    episode_step = np.asarray(episode_step, dtype=np.int32)
    next_anchor_x = np.asarray(anchor_x, dtype=np.float32).copy()
    next_anchor_step = np.asarray(anchor_step, dtype=np.int32).copy()
    speed = np.zeros_like(position_x, dtype=np.float32)
    stalled = np.zeros_like(position_x, dtype=bool)
    if min_speed <= 0.0 or window_steps <= 0:
        return speed, stalled, next_anchor_x, next_anchor_step

    grace_steps = max(int(grace_steps), 0)
    start_tracking = episode_step == grace_steps
    next_anchor_x[start_tracking] = position_x[start_tracking]
    next_anchor_step[start_tracking] = episode_step[start_tracking]

    elapsed = episode_step - next_anchor_step
    active = episode_step > grace_steps
    finite = np.isfinite(position_x) & np.isfinite(next_anchor_x)
    valid_elapsed = np.maximum(elapsed, 1)
    measured = active & finite
    speed[measured] = (
        (position_x[measured] - next_anchor_x[measured])
        / (valid_elapsed[measured] * max(float(dt), 1.0e-6))
    )
    check = active & (elapsed >= int(window_steps))
    stalled = check & (~finite | (speed < float(min_speed)))
    next_anchor_x[check] = position_x[check]
    next_anchor_step[check] = episode_step[check]
    return speed, stalled, next_anchor_x, next_anchor_step


def scaled_randomization_range(
    nominal: float,
    low: float,
    high: float,
    scale: float,
) -> tuple[float, float]:
    """Interpolate a domain-randomization range out from its nominal value."""

    scale = float(np.clip(scale, 0.0, 1.0))
    return (
        float(nominal + scale * (low - nominal)),
        float(nominal + scale * (high - nominal)),
    )


@dataclass
class LytRobotEnvCfg:
    model_xml: str = "robot/MJCFmodelposition.xml"
    motion_file: str = "robot/reference_data/walk1_mjcf.npz"
    motion_files: tuple[str, ...] = ()
    motion_names: tuple[str, ...] = ()
    motion_weights: tuple[float, ...] = ()
    motion_loop: tuple[bool, ...] = ()
    training_task: str = ""
    skill_conditioning: bool = True
    use_reference_velocity: bool = False
    terrain_types: tuple[str, ...] = ()
    terrain_asset_dir: str = ""
    stair_terrain: bool = False
    stair_step_height: float = 0.13
    stair_step_depth: float = 0.30
    stair_step_count: int = 5
    stair_width: float = 2.0
    stair_run_in: float = 1.5
    stair_top_length: float = 1.2
    stair_run_out: float = 1.5
    stair_start_x: float = -1.5
    stair_down_offset: float = 0.0
    flat_lane_y: float = 3.0
    rough_lane_y: float = 6.0
    pit_lane_y: float = 9.0
    gap_lane_y: float = 12.0
    terrain_lane_width: float = 2.0
    rough_terrain_height: float = 0.05
    slope_angle_deg: float = 6.0
    gap_terrain_width: float = 0.20
    serial_terrain_course: bool = False
    # Reuse one looping walk clip as the AMP expert for every serial-course
    # stage.  This is explicit because ordinary courses still require a
    # terrain-aligned multi-motion library.
    serial_course_single_motion: bool = False
    terrain_course_variants: int = 8
    terrain_course_lane_spacing: float = 3.0
    terrain_course_run_in: float = 2.90
    terrain_course_run_out: float = 3.10
    terrain_course_base_height: float = 0.30
    terrain_course_random_assignment: bool = True
    # Training-only AMP state initialization for serial courses.  When
    # enabled, ``motion_reset_prob`` samples a physically moving reference
    # pose/velocity/phase while keeping the robot at the selected stage's
    # spatial entrance.  Deployment playback sets motion_reset_prob=0 and
    # therefore still starts from rest at phase zero.
    serial_reference_state_init: bool = False
    terrain_course_completion_reward: float = 25.0
    terrain_course_stage_reward: float = 0.0
    terrain_course_stage_tolerance: float = 0.35
    terrain_course_random_start_prob: float = 0.0
    # Keep finite, non-looping terrain references spatially aligned with the
    # robot. Without this guard the reference clip can end while a learning
    # policy is still short of the physical stage boundary.
    terrain_course_phase_sync: bool = False
    terrain_course_phase_lookahead: float = 0.10
    legacy_0806_terrain_model: bool = False
    num_envs: int = 4096
    seed: int = 42
    randomize_episode_seed: bool = False
    decimation: int = CONTROL_DECIMATION
    timestep: float = SIMULATION_TIMESTEP_S
    # Full-body is both the training default and the backward-compatible
    # interpretation for checkpoints that predate this explicit field.
    robot_collision_profile: str = "full_body"
    allow_collision_profile_migration: bool = False
    solver_iterations: int = 20
    solver_ls_iterations: int = 20
    warp_nconmax: int = 128
    warp_njmax: int = 512
    warp_cuda_graph: bool = True
    terrain_penetration_tolerance: float = 0.01
    terrain_penetration_penalty_coef: float = 0.0
    episode_steps: int = 2000
    episode_timeout_jitter_steps: int = 480
    target_velocity: float = 1.0
    reward_mode: str = "default"
    # Keep retargeted data exclusively inside the AMP discriminator.  The task
    # reward then contains only physical walking objectives, never reference
    # pose/contact/control matching.
    amp_task_only: bool = False
    forward_velocity_tracking_sigma: float = 0.25
    forward_velocity_reward_coef: float = 0.0
    forward_progress_reward_coef: float = 0.0
    forward_velocity_deficit_penalty_coef: float = 0.0
    forward_velocity_excess_penalty_coef: float = 0.0
    min_forward_progress_speed: float = 0.0
    forward_progress_grace_steps: int = 100
    forward_progress_window_steps: int = 100
    forward_progress_termination_penalty: float = 2.0
    fall_penalty: float = 2.0
    terminate_incomplete_motion_end: bool = False
    reset_noise_scale: float = 0.01
    motion_reset_prob: float = 0.5
    action_residual_scale: float = 0.5
    distal_action_scale: float = 1.0
    reference_ctrl_blend: float = 1.0
    motion_transition_blend_steps: int = 0
    fall_height: float = 0.45
    max_height: float = 1.55
    min_upright_z: float = 0.45
    foot_vel_tracking_sigma: float = 1.0
    stance_foot_vel_sigma: float = 0.15
    foot_vel_reward_coef: float = 0.35
    stance_foot_reward_coef: float = 1.00
    foot_slip_velocity_threshold: float = 0.08
    foot_slip_penalty_coef: float = 0.0
    contact_reward_coef: float = 0.70
    missed_contact_penalty_coef: float = 0.80
    side_foot_pos_sigma: float = 0.08
    step_gap_sigma: float = 0.12
    swing_advance_sigma: float = 0.10
    side_foot_pos_coef: float = 0.0
    step_gap_coef: float = 0.0
    swing_advance_coef: float = 0.0
    alternating_step_reward_coef: float = 0.0
    stair_vertical_velocity_sigma: float = 0.25
    stair_vertical_velocity_reward_coef: float = 1.0
    stair_swing_clearance: float = 0.08
    stair_swing_clearance_reward_coef: float = 0.75
    stair_tread_support_reward_coef: float = 1.25
    stair_min_root_clearance: float = 0.60
    distal_joint_vel_sigma: float = 2.0
    symmetry_reward_coef: float = 1.20
    symmetry_foot_pos_sigma: float = 0.08
    symmetry_balance_sigma: float = 0.25
    symmetry_ema_decay: float = 0.99
    symmetry_foot_weight: float = 0.60
    symmetry_contact_weight: float = 0.25
    symmetry_action_weight: float = 0.15
    startup_action_ramp_steps: int = 40
    stand_start_prob: float = 0.5
    stand_start_phase_delay_steps: int = 50
    stand_start_transition_min_steps: int = 160
    stand_start_transition_max_steps: int = 320
    stand_start_push_delay_steps: int = 200
    # Used only by squat_training's phase-clock supervisor.  Other tasks do
    # not read these fields and therefore retain their original phase logic.
    squat_phase_feedback: bool = False
    squat_phase_lag_tolerance: float = 0.08
    squat_phase_min_scale: float = 0.0
    squat_bottom_depth_threshold: float = 0.85
    squat_stand_depth_threshold: float = 0.15
    squat_upright_threshold: float = 0.75
    squat_phase_stall_threshold: float = 0.25
    squat_phase_max_stall_steps: int = 80
    squat_phase_stall_penalty: float = 0.25
    squat_phase_slowdown_penalty_coef: float = 2.0
    squat_cycle_completion_bonus: float = 5.0
    squat_terminate_on_cycle_completion: bool = False
    ctrl_penalty_coef: float = 0.040
    action_rate_penalty_coef: float = 0.100
    distal_action_penalty_coef: float = 0.030
    distal_action_rate_penalty_coef: float = 0.120
    domain_randomization: bool = False
    domain_randomization_scale: float = 1.0
    warp_global_domain_randomization: bool = False
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
    push_interval_min_steps: int = 50
    push_interval_max_steps: int = 200
    push_duration_min_steps: int = 4
    push_duration_max_steps: int = 16
    obs_noise_std: float = 0.02
    action_noise_std: float = 0.01
    action_delay_min_ms: float = 0.02
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

        terrain_types = validate_terrain_types(cfg.terrain_types)
        if cfg.reward_mode not in {"default", "stairs"}:
            raise ValueError(f"Unknown reward mode: {cfg.reward_mode!r}.")
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
            course_run_in=cfg.terrain_course_run_in,
            course_run_out=cfg.terrain_course_run_out,
            course_base_height=cfg.terrain_course_base_height,
            flat_lane_y=cfg.flat_lane_y,
            rough_lane_y=cfg.rough_lane_y,
            pit_lane_y=cfg.pit_lane_y,
            gap_lane_y=cfg.gap_lane_y,
            lane_width=cfg.terrain_lane_width,
            rough_height=cfg.rough_terrain_height,
            slope_angle_deg=cfg.slope_angle_deg,
            gap_width=cfg.gap_terrain_width,
            legacy_0806_model=bool(cfg.legacy_0806_terrain_model),
        )
        self.complex_terrain_enabled = bool(
            self.stair_cfg.enabled or self.more_terrain_cfg.enabled
        )
        self.model = load_model(
            self.model_path,
            self.stair_cfg,
            self.more_terrain_cfg,
        )
        self.cfg.robot_collision_profile = configure_robot_collision_profile(
            self.model, self.cfg.robot_collision_profile
        )
        self.model.opt.timestep = float(cfg.timestep)
        minimum_iterations = (
            20 if self.cfg.robot_collision_profile == "full_body" else 1
        )
        self.model.opt.iterations = max(
            minimum_iterations, int(cfg.solver_iterations)
        )
        self.model.opt.ls_iterations = max(1, int(cfg.solver_ls_iterations))
        self.position_actuator_ids = np.arange(self.model.nu, dtype=np.int32)
        self.position_action_dim = int(len(self.position_actuator_ids))
        self.action_dim = self.position_action_dim
        self.ctrl_dim = int(self.model.nu)
        motion_files = cfg.motion_files or (cfg.motion_file,)
        self.motion = load_motion_library(
            motion_files,
            self.model,
            actuator_ids=self.position_actuator_ids,
            motion_weights=cfg.motion_weights or None,
            motion_names=cfg.motion_names or None,
            motion_loop=cfg.motion_loop or None,
            force_library=bool(cfg.serial_terrain_course and cfg.serial_course_single_motion),
        )
        self.motion_count = (
            self.motion.motion_count if isinstance(self.motion, MotionLibrary) else 1
        )
        motion_names = (
            self.motion.motion_names
            if isinstance(self.motion, MotionLibrary)
            else (Path(cfg.motion_file).stem,)
        )
        self.stair_motion_mask = np.asarray(
            ["stair" in name.lower() for name in motion_names], dtype=bool
        )
        self.stair_down_motion_mask = np.asarray(
            ["stair_down" in name.lower() for name in motion_names], dtype=bool
        )
        if cfg.reward_mode == "stairs":
            if not self.stair_cfg.enabled:
                raise ValueError("The stairs reward mode requires enabled stair geometry.")
            if self.more_terrain_cfg.serial_course:
                required = {"walk", "stair_up", "stair_down"}
                missing = required - {name.lower() for name in motion_names}
                if missing:
                    raise ValueError(
                        "Serial stairs reward requires walk, stair_up, and "
                        f"stair_down motion clips; missing {sorted(missing)}."
                    )
            elif not np.all(self.stair_motion_mask):
                raise ValueError("The stairs reward mode requires only stair motion clips.")
        motion_references = (
            self.motion.motions
            if isinstance(self.motion, MotionLibrary)
            else (self.motion,)
        )
        self.motion_stair_height_anchor = np.zeros(self.motion_count, dtype=np.float32)
        self.motion_stair_height_scale = np.ones(self.motion_count, dtype=np.float32)
        for motion_id, motion_reference in enumerate(motion_references):
            if not self.stair_motion_mask[motion_id]:
                continue
            anchor, scale = _stair_root_height_transform(
                motion_reference,
                step_height=self.stair_cfg.step_height,
                step_count=self.stair_cfg.step_count,
                descending=bool(self.stair_down_motion_mask[motion_id]),
            )
            self.motion_stair_height_anchor[motion_id] = anchor
            self.motion_stair_height_scale[motion_id] = scale
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
        self.course_motion_sequences = self._build_course_motion_sequences(motion_names)
        self.course_stair_cfgs = (
            tuple(
                serial_course_stair_cfg(self.more_terrain_cfg, self.stair_cfg, variant)
                for variant in range(len(self.more_terrain_cfg.normalized_course_orders))
            )
            if self.more_terrain_cfg.serial_course and "stairs" in terrain_types
            else ()
        )
        self.course_stage_scenario_names = tuple(
            course_stage_scenarios(
                order,
                legacy_0806_model=self.more_terrain_cfg.legacy_0806_model,
            )
            for order in self.more_terrain_cfg.normalized_course_orders
        )
        self.course_stage_end_x = self._build_course_stage_end_x()
        self.course_finish_x = (
            course_end_x(self.more_terrain_cfg, 0)
            if self.more_terrain_cfg.serial_course
            else float("inf")
        )
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
        self.robot_mass = float(np.sum(self.model.body_mass))
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
        self.terrain_geom_ids = set(int(value) for value in terrain_geom_ids(self.model))
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
        self.forward_progress_anchor_x = np.zeros(self.num_envs, dtype=np.float32)
        self.forward_progress_anchor_step = np.zeros(self.num_envs, dtype=np.int32)
        self.motion_phase = np.zeros(self.num_envs, dtype=np.float32)
        self.motion_ids = np.zeros(self.num_envs, dtype=np.int32)
        self.course_variant_ids = np.zeros(self.num_envs, dtype=np.int32)
        self.course_stage_ids = np.zeros(self.num_envs, dtype=np.int32)
        self.course_start_stage_ids = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_seed = np.full(self.num_envs, int(cfg.seed), dtype=np.uint64)
        self.last_action = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.last_reference_ctrl = np.zeros(
            (self.num_envs, self.action_dim), dtype=np.float32
        )
        self.motion_transition_from_ctrl = np.zeros_like(self.last_reference_ctrl)
        self.motion_transition_steps_left = np.zeros(self.num_envs, dtype=np.int32)
        self.last_foot_pos = np.zeros((self.num_envs, len(CONTACT_GEOM_NAMES), 3), dtype=np.float32)
        self.last_foot_world_pos = np.zeros_like(self.last_foot_pos)
        self.last_foot_contact = np.zeros((self.num_envs, 2), dtype=np.float32)
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
        skill_dim = self.motion_count if self.cfg.skill_conditioning and self.motion_count > 1 else 0
        return int(current_state_dim + tracking_dim + skill_dim)

    @property
    def amp_obs_dim(self) -> int:
        return self.motion.feature_dim

    def _build_course_motion_sequences(
        self,
        motion_names: tuple[str, ...],
    ) -> np.ndarray:
        if not self.more_terrain_cfg.serial_course:
            return np.empty((0, 0), dtype=np.int32)
        if not isinstance(self.motion, MotionLibrary):
            raise ValueError("serial terrain course requires a multi-motion library.")
        if self.cfg.serial_course_single_motion:
            if len(motion_names) != 1:
                raise ValueError(
                    "--serial-course-single-motion requires exactly one motion clip."
                )
            stage_count = len(
                course_motion_sequence(
                    self.more_terrain_cfg.normalized_course_orders[0],
                    legacy_0806_model=self.more_terrain_cfg.legacy_0806_model,
                )
            )
            return np.zeros(
                (len(self.more_terrain_cfg.normalized_course_orders), stage_count),
                dtype=np.int32,
            )
        name_to_id = {name.lower(): index for index, name in enumerate(motion_names)}
        rows: list[list[int]] = []
        for order in self.more_terrain_cfg.normalized_course_orders:
            names = course_motion_sequence(
                order,
                legacy_0806_model=self.more_terrain_cfg.legacy_0806_model,
            )
            missing = [name for name in names if name not in name_to_id]
            if missing:
                raise ValueError(
                    "Serial terrain course is missing required motion clips: "
                    f"{sorted(set(missing))}."
                )
            rows.append([name_to_id[name] for name in names])
        return np.asarray(rows, dtype=np.int32)

    def _build_course_stage_end_x(self) -> np.ndarray:
        """Physical root-X boundary for each action in every serial course."""

        if not self.more_terrain_cfg.serial_course:
            return np.empty((0, 0), dtype=np.float32)
        rows = [
            course_stage_end_positions(self.more_terrain_cfg, variant)
            for variant in range(len(self.more_terrain_cfg.normalized_course_orders))
        ]
        result = np.asarray(rows, dtype=np.float32)
        if result.shape != self.course_motion_sequences.shape:
            raise ValueError(
                "Serial course motion stages and physical boundaries do not align: "
                f"motions={self.course_motion_sequences.shape}, boundaries={result.shape}."
            )
        return result

    def _course_height_offsets(self) -> np.ndarray:
        if not self.more_terrain_cfg.serial_course:
            return np.zeros(self.num_envs, dtype=np.float32)
        # Gap/pit clips were retargeted on the original 0.30 m platforms;
        # the other references use zero as their terrain datum.
        return np.asarray(
            [
                0.0
                if self.motion_terrain_kinds[int(motion_id)] in {"gap", "pit"}
                else self.more_terrain_cfg.course_base_height
                for motion_id in self.motion_ids
            ],
            dtype=np.float32,
        )

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

    def _motion_phase_progress(self) -> np.ndarray:
        """Scale reference-clock advancement without changing target blending."""

        return self._stand_start_progress()

    def _sync_serial_course_phase(
        self,
        next_phase: np.ndarray,
        frame_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Limit non-looping references to the robot's physical stage progress.

        A small look-ahead keeps the controller predictive. Looping approach
        and exit clips retain their normal clock because they can safely replay
        until the corresponding boundary is reached.
        """

        limited = np.zeros(self.num_envs, dtype=bool)
        progress = np.zeros(self.num_envs, dtype=np.float32)
        if (
            not self.more_terrain_cfg.serial_course
            or not self.cfg.terrain_course_phase_sync
            or not isinstance(self.motion, MotionLibrary)
        ):
            return next_phase, limited, progress

        stage_ids = self.course_stage_ids
        variants = self.course_variant_ids
        stage_end = self.course_stage_end_x[variants, stage_ids]
        previous_stage = np.maximum(stage_ids - 1, 0)
        previous_end = self.course_stage_end_x[variants, previous_stage]
        stage_start = np.where(stage_ids > 0, previous_end, 0.0).astype(
            np.float32
        )
        root_x = np.asarray(
            [data.qpos[0] for data in self.datas], dtype=np.float32
        )
        stage_length = np.maximum(stage_end - stage_start, 1.0e-6)
        progress = np.clip((root_x - stage_start) / stage_length, 0.0, 1.0)
        lookahead = float(
            np.clip(self.cfg.terrain_course_phase_lookahead, 0.0, 1.0)
        )
        phase_limit = np.clip(progress + lookahead, 0.0, 1.0) * np.maximum(
            frame_counts - 1.0, 0.0
        )
        # Spatial synchronization may pause advancement but must never rewind
        # the reference when the robot briefly steps backward.
        phase_limit = np.maximum(phase_limit, self.motion_phase)
        active_looping = self.motion.motion_loop[self.motion_ids]
        synced = np.where(
            active_looping,
            next_phase,
            np.minimum(next_phase, phase_limit),
        ).astype(np.float32)
        limited = (~active_looping) & (synced < next_phase - 1.0e-6)
        return synced, limited, progress.astype(np.float32)

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
        if isinstance(self.motion, MotionLibrary):
            targets = self.motion.targets_at_phase(self.motion_phase, self.motion_ids)
        else:
            targets = self.motion.targets_at_phase(self.motion_phase)
        stair_anchor = self.motion_stair_height_anchor[self.motion_ids]
        stair_scale = self.motion_stair_height_scale[self.motion_ids]
        scaled_root_lin_vel = targets.root_lin_vel.copy()
        scaled_root_lin_vel[:, 2] *= stair_scale
        targets = MotionTargets(
            root_height=(
                stair_anchor + (targets.root_height - stair_anchor) * stair_scale
            ).astype(np.float32),
            root_quat=targets.root_quat,
            root_lin_vel=scaled_root_lin_vel.astype(np.float32),
            root_ang_vel=targets.root_ang_vel,
            joint_pos=targets.joint_pos,
            joint_vel=targets.joint_vel,
            tendon_ctrl=targets.tendon_ctrl,
            foot_pos=targets.foot_pos,
            foot_vel=targets.foot_vel,
            foot_contact=targets.foot_contact,
        )
        targets = self._blend_start_targets(targets)
        if not self.more_terrain_cfg.serial_course:
            return targets
        return MotionTargets(
            root_height=(targets.root_height + self._course_height_offsets()).astype(np.float32),
            root_quat=targets.root_quat,
            root_lin_vel=targets.root_lin_vel,
            root_ang_vel=targets.root_ang_vel,
            joint_pos=targets.joint_pos,
            joint_vel=targets.joint_vel,
            tendon_ctrl=targets.tendon_ctrl,
            foot_pos=targets.foot_pos,
            foot_vel=targets.foot_vel,
            foot_contact=targets.foot_contact,
        )

    def _init_domain_randomization(self) -> None:
        self.domain_randomization = bool(self.cfg.domain_randomization)
        self.domain_randomization_scale = float(np.clip(self.cfg.domain_randomization_scale, 0.0, 1.0))
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

    def set_domain_randomization_scale(self, scale: float) -> None:
        """Set curriculum strength for subsequently sampled and step-wise noise."""

        self.domain_randomization_scale = float(np.clip(scale, 0.0, 1.0))
        self.cfg.domain_randomization_scale = self.domain_randomization_scale

    def set_terrain_course_random_start_prob(self, probability: float) -> None:
        """Set the random-stage curriculum probability used by future resets."""

        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("terrain course random-start probability must be in [0, 1].")
        self.cfg.terrain_course_random_start_prob = probability

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
        scale = self.domain_randomization_scale
        payload_low, payload_high = scaled_randomization_range(
            0.0, self.cfg.payload_mass_min, self.cfg.payload_mass_max, scale
        )
        body_low, body_high = scaled_randomization_range(
            1.0, self.cfg.body_mass_scale_min, self.cfg.body_mass_scale_max, scale
        )
        tendon_low, tendon_high = scaled_randomization_range(
            1.0, self.cfg.tendon_length_scale_min, self.cfg.tendon_length_scale_max, scale
        )
        nominal_friction = float(self.base_geom_friction[self.floor_geom_id, 0])
        friction_low, friction_high = scaled_randomization_range(
            nominal_friction, self.cfg.ground_friction_min, self.cfg.ground_friction_max, scale
        )
        nominal_gravity = abs(float(self.base_gravity[2]))
        gravity_low, gravity_high = scaled_randomization_range(
            nominal_gravity, self.cfg.gravity_min, self.cfg.gravity_max, scale
        )
        self.env_payload_mass[ids] = self._uniform(payload_low, payload_high, len(ids))
        self.env_body_mass_scale[ids] = self._uniform(
            body_low,
            body_high,
            (len(ids), self.model.nbody),
        )
        self.env_body_mass_scale[ids, 0] = 1.0
        self.env_tendon_length_scale[ids] = self._uniform(
            tendon_low,
            tendon_high,
            (len(ids), self.position_action_dim),
        )
        self.env_ground_friction[ids] = self._uniform(
            friction_low,
            friction_high,
            len(ids),
        )
        self.env_gravity[ids] = self._uniform(gravity_low, gravity_high, len(ids))

        min_delay = scale * max(0.0, float(self.cfg.action_delay_min_ms)) / 1000.0
        max_delay = scale * max(0.0, float(self.cfg.action_delay_max_ms)) / 1000.0
        max_delay = max(min_delay, max_delay)
        delay_seconds = self.rng.uniform(min_delay, max_delay, size=len(ids))
        # Round to the nearest physics tick instead of ceiling: the configured
        # 0.02-5 ms range spans sub-tick delays, and ceiling would quantise
        # every positive delay to a full 5 ms tick (losing the near-zero end).
        delay_ticks = np.floor(
            delay_seconds / max(float(self.cfg.timestep), 1.0e-9) + 0.5
        ).astype(np.int32)
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
        terrain_ids = np.fromiter(self.terrain_geom_ids, dtype=np.int32)
        self.model.geom_friction[terrain_ids, 0] = self.env_ground_friction[env_id]
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
            action
            + self.rng.normal(
                0.0,
                self.domain_randomization_scale * float(self.cfg.action_noise_std),
                size=action.shape,
            ).astype(np.float32),
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
        magnitude = float(
            self._uniform(
                self.domain_randomization_scale * float(self.cfg.push_force_min),
                self.domain_randomization_scale * float(self.cfg.push_force_max),
            )
        )
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

    def _transition_reference_ctrl(
        self, reference_ctrl: np.ndarray, env_id: int
    ) -> np.ndarray:
        """Blend only a newly selected motion onto the previous reference."""
        target = np.asarray(reference_ctrl, dtype=np.float32)
        total_steps = max(0, int(self.cfg.motion_transition_blend_steps))
        steps_left = int(self.motion_transition_steps_left[env_id])
        if total_steps > 0 and steps_left > 0:
            alpha = float(total_steps - steps_left + 1) / float(total_steps)
            transitioned = (
                (1.0 - alpha) * self.motion_transition_from_ctrl[env_id]
                + alpha * target
            ).astype(np.float32)
            self.motion_transition_steps_left[env_id] = steps_left - 1
        else:
            transitioned = target.copy()
            self.motion_transition_steps_left[env_id] = 0
        self.last_reference_ctrl[env_id] = transitioned
        return transitioned

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
            if geom1 in self.terrain_geom_ids:
                other = geom2
            elif geom2 in self.terrain_geom_ids:
                other = geom1
            if other is None:
                continue
            contact_index = self.contact_geom_id_to_index.get(other)
            if contact_index is not None:
                contact[contact_index] = 1.0
        return contact

    def _foot_contacts(self, data: mujoco.MjData) -> np.ndarray:
        return self._side_contacts_from_geom(self._foot_geom_contacts(data))

    def _terrain_penetration_depth(self, data: mujoco.MjData) -> float:
        penetration = 0.0
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1_terrain = int(contact.geom1) in self.terrain_geom_ids
            geom2_terrain = int(contact.geom2) in self.terrain_geom_ids
            if geom1_terrain == geom2_terrain:
                continue
            penetration = max(penetration, -float(contact.dist))
        return penetration

    def _resolve_terrain_reset_penetration(self, data: mujoco.MjData) -> None:
        """Lift a reset state until no robot geom starts inside terrain."""

        for _ in range(4):
            mujoco.mj_forward(self.model, data)
            penetration = self._terrain_penetration_depth(data)
            if penetration <= 0.0:
                return
            data.qpos[2] += penetration + STRICT_RESET_CLEARANCE
        mujoco.mj_forward(self.model, data)

    def reset(self, env_ids: np.ndarray | list[int] | None = None) -> np.ndarray:
        if env_ids is None:
            ids = np.arange(self.num_envs, dtype=np.int32)
        else:
            ids = np.asarray(env_ids, dtype=np.int32)
        self._refresh_episode_rng(ids)

        serial_course = self.more_terrain_cfg.serial_course
        if serial_course:
            variant_count = len(self.more_terrain_cfg.normalized_course_orders)
            variants = (
                self.rng.integers(0, variant_count, size=len(ids), dtype=np.int32)
                if self.cfg.terrain_course_random_assignment
                else np.mod(ids, variant_count).astype(np.int32)
            )
            start_stages = sample_course_start_stages(
                self.rng,
                len(ids),
                self.course_motion_sequences.shape[1],
                self.cfg.terrain_course_random_start_prob,
            )
            self.course_variant_ids[ids] = variants
            self.course_stage_ids[ids] = start_stages
            self.course_start_stage_ids[ids] = start_stages
            sampled_motion_ids = self.course_motion_sequences[
                variants, start_stages
            ]
            assert isinstance(self.motion, MotionLibrary)
            states = self.motion.sample_states(
                len(ids), self.rng, motion_ids=sampled_motion_ids
            )
            if self.cfg.serial_reference_state_init:
                for row, motion_id in enumerate(sampled_motion_ids):
                    # MotionReference root X/Y is local clip travel.  Preserve
                    # the sampled gait pose, velocity and phase, but let the
                    # terrain spawn offset place it at the physical stage
                    # entrance.  This keeps full-route success honest instead
                    # of moving a reset several metres down the course.
                    motion = self.motion.motions[int(motion_id)]
                    states.qpos[row, :2] = motion.reference_qpos[0, :2]
            else:
                for row, motion_id in enumerate(sampled_motion_ids):
                    states.qpos[row] = self.motion.motions[
                        int(motion_id)
                    ].reference_qpos[0]
                    # Tasks that do not explicitly opt into serial reference
                    # initialization retain the legacy phase-zero rest reset.
                    states.qvel[row] = 0.0
                    states.indices[row] = 0
        elif isinstance(self.motion, MotionLibrary):
            sampled_motion_ids = self.motion.sample_motion_ids(len(ids), self.rng)
            states = self.motion.sample_states(len(ids), self.rng, motion_ids=sampled_motion_ids)
        else:
            sampled_motion_ids = np.zeros(len(ids), dtype=np.int32)
            states = self.motion.sample_states(len(ids), self.rng)
        self.motion_ids[ids] = sampled_motion_ids
        # Serial-course deployment and non-serial flat playback both need to
        # launch from the robot's natural standing pose.  Previously serial
        # courses silently forced this mask to false, so a nonzero
        # ``stand_start_prob`` never exposed the policy to its deployed start
        # state.
        use_stand_start = self.rng.random(len(ids)) < float(
            np.clip(self.cfg.stand_start_prob, 0.0, 1.0)
        )
        if serial_course and not self.cfg.serial_reference_state_init:
            use_motion_reset = ~use_stand_start
        else:
            sample_motion_state = self.rng.random(len(ids)) < float(
                np.clip(self.cfg.motion_reset_prob, 0.0, 1.0)
            )
            use_motion_reset = ~use_stand_start & sample_motion_state
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

            if use_motion_reset[row]:
                motion_id = int(sampled_motion_ids[row])
                height_anchor = self.motion_stair_height_anchor[motion_id]
                height_scale = self.motion_stair_height_scale[motion_id]
                qpos[2] = height_anchor + (qpos[2] - height_anchor) * height_scale
                qvel[2] *= height_scale

            if self.complex_terrain_enabled:
                if serial_course:
                    scenario = self.course_stage_scenario_names[
                        int(variants[row])
                    ][int(start_stages[row])]
                    terrain_kind = (
                        "stairs" if scenario in {"stair_up", "stair_down"} else scenario
                    )
                    stair_down = scenario == "stair_down"
                else:
                    terrain_kind = self.motion_terrain_kinds[sampled_motion_ids[row]]
                    stair_down = bool(
                        self.stair_down_motion_mask[sampled_motion_ids[row]]
                    )
                offset = terrain_spawn_offset(
                    terrain_kind,
                    self.more_terrain_cfg,
                    self.stair_cfg,
                    stair_down=stair_down,
                    stair_down_offset=self.cfg.stair_down_offset,
                    course_variant=int(self.course_variant_ids[int(env_id)]),
                    course_stage=(int(start_stages[row]) if serial_course else None),
                )
                qpos[:3] += offset

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
            self._resolve_terrain_reset_penetration(data)
            if isinstance(self.motion, MotionLibrary):
                reference_ctrl = self.motion.targets_at_phase(
                    np.asarray([phase], dtype=np.float32),
                    np.asarray([sampled_motion_ids[row]], dtype=np.int32),
                ).tendon_ctrl[0]
            else:
                reference_ctrl = self.motion.targets_at_phase(
                    np.asarray([phase], dtype=np.float32)
                ).tendon_ctrl[0]
            if use_stand_start[row]:
                reference_ctrl = self.stand_tendon_ctrl[0]
            self.last_reference_ctrl[int(env_id)] = reference_ctrl
            self.motion_transition_from_ctrl[int(env_id)] = reference_ctrl
            self.motion_transition_steps_left[int(env_id)] = 0
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
            self.last_foot_contact[int(env_id)] = self._foot_contacts(data)
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
            self.forward_progress_anchor_x[int(env_id)] = float(data.qpos[0])
            self.forward_progress_anchor_step[int(env_id)] = 0
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

    def observe(self, *, add_noise: bool = True) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        reference_targets = self._targets_at_current_phase()
        stand_start_progress = self._stand_start_progress()
        for env_id, data in enumerate(self.datas):
            quat = _canonical_quat(data.qpos[3:7])
            up_z = float(data.xmat[self.pelvis_body_id, 8])
            frame_count = (
                int(self.motion.frame_counts[self.motion_ids[env_id]])
                if isinstance(self.motion, MotionLibrary)
                else self.motion.frame_count
            )
            phase = 2.0 * np.pi * float(self.motion_phase[env_id]) / float(frame_count)
            joint_pos = np.asarray(data.qpos[self.motion.robot_qpos_adrs], dtype=np.float32)
            joint_vel = np.asarray(data.qvel[self.motion.robot_qvel_adrs], dtype=np.float32)
            foot_pos = self._foot_positions_local(data).reshape(-1)
            ref_foot_pos = reference_targets.foot_pos[env_id].reshape(-1)
            ref_ctrl_action = self.ctrl_to_action(reference_targets.tendon_ctrl[env_id])
            root_vel_target = np.concatenate(
                (reference_targets.root_lin_vel[env_id], reference_targets.root_ang_vel[env_id])
            ).astype(np.float32)
            if not self.cfg.use_reference_velocity:
                root_vel_target[0] = float(self.cfg.target_velocity) * float(
                    stand_start_progress[env_id]
                )
            parts = [
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
            ]
            if self.cfg.skill_conditioning and self.motion_count > 1:
                skill = np.zeros(self.motion_count, dtype=np.float32)
                skill[self.motion_ids[env_id]] = 1.0
                parts.append(skill)
            obs[env_id] = np.concatenate(parts)
        obs = np.nan_to_num(obs, nan=0.0, posinf=20.0, neginf=-20.0)
        if add_noise and self.domain_randomization and self.cfg.obs_noise_std > 0.0:
            obs += self.rng.normal(
                0.0,
                self.domain_randomization_scale * float(self.cfg.obs_noise_std),
                size=obs.shape,
            ).astype(np.float32)
        return np.clip(obs, -20.0, 20.0).astype(np.float32)

    def amp_observe(self) -> np.ndarray:
        qpos = np.stack([np.asarray(data.qpos, dtype=np.float32) for data in self.datas], axis=0)
        qvel = np.stack([np.asarray(data.qvel, dtype=np.float32) for data in self.datas], axis=0)
        foot_pos = np.stack([self._foot_positions_local(data) for data in self.datas], axis=0)
        qpos = qpos.copy()
        if self.more_terrain_cfg.serial_course:
            qpos[:, 2] -= self._course_height_offsets()
        stair_anchor = self.motion_stair_height_anchor[self.motion_ids]
        stair_scale = self.motion_stair_height_scale[self.motion_ids]
        qpos[:, 2] = stair_anchor + (qpos[:, 2] - stair_anchor) / stair_scale
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
            "actuator_power": [],
            "mechanical_energy": [],
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
        phase_scale = self._motion_phase_progress()
        course_complete = np.zeros(self.num_envs, dtype=bool)
        course_suffix_complete = np.zeros(self.num_envs, dtype=bool)
        course_stage_advance = np.zeros(self.num_envs, dtype=bool)
        course_phase_sync_limited = np.zeros(self.num_envs, dtype=bool)
        course_stage_progress = np.zeros(self.num_envs, dtype=np.float32)
        incomplete_motion_end_candidate = np.zeros(self.num_envs, dtype=bool)
        incomplete_motion_required_x = np.full(self.num_envs, np.inf, dtype=np.float32)
        incomplete_motion_end_flags = np.zeros(self.num_envs, dtype=bool)
        if isinstance(self.motion, MotionLibrary):
            frequencies = self.motion.frequencies[self.motion_ids]
            frame_counts = self.motion.frame_counts[self.motion_ids].astype(np.float32)
            next_phase = self.motion_phase + frequencies * self.dt * phase_scale
            if self.more_terrain_cfg.serial_course:
                (
                    next_phase,
                    course_phase_sync_limited,
                    course_stage_progress,
                ) = self._sync_serial_course_phase(next_phase, frame_counts)
                reference_complete = next_phase >= frame_counts - 1.0
                active_looping = self.motion.motion_loop[self.motion_ids].copy()
                last_stage = self.course_motion_sequences.shape[1] - 1
                at_last_stage = self.course_stage_ids >= last_stage
                reached_finish = np.asarray(
                    [data.qpos[0] for data in self.datas], dtype=np.float32
                ) >= (
                    self.course_finish_x
                    - self.more_terrain_cfg.course_completion_tolerance
                )
                stage_end_x = self.course_stage_end_x[
                    self.course_variant_ids, self.course_stage_ids
                ]
                reached_stage_end = np.asarray(
                    [data.qpos[0] for data in self.datas], dtype=np.float32
                ) >= (
                    stage_end_x - float(self.cfg.terrain_course_stage_tolerance)
                )
                physical_stage_complete = np.where(
                    at_last_stage, reached_finish, reached_stage_end
                )
                phase_synced_nonloop = (
                    bool(self.cfg.terrain_course_phase_sync) & ~active_looping
                )
                motion_complete = reference_complete & (
                    ~phase_synced_nonloop | physical_stage_complete
                )
                course_suffix_complete = (
                    motion_complete & at_last_stage & reached_finish
                )
                course_complete = course_suffix_complete & (
                    self.course_start_stage_ids == 0
                )
                # Do not let the reference sequence run through every terrain
                # while a high-reward policy remains near the starting line.
                advance = motion_complete & reached_stage_end & ~at_last_stage
                course_stage_advance = advance.copy()
                transition_steps = max(0, int(self.cfg.motion_transition_blend_steps))
                if transition_steps > 0 and np.any(advance):
                    self.motion_transition_from_ctrl[advance] = self.last_reference_ctrl[
                        advance
                    ]
                    self.motion_transition_steps_left[advance] = transition_steps
                incomplete_motion_required_x = np.where(
                    at_last_stage,
                    self.course_finish_x
                    - self.more_terrain_cfg.course_completion_tolerance,
                    stage_end_x - float(self.cfg.terrain_course_stage_tolerance),
                ).astype(np.float32)
                incomplete_motion_end_candidate = (
                    bool(self.cfg.terminate_incomplete_motion_end)
                    & motion_complete
                    & ~active_looping
                    & ~advance
                    & ~course_suffix_complete
                )
                self.course_stage_ids[advance] += 1
                if np.any(advance):
                    self.motion_ids[advance] = self.course_motion_sequences[
                        self.course_variant_ids[advance], self.course_stage_ids[advance]
                    ]
                looping = self.motion.motion_loop[self.motion_ids]
                retry_loop = motion_complete & looping & ~reached_stage_end & ~at_last_stage
                self.motion_phase = np.where(
                    advance,
                    0.0,
                    np.where(
                        retry_loop,
                        np.mod(next_phase, frame_counts),
                        np.minimum(next_phase, frame_counts - 1.0),
                    ),
                ).astype(np.float32)
            else:
                looping = self.motion.motion_loop[self.motion_ids]
                self.motion_phase = np.where(
                    looping,
                    np.mod(next_phase, frame_counts),
                    np.minimum(next_phase, frame_counts - 1.0),
                ).astype(np.float32)
                motion_complete = (~looping) & (next_phase >= frame_counts - 1.0)
        else:
            self.motion_phase = (
                self.motion_phase + self.motion.frequency * self.dt * phase_scale
            ) % self.motion.frame_count
            motion_complete = np.zeros(self.num_envs, dtype=bool)
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
            transitioned_reference_ctrl = self._transition_reference_ctrl(
                reference_targets.tendon_ctrl[env_id], env_id
            )
            base_ctrl = self._reference_base_ctrl(transitioned_reference_ctrl, data)
            start_x = float(data.qpos[0])
            mechanical_energy = 0.0
            mechanical_energy_invalid = False
            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[self.pelvis_body_id] = self.push_force[env_id]
            applied_action = commanded_action
            for _ in range(self.cfg.decimation):
                applied_action = self._delayed_action(env_id, commanded_action)
                data.ctrl[:] = self.action_to_ctrl(applied_action, base_ctrl=base_ctrl, env_id=env_id)
                mujoco.mj_step(self.model, data)
                substep_energy = float(
                    np.sum(np.abs(data.actuator_force * data.actuator_velocity))
                    * float(self.cfg.timestep)
                )
                if np.isfinite(substep_energy):
                    mechanical_energy += substep_energy
                else:
                    mechanical_energy_invalid = True

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
                (
                    float(reference_targets.root_lin_vel[env_id, 0])
                    if self.cfg.use_reference_velocity
                    else float(self.cfg.target_velocity) * float(phase_scale[env_id])
                ),
                reference_targets,
                foot_pos,
                foot_vel,
                foot_world_pos,
                foot_world_vel,
                foot_geom_contact,
                foot_contact,
            )
            invalid_step = bool(
                mechanical_energy_invalid
                or reward_parts.get("invalid_state", 0.0) > 0.5
            )
            if invalid_step:
                reward = -2.0
                done = True
                timed_out = False
                mechanical_energy = 0.0
                reward_parts = {key: 0.0 for key in reward_parts}
                reward_parts["invalid_state"] = 1.0
                reward_parts["termination"] = 1.0
            rewards[env_id] = reward
            reward_parts.update(
                {
                    "actuator_power": mechanical_energy / max(self.dt, 1.0e-6),
                    "mechanical_energy": mechanical_energy,
                    "course_phase_sync_limited": float(
                        course_phase_sync_limited[env_id]
                    ),
                    "course_stage_progress": float(
                        course_stage_progress[env_id]
                    ),
                }
            )
            if self.more_terrain_cfg.serial_course:
                rewards[env_id] += (
                    float(self.cfg.terrain_course_completion_reward)
                    * float(course_suffix_complete[env_id])
                    + float(self.cfg.terrain_course_stage_reward)
                    * float(course_stage_advance[env_id])
                )
            successful_motion_end = (
                bool(course_suffix_complete[env_id])
                if self.more_terrain_cfg.serial_course
                else bool(motion_complete[env_id])
            )
            incomplete_motion_end = bool(
                incomplete_motion_end_candidate[env_id]
                and float(data.qpos[0]) < float(incomplete_motion_required_x[env_id])
            )
            incomplete_motion_end_flags[env_id] = incomplete_motion_end
            reward_parts["incomplete_motion_end"] = float(incomplete_motion_end)
            dones[env_id] = done or successful_motion_end or incomplete_motion_end
            time_outs[env_id] = timed_out or successful_motion_end
            episode_steps[env_id] = self.episode_step[env_id]
            self.last_action[env_id] = commanded_action
            self.last_foot_pos[env_id] = foot_pos
            self.last_foot_world_pos[env_id] = foot_world_pos
            self.last_foot_contact[env_id] = foot_contact
            for key, value in reward_parts.items():
                components.setdefault(key, []).append(float(value))
            if self.more_terrain_cfg.serial_course:
                components.setdefault("course_completion", []).append(
                    float(course_suffix_complete[env_id])
                )
                components.setdefault("course_stage_completion", []).append(
                    float(course_stage_advance[env_id])
                )

        progress_speed, insufficient_progress, next_anchor_x, next_anchor_step = (
            _forward_progress_guard(
                np.asarray([data.qpos[0] for data in self.datas], dtype=np.float32),
                self.episode_step,
                self.forward_progress_anchor_x,
                self.forward_progress_anchor_step,
                dt=self.dt,
                min_speed=float(self.cfg.min_forward_progress_speed),
                grace_steps=int(self.cfg.forward_progress_grace_steps),
                window_steps=int(self.cfg.forward_progress_window_steps),
            )
        )
        self.forward_progress_anchor_x[:] = next_anchor_x
        self.forward_progress_anchor_step[:] = next_anchor_step
        forward_progress_failure = insufficient_progress | incomplete_motion_end_flags
        rewards -= (
            float(self.cfg.forward_progress_termination_penalty)
            * forward_progress_failure.astype(np.float32)
        )
        dones |= insufficient_progress
        components["forward_progress_window_speed"] = progress_speed.tolist()
        components["insufficient_progress"] = insufficient_progress.astype(np.float32).tolist()

        reported_course_stage = self.course_stage_ids.copy()
        reported_course_start_stage = self.course_start_stage_ids.copy()
        done_ids = np.nonzero(dones)[0].astype(np.int32)
        if len(done_ids) > 0:
            self.reset(done_ids)
        else:
            self.last_obs = self.observe()

        info = {key: np.asarray(value, dtype=np.float32) for key, value in components.items()}
        info["episode_step"] = episode_steps
        info["time_outs"] = time_outs.astype(np.float32)
        if self.more_terrain_cfg.serial_course:
            info["course_complete"] = course_complete.astype(np.float32)
            info["course_suffix_complete"] = course_suffix_complete.astype(np.float32)
            info["course_stage"] = reported_course_stage.astype(np.float32)
            info["course_start_stage"] = reported_course_start_stage.astype(np.float32)
            info["successes"] = course_complete.astype(np.float32)
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
        foot_world_pos: np.ndarray,
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
        foot_slip = self._foot_slip_penalty(
            foot_world_vel,
            foot_geom_contact,
        )
        contact_track = float(
            np.clip(1.0 - np.mean(np.abs(foot_contact - reference_targets.foot_contact[env_id])), 0.0, 1.0)
        )
        missed_contact = float(
            np.mean(reference_targets.foot_contact[env_id] * (1.0 - foot_contact))
        )
        stair_terms = (
            self._stair_terrain_reward_terms(
                foot_world_pos,
                foot_contact,
                reference_targets.foot_contact[env_id],
                env_id=env_id,
                vertical_velocity=float(data.qvel[2]),
                reference_vertical_velocity=float(reference_targets.root_lin_vel[env_id, 2]),
            )
            if self.cfg.reward_mode == "stairs"
            else {}
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

        if self.cfg.reward_mode == "stairs":
            # Stair locomotion is intentionally less dominated by flat-ground
            # pose symmetry.  It explicitly rewards matching vertical motion,
            # clearing the next riser, and supporting the body on a tread.
            reward = (
                0.05
                + 1.15 * pose_track
                + 0.35 * vel_track
                + 1.20 * height
                + 1.00 * root_orientation
                + 0.55 * root_vel
                + self.cfg.forward_velocity_reward_coef * stable_forward_velocity_track
                + self.cfg.forward_progress_reward_coef * stable_forward_progress
                - self.cfg.forward_velocity_deficit_penalty_coef * forward_velocity_deficit
                - self.cfg.forward_velocity_excess_penalty_coef * forward_velocity_excess
                + 0.65 * foot_pos_track
                + 0.40 * foot_vel_track
                + self.cfg.stance_foot_reward_coef * stance_foot_track
                - self.cfg.foot_slip_penalty_coef * foot_slip
                + self.cfg.contact_reward_coef * contact_track
                + 0.35 * distal_vel_track
                + 0.25 * ctrl_track
                + self.cfg.stair_vertical_velocity_reward_coef
                * stair_terms["stair_vertical_velocity_track"]
                + self.cfg.stair_swing_clearance_reward_coef
                * stair_terms["stair_swing_clearance"]
                + self.cfg.stair_tread_support_reward_coef
                * stair_terms["stair_tread_support"]
                - self.cfg.ctrl_penalty_coef * ctrl_penalty
                - self.cfg.action_rate_penalty_coef * action_rate
                - self.cfg.distal_action_penalty_coef * distal_action
                - self.cfg.distal_action_rate_penalty_coef * distal_action_rate
                - self.cfg.missed_contact_penalty_coef * missed_contact
            )
        else:
            reward = (
                0.05
                + 1.40 * pose_track
                + 0.45 * vel_track
                + 0.75 * height
                + 0.90 * root_orientation
                + 0.90 * root_vel
                + self.cfg.forward_velocity_reward_coef * stable_forward_velocity_track
                + self.cfg.forward_progress_reward_coef * stable_forward_progress
                - self.cfg.forward_velocity_deficit_penalty_coef * forward_velocity_deficit
                - self.cfg.forward_velocity_excess_penalty_coef * forward_velocity_excess
                + 1.25 * foot_pos_track
                + self.cfg.side_foot_pos_coef * side_metrics["side_foot_pos"]
                + self.cfg.step_gap_coef * side_metrics["step_gap"]
                + self.cfg.swing_advance_coef * side_metrics["swing_advance"]
                + self.cfg.foot_vel_reward_coef * foot_vel_track
                + self.cfg.stance_foot_reward_coef * stance_foot_track
                - self.cfg.foot_slip_penalty_coef * foot_slip
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

        if self.cfg.amp_task_only:
            # The retargeted sequence must shape this policy only through AMP.
            # Touchdown is rewarded when it alternates from the foot that was
            # supporting the previous step; this uses simulator contacts only.
            previous_contact = self.last_foot_contact[env_id]
            touchdown = (foot_contact > 0.5) & (previous_contact <= 0.5)
            alternating_step = float(
                (touchdown[0] and previous_contact[1] > 0.5)
                or (touchdown[1] and previous_contact[0] > 0.5)
            )
            terrain_height = (
                float(self.more_terrain_cfg.course_base_height)
                if self.more_terrain_cfg.serial_course
                else 0.0
            )
            physical_height = float(
                np.exp(
                    -((root_height - (self.nominal_root_height + terrain_height)) ** 2)
                    / (0.16**2)
                )
            )
            upright = float(
                np.clip(
                    (up_z - float(self.cfg.min_upright_z))
                    / max(1.0 - float(self.cfg.min_upright_z), 1.0e-6),
                    0.0,
                    1.0,
                )
            )
            stable_velocity = forward_velocity_track * upright * physical_height
            forward_only_progress = float(np.clip(forward_progress, 0.0, 1.0))
            support = float(np.any(foot_contact > 0.5))
            reward = (
                0.05
                + 1.20 * upright
                + 0.80 * physical_height
                + self.cfg.forward_velocity_reward_coef * stable_velocity
                + self.cfg.forward_progress_reward_coef * forward_only_progress
                - self.cfg.forward_velocity_deficit_penalty_coef * forward_velocity_deficit
                - self.cfg.forward_velocity_excess_penalty_coef * forward_velocity_excess
                + self.cfg.contact_reward_coef * support
                + self.cfg.alternating_step_reward_coef * alternating_step
                - self.cfg.foot_slip_penalty_coef * foot_slip
                - self.cfg.ctrl_penalty_coef * ctrl_penalty
                - self.cfg.action_rate_penalty_coef * action_rate
                - self.cfg.distal_action_penalty_coef * distal_action
                - self.cfg.distal_action_rate_penalty_coef * distal_action_rate
            )

        bad_state = not (
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.ctrl).all()
        )
        physical_root_height = self.nominal_root_height + (
            float(self.more_terrain_cfg.course_base_height)
            if self.more_terrain_cfg.serial_course
            else 0.0
        )
        height_reference = (
            physical_root_height if self.cfg.amp_task_only else ref_root_height
        )
        min_root_height = height_reference + (
            float(self.cfg.fall_height) - self.nominal_root_height
        )
        max_root_height = height_reference + (
            float(self.cfg.max_height) - self.nominal_root_height
        )
        fallen = (
            root_height < min_root_height
            or root_height > max_root_height
            or up_z < self.cfg.min_upright_z
        )
        terrain_penetration_depth = self._terrain_penetration_depth(data)
        terrain_penetration = terrain_penetration_depth > max(
            float(self.cfg.terrain_penetration_tolerance), 0.0
        )
        reward -= (
            max(float(self.cfg.terrain_penetration_penalty_coef), 0.0)
            * terrain_penetration_depth
        )
        fallen = fallen or terrain_penetration
        stair_root_clearance = 0.0
        stair_terrain_penetration = False
        if self.cfg.reward_mode == "stairs" and self.stair_motion_mask[
            self.motion_ids[env_id]
        ]:
            stair_surface = float(
                self._stair_surface_height(
                    np.asarray([data.qpos[0]], dtype=np.float32), env_id
                )[0]
            )
            stair_root_clearance = root_height - stair_surface
            stair_terrain_penetration = (
                stair_root_clearance < float(self.cfg.stair_min_root_clearance)
            )
            fallen = fallen or stair_terrain_penetration
        components = {
            "root_height_value": root_height,
            "reference_root_height_value": ref_root_height,
            "terrain_penetration_depth": terrain_penetration_depth,
            "terrain_penetration": float(terrain_penetration),
            "stair_root_clearance_value": stair_root_clearance,
            "stair_terrain_penetration": float(stair_terrain_penetration),
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
            "foot_slip": foot_slip,
            "contact": contact_track,
            "missed_contact": missed_contact,
            **stair_terms,
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
            "alternating_step": (
                alternating_step if self.cfg.amp_task_only else 0.0
            ),
        }
        reward, bad_state, components = _sanitize_reward_outputs(
            reward,
            bad_state,
            fallen,
            components,
            fall_penalty=self.cfg.fall_penalty,
        )
        timed_out = self.episode_step[env_id] >= self.episode_timeout_step[env_id]
        done = bool(bad_state or fallen or timed_out)
        termination = bool(bad_state or fallen)
        components["termination"] = float(termination)
        return float(reward), done, bool(timed_out), components

    def _stair_terrain_reward_terms(
        self,
        foot_world_pos: np.ndarray,
        foot_contact: np.ndarray,
        reference_foot_contact: np.ndarray,
        *,
        env_id: int = 0,
        vertical_velocity: float,
        reference_vertical_velocity: float,
    ) -> dict[str, float]:
        """Return stair-surface-aware support, clearance, and ascent/descent terms."""

        if self.cfg.reward_mode != "stairs" or (
            hasattr(self, "motion_ids")
            and hasattr(self, "stair_motion_mask")
            and not self.stair_motion_mask[self.motion_ids[int(env_id)]]
        ):
            return {
                "stair_vertical_velocity_track": 0.0,
                "stair_swing_clearance": 0.0,
                "stair_tread_support": 0.0,
                "stair_min_swing_clearance_value": 0.0,
            }

        world_pos = np.asarray(foot_world_pos, dtype=np.float32)
        surface_height = LytRobotAmpEnv._stair_surface_height(
            self, world_pos[:, 0], env_id
        )
        geom_clearance = world_pos[:, 2] - surface_height
        side_clearance = np.asarray(
            [np.min(geom_clearance[0:2]), np.min(geom_clearance[2:4])],
            dtype=np.float32,
        )
        reference_contact = np.clip(
            np.asarray(reference_foot_contact, dtype=np.float32), 0.0, 1.0
        )
        actual_contact = np.clip(np.asarray(foot_contact, dtype=np.float32), 0.0, 1.0)

        stance_weight = reference_contact
        stance_sum = float(np.sum(stance_weight))
        # The contact mesh origin sits a few centimetres above/below the exact
        # collision point. A bounded Gaussian around the tread surface keeps
        # this term stable across foot and toe mesh origins.
        support_height_track = np.exp(
            -np.square(side_clearance) / (0.04**2)
        ).astype(np.float32)
        tread_support = (
            float(np.sum(stance_weight * actual_contact * support_height_track) / stance_sum)
            if stance_sum > 1.0e-6
            else 1.0
        )

        swing_weight = 1.0 - reference_contact
        swing_sum = float(np.sum(swing_weight))
        clearance_target = max(float(self.cfg.stair_swing_clearance), 1.0e-6)
        clearance_score = np.clip(side_clearance / clearance_target, 0.0, 1.0)
        swing_clearance = (
            float(np.sum(swing_weight * clearance_score) / swing_sum)
            if swing_sum > 1.0e-6
            else 1.0
        )
        min_swing_clearance = (
            float(np.min(side_clearance[swing_weight > 1.0e-6]))
            if swing_sum > 1.0e-6
            else 0.0
        )

        vertical_sigma = max(float(self.cfg.stair_vertical_velocity_sigma), 1.0e-6)
        vertical_velocity_track = float(
            np.exp(
                -((float(vertical_velocity) - float(reference_vertical_velocity)) ** 2)
                / (vertical_sigma**2)
            )
        )
        return {
            "stair_vertical_velocity_track": vertical_velocity_track,
            "stair_swing_clearance": swing_clearance,
            "stair_tread_support": tread_support,
            "stair_min_swing_clearance_value": min_swing_clearance,
        }

    def _stair_surface_height(
        self,
        x: np.ndarray | float,
        env_id: int = 0,
    ) -> np.ndarray:
        """Return the exact analytic stair tread height used by this world."""

        serial = bool(
            getattr(getattr(self, "more_terrain_cfg", None), "serial_course", False)
        )
        if not serial:
            return terrain_height_at(x, self.stair_cfg)
        variant = int(self.course_variant_ids[int(env_id)])
        return terrain_height_at(x, self.course_stair_cfgs[variant]) + float(
            self.more_terrain_cfg.course_base_height
        )

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

    def _foot_slip_penalty(
        self,
        foot_world_vel: np.ndarray,
        foot_geom_contact: np.ndarray,
    ) -> float:
        """Return bounded tangential-speed violation for contacting foot geoms.

        The existing stance reward is bounded in ``[0, 1]`` and therefore only
        removes a bonus when a foot slides.  This non-negative term gives the
        trainer an explicit cost whose coefficient can dominate persistent
        slip without penalizing a swing foot that is clear of the ground.
        """

        contact = np.clip(
            np.asarray(foot_geom_contact, dtype=np.float32), 0.0, 1.0
        )
        contact_count = float(np.sum(contact))
        if contact_count <= 1.0e-6:
            return 0.0
        horizontal_speed = np.linalg.norm(
            np.asarray(foot_world_vel, dtype=np.float32)[:, 0:2], axis=1
        )
        threshold = max(float(self.cfg.foot_slip_velocity_threshold), 0.0)
        scale = max(float(self.cfg.stance_foot_vel_sigma), 1.0e-6)
        violation = np.clip((horizontal_speed - threshold) / scale, 0.0, 4.0)
        return float(np.sum(violation * contact) / contact_count)

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
