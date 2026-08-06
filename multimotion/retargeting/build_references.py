#!/usr/bin/env python3
"""Convert OmniRetarget 14-DoF clips to the tendon-driven LYT training format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
LEGACY_RETARGET = REPO_ROOT / "squat_training" / "retargeting"
sys.path[:0] = [str(SRC_ROOT), str(LEGACY_RETARGET)]

from lytrobot_amp.terrain import (  # noqa: E402
    MoreTerrainCfg,
    StairTerrainCfg,
    load_model,
    terrain_geom_ids,
    terrain_height_from_model,
    terrain_height_at,
    terrain_kind_from_motion,
    terrain_spawn_offset,
)
from lytrobot_amp.benchmark import TRAINING_TARGET_SPEED_MPS  # noqa: E402
from prepare_walk1 import (  # noqa: E402
    HEEL_PHASE_END,
    ROLL_FLAT,
    ROLL_HEEL,
    ROLL_TOE,
    TOE_PHASE_START,
)

MODEL_XML = REPO_ROOT / "robot" / "MJCFmodelposition.xml"
OUTPUT_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"

JOINT_NAMES = (
    "root",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "patella_r_j1",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "patella_l_j1",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
)
RAW_JOINT_QPOS = np.arange(7, 21, dtype=np.int64)
MAIN_JOINT_QPOS = np.asarray(
    [7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22],
    dtype=np.int64,
)
# Foot-contact IK uses each knee plus ankle pitch, subtalar roll and toe pitch.
# The knee is needed during double support to adjust each leg height
# independently.  Keeping the hip targets fixed makes the correction local and
# still reduces finite-difference work from 15 variables to 9 including root.
CONTACT_JOINT_QPOS = np.asarray(
    [10, 12, 13, 14, 18, 20, 21, 22],
    dtype=np.int64,
)
TRAIN_STAIR_CFG = StairTerrainCfg(
    enabled=True,
    step_height=0.13,
    step_depth=0.58,
    step_count=3,
    start_x=-0.95,
)
STAIR_DOWN_OFFSET = 0.25
CONTACT_CLEARANCE = 0.0015
HEEL_STRIKE_TOE_LIFT = 0.025
TOE_OFF_HEEL_LIFT = 0.020
SWING_CLEARANCE = 0.003
# Signed MuJoCo geometry distance: zero is touching and negative is overlap.
# The post-pass removes all negative distances without unnecessarily widening
# already valid stair poses (which can lower a swing heel into the next tread).
LEG_CLEARANCE = 0.0
OUTPUT_FREQUENCY = 40.0
TURN_TIME_SCALE = 2.0
TURN_RAMP_SECONDS = 0.75
FOOT_GEOM_NAMES = ("foot_r", "toe_r", "foot_l", "toe_l")
COLLISION_BODY_PAIRS = (
    ("femur_r", "femur_l"),
    ("tibia_r", "tibia_l"),
    ("foot_r", "foot_l"),
    ("foot_r", "toe_l"),
    ("toe_r", "foot_l"),
    ("toe_r", "toe_l"),
)
TURN_TASKS = ("turn_left", "turn_right")
TERRAIN_TASKS = ("gap_cross", "pit_walk", "rough_walk", "stair_up", "stair_down")


def normalize_quaternions(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-12)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def match_demonstrated_heading(qpos: np.ndarray, target_yaw: np.ndarray) -> None:
    """Replace SQP yaw drift while retaining its demonstrated pelvis tilt."""

    quaternions = normalize_quaternions(qpos[:, 3:7])
    current_yaw = np.unwrap(
        np.arctan2(
            2.0
            * (
                quaternions[:, 0] * quaternions[:, 3]
                + quaternions[:, 1] * quaternions[:, 2]
            ),
            1.0
            - 2.0
            * (quaternions[:, 2] ** 2 + quaternions[:, 3] ** 2),
        )
    )
    delta = np.unwrap(np.asarray(target_yaw, dtype=np.float64)) - current_yaw
    correction = np.column_stack(
        (
            np.cos(0.5 * delta),
            np.zeros((len(delta), 2), dtype=np.float64),
            np.sin(0.5 * delta),
        )
    )
    w0, x0, y0, z0 = correction.T
    w1, x1, y1, z1 = quaternions.T
    qpos[:, 3:7] = normalize_quaternions(
        np.column_stack(
            (
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            )
        )
    )


def quaternion_yaw(qpos: np.ndarray) -> np.ndarray:
    quaternions = normalize_quaternions(qpos[:, 3:7])
    return np.unwrap(
        np.arctan2(
            2.0
            * (
                quaternions[:, 0] * quaternions[:, 3]
                + quaternions[:, 1] * quaternions[:, 2]
            ),
            1.0
            - 2.0
            * (quaternions[:, 2] ** 2 + quaternions[:, 3] ** 2),
        )
    )


def apply_controlled_turn_heading(qpos: np.ndarray, frequency: float) -> float:
    """Use an eased constant-rate yaw profile and return its peak rate."""

    source_yaw = quaternion_yaw(qpos)
    total_yaw = float(source_yaw[-1] - source_yaw[0])
    duration = (len(qpos) - 1) / frequency
    ramp = min(TURN_RAMP_SECONDS, 0.45 * duration)
    peak_rate = total_yaw / max(duration - ramp, 1.0e-6)
    times = np.arange(len(qpos), dtype=np.float64) / frequency
    travelled = np.empty_like(times)
    acceleration = peak_rate / ramp
    start_distance = 0.5 * peak_rate * ramp
    for index, value in enumerate(times):
        if value < ramp:
            travelled[index] = 0.5 * acceleration * value * value
        elif value <= duration - ramp:
            travelled[index] = start_distance + peak_rate * (value - ramp)
        else:
            remaining = duration - value
            travelled[index] = (
                total_yaw - 0.5 * acceleration * remaining * remaining
            )
    travelled[-1] = total_yaw
    match_demonstrated_heading(qpos, source_yaw[0] + travelled)
    return abs(float(peak_rate))


def clip_joint_limits(model: mujoco.MjModel, qpos: np.ndarray) -> None:
    for joint_id in range(model.njnt):
        if (
            model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            or not model.jnt_limited[joint_id]
        ):
            continue
        address = int(model.jnt_qposadr[joint_id])
        qpos[:, address] = np.clip(qpos[:, address], *model.jnt_range[joint_id])


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def roll_clearance_targets(phase: float, mode: int) -> np.ndarray:
    """Desired heel/toe clearances for heel -> sole -> toe support."""

    if mode == ROLL_HEEL:
        progress = smoothstep(phase / HEEL_PHASE_END)
        return np.asarray(
            [
                CONTACT_CLEARANCE,
                CONTACT_CLEARANCE + HEEL_STRIKE_TOE_LIFT * (1.0 - progress),
            ]
        )
    if mode == ROLL_FLAT:
        return np.full(2, CONTACT_CLEARANCE, dtype=np.float64)
    if mode == ROLL_TOE:
        progress = smoothstep((phase - TOE_PHASE_START) / (1.0 - TOE_PHASE_START))
        return np.asarray(
            [
                CONTACT_CLEARANCE + TOE_OFF_HEEL_LIFT * progress,
                CONTACT_CLEARANCE,
            ]
        )
    raise ValueError(f"Unexpected support roll mode: {mode}.")


def interpolate_qpos(
    qpos: np.ndarray,
    source_frequency: float,
    target_frequency: float,
    time_scale: float,
    loop: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample qpos and return nearest source indices for discrete labels."""

    if loop and np.isclose(source_frequency, target_frequency) and time_scale == 1.0:
        return np.asarray(qpos, dtype=np.float64).copy(), np.arange(len(qpos))
    source_duration = (
        len(qpos) / source_frequency if loop else (len(qpos) - 1) / source_frequency
    )
    output_duration = source_duration * time_scale
    output_count = max(2, int(round(output_duration * target_frequency)) + (0 if loop else 1))
    if loop:
        # Cover one complete cycle uniformly.  A time-scaled translating loop
        # is periodic in joint pose, but not in root XY position, so the seam
        # translation must be restored when the upper sample wraps to frame 0.
        source_positions = (
            np.arange(output_count, dtype=np.float64) * len(qpos) / output_count
        )
        low = np.floor(source_positions).astype(np.int64)
        high = (low + 1) % len(qpos)
    else:
        output_times = np.arange(output_count, dtype=np.float64) / target_frequency
        source_positions = output_times / time_scale * source_frequency
        source_positions = np.clip(source_positions, 0.0, len(qpos) - 1)
        low = np.floor(source_positions).astype(np.int64)
        high = np.minimum(low + 1, len(qpos) - 1)
    blend = (source_positions - low)[:, None]
    upper = np.asarray(qpos[high], dtype=np.float64).copy()
    if loop:
        seam = high == 0
        root_steps = np.diff(np.asarray(qpos[:, :2], dtype=np.float64), axis=0)
        cycle_translation = (
            np.asarray(qpos[-1, :2], dtype=np.float64)
            - np.asarray(qpos[0, :2], dtype=np.float64)
            + np.median(root_steps, axis=0)
        )
        upper[seam, :2] += cycle_translation
    result = (1.0 - blend) * qpos[low] + blend * upper
    quat_low = normalize_quaternions(qpos[low, 3:7])
    quat_high = normalize_quaternions(qpos[high, 3:7])
    signs = np.where(np.sum(quat_low * quat_high, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    result[:, 3:7] = normalize_quaternions(
        (1.0 - blend) * quat_low + blend * (quat_high * signs)
    )
    nearest = np.where((source_positions - low) < 0.5, low, high)
    return result, nearest.astype(np.int64)


def geom_for_body(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    geom_ids = np.flatnonzero(np.asarray(model.geom_bodyid) == body_id)
    if body_id < 0 or len(geom_ids) == 0:
        raise ValueError(f"No geometry found for body {body_name!r}.")
    return int(geom_ids[0])


def sole_landmarks(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract heel and toe bottom landmarks from the full robot STL meshes."""

    geom_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in FOOT_GEOM_NAMES
        ],
        dtype=np.int32,
    )
    if np.any(geom_ids < 0):
        raise ValueError(f"Missing one of the foot geometries: {FOOT_GEOM_NAMES}.")
    body_ids = np.asarray(model.geom_bodyid[geom_ids], dtype=np.int32)
    points = np.zeros((4, 3), dtype=np.float64)
    for index, (name, geom_id) in enumerate(zip(FOOT_GEOM_NAMES, geom_ids, strict=True)):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        stop = start + int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start:stop], dtype=np.float64)
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        vertices = model.geom_pos[geom_id] + vertices @ rotation.reshape(3, 3).T
        if name.startswith("foot"):
            x_min = float(vertices[:, 0].min())
            candidates = vertices[
                vertices[:, 0] <= x_min + 0.25 * float(np.ptp(vertices[:, 0]))
            ]
        else:
            candidates = vertices
        z_min = float(candidates[:, 2].min())
        bottom = candidates[candidates[:, 2] <= z_min + 5.0e-4]
        points[index] = np.mean(bottom, axis=0)
        points[index, 2] = z_min
    return geom_ids, body_ids, points


def landmark_positions(
    data: mujoco.MjData,
    body_ids: np.ndarray,
    local_points: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            data.xpos[body_id]
            + data.xmat[body_id].reshape(3, 3) @ local_point
            for body_id, local_point in zip(body_ids, local_points, strict=True)
        ],
        dtype=np.float64,
    )


def retarget_terrain_model(
    task: str,
    model_xml: Path,
) -> tuple[mujoco.MjModel, np.ndarray, str]:
    """Build the task's exact terrain and return its local-to-world offset."""

    kind = terrain_kind_from_motion(task)
    if kind == "stairs":
        model = load_model(model_xml, TRAIN_STAIR_CFG)
        offset = terrain_spawn_offset(
            kind,
            MoreTerrainCfg(),
            TRAIN_STAIR_CFG,
            stair_down=task == "stair_down",
            stair_down_offset=STAIR_DOWN_OFFSET,
        )
    elif kind in {"gap", "pit", "rough"}:
        more = MoreTerrainCfg(enabled=True, terrain_types=(kind,))
        model = load_model(model_xml, None, more)
        offset = terrain_spawn_offset(kind, more, StairTerrainCfg())
    else:
        model = load_model(model_xml)
        offset = np.zeros(3, dtype=np.float32)
    return model, np.asarray(offset, dtype=np.float64), kind


def refine_contacts_and_collisions(
    qpos: np.ndarray,
    contact: np.ndarray,
    roll_phase: np.ndarray,
    roll_mode: np.ndarray,
    task: str,
    model_xml: Path,
) -> tuple[np.ndarray, dict[str, float | int | list[float]]]:
    """Full-model IK for leg clearance and ordered heel/sole/toe contact."""

    model, pose_offset, terrain_kind = retarget_terrain_model(task, model_xml)
    data = mujoco.MjData(model)
    _, landmark_body_ids, landmark_points = sole_landmarks(model)
    collision_pairs = [
        (geom_for_body(model, left), geom_for_body(model, right))
        for left, right in COLLISION_BODY_PAIRS
    ]
    joint_indices = CONTACT_JOINT_QPOS
    by_address = {
        int(model.jnt_qposadr[joint_id]): joint_id for joint_id in range(model.njnt)
    }
    joint_ids = [by_address[int(index)] for index in joint_indices]
    joint_lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in joint_ids])
    joint_upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in joint_ids])
    refined = np.asarray(qpos, dtype=np.float64).copy()
    fromto = np.zeros(6, dtype=np.float64)

    def collision_distance(pose: np.ndarray) -> float:
        evaluated = pose.copy()
        evaluated[:3] += pose_offset
        data.qpos[:] = evaluated
        mujoco.mj_forward(model, data)
        return float(
            min(
                mujoco.mj_geomDistance(
                    model, data, geom_a, geom_b, 0.20, fromto
                )
                for geom_a, geom_b in collision_pairs
            )
        )

    def ground_heights(xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        return terrain_height_from_model(
            model,
            data,
            points[:, 0],
            points[:, 1],
        )

    minimum_before = np.inf
    minimum_after = np.inf
    roll_errors: list[float] = []
    heel_order_values: list[float] = []
    toe_order_values: list[float] = []
    previous = np.concatenate(([refined[-1, 2]], refined[-1, joint_indices]))

    for frame in range(len(refined)):
        nominal_pose = refined[frame].copy()
        nominal_world = nominal_pose.copy()
        nominal_world[:3] += pose_offset
        data.qpos[:] = nominal_world
        mujoco.mj_forward(model, data)
        nominal_landmarks = landmark_positions(data, landmark_body_ids, landmark_points)
        fixed_ground = ground_heights(nominal_landmarks[:, :2])
        nominal_clearance = nominal_landmarks[:, 2] - fixed_ground
        minimum_before = min(minimum_before, float(nominal_clearance.min()))

        support_constraints: list[tuple[slice, np.ndarray, int]] = []
        root_offsets: list[float] = []
        for side in np.flatnonzero(contact[frame] > 0.5):
            support_slice = slice(2 * int(side), 2 * int(side) + 2)
            targets = roll_clearance_targets(
                float(roll_phase[frame, side]),
                int(roll_mode[frame, side]),
            )
            support_constraints.append((support_slice, targets, int(side)))
            root_offsets.extend(targets - nominal_clearance[support_slice])
        if not support_constraints:
            lower_side = int(np.argmin([nominal_clearance[:2].min(), nominal_clearance[2:].min()]))
            support_slice = slice(2 * lower_side, 2 * lower_side + 2)
            targets = np.full(2, CONTACT_CLEARANCE, dtype=np.float64)
            support_constraints.append((support_slice, targets, lower_side))
            root_offsets.extend(targets - nominal_clearance[support_slice])

        nominal_root = float(nominal_pose[2] + np.mean(root_offsets))
        nominal_variables = np.concatenate(
            ([nominal_root], nominal_pose[joint_indices])
        )
        variable_lower = np.concatenate(([nominal_root - 0.05], joint_lower))
        variable_upper = np.concatenate(([nominal_root + 0.05], joint_upper))

        def residual(variables: np.ndarray) -> np.ndarray:
            pose = nominal_pose.copy()
            pose[:3] += pose_offset
            pose[2] = variables[0]
            pose[joint_indices] = variables[1:]
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            landmarks = landmark_positions(data, landmark_body_ids, landmark_points)
            current_ground = ground_heights(landmarks[:, :2])
            clearance = landmarks[:, 2] - current_ground
            support_error = np.concatenate(
                [
                    clearance[support_slice] - targets
                    for support_slice, targets, _ in support_constraints
                ]
            )
            support_x_error = np.concatenate(
                [
                    landmarks[support_slice, 0]
                    - nominal_landmarks[support_slice, 0]
                    for support_slice, _, _ in support_constraints
                ]
            )
            # Use the complete STL geometry for foot/leg non-interference.
            current_clearance = clearance
            ground_penetration = np.minimum(
                current_clearance - 0.25 * CONTACT_CLEARANCE,
                0.0,
            )
            return np.concatenate(
                (
                    30.0 * support_error,
                    4.0 * support_x_error,
                    18.0 * ground_penetration,
                    [0.18 * (variables[0] - nominal_root)],
                    0.035 * (variables[1:] - nominal_variables[1:]),
                    [0.03 * (variables[0] - previous[0])],
                    0.012 * (variables[1:] - previous[1:]),
                )
            )

        solution = least_squares(
            residual,
            np.clip(nominal_variables, variable_lower, variable_upper),
            bounds=(variable_lower, variable_upper),
            max_nfev=60,
            ftol=1.0e-7,
            xtol=1.0e-7,
            gtol=1.0e-7,
        )
        refined[frame, 2] = solution.x[0]
        refined[frame, joint_indices] = solution.x[1:]
        previous = solution.x.copy()

        evaluated = refined[frame].copy()
        evaluated[:3] += pose_offset
        data.qpos[:] = evaluated
        mujoco.mj_forward(model, data)
        landmarks = landmark_positions(data, landmark_body_ids, landmark_points)
        clearance = landmarks[:, 2] - ground_heights(landmarks[:, :2])
        minimum_after = min(minimum_after, float(clearance.min()))
        for support_slice, targets, _ in support_constraints:
            roll_errors.extend(np.abs(clearance[support_slice] - targets))
        for side in range(2):
            mode = int(roll_mode[frame, side])
            if contact[frame, side] <= 0.5:
                continue
            toe_minus_heel = float(clearance[2 * side + 1] - clearance[2 * side])
            if mode == ROLL_HEEL:
                heel_order_values.append(toe_minus_heel)
            elif mode == ROLL_TOE:
                toe_order_values.append(toe_minus_heel)

    # A turn can legitimately cross the swing-foot path in the source human
    # motion, but the wider robot feet must not occupy the same volume.  Apply
    # a smooth outward hip-abduction correction only where the full STL
    # geometry violates the requested clearance.  The stance leg is left
    # untouched whenever one foot is in swing, preserving the contact IK.
    hip_adduction_indices = (8, 16)
    hip_lower = np.asarray(
        [
            model.jnt_range[
                by_address[index],
                0,
            ]
            for index in hip_adduction_indices
        ]
    )
    total_abduction = np.zeros((len(refined), 2), dtype=np.float64)
    for _ in range(3):
        required = np.zeros_like(total_abduction)
        for frame, pose in enumerate(refined):
            if collision_distance(pose) >= LEG_CLEARANCE:
                continue
            swing = np.flatnonzero(contact[frame] <= 0.5)
            sides = swing.tolist() if len(swing) == 1 else [0, 1]
            candidate = pose.copy()
            for _step in range(40):
                for side in sides:
                    address = hip_adduction_indices[side]
                    candidate[address] = max(
                        candidate[address] - 0.01,
                        hip_lower[side],
                    )
                if collision_distance(candidate) >= LEG_CLEARANCE:
                    break
            for side in sides:
                required[frame, side] = (
                    candidate[hip_adduction_indices[side]]
                    - pose[hip_adduction_indices[side]]
                )
        if not np.any(required < 0.0):
            break
        smoothed = required.copy()
        for frame, side in np.argwhere(required < 0.0):
            delta = required[frame, side]
            for offset in range(-4, 5):
                neighbor = frame + offset
                if not 0 <= neighbor < len(refined):
                    continue
                taper = 1.0 - abs(offset) / 5.0
                smoothed[neighbor, side] = min(
                    smoothed[neighbor, side],
                    delta * taper,
                )
        for side, address in enumerate(hip_adduction_indices):
            refined[:, address] = np.maximum(
                refined[:, address] + smoothed[:, side],
                hip_lower[side],
            )
        total_abduction += smoothed

    # Recompute all quality statistics after the abduction correction.
    collision_minima: list[float] = []
    minimum_after = np.inf
    roll_errors = []
    heel_order_values = []
    toe_order_values = []
    for pose in refined:
        collision_minima.append(collision_distance(pose))
    for frame, pose in enumerate(refined):
        evaluated = pose.copy()
        evaluated[:3] += pose_offset
        data.qpos[:] = evaluated
        mujoco.mj_forward(model, data)
        landmarks = landmark_positions(data, landmark_body_ids, landmark_points)
        clearance = landmarks[:, 2] - ground_heights(landmarks[:, :2])
        minimum_after = min(minimum_after, float(clearance.min()))
        for side in np.flatnonzero(contact[frame] > 0.5):
            support_slice = slice(2 * int(side), 2 * int(side) + 2)
            targets = roll_clearance_targets(
                float(roll_phase[frame, side]),
                int(roll_mode[frame, side]),
            )
            roll_errors.extend(np.abs(clearance[support_slice] - targets))
            toe_minus_heel = float(
                clearance[2 * int(side) + 1] - clearance[2 * int(side)]
            )
            if int(roll_mode[frame, side]) == ROLL_HEEL:
                heel_order_values.append(toe_minus_heel)
            elif int(roll_mode[frame, side]) == ROLL_TOE:
                toe_order_values.append(toe_minus_heel)
    return refined, {
        "terrain_kind": terrain_kind,
        "terrain_pose_offset": pose_offset.tolist(),
        "contact_clearance": CONTACT_CLEARANCE,
        "heel_strike_toe_lift": HEEL_STRIKE_TOE_LIFT,
        "toe_off_heel_lift": TOE_OFF_HEEL_LIFT,
        "minimum_landmark_clearance_before": float(minimum_before),
        "minimum_landmark_clearance_after": float(minimum_after),
        "maximum_roll_target_error": float(max(roll_errors, default=0.0)),
        "heel_phase_toe_minus_heel_min": float(
            min(heel_order_values, default=0.0)
        ),
        "toe_phase_toe_minus_heel_max": float(max(toe_order_values, default=0.0)),
        "minimum_cross_leg_geometry_distance": float(min(collision_minima)),
        "collision_clearance_target": LEG_CLEARANCE,
        "maximum_collision_abduction_correction": float(
            np.max(np.abs(total_abduction))
        ),
        "ik_success_frames": int(len(refined)),
    }


def velocities(
    model: mujoco.MjModel, qpos: np.ndarray, frequency: float, loop: bool
) -> np.ndarray:
    """Differentiate poses without wrapping away a loop's forward travel.

    A locomotion cycle repeats its joint pose but not its horizontal root
    position.  Treating the final and first root positions as adjacent without
    the cycle displacement creates a large backwards velocity at the seam.
    """

    result = np.zeros((len(qpos), model.nv), dtype=np.float64)
    dt = 1.0 / frequency
    cycle_translation = np.zeros(2, dtype=np.float64)
    if loop and len(qpos) >= 3:
        root_steps = np.diff(np.asarray(qpos[:, :2], dtype=np.float64), axis=0)
        cycle_translation = (
            np.asarray(qpos[-1, :2], dtype=np.float64)
            - np.asarray(qpos[0, :2], dtype=np.float64)
            + np.median(root_steps, axis=0)
        )
    for index in range(len(qpos)):
        if loop:
            previous = qpos[(index - 1) % len(qpos)].copy()
            following = qpos[(index + 1) % len(qpos)].copy()
            if index == 0:
                previous[:2] -= cycle_translation
            elif index == len(qpos) - 1:
                following[:2] += cycle_translation
            interval = 2.0 * dt
        elif index == 0:
            previous, following, interval = qpos[0], qpos[1], dt
        elif index == len(qpos) - 1:
            previous, following, interval = qpos[-2], qpos[-1], dt
        else:
            previous, following, interval = qpos[index - 1], qpos[index + 1], 2.0 * dt
        mujoco.mj_differentiatePos(model, result[index], interval, previous, following)
    return result


def forward_speed(qpos: np.ndarray, frequency: float, loop: bool) -> float:
    """Return net world-X speed, including a translating loop's seam step."""

    duration = (len(qpos) - (0 if loop else 1)) / max(float(frequency), 1.0e-6)
    displacement = float(qpos[-1, 0] - qpos[0, 0])
    if loop and len(qpos) >= 2:
        displacement += float(np.median(np.diff(qpos[:, 0])))
    return displacement / max(duration, 1.0e-6)


def match_forward_speed(
    qpos: np.ndarray,
    frequency: float,
    loop: bool,
    target_speed: float,
) -> float:
    """Scale only root world-X travel so the discrete clip matches target."""

    measured = forward_speed(qpos, frequency, loop)
    if measured <= 0.0 or target_speed <= 0.0:
        raise ValueError(
            f"forward speed must be positive, got measured={measured}, "
            f"target={target_speed}."
        )
    origin = float(qpos[0, 0])
    qpos[:, 0] = origin + (qpos[:, 0] - origin) * (float(target_speed) / measured)
    return forward_speed(qpos, frequency, loop)


def align_stair_contacts(
    qpos: np.ndarray,
    contact: np.ndarray,
    task: str,
    model_xml: Path,
) -> dict[str, float]:
    if task not in {"stair_up", "stair_down"}:
        return {}
    terrain_model = load_model(model_xml, TRAIN_STAIR_CFG)
    terrain_ids = terrain_geom_ids(terrain_model)
    foot_ids = np.asarray(
        [
            mujoco.mj_name2id(terrain_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("foot_r", "toe_r", "foot_l", "toe_l")
        ],
        dtype=np.int32,
    )
    data = mujoco.MjData(terrain_model)
    fromto = np.zeros(6, dtype=np.float64)
    x_offset = (
        TRAIN_STAIR_CFG.descent_start + STAIR_DOWN_OFFSET
        if task == "stair_down"
        else 0.0
    )
    corrections = np.zeros(len(qpos), dtype=np.float64)
    for frame in range(len(qpos)):
        active_ids: list[int] = []
        if contact[frame, 0] > 0.5:
            active_ids.extend(int(value) for value in foot_ids[:2])
        if contact[frame, 1] > 0.5:
            active_ids.extend(int(value) for value in foot_ids[2:])
        if not active_ids:
            continue
        data.qpos[:] = qpos[frame]
        data.qpos[0] += x_offset
        mujoco.mj_forward(terrain_model, data)
        distances = [
            mujoco.mj_geomDistance(
                terrain_model,
                data,
                int(terrain_id),
                foot_id,
                1.0,
                fromto,
            )
            for terrain_id in terrain_ids
            for foot_id in active_ids
        ]
        correction = CONTACT_CLEARANCE - float(np.min(distances))
        qpos[frame, 2] += correction
        corrections[frame] = correction
    return {
        "terrain_contact_correction_min": float(corrections.min()),
        "terrain_contact_correction_max": float(corrections.max()),
        "terrain_contact_correction_rms": float(np.sqrt(np.mean(corrections**2))),
        "terrain_contact_clearance": CONTACT_CLEARANCE,
        "terrain_step_height": TRAIN_STAIR_CFG.step_height,
        "terrain_step_depth": TRAIN_STAIR_CFG.step_depth,
        "terrain_step_count": TRAIN_STAIR_CFG.step_count,
        "terrain_down_offset": STAIR_DOWN_OFFSET,
    }


def build(
    raw_path: Path,
    output: Path,
    model: mujoco.MjModel,
    model_xml: Path,
    target_walk_speed: float = TRAINING_TARGET_SPEED_MPS,
) -> None:
    with np.load(raw_path) as archive:
        raw = np.asarray(archive["qpos"], dtype=np.float64)
        frequency = float(archive["fps"])
        contact = np.asarray(archive["foot_contact"], dtype=np.float32)
        roll_phase = np.asarray(archive["foot_roll_phase"], dtype=np.float32)
        roll_mode = np.asarray(archive["foot_roll_mode"], dtype=np.int8)
        target_yaw = np.asarray(archive["target_yaw"], dtype=np.float64)
        task = str(archive["task"])
        loop = bool(np.asarray(archive["loop"]).item())
        source = str(archive["source"])
        source_url = str(archive["source_url"])
        source_frames = np.asarray(archive["source_frames"], dtype=np.int32)
        retargeter = str(archive["retargeter"])
        retargeter_commit = str(archive["retargeter_commit"])
        cost = float(np.asarray(archive["cost"]).item())
    if raw.shape[1] != 21:
        raise ValueError(f"{raw_path}: expected OmniRetarget nq=21, got {raw.shape}.")
    # Prepared cyclic clips contain an explicit duplicate endpoint. The final
    # SQP frame is also the hardest foot-sticking seam; keep one unique cycle.
    if loop:
        raw = raw[:-1]
        contact = contact[:-1]
        roll_phase = roll_phase[:-1]
        roll_mode = roll_mode[:-1]
        target_yaw = target_yaw[:-1]

    qpos = np.repeat(np.asarray(model.qpos0)[None, :], len(raw), axis=0)
    qpos[:, :7] = raw[:, :7]
    qpos[:, MAIN_JOINT_QPOS] = raw[:, RAW_JOINT_QPOS]
    qpos[:, 3:7] = normalize_quaternions(qpos[:, 3:7])
    match_demonstrated_heading(qpos, target_yaw)
    qpos[:, :2] -= qpos[0, :2]
    is_turn = task in TURN_TASKS
    demonstrated_turn_degrees = float(
        np.rad2deg(np.unwrap(target_yaw)[-1] - np.unwrap(target_yaw)[0])
    )
    if is_turn and abs(demonstrated_turn_degrees) > 90.0 + 1.0e-3:
        raise ValueError(
            f"{task}: demonstrated turn is {demonstrated_turn_degrees:.3f} degrees; "
            "the limit is 90 degrees."
        )
    demonstrated_speed = forward_speed(qpos, frequency, loop)
    time_scale = TURN_TIME_SCALE if is_turn else 1.0
    if task == "walk":
        if demonstrated_speed <= 0.0 or target_walk_speed <= 0.0:
            raise ValueError(
                f"walk speed must be positive, got demonstrated={demonstrated_speed}, "
                f"target={target_walk_speed}."
            )
        time_scale = demonstrated_speed / float(target_walk_speed)
    qpos, nearest = interpolate_qpos(
        qpos,
        source_frequency=frequency,
        target_frequency=OUTPUT_FREQUENCY,
        time_scale=time_scale,
        loop=loop,
    )
    controlled_turn_peak_rate = (
        apply_controlled_turn_heading(qpos, OUTPUT_FREQUENCY)
        if is_turn
        else None
    )
    contact = contact[nearest]
    roll_phase = roll_phase[nearest]
    roll_mode = roll_mode[nearest]
    frequency = OUTPUT_FREQUENCY
    clip_joint_limits(model, qpos)
    terrain_alignment = align_stair_contacts(qpos, contact, task, model_xml)
    qpos, refinement = refine_contacts_and_collisions(
        qpos,
        contact,
        roll_phase,
        roll_mode,
        task,
        model_xml,
    )
    clip_joint_limits(model, qpos)
    retargeted_forward_speed = forward_speed(qpos, frequency, loop)
    if task == "walk":
        retargeted_forward_speed = match_forward_speed(
            qpos,
            frequency,
            loop,
            target_walk_speed,
        )
    qvel = velocities(model, qpos, frequency, loop)
    metadata = {
        "task": task,
        "loop": loop,
        "source": source,
        "source_url": source_url,
        "source_frames": source_frames.tolist(),
        "retargeter": retargeter,
        "retargeter_url": "https://github.com/amazon-far/holosoma",
        "retargeter_commit": retargeter_commit,
        "omniretarget_final_cost": cost,
        "frequency": frequency,
        "time_scale": time_scale,
        "demonstrated_forward_speed_mps": demonstrated_speed,
        "target_forward_speed_mps": (
            float(target_walk_speed) if task == "walk" else None
        ),
        "retargeted_forward_speed_mps": retargeted_forward_speed,
        "turn_yaw_rate_limit_method": (
            "2x time stretch plus 0.75s eased constant-rate yaw"
            if is_turn
            else "not applicable"
        ),
        "controlled_turn_peak_yaw_rate_rad_s": controlled_turn_peak_rate,
        "demonstrated_turn_degrees": demonstrated_turn_degrees if is_turn else None,
        "contact_sequence": "heel -> full sole -> toe",
        **refinement,
        **terrain_alignment,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        joint_names=np.asarray(JOINT_NAMES),
        frequency=np.float64(frequency),
        foot_contact=contact,
        foot_roll_phase=roll_phase,
        foot_roll_mode=roll_mode,
        task=np.asarray(task),
        loop=np.asarray(loop),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    print(
        f"{task}: {len(qpos)} frames @ {frequency:g}Hz, "
        f"root_z=[{qpos[:, 2].min():.3f}, {qpos[:, 2].max():.3f}], "
        f"forward_speed={retargeted_forward_speed:.3f}m/s, "
        f"leg_clearance={refinement['minimum_cross_leg_geometry_distance']:.4f}m, "
        f"roll_error={refinement['maximum_roll_target_error']:.4f}m -> {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", type=Path, default=MODEL_XML)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--target-walk-speed",
        type=float,
        default=TRAINING_TARGET_SPEED_MPS,
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        choices=("walk", "turn_left", "turn_right", "stair_up", "stair_down"),
        default=("walk", "turn_left", "turn_right", "stair_up", "stair_down"),
    )
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    for clip in args.clips:
        build(
            args.results_dir / f"{clip}_lyt_raw.npz",
            args.output_dir / f"{clip}_omniretarget.npz",
            model,
            args.model_xml,
            args.target_walk_speed,
        )


if __name__ == "__main__":
    main()
