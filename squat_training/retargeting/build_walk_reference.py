#!/usr/bin/env python3
"""Convert a retargeted 14-DoF walk cycle to the main robot motion format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from build_training_reference import (
    clip_joint_limits,
    cyclic_velocity,
    make_cyclic,
    resample_qpos,
)
from prepare_walk1 import (
    HEEL_PHASE_END,
    ROLL_FLAT,
    ROLL_HEEL,
    ROLL_TOE,
    TOE_PHASE_START,
    foot_roll_schedule,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

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

# The retarget model omits the two passive patella slide joints.  Map the 14
# actuated joints around those two qpos slots in the full tendon-driven model.
RAW_JOINT_QPOS = np.arange(7, 21, dtype=np.int64)
MAIN_JOINT_QPOS = np.asarray(
    [7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22], dtype=np.int64
)
FOOT_GEOM_NAMES = ("foot_r", "toe_r", "foot_l", "toe_l")
HEEL_STRIKE_TOE_LIFT = 0.018
HEEL_STRIKE_GEOM_CLEARANCE = 0.012
TOE_OFF_HEEL_LIFT = 0.015
SWING_FOOT_MIN_CLEARANCE = 0.003
LANDING_APPROACH_FRAMES = 10
LANDING_APPROACH_HEEL_ADVANCE = 0.008
LANDING_APPROACH_START_CLEARANCE = 0.050
LANDING_HEEL_CLEARANCE = 0.004
EARLY_SWING_FRAMES = 10
EARLY_SWING_TEMPORAL_WEIGHT = 0.12
OUTGOING_STANCE_TEMPORAL_WEIGHT = 0.10


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Normalize a quaternion sequence while keeping a continuous sign."""
    result = np.asarray(quaternions, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-12)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def smooth_periodic_qpos(qpos: np.ndarray, window: int) -> np.ndarray:
    """Low-pass a cycle without introducing a discontinuity at its seam."""
    if window < 5 or window % 2 == 0:
        raise ValueError("--smooth-window must be an odd integer of at least 5.")
    if len(qpos) < 2 or not np.allclose(qpos[0], qpos[-1], atol=1.0e-10):
        raise ValueError("Periodic smoothing requires an explicit duplicate endpoint.")

    cycle = np.asarray(qpos[:-1], dtype=np.float64).copy()
    if window > len(cycle):
        raise ValueError(
            f"--smooth-window={window} exceeds the {len(cycle)} unique cycle frames."
        )
    cycle[:, 3:7] = normalize_quaternions(cycle[:, 3:7])
    cycle = savgol_filter(cycle, window, polyorder=3, axis=0, mode="wrap")
    cycle[:, 3:7] = normalize_quaternions(cycle[:, 3:7])
    return np.vstack((cycle, cycle[0]))


def foot_geom_ids(model: mujoco.MjModel) -> tuple[int, np.ndarray]:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    geom_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in FOOT_GEOM_NAMES
        ],
        dtype=np.int32,
    )
    if floor_id < 0 or np.any(geom_ids < 0):
        raise ValueError(
            "The main model must define floor, foot_r, toe_r, foot_l and toe_l geoms."
        )
    return floor_id, geom_ids


def sole_landmarks(model: mujoco.MjModel, geom_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed heel/toe bottom landmarks from the full-model STL meshes."""
    body_ids = model.geom_bodyid[geom_ids].astype(np.int32)
    points = np.zeros((len(geom_ids), 3), dtype=np.float64)
    for index, geom_id in enumerate(geom_ids):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        stop = start + int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start:stop], dtype=np.float64)
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        body_vertices = model.geom_pos[geom_id] + vertices @ rotation.reshape(3, 3).T
        if FOOT_GEOM_NAMES[index].startswith("foot"):
            x_min = float(body_vertices[:, 0].min())
            x_range = float(np.ptp(body_vertices[:, 0]))
            candidates = body_vertices[body_vertices[:, 0] <= x_min + 0.25 * x_range]
        else:
            candidates = body_vertices
        z_min = float(candidates[:, 2].min())
        bottom = candidates[candidates[:, 2] <= z_min + 5.0e-4]
        points[index] = np.mean(bottom, axis=0)
        points[index, 2] = z_min
    return body_ids, points


def sole_landmark_heights(
    data: mujoco.MjData, body_ids: np.ndarray, local_points: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            data.xpos[body_id, 2]
            + data.xmat[body_id].reshape(3, 3)[2] @ local_point
            for body_id, local_point in zip(body_ids, local_points, strict=True)
        ],
        dtype=np.float64,
    )


def foot_clearances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    floor_id: int,
    geom_ids: np.ndarray,
) -> np.ndarray:
    """Return signed floor distances for right foot/toe then left foot/toe."""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    fromto = np.zeros(6, dtype=np.float64)
    return np.asarray(
        [
            mujoco.mj_geomDistance(model, data, floor_id, int(geom_id), 0.5, fromto)
            for geom_id in geom_ids
        ],
        dtype=np.float64,
    )


def joint_bounds_for_qpos(
    model: mujoco.MjModel, qpos_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    by_address = {int(model.jnt_qposadr[joint_id]): joint_id for joint_id in range(model.njnt)}
    joint_ids = [by_address[int(index)] for index in qpos_indices]
    lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in joint_ids])
    upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in joint_ids])
    return lower, upper


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def roll_clearance_targets(
    phase: float, mode: int, ground_clearance: float
) -> np.ndarray:
    """Desired full-model foot/toe distances for a stance roll phase."""
    if mode == ROLL_HEEL:
        progress = smoothstep(phase / HEEL_PHASE_END)
        return np.asarray(
            [ground_clearance, ground_clearance + HEEL_STRIKE_TOE_LIFT * (1.0 - progress)]
        )
    if mode == ROLL_FLAT:
        return np.full(2, ground_clearance, dtype=np.float64)
    if mode == ROLL_TOE:
        progress = smoothstep((phase - TOE_PHASE_START) / (1.0 - TOE_PHASE_START))
        return np.asarray(
            [ground_clearance + TOE_OFF_HEEL_LIFT * progress, ground_clearance]
        )
    raise ValueError(f"Invalid stance roll mode {mode}.")


def enforce_main_model_contacts(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    source_contact: np.ndarray,
    source_roll_phase: np.ndarray,
    source_roll_mode: np.ndarray,
    target_clearance: float,
    ground_sigma: float,
    temporal_weight: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float | int | list[float]],
]:
    """Ground each stance foot on the full robot STL geoms with a small smooth IK."""
    if len(qpos) < 2 or not np.allclose(qpos[0], qpos[-1], atol=1.0e-10):
        raise ValueError("Contact correction requires an explicit duplicate endpoint.")
    expected_shape = (len(qpos), 2)
    if (
        source_contact.shape != expected_shape
        or source_roll_phase.shape != expected_shape
        or source_roll_mode.shape != expected_shape
    ):
        raise ValueError(
            f"Expected contact/roll shape {expected_shape}, got "
            f"{source_contact.shape}, {source_roll_phase.shape}, {source_roll_mode.shape}."
        )
    if ground_sigma <= 0.0:
        raise ValueError("--ground-sigma must be positive.")
    if temporal_weight < 0.0:
        raise ValueError("--temporal-weight must be non-negative.")

    cycle = np.asarray(qpos[:-1], dtype=np.float64).copy()
    # Preserve natural double support: outgoing toe and incoming heel may both
    # be constrained in the same frame.
    contact = (source_contact[:-1] > 0.5).astype(np.float32)
    if np.any(np.sum(contact, axis=1) == 0):
        raise ValueError("Every walk frame must contain at least one support foot.")

    # Mark the final swing frames before each heel strike.  Contact constraints
    # alone are too late: the toe mesh can otherwise graze the floor a few
    # frames before the source contact label switches on, which looks exactly
    # like a toe-first landing in the full-model viewer.
    landing_countdown = np.zeros_like(contact, dtype=np.int16)
    early_swing_count = np.zeros_like(contact, dtype=np.int16)
    for side in range(contact.shape[1]):
        previous = np.roll(contact[:, side], 1)
        strikes = np.flatnonzero((contact[:, side] > 0.5) & (previous <= 0.5))
        for strike in strikes:
            for countdown in range(1, LANDING_APPROACH_FRAMES + 1):
                frame = (int(strike) - countdown) % len(contact)
                if contact[frame, side] > 0.5:
                    break
                landing_countdown[frame, side] = countdown
        takeoffs = np.flatnonzero((contact[:, side] <= 0.5) & (previous > 0.5))
        for takeoff in takeoffs:
            for count in range(1, EARLY_SWING_FRAMES + 1):
                frame = (int(takeoff) + count - 1) % len(contact)
                if contact[frame, side] > 0.5:
                    break
                early_swing_count[frame, side] = count

    floor_id, geom_ids = foot_geom_ids(model)
    landmark_body_ids, landmark_points = sole_landmarks(model, geom_ids)
    data = mujoco.MjData(model)

    before = np.asarray(
        [foot_clearances(model, data, pose, floor_id, geom_ids) for pose in cycle]
    )
    before_landmarks = []
    for pose in cycle:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        before_landmarks.append(
            sole_landmark_heights(data, landmark_body_ids, landmark_points)
        )
    before_landmarks = np.asarray(before_landmarks)
    frame_constraints: list[list[tuple[slice, np.ndarray, int, float, int]]] = []
    support_before_values: list[float] = []
    support_target_values: list[float] = []
    raw_root_offset = np.zeros(len(cycle), dtype=np.float64)
    for frame in range(len(cycle)):
        constraints: list[tuple[slice, np.ndarray, int, float, int]] = []
        offsets: list[float] = []
        for side in np.flatnonzero(contact[frame]):
            support_slice = slice(0, 2) if side == 0 else slice(2, 4)
            phase = float(source_roll_phase[frame, side])
            mode = int(source_roll_mode[frame, side])
            targets = roll_clearance_targets(
                phase,
                mode,
                target_clearance,
            )
            constraints.append((support_slice, targets, int(side), phase, mode))
            current = before_landmarks[frame, support_slice]
            offsets.extend(targets - current)
            support_before_values.extend(current)
            support_target_values.extend(targets)
        frame_constraints.append(constraints)
        raw_root_offset[frame] = float(np.mean(offsets))
    root_offset = gaussian_filter1d(raw_root_offset, ground_sigma, mode="wrap")
    cycle[:, 2] += root_offset

    lower, upper = joint_bounds_for_qpos(model, MAIN_JOINT_QPOS)
    frame_indices = np.arange(len(cycle), dtype=np.int32)
    processing_start = 0
    processing_order = (processing_start + frame_indices) % len(cycle)
    previous_frame = int((processing_start - 1) % len(cycle))
    previous_root_z = float(cycle[previous_frame, 2])
    previous_joints = cycle[previous_frame, MAIN_JOINT_QPOS].copy()
    for frame in processing_order:
        frame = int(frame)
        constraints = frame_constraints[frame]
        nominal_root_z = float(cycle[frame, 2])
        nominal_joints = cycle[frame, MAIN_JOINT_QPOS].copy()
        nominal = np.concatenate(([nominal_root_z], nominal_joints))
        variable_lower = np.concatenate(([nominal_root_z - 0.03], lower))
        variable_upper = np.concatenate(([nominal_root_z + 0.03], upper))
        temporal_root_z = previous_root_z
        temporal_joints = previous_joints

        def residual(variables: np.ndarray) -> np.ndarray:
            pose = cycle[frame].copy()
            pose[2] = variables[0]
            pose[MAIN_JOINT_QPOS] = variables[1:]
            distances = foot_clearances(model, data, pose, floor_id, geom_ids)
            heights = sole_landmark_heights(data, landmark_body_ids, landmark_points)
            contact_residuals = np.concatenate(
                [
                    heights[support_slice] - targets
                    for support_slice, targets, _, _, _ in constraints
                ]
            )
            heel_separation = []
            for _, _, side, phase, mode in constraints:
                if mode != ROLL_HEEL:
                    continue
                progress = smoothstep(phase / HEEL_PHASE_END)
                toe_bound = target_clearance + HEEL_STRIKE_GEOM_CLEARANCE * (
                    1.0 - progress
                )
                toe_geom_index = 1 if side == 0 else 3
                heel_separation.append(
                    3.0 * min(distances[toe_geom_index] - toe_bound, 0.0)
                )
            swing_separation = []
            landing_residuals = []
            for side in np.flatnonzero(contact[frame] <= 0.5):
                swing_slice = slice(0, 2) if side == 0 else slice(2, 4)
                swing_separation.extend(
                    5.0
                    * np.minimum(
                        distances[swing_slice] - SWING_FOOT_MIN_CLEARANCE,
                        0.0,
                    )
                )
                countdown = int(landing_countdown[frame, side])
                if countdown == 0:
                    continue
                landing_progress = smoothstep(
                    (LANDING_APPROACH_FRAMES - countdown)
                    / max(LANDING_APPROACH_FRAMES - 1, 1)
                )
                toe_bound = SWING_FOOT_MIN_CLEARANCE + (
                    HEEL_STRIKE_GEOM_CLEARANCE - SWING_FOOT_MIN_CLEARANCE
                ) * landing_progress
                heel_bound = LANDING_APPROACH_START_CLEARANCE + (
                    LANDING_HEEL_CLEARANCE - LANDING_APPROACH_START_CLEARANCE
                ) * landing_progress
                foot_geom_index = 0 if side == 0 else 2
                toe_geom_index = foot_geom_index + 1
                heel_advance = LANDING_APPROACH_HEEL_ADVANCE * landing_progress
                landing_residuals.extend(
                    (
                        5.0
                        * min(distances[toe_geom_index] - toe_bound, 0.0),
                        3.0
                        * landing_progress
                        * max(distances[foot_geom_index] - heel_bound, 0.0),
                        3.0
                        * landing_progress
                        * max(
                            distances[foot_geom_index]
                            - distances[toe_geom_index]
                            + heel_advance,
                            0.0,
                        ),
                    )
                )
            phase_continuity = []
            if np.count_nonzero(contact[frame]) > 1:
                for _, _, side, _, mode in constraints:
                    if mode != ROLL_TOE:
                        continue
                    joint_slice = slice(7 * side, 7 * (side + 1))
                    phase_continuity.extend(
                        OUTGOING_STANCE_TEMPORAL_WEIGHT
                        * (variables[1:][joint_slice] - temporal_joints[joint_slice])
                    )
            for side in np.flatnonzero(early_swing_count[frame] > 0):
                count = int(early_swing_count[frame, side])
                weight = EARLY_SWING_TEMPORAL_WEIGHT * (
                    (EARLY_SWING_FRAMES - count + 1) / EARLY_SWING_FRAMES
                )
                joint_slice = slice(7 * side, 7 * (side + 1))
                phase_continuity.extend(
                    weight
                    * (variables[1:][joint_slice] - temporal_joints[joint_slice])
                )
            # Track the roll-specific foot/toe distances while all four foot
            # meshes remain above ground. The weak nominal term keeps IK local.
            return np.concatenate(
                (
                    contact_residuals,
                    np.asarray(heel_separation, dtype=np.float64),
                    np.asarray(swing_separation, dtype=np.float64),
                    np.asarray(landing_residuals, dtype=np.float64),
                    np.asarray(phase_continuity, dtype=np.float64),
                    3.0 * np.minimum(distances, 0.0),
                    [0.10 * (variables[0] - nominal_root_z)],
                    0.03 * (variables[1:] - nominal_joints),
                    [temporal_weight * (variables[0] - temporal_root_z)],
                    temporal_weight * (variables[1:] - temporal_joints),
                )
            )

        solution = least_squares(
            residual,
            nominal,
            bounds=(variable_lower, variable_upper),
            max_nfev=200,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        if not solution.success:
            candidate = cycle[frame].copy()
            candidate[2] = solution.x[0]
            candidate[MAIN_JOINT_QPOS] = solution.x[1:]
            distances = foot_clearances(model, data, candidate, floor_id, geom_ids)
            heights = sole_landmark_heights(data, landmark_body_ids, landmark_points)
            support_error = max(
                float(np.max(np.abs(heights[support_slice] - targets)))
                for support_slice, targets, _, _, _ in constraints
            )
            # Mesh distance has contact-feature discontinuities near heel/toe
            # transitions, so least_squares can exhaust its evaluation budget
            # after the physical constraints are already satisfied.
            if support_error > 8.0e-3 or distances.min() < -8.0e-4:
                raise RuntimeError(
                    f"Contact IK failed at frame {frame}: {solution.message}; "
                    f"support_error={support_error:.3e}, min_clearance={distances.min():.3e}, "
                    f"distances={distances}"
                )
        cycle[frame, 2] = solution.x[0]
        cycle[frame, MAIN_JOINT_QPOS] = solution.x[1:]
        previous_root_z = float(solution.x[0])
        previous_joints = solution.x[1:].copy()

    after = np.asarray(
        [foot_clearances(model, data, pose, floor_id, geom_ids) for pose in cycle]
    )
    after_landmarks = []
    for pose in cycle:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        after_landmarks.append(
            sole_landmark_heights(data, landmark_body_ids, landmark_points)
        )
    after_landmarks = np.asarray(after_landmarks)
    support_after_values: list[float] = []
    for frame, constraints in enumerate(frame_constraints):
        for support_slice, _, _, _, _ in constraints:
            support_after_values.extend(after_landmarks[frame, support_slice])
    support_before = np.asarray(support_before_values)
    support_targets = np.asarray(support_target_values)
    support_after = np.asarray(support_after_values)
    contact_frames = 0
    for pose in cycle:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        contact_frames += int(data.ncon > 0)

    result = np.vstack((cycle, cycle[0]))
    contact = np.vstack((contact, contact[0]))
    roll_phase = np.asarray(source_roll_phase, dtype=np.float32).copy()
    roll_mode = np.asarray(source_roll_mode, dtype=np.int8).copy()
    roll_phase[-1] = roll_phase[0]
    roll_mode[-1] = roll_mode[0]
    roll_error = np.abs(support_after - support_targets)
    statistics: dict[str, float | int | list[float]] = {
        "target_foot_clearance": float(target_clearance),
        "support_clearance_before_min": float(support_before.min()),
        "support_clearance_before_max": float(support_before.max()),
        "support_clearance_after_min": float(support_after.min()),
        "support_clearance_after_max": float(support_after.max()),
        "roll_target_error_max": float(roll_error.max()),
        "roll_mode_counts": [
            int(np.count_nonzero(source_roll_mode[:-1] == mode))
            for mode in (ROLL_HEEL, ROLL_FLAT, ROLL_TOE)
        ],
        "all_foot_clearance_after_min": float(after.min()),
        "main_model_contact_frames": contact_frames,
        "root_ground_offset_range": [float(root_offset.min()), float(root_offset.max())],
    }
    return result, contact, roll_phase, roll_mode, statistics


def smooth_cyclic_scalar(values: np.ndarray, blend_frames: int) -> np.ndarray:
    """Match the scalar endpoint while preserving the interior samples."""
    result = np.asarray(values, dtype=np.float64).copy()
    blend_frames = min(max(blend_frames, 2), max(2, len(result) // 4))
    endpoint = 0.5 * (result[0] + result[-1])
    for index in range(blend_frames):
        phase = index / (blend_frames - 1)
        weight = phase * phase * (3.0 - 2.0 * phase)
        result[index] = (1.0 - weight) * endpoint + weight * result[index]
        tail = len(result) - 1 - index
        result[tail] = (1.0 - weight) * endpoint + weight * result[tail]
    result[0] = endpoint
    result[-1] = endpoint
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-xml", type=Path, default=REPO_ROOT / "robot" / "MJCFmodelposition.xml"
    )
    parser.add_argument("--frequency", type=float, default=40.0)
    parser.add_argument("--blend-frames", type=int, default=6)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=17,
        help="Odd Savitzky-Golay window applied around the complete gait cycle.",
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=0.0005,
        help="Signed stance-foot distance above the main model floor in metres.",
    )
    parser.add_argument(
        "--ground-sigma",
        type=float,
        default=2.0,
        help="Circular Gaussian sigma for the root-height contact correction.",
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.03,
        help="Sequential regularization for smooth contact-IK solutions.",
    )
    args = parser.parse_args()

    with np.load(args.input) as archive:
        raw_qpos = np.asarray(archive["qpos"], dtype=np.float64)
        source_frequency = float(archive["fps"])
        source_foot_contact = np.asarray(archive["foot_contact"], dtype=np.float32)
        fallback_phase, fallback_mode = foot_roll_schedule(source_foot_contact)
        source_roll_phase = np.asarray(
            archive["foot_roll_phase"]
            if "foot_roll_phase" in archive.files
            else fallback_phase,
            dtype=np.float32,
        )
        source_roll_mode = np.asarray(
            archive["foot_roll_mode"]
            if "foot_roll_mode" in archive.files
            else fallback_mode,
            dtype=np.int8,
        )
        forward_velocity = np.asarray(archive["target_forward_velocity"], dtype=np.float64)
        source_frames = np.asarray(archive["source_frames"], dtype=np.int32)
        source = str(archive["source"]) if "source" in archive.files else "unknown"
        cost = float(np.asarray(archive["cost"]).item())
        retarget_step_size = (
            float(archive["retarget_step_size"])
            if "retarget_step_size" in archive.files
            else None
        )
        retarget_foot_tolerance = (
            float(archive["retarget_foot_tolerance"])
            if "retarget_foot_tolerance" in archive.files
            else None
        )
        mapping = [str(item) for item in archive["mapping_human_to_robot"]]

    frame_count = len(raw_qpos)
    if raw_qpos.shape != (frame_count, 21):
        raise ValueError(f"Expected retarget qpos shape (T, 21), got {raw_qpos.shape}.")
    if (
        source_foot_contact.shape != (frame_count, 2)
        or source_roll_phase.shape != (frame_count, 2)
        or source_roll_mode.shape != (frame_count, 2)
        or forward_velocity.shape != (frame_count,)
    ):
        raise ValueError(
            "Unexpected auxiliary shapes: "
            f"contact={source_foot_contact.shape}, roll_phase={source_roll_phase.shape}, "
            f"roll_mode={source_roll_mode.shape}, velocity={forward_velocity.shape}."
        )

    duration = (frame_count - 1) / source_frequency
    raw_qpos = resample_qpos(raw_qpos, source_frequency, args.frequency)
    target_times = np.linspace(0.0, duration, len(raw_qpos))
    source_times = np.arange(frame_count, dtype=np.float64) / source_frequency
    nearest = np.clip(
        np.rint(target_times * source_frequency).astype(np.int64), 0, frame_count - 1
    )
    source_foot_contact = source_foot_contact[nearest]
    source_roll_phase = source_roll_phase[nearest]
    source_roll_mode = source_roll_mode[nearest]
    forward_velocity = np.interp(target_times, source_times, forward_velocity)
    frame_count = len(raw_qpos)

    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    if (model.nq, model.nv) != (23, 22):
        raise ValueError(f"Expected main robot (nq, nv)=(23, 22), got {(model.nq, model.nv)}.")

    qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float64)[None, :], frame_count, axis=0)
    qpos[:, :7] = raw_qpos[:, :7]
    qpos[:, MAIN_JOINT_QPOS] = raw_qpos[:, RAW_JOINT_QPOS]
    # The walking task represents forward translation through qvel; keeping the
    # reference root in place prevents a finite clip from drifting out of scene.
    qpos[:, :2] = 0.0
    qpos = make_cyclic(qpos, blend_frames=args.blend_frames, hold_frames=0)
    qpos = smooth_periodic_qpos(qpos, args.smooth_window)
    qpos[:, :2] = 0.0
    clip_joint_limits(model, qpos)
    qpos, foot_contact, foot_roll_phase, foot_roll_mode, contact_statistics = (
        enforce_main_model_contacts(
            model,
            qpos,
            source_foot_contact,
            source_roll_phase,
            source_roll_mode,
            target_clearance=args.ground_clearance,
            ground_sigma=args.ground_sigma,
            temporal_weight=args.temporal_weight,
        )
    )
    clip_joint_limits(model, qpos)

    qvel = cyclic_velocity(model, qpos, args.frequency)
    qvel[:, 0] = smooth_cyclic_scalar(forward_velocity, args.blend_frames)
    # qpos contains an explicit duplicate endpoint.  Match its velocity as well
    # so phase interpolation is continuous when the reference wraps.
    qvel[-1] = qvel[0]

    metadata = {
        "task": "walk",
        "source": f"{source}, frames {int(source_frames[0])}-{int(source_frames[1])}",
        "source_frames_inclusive": source_frames.tolist(),
        "cycle_selection": "minimum normalized pose seam near the dominant 45-frame gait period",
        "retargeter": "OmniRetarget/Holosoma interaction-mesh SQP",
        "retargeter_url": "https://github.com/amazon-far/holosoma",
        "retargeter_commit": "79d728c2c40e7f9ce8c28f9d9cab117ed2350bb5",
        "human_to_robot_mapping": mapping,
        "joint_mapping": {
            "retarget_14dof": [
                JOINT_NAMES[index]
                for index in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16)
            ],
            "passive_main_model_only": ["patella_r_j1", "patella_l_j1"],
        },
        "root_convention": "root x/y fixed; qvel x preserves source forward speed",
        "contact_convention": (
            "source stance phase split into heel/flat/toe roll, then full-model "
            "foot/toe STL contact IK; natural double-support labels preserved"
        ),
        "source_frequency": source_frequency,
        "frequency": args.frequency,
        "frame_count": frame_count,
        "blend_frames": args.blend_frames,
        "periodic_smoothing": {
            "method": "circular Savitzky-Golay",
            "window": args.smooth_window,
            "polyorder": 3,
        },
        "grounding": {
            "method": "smoothed root-height correction plus local full-model contact IK",
            "gaussian_sigma": args.ground_sigma,
            "temporal_weight": args.temporal_weight,
            "roll_modes": {"swing": 0, "heel": 1, "flat": 2, "toe": 3},
            "heel_phase_end": HEEL_PHASE_END,
            "toe_phase_start": TOE_PHASE_START,
            "heel_strike_toe_lift": HEEL_STRIKE_TOE_LIFT,
            "heel_strike_geom_clearance": HEEL_STRIKE_GEOM_CLEARANCE,
            "toe_off_heel_lift": TOE_OFF_HEEL_LIFT,
            "swing_foot_min_clearance": SWING_FOOT_MIN_CLEARANCE,
            "landing_approach_frames": LANDING_APPROACH_FRAMES,
            "landing_approach_heel_advance": LANDING_APPROACH_HEEL_ADVANCE,
            "landing_approach_start_clearance": LANDING_APPROACH_START_CLEARANCE,
            "landing_heel_clearance": LANDING_HEEL_CLEARANCE,
            "early_swing_frames": EARLY_SWING_FRAMES,
            "early_swing_temporal_weight": EARLY_SWING_TEMPORAL_WEIGHT,
            "outgoing_stance_temporal_weight": OUTGOING_STANCE_TEMPORAL_WEIGHT,
            **contact_statistics,
        },
        "omniretarget_final_cost": cost,
        "omniretarget_step_size": retarget_step_size,
        "omniretarget_foot_tolerance": retarget_foot_tolerance,
        "mean_forward_velocity": float(np.mean(qvel[:, 0])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        joint_names=np.asarray(JOINT_NAMES),
        frequency=np.float64(args.frequency),
        foot_contact=foot_contact,
        foot_roll_phase=foot_roll_phase,
        foot_roll_mode=foot_roll_mode,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    print(
        f"saved {args.output}: qpos={qpos.shape}, qvel={qvel.shape}, "
        f"frequency={args.frequency:g}Hz, mean_vx={np.mean(qvel[:, 0]):.3f}m/s, "
        f"contacts={contact_statistics['main_model_contact_frames']}/{frame_count - 1}, "
        f"cyclic_seam={np.max(np.abs(qpos[-1] - qpos[0])):.2e}"
    )


if __name__ == "__main__":
    main()
