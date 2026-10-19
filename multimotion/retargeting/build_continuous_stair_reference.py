#!/usr/bin/env python3
"""Build one continuous five-step up/platform/down stair reference.

The two official OmniRetarget clips provide the pose prior.  A sequential
full-body IK pass replaces their independent world trajectories with an
explicit alternating footstep plan, so ascent, top-platform transfer and
descent live in one non-looping motion without a playback seam.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path[:0] = [str(SRC_ROOT), str(HERE)]

from build_references import (  # noqa: E402
    CONTACT_CLEARANCE,
    FOOT_GEOM_NAMES,
    JOINT_NAMES,
    MAIN_JOINT_QPOS,
    OUTPUT_FREQUENCY,
    TRAIN_STAIR_CFG,
    landmark_positions,
    normalize_quaternions,
    sole_landmarks,
    terrain_geom_ids,
    velocities,
)
from lytrobot_amp.terrain import load_model, terrain_height_at  # noqa: E402
from prepare_walk1 import ROLL_FLAT  # noqa: E402


MODEL_XML = REPO_ROOT / "robot" / "MJCFmodelfinal.xml"
MOTION_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
OUTPUT = MOTION_DIR / "stair_up_down_omniretarget.npz"
FRAMES_PER_STEP = 32
SWING_CLEARANCE_M = 0.020

# Frame-indexed pelvic X waypoints, fitted from individually reachable IK
# poses on this *fixed* 0.13 m / 0.32 m course.  In particular the pelvis
# needs to move ahead of the trailing leg at the first descending tread; a
# fixed offset from the mean foot position makes the strict knee/ankle ranges
# locally infeasible.  PCHIP supplies a continuous, non-overshooting guide.
PELVIS_X_WAYPOINTS = np.asarray(
    [[0, 0.34], [16, 0.34]]
    + [
        waypoint
        for pair in range(13)
        for waypoint in (
            # Stay over the trailing stance foot through most of the leading
            # swing.  Begin the catch step only during its last 0.10 s.
            [28 + 64 * pair, 0.71 + 0.32 * pair - 0.37],
            # The leading right foot has just landed.  Keep the root near the
            # midpoint while both feet bridge adjacent treads, then transfer
            # the COM over the right stance foot before lifting the left foot.
            [32 + 64 * pair, 0.71 + 0.32 * pair - 0.21],
            [42 + 64 * pair, 0.71 + 0.32 * pair - 0.09],
            # The trailing left foot has joined on the same tread.
            [64 + 64 * pair, 0.71 + 0.32 * pair - 0.05],
        )
    ],
    dtype=np.float64,
)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def load_pose_prior(frame_count: int) -> np.ndarray:
    clips = []
    for name in ("stair_up", "stair_down"):
        with np.load(MOTION_DIR / f"{name}_omniretarget.npz") as archive:
            clips.append(np.asarray(archive["qpos"], dtype=np.float64))
    source = np.concatenate(clips, axis=0)
    positions = np.linspace(0.0, len(source) - 1.0, frame_count)
    low = np.floor(positions).astype(np.int64)
    high = np.minimum(low + 1, len(source) - 1)
    blend = (positions - low)[:, None]
    prior = (1.0 - blend) * source[low] + blend * source[high]
    prior[:, 3:7] = normalize_quaternions(prior[:, 3:7])
    return prior


def footstep_plan() -> tuple[np.ndarray, np.ndarray]:
    """Return a one-tread-at-a-time step-to gait for the whole course."""

    # With the clinical hip range, an alternating gait would require each leg
    # to span 0.64 m and is kinematically unreachable.  The leading right foot
    # advances one 0.32 m tread and the left foot joins it before the next
    # riser.  The same step-to pattern continues over the platform and descent.
    tread_centres = [
        0.71, 1.03, 1.35, 1.67, 1.99,  # five up
        2.31, 2.63,                    # top-platform transfer
        2.95, 3.27, 3.59, 3.91, 4.23, # five down
        4.55,                          # run-out
    ]
    sides = np.asarray([side for _ in tread_centres for side in (0, 1)], dtype=np.int64)
    landings = np.asarray([x for x in tread_centres for _ in (0, 1)], dtype=np.float64)
    return sides, landings


def desired_foot_centres() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    sides, landings = footstep_plan()
    frame_count = len(landings) * FRAMES_PER_STEP + 1
    centres = np.zeros((frame_count, 2, 3), dtype=np.float64)
    contacts = np.zeros((frame_count, 2), dtype=np.float32)
    pelvis_levels = np.zeros(frame_count, dtype=np.float64)
    pelvis_lateral = np.zeros(frame_count, dtype=np.float64)
    # Start one tread behind the first landing.  The pelvis guide, rather than
    # an overlong 0.41 m first stride, puts the initial COM inside both feet.
    initial_x = np.asarray([0.39, 0.39], dtype=np.float64)
    current = np.column_stack(
        (
            initial_x,
            np.asarray([-0.075, 0.075]),
            terrain_height_at(initial_x, TRAIN_STAIR_CFG) + CONTACT_CLEARANCE,
        )
    )
    centres[0] = current
    contacts[0] = 1.0
    current_level = 0.0
    current_pelvis_y = 0.0
    for step, (side, landing_x) in enumerate(zip(sides, landings, strict=True)):
        start = step * FRAMES_PER_STEP
        target = current[side].copy()
        target[0] = landing_x
        target_level = float(terrain_height_at(np.asarray([landing_x]), TRAIN_STAIR_CFG)[0])
        target[2] = target_level + CONTACT_CLEARANCE
        stance = 1 - int(side)
        support_pelvis_y = float(current[stance, 1])
        # Shift over the landing/support foot on both sides.  Keeping the
        # left landing as the mirror of the right landing prevents the stair
        # prior from baking in a right-leading lateral bias.
        landing_pelvis_y = float(target[1])
        if target_level < current_level - 1.0e-6 and side == 0:
            next_level = target_level
        elif target_level > current_level + 1.0e-6 and side == 1:
            next_level = target_level
        else:
            next_level = current_level
        for local in range(1, FRAMES_PER_STEP + 1):
            frame = start + local
            phase = local / FRAMES_PER_STEP
            level_blend = smoothstep(phase)
            pelvis_levels[frame] = (1.0 - level_blend) * current_level + level_blend * next_level
            centres[frame] = current
            # First transfer weight in double support.  This keeps the pelvis
            # target continuous when the stance side changes between steps.
            shift_end = 0.30
            shift_blend = smoothstep(min(phase / shift_end, 1.0))
            pelvis_lateral[frame] = (
                (1.0 - shift_blend) * current_pelvis_y
                + shift_blend * support_pelvis_y
            )
            if phase <= shift_end:
                contacts[frame] = 1.0
                continue

            swing_phase = min((phase - shift_end) / (0.95 - shift_end), 1.0)
            landing_shift = smoothstep((swing_phase - 0.55) / 0.45)
            pelvis_lateral[frame] = (
                (1.0 - landing_shift) * support_pelvis_y
                + landing_shift * landing_pelvis_y
            )
            high_z = max(float(current[side, 2]), float(target[2])) + SWING_CLEARANCE_M
            descent_drop = max(0.0, float(current[side, 2] - target[2]))
            if swing_phase < 0.50:
                blend = smoothstep(swing_phase / 0.50)
                # Fold the leg forward while lifting; the full-mesh collision
                # residual below prevents a toe from entering the riser.
                swing = current[side].copy()
                swing[0] = (
                    (1.0 - 0.15 * blend) * current[side, 0]
                    + 0.15 * blend * target[0]
                )
                swing[2] = (1.0 - blend) * current[side, 2] + blend * high_z
            elif swing_phase < 0.82:
                blend = smoothstep((swing_phase - 0.50) / 0.32)
                swing = (1.0 - blend) * current[side] + blend * target
                swing[0] = (
                    (1.0 - blend) * (0.85 * current[side, 0] + 0.15 * target[0])
                    + blend * target[0]
                )
                # On descent the landing tread is 13 cm below the support
                # foot.  Start lowering the swing foot once it has travelled
                # most of the way over the edge, instead of demanding the
                # whole drop in the final few 40 Hz frames.
                early_drop = 0.65 * descent_drop * smoothstep((blend - 0.55) / 0.45)
                swing[2] = high_z + 0.02 * np.sin(np.pi * blend) - early_drop
            else:
                blend = smoothstep((swing_phase - 0.82) / 0.18)
                swing = target.copy()
                landing_height = high_z - 0.65 * descent_drop
                swing[2] = (1.0 - blend) * landing_height + blend * target[2]
            centres[frame, side] = swing
            contacts[frame, stance] = 1.0
            # The MuJoCo sole meshes sit about 2--3 mm inside their compliant
            # contact surface at rest.  Keep double support labelled through
            # that initial unloading interval; declare swing only after the
            # foot has separated from the tread.
            if swing_phase < 0.50:
                contacts[frame, side] = 1.0
            if local >= FRAMES_PER_STEP - 2:
                contacts[frame, side] = 1.0
        current[side] = target
        current_level = next_level
        current_pelvis_y = landing_pelvis_y
    return centres, contacts, landings, pelvis_levels, pelvis_lateral


def build() -> None:
    model = load_model(MODEL_XML, TRAIN_STAIR_CFG)
    data = mujoco.MjData(model)
    _, landmark_body_ids, landmark_local = sole_landmarks(model)
    terrain_ids = terrain_geom_ids(model)
    foot_geom_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_GEOM_NAMES],
        dtype=np.int32,
    )
    fromto = np.zeros(6, dtype=np.float64)
    centres, contacts, landings, pelvis_levels, pelvis_lateral = desired_foot_centres()
    prior = load_pose_prior(len(centres))
    if len(centres) - 1 != int(PELVIS_X_WAYPOINTS[-1, 0]):
        raise ValueError("Pelvis guide must match the 0.13 m course frame count.")
    pelvis_x_guide = PchipInterpolator(
        PELVIS_X_WAYPOINTS[:, 0], PELVIS_X_WAYPOINTS[:, 1]
    )(np.arange(len(centres)))

    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    neutral_landmarks = landmark_positions(data, landmark_body_ids, landmark_local)
    neutral_centres = neutral_landmarks.reshape(2, 2, 3).mean(axis=1)
    landmark_offsets = neutral_landmarks.reshape(2, 2, 3) - neutral_centres[:, None, :]
    # ``centres`` describe the rigid foot centre, while contact clearance is a
    # sole-minimum requirement.  Lift each centre by its lowest landmark offset
    # so heel targets never ask the geometry to enter the tread.
    centres[:, :, 2] -= np.min(landmark_offsets[:, :, 2], axis=1)[None, :]

    by_address = {
        int(model.jnt_qposadr[joint_id]): joint_id for joint_id in range(model.njnt)
    }
    joint_ids = [by_address[int(address)] for address in MAIN_JOINT_QPOS]
    joint_lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in joint_ids])
    joint_upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in joint_ids])
    qpos = np.repeat(np.asarray(model.qpos0)[None, :], len(centres), axis=0)
    # Variables: root xyz, root roll/pitch, then the 14 actuated joint angles.
    previous = np.concatenate(
        (np.asarray([0.34, 0.0, 0.875, 0.0, 0.0]), model.qpos0[MAIN_JOINT_QPOS])
    )
    landmark_errors = []
    contact_landmark_errors = []
    swing_landmark_errors = []
    actual_landmark_log: list[np.ndarray] = []
    target_landmark_log: list[np.ndarray] = []
    geometry_distances = []

    for frame in range(len(centres)):
        if frame % FRAMES_PER_STEP == 0:
            print(f"solving frame {frame}/{len(centres) - 1}", flush=True)
        target_landmarks = centres[frame, :, None, :] + landmark_offsets
        target_landmarks = target_landmarks.reshape(4, 3)
        target_by_foot = target_landmarks.reshape(2, 2, 3)
        target_surface = (
            terrain_height_at(centres[frame, :, 0], TRAIN_STAIR_CFG)
            + CONTACT_CLEARANCE
        )
        stable_contact = (contacts[frame] > 0.5) & (
            np.min(target_by_foot[:, :, 2], axis=1) <= target_surface + 0.005
        )
        landmark_weights = np.repeat(
            np.where(
                stable_contact[:, None],
                np.asarray([180.0, 180.0, 600.0]),
                np.asarray([130.0, 130.0, 190.0]),
            ),
            2,
            axis=0,
        )
        pelvis_x = float(pelvis_x_guide[frame])
        # Follow the smooth course level, independent of temporary swing-foot
        # clearance.  Each stair advances the pelvis by exactly one riser.
        lower_leg_relief = float(
            np.interp(
                pelvis_levels[frame],
                [0.0, 0.13, 0.26, 0.39, 0.52, 0.65],
                [0.025, 0.022, 0.028, 0.033, 0.0, 0.0],
            )
        )
        descent_blend = smoothstep(
            (frame - 16 * FRAMES_PER_STEP) / (3 * FRAMES_PER_STEP)
        )
        pelvis_z = float(
            0.835 + 0.040 * smoothstep(frame / (2 * FRAMES_PER_STEP))
            + pelvis_levels[frame] - descent_blend * lower_leg_relief
        )
        pelvis_y = float(pelvis_lateral[frame])
        desired_root = np.asarray([pelvis_x, pelvis_y, pelvis_z], dtype=np.float64)
        nominal_joints = np.clip(prior[frame, MAIN_JOINT_QPOS], joint_lower, joint_upper)
        initial = previous.copy()
        if frame == 0:
            initial[:3] = desired_root
            initial[3] = float(
                np.clip(-1.5 * pelvis_y, np.deg2rad(-15.0), np.deg2rad(15.0))
            )
            initial[4] = np.deg2rad(-20.0)
        lower = np.concatenate(
            (desired_root + [-0.28, -0.14, -0.25], np.deg2rad([-15.0, -30.0]), joint_lower)
        )
        upper = np.concatenate(
            (desired_root + [0.28, 0.14, 0.25], np.deg2rad([15.0, 30.0]), joint_upper)
        )
        if frame > 0:
            # Hard continuity limits prevent the optimizer from switching to a
            # distant inverse-kinematics branch between adjacent 40 Hz frames.
            # These are trajectory-rate constraints, not changes to the robot's
            # clinical joint limits.
            frame_delta = np.concatenate(
                (
                    np.asarray([0.040, 0.040, 0.030]),
                    np.deg2rad([3.0, 3.0]),
                    np.full(len(MAIN_JOINT_QPOS), np.deg2rad(7.0)),
                )
            )
            lower = np.maximum(lower, previous - frame_delta)
            upper = np.minimum(upper, previous + frame_delta)
        def residual(variables: np.ndarray) -> np.ndarray:
            pose = np.asarray(model.qpos0).copy()
            pose[:3] = variables[:3]
            roll, pitch = variables[3:5]
            cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
            cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
            pose[3:7] = (cr * cp, sr * cp, cr * sp, sr * sp)
            pose[MAIN_JOINT_QPOS] = variables[5:]
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            landmarks = landmark_positions(data, landmark_body_ids, landmark_local)
            geom_dist = np.asarray(
                [
                    min(
                        mujoco.mj_geomDistance(
                            model, data, int(terrain_id), int(foot_id), 0.25, fromto
                        )
                        for terrain_id in terrain_ids
                    )
                    for foot_id in foot_geom_ids
                ]
            )
            return np.concatenate(
                (
                    (landmark_weights * (landmarks - target_landmarks)).ravel(),
                    1200.0 * np.minimum(geom_dist - 0.0010, 0.0),
                    np.asarray([30.0, 12.0, 10.0]) * (variables[:3] - desired_root),
                    0.50
                    * (
                        variables[3:5]
                        - np.asarray(
                            [
                                np.clip(
                                    -1.5 * pelvis_y,
                                    np.deg2rad(-15.0),
                                    np.deg2rad(15.0),
                                ),
                                np.deg2rad(-20.0),
                            ]
                        )
                    ),
                    0.10 * (variables[5:] - nominal_joints),
                    0.50 * (variables - previous),
                )
            )

        solution = least_squares(
            residual,
            np.clip(initial, lower, upper),
            bounds=(lower, upper),
            max_nfev=120,
            ftol=1.0e-7,
            xtol=1.0e-7,
            gtol=1.0e-7,
        )

        def candidate_metrics(variables: np.ndarray) -> tuple[float, float]:
            # ``residual`` also installs the candidate in ``data``.
            residual(variables)
            actual_landmarks = landmark_positions(data, landmark_body_ids, landmark_local)
            foot_error = float(
                np.max(np.linalg.norm(actual_landmarks - target_landmarks, axis=1))
            )
            geometry_distance = (
                min(
                    mujoco.mj_geomDistance(
                        model, data, int(terrain_id), int(foot_id), 0.25, fromto
                    )
                    for terrain_id in terrain_ids
                    for foot_id in foot_geom_ids
                )
            )
            return foot_error, float(geometry_distance)

        best_error, best_distance = candidate_metrics(solution.x)
        if best_error > 0.040 or best_distance < -1.0e-4:
            # Difficult stair-edge frames can have several local IK minima.
            # Retry only those frames from varied, still strictly bounded root
            # and joint seeds, then select by physical foot tracking quality.
            base_roll = float(
                np.clip(-1.5 * pelvis_y, np.deg2rad(-15.0), np.deg2rad(15.0))
            )
            seeds: list[np.ndarray] = []
            for root_dx, root_dz, pitch_deg, use_nominal in (
                (0.0, 0.0, -25.0, False),
                (0.0, 0.0, -25.0, True),
                (-0.15, 0.0, -25.0, False),
                (0.15, 0.0, -10.0, False),
                (0.0, -0.10, -25.0, True),
                (0.0, 0.10, 0.0, True),
            ):
                seed = previous.copy()
                seed[:3] = desired_root + [root_dx, 0.0, root_dz]
                seed[3:5] = (base_roll, np.deg2rad(pitch_deg))
                if use_nominal:
                    seed[5:] = nominal_joints
                seeds.append(np.clip(seed, lower, upper))

            best_quality = best_error + 10.0 * max(-best_distance, 0.0)
            for seed in seeds:
                candidate = least_squares(
                    residual,
                    seed,
                    bounds=(lower, upper),
                    max_nfev=180,
                    ftol=1.0e-7,
                    xtol=1.0e-7,
                    gtol=1.0e-7,
                )
                error, distance = candidate_metrics(candidate.x)
                quality = error + 10.0 * max(-distance, 0.0)
                if quality < best_quality:
                    solution = candidate
                    best_error = error
                    best_distance = distance
                    best_quality = quality
                if best_error <= 0.040 and best_distance >= -1.0e-4:
                    break
            print(
                f"  multistart frame {frame}: error={best_error:.4f}m "
                f"clearance={best_distance:.5f}m",
                flush=True,
            )
        previous = solution.x
        qpos[frame, :3] = solution.x[:3]
        roll, pitch = solution.x[3:5]
        cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
        cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
        qpos[frame, 3:7] = (cr * cp, sr * cp, cr * sp, sr * sp)
        qpos[frame, MAIN_JOINT_QPOS] = solution.x[5:]
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        actual = landmark_positions(data, landmark_body_ids, landmark_local)
        point_errors = np.linalg.norm(actual - target_landmarks, axis=1)
        landmark_errors.append(float(np.max(point_errors)))
        # A foot can remain labelled as unloading double support while its heel
        # is already rising.  Contact accuracy is evaluated only where the
        # target sole is actually on the terrain; toe-off is a swing-pose test.
        contact_points = np.repeat(stable_contact, 2)
        contact_landmark_errors.append(
            float(np.max(point_errors[contact_points])) if np.any(contact_points) else 0.0
        )
        swing_landmark_errors.append(
            float(np.max(point_errors[~contact_points])) if np.any(~contact_points) else 0.0
        )
        actual_landmark_log.append(actual.copy())
        target_landmark_log.append(target_landmarks.copy())
        geometry_distances.append(
            min(
                mujoco.mj_geomDistance(
                    model, data, int(terrain_id), int(foot_id), 0.25, fromto
                )
                for terrain_id in terrain_ids
                for foot_id in foot_geom_ids
            )
        )

    # Signed mesh distances at a few stair-edge frames can retain a shallow
    # numerical overlap even after sequential IK.  Lift the *floating root*
    # by only the measured overlap, spread the correction over neighbouring
    # frames, and re-evaluate every affected foot and collision.  The model's
    # joint ranges and all relative leg angles remain exactly unchanged.
    overlap = np.maximum(0.0006 - np.asarray(geometry_distances), 0.0)
    lift = np.zeros(len(qpos), dtype=np.float64)
    for centre in np.flatnonzero(overlap > 0.0):
        for offset, weight in ((0, 1.0), (-1, 0.5), (1, 0.5), (-2, 0.25), (2, 0.25)):
            index = int(centre + offset)
            if 0 <= index < len(qpos):
                lift[index] = max(lift[index], weight * overlap[centre])
    for frame in np.flatnonzero(lift > 0.0):
        qpos[frame, 2] += lift[frame]
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        target_landmarks = target_landmark_log[frame]
        actual = landmark_positions(data, landmark_body_ids, landmark_local)
        point_errors = np.linalg.norm(actual - target_landmarks, axis=1)
        stable_contact = (contacts[frame] > 0.5) & (np.min(
            target_landmarks.reshape(2, 2, 3)[:, :, 2], axis=1
        ) <= (
            terrain_height_at(centres[frame, :, 0], TRAIN_STAIR_CFG)
            + CONTACT_CLEARANCE + 0.005
        ))
        contact_points = np.repeat(stable_contact, 2)
        landmark_errors[frame] = float(np.max(point_errors))
        contact_landmark_errors[frame] = (
            float(np.max(point_errors[contact_points])) if np.any(contact_points) else 0.0
        )
        swing_landmark_errors[frame] = (
            float(np.max(point_errors[~contact_points])) if np.any(~contact_points) else 0.0
        )
        actual_landmark_log[frame] = actual
        geometry_distances[frame] = min(
            mujoco.mj_geomDistance(model, data, int(g), int(foot), 0.25, fromto)
            for g in terrain_ids for foot in foot_geom_ids
        )

    max_landmark_error = float(max(landmark_errors))
    max_contact_error = float(max(contact_landmark_errors))
    max_swing_error = float(max(swing_landmark_errors))
    max_root_delta = float(np.max(np.abs(np.diff(qpos[:, :3], axis=0))))
    max_joint_delta = float(np.max(np.abs(np.diff(qpos[:, MAIN_JOINT_QPOS], axis=0))))
    worst_landmark_frame = int(np.argmax(landmark_errors))
    min_geometry_distance = float(min(geometry_distances))
    if (
        min_geometry_distance < -1.0e-4
        or max_contact_error > 0.030
        or max_swing_error > 0.055
        or max_root_delta > 0.041
        or max_joint_delta > np.deg2rad(7.0) + 1e-6
    ):
        diagnostics = HERE / "diagnostics" / "stair_up_down_candidate.npz"
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            diagnostics,
            qpos=qpos.astype(np.float32),
            foot_contact=contacts,
            foot_landmark_error=np.asarray(landmark_errors),
            contact_landmark_error=np.asarray(contact_landmark_errors),
            swing_landmark_error=np.asarray(swing_landmark_errors),
            all_foot_geometry_distance=np.asarray(geometry_distances),
            target_landmarks=np.asarray(target_landmark_log),
        )
        worst_actual = actual_landmark_log[worst_landmark_frame]
        worst_target = target_landmark_log[worst_landmark_frame]
        raise RuntimeError(
            f"continuous stair reference penetrates terrain: {min_geometry_distance:.6f} m; "
            f"max contact error={max_contact_error:.6f} m; "
            f"max swing error={max_swing_error:.6f} m; "
            f"max root/joint delta={max_root_delta:.5f} m/{max_joint_delta:.5f} rad; "
            f"max landmark error={max_landmark_error:.6f} m at frame {worst_landmark_frame}; "
            f"min geometry frame={int(np.argmin(geometry_distances))}; "
            f"diagnostics={diagnostics}; "
            f"actual={np.round(worst_actual, 3).tolist()}; "
            f"target={np.round(worst_target, 3).tolist()}"
        )
    qvel = velocities(model, qpos, OUTPUT_FREQUENCY, False)
    metadata = {
        "task": "stair_up_down",
        "loop": False,
        "retargeter": "OmniRetarget pose prior + sequential full-body stair IK",
        "frequency": OUTPUT_FREQUENCY,
        "step_height_m": TRAIN_STAIR_CFG.step_height,
        "step_depth_m": TRAIN_STAIR_CFG.step_depth,
        "step_count_each_direction": TRAIN_STAIR_CFG.step_count,
        "top_platform_length_m": TRAIN_STAIR_CFG.top_length,
        "footfall_target_x_m": landings.tolist(),
        "frames_per_step": FRAMES_PER_STEP,
        "maximum_foot_landmark_error_m": max_landmark_error,
        "maximum_contact_foot_landmark_error_m": max_contact_error,
        "maximum_swing_foot_landmark_error_m": max_swing_error,
        "maximum_root_translation_per_frame_m": max_root_delta,
        "maximum_joint_angle_per_frame_rad": max_joint_delta,
        "minimum_foot_terrain_geometry_distance_m": min_geometry_distance,
        "lateral_symmetry": "right/left foot landings mirror across the stair centerline",
        "continuous_course": "human stair prior, mirror-symmetric step-to five up -> two top-platform transfers -> five down -> run-out",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        joint_names=np.asarray(JOINT_NAMES),
        frequency=np.float64(OUTPUT_FREQUENCY),
        foot_contact=contacts,
        foot_roll_phase=np.zeros_like(contacts, dtype=np.float32),
        foot_roll_mode=np.full_like(contacts, ROLL_FLAT, dtype=np.int8),
        task=np.asarray("stair_up_down"),
        loop=np.asarray(False),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    print(
        f"saved {len(qpos)} frames -> {OUTPUT}; "
        f"max_contact_error={max_contact_error:.4f}m "
        f"max_swing_error={max_swing_error:.4f}m "
        f"min_geometry_distance={min_geometry_distance:.6f}m"
    )


if __name__ == "__main__":
    build()
