#!/usr/bin/env python3
"""Build a short, contact-corrected walking cycle from a full-model NPZ.

The exported file has unique cycle frames (no duplicated endpoint), translating
root positions, and velocities differentiated from those positions. Foot and
toe motion is model-constrained reconstruction where the source has no motion.
Dynamic stability is assessed separately by test_reference_dynamics.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "squat_training/retargeting"))
from build_walk_reference import (MAIN_JOINT_QPOS, enforce_main_model_contacts,
                                  foot_geom_ids, foot_clearances, normalize_quaternions,
                                  sole_landmarks, roll_clearance_targets)
from prepare_walk1 import foot_roll_schedule


def enforce_passive(model, qpos):
    for eq in range(model.neq):
        if model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        first, second = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
        a, b = int(model.jnt_qposadr[first]), int(model.jnt_qposadr[second])
        x = qpos[:, b] - model.qpos0[b]
        qpos[:, a] = model.qpos0[a] + sum(model.eq_data[eq, k] * x**k for k in range(5))


def periodic_fit(values, harmonics=5):
    phase = np.arange(len(values)) * 2 * np.pi / len(values)
    basis = np.column_stack([np.ones(len(values))] + [f(k * phase) for k in range(1, harmonics + 1) for f in (np.cos, np.sin)])
    return basis @ np.linalg.lstsq(basis, values, rcond=None)[0]


def refine_cycle(model, poses, frequency, speed, contact, phases, modes, *, max_nfev=100, min_knee_flexion=0.25):
    """Solve the whole loop together: planted support, swing and periodic smoothness.

    A linear inverted pendulum ZMP residual is a planning approximation. It
    improves weight transfer but does not substitute for a free-base rollout.
    """
    count = len(poses)
    floor, geom_ids = foot_geom_ids(model)
    body_ids, points = sole_landmarks(model, geom_ids)
    mesh_vertices = []
    for geom in geom_ids:
        mesh = int(model.geom_dataid[geom])
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Walking contact refinement requires mesh foot colliders.")
        first, count_vertices = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = np.asarray(model.mesh_vert[first:first+count_vertices], dtype=np.float64)
        mesh_vertices.append(vertices[ConvexHull(vertices).vertices])
    data = mujoco.MjData(model)
    travel = count * speed / frequency
    root_x = np.arange(count) * speed / frequency
    boundary = max(1, int(np.ceil(0.05*frequency)))
    swing_interior = np.logical_and.reduce([
        np.roll(contact < 0.5, shift, axis=0) for shift in range(-boundary, boundary+1)
    ])
    poses = poses.copy()
    poses[:, 0] = root_x
    world_points = np.empty((count, 4, 3))
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        world_points[i] = [data.xpos[b] + data.xmat[b].reshape(3, 3) @ p for b, p in zip(body_ids, points)]
    desired = world_points.copy()
    support_centers = np.empty((count, 2))
    for side in range(2):
        active = contact[:, side] > 0.5
        strike = int(np.flatnonzero(active & ~np.roll(active, 1))[0])
        stance_count = int(active.sum())
        heel, toe = 2*side, 2*side+1
        # Keep the human's landing location, but lock it in the world during
        # stance. Each next landing advances by the full cycle translation.
        anchor = world_points[strike, heel].copy()
        lateral = float(np.median(world_points[:, heel, 1]))
        toe_offset = float(np.median(world_points[:, toe, 0] - world_points[:, heel, 0]))
        for i in range(count):
            phase_frame = (i - strike) % count
            base = anchor.copy()
            if i < strike:
                base[0] -= travel
            base[1] = lateral
            if active[i]:
                targets = roll_clearance_targets(float(phases[i, side]), int(modes[i, side]), 0.0005)
                desired[i, heel, :2] = base[:2]
                desired[i, toe, :2] = base[:2] + [toe_offset, 0]
                desired[i, heel, 2], desired[i, toe, 2] = targets
            else:
                u = (phase_frame - stance_count) / max(count-stance_count, 1)
                u = float(np.clip(u, 0, 1))
                ease = u*u*u*(10 - 15*u + 6*u*u)
                lift = 0.055 * np.sin(np.pi*u)**2
                desired[i, heel, :2] = base[:2] + [travel * ease, 0]
                desired[i, toe, :2] = desired[i, heel, :2] + [toe_offset, 0]
                desired[i, heel, 2] = 0.0005 + lift + 0.012*(1-u)
                desired[i, toe, 2] = 0.0005 + lift + 0.012*u
    for i in range(count):
        centers = []
        for side in range(2):
            if contact[i, side] > 0.5:
                sole = desired[i, 2*side:2*side+2]
                loaded = sole[:, 2] <= sole[:, 2].min() + 0.0015
                centers.append(sole[loaded, :2].mean(0))
        support_centers[i] = np.mean(centers, axis=0)
    # Columns: periodic root X offset, root Y/Z, roll/pitch, 14 hinge joints.
    variables = np.column_stack((np.zeros(count), poses[:, 1:3], np.zeros((count, 2)), poses[:, MAIN_JOINT_QPOS]))
    joint_ids = [int(np.flatnonzero(model.jnt_qposadr == address)[0]) for address in MAIN_JOINT_QPOS]
    lower = np.r_[[-0.065, -0.085, 0.70, -0.14, -0.18], model.jnt_range[joint_ids, 0] + 0.002]
    upper = np.r_[[0.065, 0.085, 0.94, 0.14, 0.18], model.jnt_range[joint_ids, 1] - 0.002]
    for column, address in enumerate(MAIN_JOINT_QPOS, 5):
        if address in (10, 18):
            upper[column] = -min_knee_flexion
    variables = np.clip(variables, lower+1e-7, upper-1e-7)
    nominal = variables.copy()
    fromto = np.zeros(6)
    weights = np.array([0.3, 0.3, 0.35, 0.18, 0.18] + [0.035]*14)

    def decode(x):
        output = poses.copy()
        output[:, 0] = root_x + x[:, 0]
        output[:, 1:3] = x[:, 1:3]
        quaternion = Rotation.from_euler('xyz', np.column_stack((x[:, 3:5], np.zeros(count)))).as_quat()
        output[:, 3:7] = quaternion[:, [3, 0, 1, 2]]
        output[:, MAIN_JOINT_QPOS] = x[:, 5:]
        enforce_passive(model, output)
        return output

    def residual(flat):
        x = flat.reshape(count, -1)
        output = decode(x)
        com = np.empty((count, 3))
        actual_landmarks = np.empty((count, 4, 3))
        velocities = np.empty((count, model.nv))
        for i in range(count):
            previous, following = output[(i-1) % count].copy(), output[(i+1) % count].copy()
            if i == 0:
                previous[0] -= travel
            if i == count-1:
                following[0] += travel
            mujoco.mj_differentiatePos(model, velocities[i], 2/frequency, previous, following)
        object_velocity = np.empty(6)
        local = []
        for i, pose in enumerate(output):
            data.qpos[:] = pose
            data.qvel[:] = velocities[i]
            mujoco.mj_forward(model, data)
            landmarks = np.array([data.xpos[b] + data.xmat[b].reshape(3, 3) @ p for b, p in zip(body_ids, points)])
            actual_landmarks[i] = landmarks
            com[i] = data.subtree_com[model.body('pelvis').id]
            distances, surface_residuals = [], []
            for geom, vertices in zip(geom_ids, mesh_vertices):
                distances.append(mujoco.mj_geomDistance(model, data, floor, int(geom), 0.5, fromto))
                rotation = data.geom_xmat[geom].reshape(3, 3)
                offsets = vertices @ rotation.T
                heights = offsets[:, 2] + data.geom_xpos[geom, 2]
                surface_weights = np.exp(-(heights-heights.min())/0.003)
                surface_weights /= surface_weights.sum()
                center = surface_weights @ offsets
                centered = offsets-center
                covariance = (centered.T * surface_weights) @ centered
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                square_root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM,
                                        int(geom), object_velocity, 0)
                omega = object_velocity[:3]
                cross = np.array([[0., -omega[2], omega[1]],
                                  [omega[2], 0., -omega[0]],
                                  [-omega[1], omega[0], 0.]])
                mean_velocity = object_velocity[3:] + cross @ center
                # Exact weighted velocity energy of the near-ground mesh
                # patch: a mean plus covariance, without a discontinuous
                # choice of one closest vertex or cancellation of foot spin.
                surface_residuals.append(np.r_[mean_velocity[:2],
                                               (cross @ square_root)[:2].ravel()])
            distances = np.asarray(distances)
            side_distances = distances.reshape(2, 2)
            # Constrain the lowest material point even just above contact;
            # raising a foot cannot turn off the no-slip objective.
            load_weights = np.exp(-(side_distances-side_distances.min(axis=1, keepdims=True))/0.005).ravel()
            load_weights *= np.repeat(contact[i], 2)
            no_slip = np.asarray(surface_residuals) * load_weights[:, None]
            foot_weights = np.repeat(np.where(contact[i] > 0.5, 10.0, 0.4), 2)
            position_error = landmarks - desired[i]
            position_error[:, :2] *= foot_weights[:, None]
            position_error[:, 2] *= np.repeat(np.where(contact[i] > 0.5, 12.0, 1.5), 2)
            local.append(np.r_[position_error.ravel(), 100*np.minimum(distances, 0),
                               100*np.minimum(distances-0.01, 0)*np.repeat(swing_interior[i], 2),
                               weights*(x[i]-nominal[i]), 0.5*no_slip.ravel()])
        before, after = np.roll(com, 1, axis=0), np.roll(com, -1, axis=0)
        before[0, 0] -= travel
        after[-1, 0] += travel
        acceleration = (before - 2*com + after) * frequency**2
        zmp = com[:, :2] - com[:, 2:3]/9.81 * acceleration[:, :2]
        # Constraint uses a smooth weight transfer target; no root force is
        # added in dynamics tests. Conservative acceleration regularization
        # keeps the IK's contact-feature switches from making velocity spikes.
        curvature = np.roll(x, 1, axis=0) - 2*x + np.roll(x, -1, axis=0)
        def horizontal_velocity(points):
            previous, following = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
            previous[0, :, 0] -= travel
            following[-1, :, 0] += travel
            return (following[:, :, :2] - previous[:, :, :2]) * (frequency / 2)
        stance_velocity_error = horizontal_velocity(actual_landmarks) - horizontal_velocity(desired)
        stance_velocity_error *= np.repeat(contact, 2, axis=1)[:, :, None]
        rows = np.column_stack((np.asarray(local), 0.15*stance_velocity_error.reshape(count, -1),
                                1.0*(zmp-support_centers), 0.60*curvature))
        return rows.ravel()

    rows = len(residual(variables.ravel())) // count
    columns = variables.shape[1]
    sparsity = lil_matrix((count*rows, count*columns), dtype=int)
    for i in range(count):
        for neighbor in ((i-1) % count, i, (i+1) % count):
            sparsity[i*rows:(i+1)*rows, neighbor*columns:(neighbor+1)*columns] = 1
    solution = least_squares(residual, variables.ravel(), bounds=(np.tile(lower, count), np.tile(upper, count)),
                             jac_sparsity=sparsity.tocsr(), max_nfev=max_nfev, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    print(f"Whole-cycle contact refinement: {solution.message}; cost={solution.cost:.6f}", flush=True)
    return decode(solution.x.reshape(count, -1)), {"least_squares_cost": float(solution.cost),
        "optimizer_success": bool(solution.success), "function_evaluations": int(solution.nfev)}


def choose_cycle(qpos, model):
    hip = qpos[:, int(model.jnt_qposadr[model.joint("hip_flexion_r").id])]
    correlations = [np.corrcoef(hip[:-lag], hip[lag:])[0, 1] for lag in range(32, 65)]
    period = 32 + int(np.argmax(correlations))
    columns = np.r_[2, MAIN_JOINT_QPOS]
    scale = np.maximum(np.std(qpos[:, columns], axis=0), 0.05)
    pose_seam = np.mean(((qpos[period:, columns] - qpos[:-period, columns]) / scale)**2, axis=1)
    difference = np.diff(qpos[:, columns], axis=0)
    velocity_seam = np.mean(((difference[period:] - difference[:-period]) / scale)**2, axis=1)
    cost = pose_seam[:-1] + 4 * velocity_seam
    # Avoid the boundary and choose one genuine right/left walking cycle.
    cost[:period] = np.inf
    start = int(np.argmin(cost))
    return qpos[start:start+period].copy(), start, period


def audit(model, qpos, qvel, frequency, contact, translation):
    floor, geom_ids = foot_geom_ids(model)
    data = mujoco.MjData(model)
    distances = np.array([foot_clearances(model, data, pose, floor, geom_ids) for pose in qpos])
    support_distances = np.column_stack((distances[:, :2].min(1), distances[:, 2:].min(1)))
    excess = []
    for j in range(1, model.njnt):
        if model.jnt_limited[j]:
            values = qpos[:, int(model.jnt_qposadr[j])]
            excess.append(np.maximum(model.jnt_range[j, 0] - values, values - model.jnt_range[j, 1]).clip(0))
    fd = np.empty_like(qvel)
    contact_velocities = []
    interior_velocities = []
    swing_contact_frames = set()
    boundary = max(1, int(np.ceil(0.05*frequency)))
    stance_interior = np.logical_and.reduce([
        np.roll(contact > 0.5, shift, axis=0) for shift in range(-boundary, boundary+1)
    ])
    swing_interior = np.logical_and.reduce([
        np.roll(contact < 0.5, shift, axis=0) for shift in range(-boundary, boundary+1)
    ])
    jacobian = np.empty((3, model.nv))
    foot_ids = set(int(geom) for geom in geom_ids)
    for i in range(len(qpos)):
        before, after = qpos[(i-1) % len(qpos)].copy(), qpos[(i+1) % len(qpos)].copy()
        if i == 0:
            before[:2] -= translation
        if i == len(qpos)-1:
            after[:2] += translation
        mujoco.mj_differentiatePos(model, fd[i], 2/frequency, before, after)
        data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(model, data)
        for point in data.contact:
            if point.dist > 0.002:
                continue
            geom = (int(point.geom1) if point.geom1 in foot_ids and point.geom2 == floor
                    else int(point.geom2) if point.geom2 in foot_ids and point.geom1 == floor else None)
            if geom is not None:
                mujoco.mj_jac(model, data, jacobian, None, point.pos, int(model.geom_bodyid[geom]))
                velocity = (jacobian @ qvel[i])[:2]
                contact_velocities.append(velocity)
                side = int(np.flatnonzero(geom_ids == geom)[0]) // 2
                if stance_interior[i, side]:
                    interior_velocities.append(velocity)
                if swing_interior[i, side]:
                    swing_contact_frames.add((i, side))
    return {
        "joint_limit_max_excess": float(np.max(excess)),
        "minimum_foot_clearance_m": float(distances.min()),
        "maximum_support_gap_m": float(support_distances[contact > 0.5].max()),
        "frames_without_support_label": int(np.count_nonzero(contact.sum(1) < 0.5)),
        "frames_without_foot_within_2mm": int(np.count_nonzero(distances.min(1) > 0.002)),
        "qvel_differentiation_rmse": float(np.sqrt(np.mean((qvel-fd)**2))),
        "contact_horizontal_velocity_rms_m_s": float(np.sqrt(np.mean(np.sum(np.square(contact_velocities), axis=1)))) if contact_velocities else float('inf'),
        "stance_interior_velocity_rms_m_s": float(np.sqrt(np.mean(np.sum(np.square(interior_velocities), axis=1)))) if interior_velocities else float('inf'),
        "swing_interior_contact_count": len(swing_contact_frames),
        "knee_lock_fraction": float(np.mean(np.abs(qpos[:, [10, 18]]) < 0.12)),
        "cycle_translation_m": translation.tolist(),
        "last_to_next_joint_step_max_deg": float(np.rad2deg(np.abs(qpos[0, MAIN_JOINT_QPOS] - qpos[-1, MAIN_JOINT_QPOS])).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "robot/reference_data/walk1_mjcf.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "robot/reference_data/walk1_mjcf_repaired.npz")
    parser.add_argument("--model-xml", type=Path, default=ROOT / "robot/MJCFmodelfinal.xml")
    parser.add_argument("--forward-speed", type=float, default=0.87)
    parser.add_argument("--cycle-duration", type=float,
                        help="Robot gait period in seconds; source timing is used when omitted.")
    parser.add_argument("--min-knee-flexion", type=float, default=0.25,
                        help="Minimum knee flexion in radians throughout the walking cycle.")
    parser.add_argument("--refinement-max-evaluations", type=int, default=100,
                        help="Maximum whole-cycle contact refinement evaluations.")
    parser.add_argument("--test-dynamics", action="store_true", help="Also run eight 10-second free-base trials and record their result.")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from the original reference")
    if args.forward_speed <= 0:
        parser.error("forward-speed must be positive")
    if args.cycle_duration is not None and args.cycle_duration <= 0:
        parser.error("cycle-duration must be positive")
    if args.min_knee_flexion < 0.12:
        parser.error("min-knee-flexion must be at least 0.12 rad to exclude knee locking")
    if args.refinement_max_evaluations <= 0:
        parser.error("refinement-max-evaluations must be positive")
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    with np.load(args.input, allow_pickle=True) as archive:
        source = archive["qpos"].astype(float)
        frequency = float(archive["frequency"])
        names = archive["joint_names"].astype(str)
    cycle, start, period = choose_cycle(source, model)
    source_frequency = frequency
    if args.cycle_duration is not None:
        frequency = period / args.cycle_duration
    print(f"Selected source frames {start}:{start+period}, period {period/frequency:.3f}s", flush=True)
    cycle = periodic_fit(cycle)
    cycle[:, 3:7] = normalize_quaternions(cycle[:, 3:7])
    cycle[:, :2] = 0
    # Preserve the variation of each source joint while fitting its amplitude
    # to this model, rather than creating long plateaus by hard clipping.
    for j in range(1, model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        address = int(model.jnt_qposadr[j])
        low, high = model.jnt_range[j] + np.array([0.002, -0.002])
        if names[j].startswith("knee_angle"):
            high = min(high, -args.min_knee_flexion)
        values = cycle[:, address]
        if values.min() < low or values.max() > high:
            span = values.max() - values.min()
            new_span = min(span, high-low)
            center = np.clip(values.mean(), low + new_span/2, high - new_span/2)
            cycle[:, address] = center + (values - (values.min()+values.max())/2) * (new_span/max(span, 1e-12))
    # A small heel-to-toe roll is reconstructed by STL contact IK below.
    right_strike = int(np.argmax(cycle[:, int(model.jnt_qposadr[model.joint("hip_flexion_r").id])]))
    contact = np.zeros((period, 2), dtype=np.float32)
    duty = 0.60
    for side in range(2):
        phase = ((np.arange(period) - right_strike - side*period/2) % period) / period
        contact[:, side] = phase < duty
    phases, modes = foot_roll_schedule(np.vstack((contact, contact[0])))
    duplicate = np.vstack((cycle, cycle[0]))
    corrected, contact, phases, modes, stats = enforce_main_model_contacts(
        model, duplicate, np.vstack((contact, contact[0])), phases, modes,
        target_clearance=0.0005, ground_sigma=2.0, temporal_weight=0.025)
    corrected = corrected[:-1]
    contact, phases, modes = contact[:-1], phases[:-1], modes[:-1]
    corrected, refinement = refine_cycle(model, corrected, frequency, args.forward_speed, contact, phases, modes,
                                        max_nfev=args.refinement_max_evaluations, min_knee_flexion=args.min_knee_flexion)
    enforce_passive(model, corrected)
    corrected[:, :2] -= corrected[0, :2]
    qvel = np.empty((period, model.nv))
    translation = np.array([period * args.forward_speed / frequency, 0.0])
    for i in range(period):
        before, after = corrected[(i-1) % period].copy(), corrected[(i+1) % period].copy()
        if i == 0:
            before[:2] -= translation
        if i == period-1:
            after[:2] += translation
        mujoco.mj_differentiatePos(model, qvel[i], 2/frequency, before, after)
    # Audit the values actually stored in the archive, including float32
    # quantization and the exact cycle translation used at the seam.
    corrected, qvel, translation = corrected.astype(np.float32), qvel.astype(np.float32), translation.astype(np.float32)
    statistics = audit(model, corrected.astype(float), qvel.astype(float), frequency, contact, translation)
    if (statistics["joint_limit_max_excess"] > 1e-6
            or statistics["minimum_foot_clearance_m"] < -0.0005
            or statistics["maximum_support_gap_m"] > 0.003
            or statistics["frames_without_foot_within_2mm"]
            or statistics["qvel_differentiation_rmse"] > 1e-5
            # A discrete velocity at touchdown contains the impact boundary;
            # test sustained support with the same 50 ms boundary convention
            # used by gait evaluation, while retaining the full contact RMS.
            or statistics["stance_interior_velocity_rms_m_s"] > 0.15
            or statistics["swing_interior_contact_count"] > 0
            or statistics["knee_lock_fraction"] > 0.10):
        raise RuntimeError(f"Repaired cycle failed its kinematic quality gates: {statistics}")
    metadata = {
        "task": "walk", "source": str(args.input.resolve()),
        "source_frequency_hz": source_frequency, "frequency_hz": frequency,
        "minimum_knee_flexion_rad": args.min_knee_flexion,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_frames": [start, start+period-1],
        "method": "Dominant-period selection, periodic Fourier smoothing, amplitude fit to model limits, full STL heel/flat/toe contact IK, whole-cycle planted-foot and swing-path refinement with periodic smoothness and approximate ZMP planning, joint equality repair, root translation and differentiated velocities",
        "model_xml": str(args.model_xml.resolve()),
        "root_convention": "World XY translates; velocities differentiate qpos; unique cycle samples with cycle_translation",
        "cycle_translation": translation.tolist(), "cycle_duration_s": period/frequency,
        "mean_forward_velocity": args.forward_speed,
        "enforce_passive_joint_constraints": True,
        "reconstructed_motion": ["foot_contact", "mtp_angle_r", "mtp_angle_l", "subtalar_angle_r", "subtalar_angle_l", "pelvis_orientation", "support_world_positions", "swing_world_positions"],
        "dynamic_stability": "Not asserted; see separate dynamics report",
        "contact_ik": stats, "cycle_refinement": refinement, "kinematic_audit": statistics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = dict(qpos=corrected, qvel=qvel, joint_names=names, frequency=frequency,
                  loop=True, foot_contact=contact, foot_roll_phase=phases,
                  foot_roll_mode=modes, cycle_translation=translation,
                  metadata=json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(args.output, **fields)
    if args.test_dynamics:
        from scripts.test_reference_dynamics import load_states, simulate
        model.opt.timestep, model.opt.iterations, model.opt.ls_iterations = 0.005, 10, 20
        positions, velocities, hz = load_states(args.output, model, "training")
        trials, trajectories = [], {}
        for trial, phase in enumerate(np.linspace(0, period, 8, endpoint=False).astype(int)):
            result, trajectory = simulate(model, positions, velocities, hz, phase, 10.0)
            trials.append(result)
            trajectories[f"trial_{trial}"] = trajectory
        passed = sum(t["passed"] for t in trials)
        metadata["dynamic_stability"] = f"{'PASSED' if passed == 8 else 'FAILED'} reference-only free-base 10-second tests: {passed}/8 passed. This is not a test of a trained balance policy."
        fields["metadata"] = json.dumps(metadata, ensure_ascii=False)
        np.savez_compressed(args.output, **fields)
        report_path = ROOT / "recording/reference" / f"{args.output.stem}_dynamics.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = dict(motion_file=str(args.output.resolve()),
                      motion_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      model_xml=str(args.model_xml.resolve()), source="training",
                      protocol="Free base, gravity and contacts; reference tendon length targets only; no external force or policy; no resets after initialization.",
                      physics_timestep_s=0.005, control_rate_hz=100,
                      pass_criteria="Complete requested duration, pelvis >=0.5m, pelvis tilt <=60deg, average forward speed >0.2m/s, finite state.",
                      passed_trials=passed, trials=trials)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        np.savez_compressed(report_path.with_suffix(".npz"), **trajectories)
        print(f"Dynamics: {passed}/8 passed; report {report_path}", flush=True)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(statistics, indent=2), flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
