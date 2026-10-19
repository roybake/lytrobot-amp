#!/usr/bin/env python3
"""Retarget a stair demonstration against the unchanged robot and stairs.xml.

Candidates stay in results until measured collision, contact and speed checks
pass. This is a kinematic reference generator, not a trained controller.
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
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from stairs_xml_scene import load_stairs_scene, read_stair_treads, validate_reference
from omniretarget_stairs_adapter import ExactStairsRetargeter, MAPPING, DEMO_JOINTS, stair_interaction_points


def sole_points(model):
    """True STL heel/toe sole landmarks, in their respective body frames."""
    bodies, local = [], []
    for name in ("foot_r", "toe_r", "foot_l", "toe_l"):
        g = model.geom(name).id
        mi = model.geom_dataid[g]
        adr = model.mesh_vertadr[mi]
        v = np.asarray(model.mesh_vert[adr:adr + model.mesh_vertnum[mi]], dtype=float)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, model.geom_quat[g])
        v = v @ mat.reshape(3, 3).T + model.geom_pos[g]
        if name.startswith("foot"):
            v = v[v[:, 0] < v[:, 0].min() + .25 * np.ptp(v[:, 0])]
        z = v[:, 2].min()
        p = v[v[:, 2] <= z + .0005].mean(axis=0)
        p[2] = z
        bodies.append(int(model.geom_bodyid[g]))
        local.append(p)
    return np.asarray(bodies), np.asarray(local)


def landmarks(data, bodies, local):
    return np.asarray([data.xpos[b] + data.xmat[b].reshape(3, 3) @ p for b, p in zip(bodies, local)])


def couple_patella(model, qpos):
    for side in ("r", "l"):
        k = int(model.joint(f"knee_angle_{side}").qposadr[0])
        p = int(model.joint(f"patella_{side}_j1").qposadr[0])
        qpos[..., p] = .005 + .012 * qpos[..., k]


def seed_trajectory(model, human, feet, contacts, root_targets,
                    enforce_joint_limits: bool = True):
    """IK initializer with optional planning-only relaxed joint bounds."""
    data = mujoco.MjData(model)
    bodies, local = sole_points(model)
    mujoco.mj_forward(model, data)
    neutral = landmarks(data, bodies, local).reshape(2, 2, 3)
    offsets = neutral - neutral.mean(axis=1, keepdims=True)
    offsets[:, :, 2] -= np.min(offsets[:, :, 2], axis=1)[:, None]
    joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    addresses = model.jnt_qposadr[joints]
    joint_lo = np.asarray(model.jnt_range[joints, 0], dtype=float).copy()
    joint_hi = np.asarray(model.jnt_range[joints, 1], dtype=float).copy()
    if not enforce_joint_limits:
        relaxed_degrees = {
            "hip_flexion": (-65., 80.), "hip_adduction": (-35., 35.),
            "hip_rotation": (-35., 35.), "knee_angle": (-135., 0.),
            "ankle_angle": (-50., 55.), "subtalar_angle": (-30., 30.),
            "mtp_angle": (-60., 60.),
        }
        for index, joint in enumerate(joints):
            name = model.joint(joint).name
            stem = name.rsplit("_", 1)[0]
            if stem in relaxed_degrees:
                joint_lo[index], joint_hi[index] = np.deg2rad(relaxed_degrees[stem])
    lo = np.r_[[-10., -.25, .45], np.deg2rad([-8., -20.]), joint_lo]
    hi = np.r_[[10., .25, 2.4], np.deg2rad([8., 20.]), joint_hi]
    mapped_ids = [model.body(n).id for n in MAPPING.values()]
    initial_joints = model.qpos0[addresses].copy()
    if not enforce_joint_limits:
        # Start the continuation solve on the natural flexed-knee branch.
        # A straight initial guess can converge to knee hyperextension and
        # then cannot fold the first swing leg without discontinuity.
        initial_degrees = {"hip_flexion": 18., "knee_angle": -35.,
                           "ankle_angle": 17.}
        for index, joint in enumerate(joints):
            stem = model.joint(joint).name.rsplit("_", 1)[0]
            if stem in initial_degrees:
                initial_joints[index] = np.deg2rad(initial_degrees[stem])
    previous = np.r_[root_targets[0], 0., 0., initial_joints]
    poses, errors = [], []
    def put(v):
        q = model.qpos0.copy()
        q[:3] = v[:3]
        q[3:7] = Rotation.from_euler("xy", v[3:5]).as_quat()[[3, 0, 1, 2]]
        q[addresses] = v[5:]
        couple_patella(model, q)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        return q
    for frame in range(len(human)):
        target = (feet[frame, :, None] + offsets).reshape(4, 3)
        weights = np.repeat(np.where(contacts[frame] > .5, 180., 90.), 2)[:, None]
        def residual(v):
            put(v)
            return np.r_[(weights * (landmarks(data, bodies, local) - target)).ravel(),
                         3. * (data.xpos[mapped_ids] - human[frame, :9]).ravel(),
                         (v[:3] - root_targets[frame]) * [15., 25., 10.],
                         .5 * v[3:5], .20 * (v - previous)]
        frame_lo, frame_hi = lo.copy(), hi.copy()
        if frame:
            # Keep a single continuous IK branch. At 20 Hz the hinge cap is
            # 3.84 rad/s, which stays below the final 4 rad/s acceptance limit
            # while allowing the first stair's swing knee to fold in time.
            delta = np.r_[np.full(3, .05), np.deg2rad([3., 3.]),
                          np.full(len(joints), np.deg2rad(11.))]
            frame_lo = np.maximum(frame_lo, previous - delta)
            frame_hi = np.minimum(frame_hi, previous + delta)
        fit = least_squares(residual,
                            np.clip(previous, frame_lo + 1.e-8, frame_hi - 1.e-8),
                            bounds=(frame_lo, frame_hi), max_nfev=60,
                            ftol=1.e-7, xtol=1.e-7, gtol=1.e-7)
        previous = fit.x
        poses.append(put(fit.x).copy())
        errors.append(float(np.max(np.linalg.norm(landmarks(data, bodies, local) - target, axis=1))))
        if frame % 20 == 0:
            print(f"seed {frame}/{len(human)}: maximum foot error {errors[-1]:.4f} m", flush=True)
    return np.asarray(poses), np.asarray(errors)


def qvel_from_poses(model, poses, fps):
    v = np.empty((len(poses), model.nv))
    for i in range(len(poses)):
        a, b = max(0, i - 1), min(len(poses) - 1, i + 1)
        mujoco.mj_differentiatePos(model, v[i], (b - a) / fps, poses[a], poses[b])
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=HERE / "prepared/stairs_xml_human.npz")
    p.add_argument("--model-xml", type=Path, default=ROOT / "robot/MJCFmodel_stair.xml")
    p.add_argument("--terrain-xml", type=Path, default=ROOT / "robot/terrains/stairs.xml")
    p.add_argument("--output-dir", type=Path, default=HERE / "results/stairs_xml")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--reuse-seed", action="store_true")
    p.add_argument("--ignore-joint-limits", action="store_true",
                   help="Planning mode: relax hinge limits in memory and report the required envelope; XML is unchanged.")
    p.add_argument("--nominal-weight", type=float, default=10.0,
                   help="OmniRetarget weight on the collision-aware IK pose prior.")
    p.add_argument("--omniretarget-blend", type=float, default=.05,
                   help="Blend official SQP output with its exact-foot IK prior before strict validation.")
    p.add_argument("--sole-clearance", type=float, default=.0125,
                   help="Raise the sole landmark to account for lower vertices in the convex STL foot hull.")
    p.add_argument("--segment-frames", type=int, default=100,
                   help="Reset OmniRetarget at this many frames to avoid accumulated SQP drift at contact switches.")
    p.add_argument("--reuse-raw", action="store_true",
                   help="Reuse an existing raw OmniRetarget result after shape validation.")
    p.add_argument("--export", type=Path, help="Write a training reference only if all checks pass.")
    args = p.parse_args()
    if not 0.0 <= args.omniretarget_blend <= 1.0:
        p.error("--omniretarget-blend must be in [0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_stairs_scene(args.model_xml, args.terrain_xml)
    scene = args.output_dir / "exact_scene.xml"
    mujoco.mj_saveLastXML(str(scene), model)
    with np.load(args.input, allow_pickle=False) as a:
        source = {k: a[k] for k in a.files}
    try:
        source_metadata = json.loads(str(source.get("metadata", "{}")))
    except json.JSONDecodeError:
        source_metadata = {}
    human = np.asarray(source["global_joint_positions"], dtype=float)
    count = min(args.max_frames, len(human)) if args.max_frames else len(human)
    if count < 2:
        raise ValueError("At least two frames are required.")
    human = human[:count]
    feet = np.asarray(source["foot_targets"], dtype=float)[:count]
    contacts = np.asarray(source["foot_contact"], dtype=float)[:count]
    roots = np.asarray(source["root_target"], dtype=float)[:count]
    fps = float(source["fps"])
    seed_path = args.output_dir / "ik_seed.npz"
    inputs_hash = hashlib.sha256(args.input.read_bytes() + args.model_xml.read_bytes()
                                 + args.terrain_xml.read_bytes() + str(count).encode()
                                 + str(args.ignore_joint_limits).encode()
                                 + str(args.nominal_weight).encode()).hexdigest()
    if args.reuse_seed and seed_path.exists():
        with np.load(seed_path) as a:
            if str(a["inputs_sha256"]) != inputs_hash:
                raise ValueError("Cached seed was built for different inputs or frame count.")
            nominal, errors = a["qpos"], a["foot_error"]
    else:
        nominal, errors = seed_trajectory(model, human, feet, contacts, roots,
                                          enforce_joint_limits=not args.ignore_joint_limits)
        np.savez_compressed(seed_path, qpos=nominal, foot_error=errors, inputs_sha256=inputs_hash)
    engine = ExactStairsRetargeter(scene, nominal_weight=args.nominal_weight,
                                   enforce_joint_limits=not args.ignore_joint_limits)
    raw = args.output_dir / "omniretarget_raw.npz"
    points = stair_interaction_points(read_stair_treads(args.terrain_xml))
    failures = []
    # OmniRetarget's sticking constraint compares with the immediately prior
    # pose. A newly touching foot was still in swing in that prior pose, so
    # sticking begins on the following frame. The saved contact labels retain
    # the actual touchdown frame for training and validation.
    sticking_contacts = contacts.astype(bool).copy()
    sticking_contacts[1:] &= contacts[:-1].astype(bool)
    foot_sequences = [{"RightToeBase": bool(c[0]), "LeftToeBase": bool(c[1])}
                      for c in sticking_contacts]

    def retarget_chunk(start, end, depth=0):
        """Run official OmniRetarget on a bounded window; bisect a bad switch."""
        chunk_path = args.output_dir / f"omniretarget_{start:04d}_{end:04d}.npz"
        try:
            result, *_ = engine.retarget_motion(
                human_joint_motions=human[start:end],
                object_poses=nominal[start:end, -7:].copy(),
                object_poses_augmented=nominal[start:end, -7:].copy(),
                object_points_local_demo=points, object_points_local=points,
                foot_sticking_sequences=foot_sequences[start:end],
                q_a_init=nominal[start].copy(),
                q_nominal_list=nominal[start:end].copy(), original=True,
                dest_res_path=str(chunk_path))
            return np.asarray(result, dtype=float)
        except RuntimeError as exc:
            if end - start <= 8:
                # A sticking linearization can be contradictory exactly at a
                # contact switch. Retry only that tiny window without XY
                # sticking; collision and nominal interaction costs remain.
                relaxed = [{"RightToeBase": False, "LeftToeBase": False}
                           for _ in range(end - start)]
                result, *_ = engine.retarget_motion(
                    human_joint_motions=human[start:end],
                    object_poses=nominal[start:end, -7:].copy(),
                    object_poses_augmented=nominal[start:end, -7:].copy(),
                    object_points_local_demo=points, object_points_local=points,
                    foot_sticking_sequences=relaxed,
                    q_a_init=nominal[start].copy(),
                    q_nominal_list=nominal[start:end].copy(), original=True,
                    dest_res_path=str(chunk_path))
                failures.append({"start": start, "end": end,
                                 "retry": "foot_sticking_disabled", "error": str(exc)})
                return np.asarray(result, dtype=float)
            middle = start + (end - start) // 2
            failures.append({"start": start, "end": end,
                             "retry": f"bisected_at_{middle}", "error": str(exc)})
            return np.concatenate((retarget_chunk(start, middle, depth + 1),
                                   retarget_chunk(middle, end, depth + 1)))

    try:
        if args.reuse_raw and raw.exists():
            with np.load(raw, allow_pickle=False) as cached:
                raw_poses = np.asarray(cached["qpos"], dtype=float)
            if raw_poses.shape != nominal.shape:
                raise ValueError(f"Cached raw shape {raw_poses.shape} does not match {nominal.shape}.")
        else:
            segment = max(8, args.segment_frames)
            raw_poses = np.concatenate([retarget_chunk(start, min(start + segment, count))
                                        for start in range(0, count, segment)])
            np.savez_compressed(raw, qpos=raw_poses, human_joints=human, fps=fps,
                                retry_log=np.asarray(json.dumps(failures)))
    except RuntimeError as exc:
        failures.append(str(exc))
        (args.output_dir / "solver_failure.json").write_text(json.dumps({
            "status": "rejected", "error": str(exc), "inputs_sha256": inputs_hash,
            "max_seed_foot_error_m": float(errors.max()),
            "note": "No training reference exported. ik_seed.npz is an unvalidated diagnostic."
        }, indent=2) + "\n")
        raise
    # Retain OmniRetarget's interaction-preserving correction while anchoring
    # it to the exact-foot, temporally bounded IK solution. The blend is
    # validated again at intermediate poses, so it cannot hide penetration.
    poses = ((1.0 - args.omniretarget_blend) * nominal
             + args.omniretarget_blend * raw_poses)
    poses[:, 3:7] /= np.maximum(np.linalg.norm(poses[:, 3:7], axis=1, keepdims=True), 1e-12)
    # Restore the existing scalar equality constraints, which are not modeled
    # by the upstream SQP. These joints move patella meshes only, not the legs.
    couple_patella(model, poses)
    # The selected heel/toe landmarks sit above lower vertices of the convex
    # STL collision hull. Translate the trajectory and its landmark targets
    # together so the true hull touches rather than penetrates the tread.
    poses[:, 2] += args.sole_clearance
    validated_feet = feet.copy()
    validated_feet[:, :, 2] += args.sole_clearance
    metadata = {
        "retargeter": "OmniRetarget/Holosoma interaction-mesh SQP with exact XML/STL collision constraints",
        "retargeter_commit": "79d728c2c40e7f9ce8c28f9d9cab117ed2350bb5",
        "inputs_sha256": inputs_hash, "source_file": str(args.input),
        "terrain_xml": str(args.terrain_xml), "model_xml": str(args.model_xml),
        "full_course": (count == len(source["global_joint_positions"]) and
                        source_metadata.get("generated_step_count") ==
                        source_metadata.get("full_plan_step_count")),
        "kinematic_only": True, "maximum_seed_foot_error_m": float(errors.max()),
        "joint_limits_enforced": not args.ignore_joint_limits,
        "omniretarget_blend": args.omniretarget_blend,
        "omniretarget_segment_frames": args.segment_frames,
        "omniretarget_retry_log": failures,
        "sole_landmark_clearance_m": args.sole_clearance,
        "effective_foot_landmark_height_above_tread_m":
            source_metadata.get("foot_target_clearance_m", 0.0) + args.sole_clearance,
        "source_metadata": source_metadata,
    }
    payload = dict(qpos=poses.astype(np.float64), qvel=qvel_from_poses(model, poses, fps),
                   joint_names=np.asarray([model.joint(j).name for j in range(model.njnt)]),
                   frequency=np.float64(fps), fps=np.float64(fps), foot_contact=contacts,
                   foot_targets=validated_feet, foot_roll_phase=np.zeros_like(contacts),
                   foot_roll_mode=np.zeros_like(contacts, dtype=np.int8),
                   task=np.asarray("stairs_xml_up_down"), loop=np.asarray(False),
                   metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    candidate = args.output_dir / "candidate.npz"
    np.savez_compressed(candidate, **payload)
    report = validate_reference(candidate, model_xml=args.model_xml,
                                terrain_xml=args.terrain_xml,
                                report_path=args.output_dir / "validation.json", subframes=2)
    # Acceptance additionally measures the explicit planned sole positions.
    bodies, local = sole_points(model)
    d = mujoco.MjData(model)
    contact_errors = []
    for q, target, contact in zip(poses, validated_feet, contacts):
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        actual = landmarks(d, bodies, local).reshape(2, 2, 3).mean(axis=1)
        contact_errors.extend(np.linalg.norm(actual[contact > .5] - target[contact > .5], axis=1).tolist())
    report["maximum_planned_contact_error_m"] = max(contact_errors, default=0.)
    report["full_course"] = metadata["full_course"]
    required_envelope = {}
    for joint in range(model.njnt):
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        address = int(model.jnt_qposadr[joint])
        required_envelope[model.joint(joint).name] = {
            "minimum_deg": float(np.rad2deg(poses[:, address].min())),
            "maximum_deg": float(np.rad2deg(poses[:, address].max())),
            "current_model_minimum_deg": float(np.rad2deg(model.jnt_range[joint, 0])),
            "current_model_maximum_deg": float(np.rad2deg(model.jnt_range[joint, 1])),
        }
    report["required_joint_envelope"] = required_envelope
    report["joint_limits_enforced"] = not args.ignore_joint_limits
    if args.ignore_joint_limits:
        accepted = bool(not report["terrain_penetrating_samples"] and
                        not report["self_penetrating_samples"] and
                        report["quaternion_norm_max_error"] < 1e-5 and
                        max(report["equality_max_error_m"].values()) < 1e-5)
    else:
        accepted = bool(report["kinematic_checks_passed"])
    accepted &= report["maximum_planned_contact_error_m"] <= .015
    accepted &= report["root_roll_abs_max_deg"] <= 10.0
    accepted &= report["root_pitch_abs_max_deg"] <= 20.0
    accepted &= report["joint_speed_rad_s"]["max"] <= 4.0
    accepted &= report["root_speed_m_s"]["max"] <= 1.2
    accepted &= .10 <= report["root_mean_forward_speed_m_s"] <= .60
    for slip in report["stance_landmark_planar_speed_m_s"].values():
        accepted &= slip["p95"] <= .05
        accepted &= slip["max_contact_bout_drift_m"] <= .015
    accepted &= metadata["full_course"]
    report["acceptance_limits"] = {
        "maximum_planned_contact_error_m": .015,
        "maximum_root_roll_deg": 10.0,
        "maximum_root_pitch_deg": 20.0,
        "maximum_joint_speed_rad_s": 4.0,
        "maximum_root_speed_m_s": 1.2,
        "mean_forward_speed_range_m_s": [.10, .60],
        "stance_p95_planar_speed_m_s": .05,
        "stance_bout_drift_m": .015,
        "requires_full_course": True,
    }
    report["accepted_for_export"] = bool(accepted)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.export:
        if not accepted:
            raise SystemExit("Reference rejected; candidate and report retained for diagnosis. Nothing exported.")
        args.export.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.export, **payload)


if __name__ == "__main__":
    main()
