#!/usr/bin/env python3
"""Test free-base MuJoCo walking with reference tendon controls and no policy.

Only the initial state is prescribed. Gravity, contact, passive joints and
actuator force limits remain active during mj_step; there is no root wrench,
pose reset, stabilizing tether or learned policy. This measures reference-only
control, not the stability of a future trained policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lytrobot_amp.motion import MotionReference, resolve_contact_geom_ids


def load_states(path, model, source):
    if source == "training":
        ref = MotionReference(path, model, zero_root_xy=False, loop_override=True)
        return ref.reference_qpos.astype(float), ref.reference_qvel.astype(float), ref.frequency
    with np.load(path, allow_pickle=True) as archive:
        names = archive["joint_names"].tolist()
        expected = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
        if names != expected:
            raise ValueError("Reference joint layout does not match model")
        return archive["qpos"].astype(float), archive["qvel"].astype(float), float(archive["frequency"])


def simulate(model, qpos, qvel, frequency, start_frame, seconds):
    data, target = mujoco.MjData(model), mujoco.MjData(model)
    count = len(qpos)
    # A translating clip contains one unique period, without a held endpoint.
    translation = np.ptp(qpos[:, :2], axis=0).max() > 1e-5
    displacement = np.sum(qvel[:, :2], axis=0) / frequency if translation else np.zeros(2)
    data.qpos[:] = qpos[start_frame]
    data.qpos[:2] = 0.0
    data.qvel[:] = qvel[start_frame]
    mujoco.mj_forward(model, data)
    root_id = model.body("pelvis").id
    foot_ids = set(resolve_contact_geom_ids(model).tolist())
    floor_id = model.geom("floor").id
    joint_addresses = [int(model.jnt_qposadr[j]) for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    tendon_ids = model.actuator_trnid[:, 0]
    ctrl_low, ctrl_high = model.actuator_ctrlrange.T
    # Same 100 Hz control rate and 200 Hz physics as the training defaults.
    control_steps = max(1, int(round(0.01 / model.opt.timestep)))
    trajectory = []
    joint_errors, velocity_errors, saturations = [], [], []
    min_height, max_tilt, max_penetration = float(data.qpos[2]), 0.0, 0.0
    unsupported, longest_unsupported = 0, 0
    reason = None
    max_steps = int(round(seconds / model.opt.timestep))
    for step in range(max_steps):
        phase = start_frame + data.time * frequency
        low = int(np.floor(phase)) % count
        high = (low + 1) % count
        weight = phase % 1.0
        target.qpos[:] = (1 - weight) * qpos[low] + weight * qpos[high]
        target.qpos[3:7] /= np.linalg.norm(target.qpos[3:7])
        target.qvel[:] = (1 - weight) * qvel[low] + weight * qvel[high]
        if translation:
            target.qpos[:2] += (int(np.floor(phase / count)) * displacement - qpos[start_frame, :2])
            if high == 0:
                target.qpos[:2] += weight * displacement
        if step % control_steps == 0:
            mujoco.mj_forward(model, target)
            data.ctrl[:] = np.clip(target.ten_length[tendon_ids], ctrl_low, ctrl_high)
        mujoco.mj_step(model, data)
        # Refresh Cartesian states for metrics without changing qpos/qvel.
        mujoco.mj_forward(model, data)
        tilt = float(np.rad2deg(np.arccos(np.clip(data.xmat[root_id, 8], -1, 1))))
        min_height, max_tilt = min(min_height, float(data.qpos[2])), max(max_tilt, tilt)
        contacts = [c for c in data.contact if (c.geom1 in foot_ids and c.geom2 == floor_id) or (c.geom2 in foot_ids and c.geom1 == floor_id)]
        max_penetration = max(max_penetration, -min([float(c.dist) for c in contacts] + [0.0]))
        supported = any(c.dist <= 0.002 for c in contacts)
        unsupported = 0 if supported else unsupported + 1
        longest_unsupported = max(longest_unsupported, unsupported)
        joint_errors.extend((data.qpos[joint_addresses] - target.qpos[joint_addresses]).tolist())
        velocity_errors.append(float(data.qvel[0] - target.qvel[0]))
        force_low, force_high = model.actuator_forcerange.T
        # Unilateral actuators resting at zero force are not overloaded.
        saturations.append(float(np.mean(data.actuator_force <= force_low + 1e-3)))
        if step % max(1, int(round(0.025 / model.opt.timestep))) == 0:
            trajectory.append(np.concatenate(([data.time], data.qpos, data.qvel)))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            reason = "nonfinite_state"
        elif data.qpos[2] < 0.5:
            reason = "pelvis_below_0.5m"
        elif tilt > 60:
            reason = "pelvis_tilt_over_60deg"
        if reason:
            break
    elapsed = float(data.time)
    passed = reason is None and data.qpos[0] > 0.2 * seconds
    return {
        "start_frame": int(start_frame), "requested_seconds": seconds,
        "simulated_seconds": elapsed, "passed": bool(passed),
        "failure_reason": reason or (None if passed else "insufficient_forward_progress"),
        "forward_distance_m": float(data.qpos[0]),
        "mean_forward_speed_mps": float(data.qpos[0] / max(elapsed, 1e-9)),
        "minimum_pelvis_height_m": min_height, "maximum_tilt_deg": max_tilt,
        "maximum_foot_penetration_m": max_penetration,
        "longest_no_foot_contact_seconds": longest_unsupported * model.opt.timestep,
        "joint_tracking_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(joint_errors))))),
        "forward_velocity_tracking_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_errors)))),
        "mean_force_limit_fraction": float(np.mean(saturations)),
        "warning_counts": [int(w.number) for w in data.warning],
    }, np.asarray(trajectory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--model-xml", type=Path, default=ROOT / "robot/MJCFmodelposition.xml")
    parser.add_argument("--source", choices=("raw", "training"), default="training")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--timestep", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.trials < 1 or args.timestep <= 0:
        parser.error("seconds, trials and timestep must be positive")
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    model.opt.timestep, model.opt.iterations, model.opt.ls_iterations = args.timestep, 10, 20
    qpos, qvel, frequency = load_states(args.motion_file, model, args.source)
    trials, trajectories = [], {}
    for trial, phase in enumerate(np.linspace(0, len(qpos), args.trials, endpoint=False).astype(int)):
        result, trajectory = simulate(model, qpos, qvel, frequency, phase, args.seconds)
        trials.append(result)
        trajectories[f"trial_{trial}"] = trajectory
        print(json.dumps(result), flush=True)
    report = {
        "motion_file": str(args.motion_file.resolve()),
        "motion_sha256": hashlib.sha256(args.motion_file.read_bytes()).hexdigest(),
        "model_xml": str(args.model_xml.resolve()), "source": args.source,
        "protocol": "Free base, gravity and contacts; reference tendon length targets only; no external force or policy; no resets after initialization.",
        "physics_timestep_s": args.timestep, "control_rate_hz": 100,
        "pass_criteria": "Complete requested duration, pelvis >=0.5m, pelvis tilt <=60deg, average forward speed >0.2m/s, finite state.",
        "passed_trials": sum(t["passed"] for t in trials), "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    np.savez_compressed(args.output.with_suffix(".npz"), **trajectories)
    print(f"Passed {report['passed_trials']}/{len(trials)}; saved {args.output}")


if __name__ == "__main__":
    main()
