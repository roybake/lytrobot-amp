#!/usr/bin/env python3
"""Audit a robot-retargeted walking reference against its MJCF ground mesh.

This is a kinematic/contact feasibility gate, not a stability or policy test.
Run dynamics separately with scripts/test_reference_dynamics.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "robot/reference_data/walk1_mjcf_repaired.npz"
DEFAULT_MODEL = ROOT / "robot/MJCFmodelposition.xml"
FOOT_GEOMS = ("foot_r", "toe_r", "foot_l", "toe_l")


def audit_reference(model: mujoco.MjModel, reference: Path) -> dict[str, float | int]:
    with np.load(reference, allow_pickle=False) as archive:
        for key in ("qpos", "qvel", "joint_names", "frequency", "foot_contact", "cycle_translation"):
            if key not in archive:
                raise ValueError(f"{reference.name} has no {key}; retarget the human recording first.")
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        names = [str(name) for name in archive["joint_names"]]
        hz = float(archive["frequency"])
        contact = np.asarray(archive["foot_contact"], dtype=np.float64)
        translation = np.asarray(archive["cycle_translation"], dtype=np.float64)
        loop = bool(archive["loop"]) if "loop" in archive else False

    expected = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
    ]
    if names != expected:
        raise ValueError("Joint names/order do not match the target MJCF model.")
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) < 3:
        raise ValueError(f"qpos must have shape (T >= 3, {model.nq}).")
    if qvel.shape != (len(qpos), model.nv):
        raise ValueError(f"qvel must have shape ({len(qpos)}, {model.nv}).")
    if contact.shape != (len(qpos), 2):
        raise ValueError(f"foot_contact must have shape ({len(qpos)}, 2).")
    if translation.shape != (2,) or not loop:
        raise ValueError("A walking cycle needs loop=true and cycle_translation=(dx, dy).")
    if hz <= 0 or not np.isfinite(hz):
        raise ValueError("Reference frequency must be positive and finite.")
    if not all(np.isfinite(x).all() for x in (qpos, qvel, contact, translation)):
        raise ValueError("Reference contains non-finite entries.")
    if np.any((contact < 0) | (contact > 1)):
        raise ValueError("foot_contact must be in [0, 1].")

    floor = model.geom("floor").id
    feet = [model.geom(name).id for name in FOOT_GEOMS]
    data = mujoco.MjData(model)
    clearances = np.empty((len(qpos), len(feet)), dtype=np.float64)
    max_joint_excess = 0.0
    max_equality_error = 0.0
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        clearances[i] = [
            mujoco.mj_geomDistance(model, data, floor, geom, 1.0, np.zeros(6))
            for geom in feet
        ]
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                adr = int(model.jnt_qposadr[j])
                lo, hi = model.jnt_range[j]
                max_joint_excess = max(max_joint_excess, lo - pose[adr], pose[adr] - hi)
        for eq in range(model.neq):
            if model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            first, second = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
            if first < 0 or second < 0:
                continue
            a, b = int(model.jnt_qposadr[first]), int(model.jnt_qposadr[second])
            x = pose[b] - model.qpos0[b]
            expected_first = model.qpos0[a] + sum(model.eq_data[eq, k] * x**k for k in range(5))
            max_equality_error = max(max_equality_error, abs(pose[a] - expected_first))

    foot_support = np.column_stack((clearances[:, :2].min(axis=1),
                                     clearances[:, 2:].min(axis=1)))
    support = contact > 0.5
    max_support_gap = float(foot_support[support].max()) if np.any(support) else float("inf")
    velocity_rmse = 0.0
    for i in range(len(qpos)):
        previous, following = qpos[(i - 1) % len(qpos)].copy(), qpos[(i + 1) % len(qpos)].copy()
        if i == 0:
            previous[:2] -= translation
        if i == len(qpos) - 1:
            following[:2] += translation
        differentiated = np.empty(model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(model, differentiated, 2.0 / hz, previous, following)
        velocity_rmse += float(np.sum((differentiated - qvel[i]) ** 2))
    velocity_rmse = float(np.sqrt(velocity_rmse / qvel.size))
    return {
        "frames": int(len(qpos)),
        "frequency_hz": hz,
        "mean_forward_speed_mps": float(translation[0] * hz / len(qpos)),
        "minimum_foot_clearance_m": float(clearances.min()),
        "maximum_support_gap_m": max_support_gap,
        "frames_without_contact_label": int(np.count_nonzero(~support.any(axis=1))),
        "frames_without_foot_near_ground": int(np.count_nonzero(clearances.min(axis=1) > 0.002)),
        "max_joint_limit_excess_rad": float(max(0.0, max_joint_excess)),
        "max_passive_joint_equality_error": float(max_equality_error),
        "qvel_differentiation_rmse": velocity_rmse,
    }


def check_quality(report: dict[str, float | int]) -> None:
    gates = {
        "minimum_foot_clearance_m": (float(report["minimum_foot_clearance_m"]) >= -0.0005),
        "maximum_support_gap_m": (float(report["maximum_support_gap_m"]) <= 0.003),
        "frames_without_contact_label": (int(report["frames_without_contact_label"]) == 0),
        "frames_without_foot_near_ground": (int(report["frames_without_foot_near_ground"]) == 0),
        "max_joint_limit_excess_rad": (float(report["max_joint_limit_excess_rad"]) <= 1e-6),
        "max_passive_joint_equality_error": (float(report["max_passive_joint_equality_error"]) <= 1e-6),
        "qvel_differentiation_rmse": (float(report["qvel_differentiation_rmse"]) <= 1e-5),
    }
    failed = [key for key, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"Walking reference failed contact/kinematic gates: {failed}\n{report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    report = audit_reference(model, args.reference)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    check_quality(report)
    print("PASS: robot kinematics and mesh-based flat-ground contacts.")


if __name__ == "__main__":
    main()
