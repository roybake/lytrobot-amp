#!/usr/bin/env python3
"""Recompute non-looping motion archives' root-local angular velocities in place."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def root_model() -> mujoco.MjModel:
    """Only the free-joint coordinate layout is needed to differentiate rotation."""
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/>'
        '<geom type="sphere" size="0.01" mass="1"/>'
        '</body></worldbody></mujoco>'
    )


def recompute_angular_velocity(
    qpos: np.ndarray, frequency: float, model: mujoco.MjModel
) -> np.ndarray:
    """Differentiate about each current pose, in that pose's local frame."""
    if qpos.ndim != 2 or qpos.shape[1] < 7 or len(qpos) < 2:
        raise ValueError("Expected at least two poses with a free-joint prefix")
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be finite and positive")
    poses = np.array(qpos[:, :7], dtype=np.float64, copy=True)
    norms = np.linalg.norm(poses[:, 3:7], axis=1, keepdims=True)
    if not np.isfinite(poses).all() or np.any(norms <= 0):
        raise ValueError("Root poses must be finite with nonzero quaternions")
    poses[:, 3:7] /= norms
    dt = 1.0 / frequency
    angular_velocity = np.empty((len(poses), 3), dtype=np.float64)
    forward, reverse = np.empty(6), np.empty(6)
    for i in range(len(poses)):
        if i < len(poses) - 1:
            mujoco.mj_differentiatePos(model, forward, dt, poses[i], poses[i + 1])
        if i > 0:
            mujoco.mj_differentiatePos(model, reverse, dt, poses[i], poses[i - 1])
        if i == 0:
            angular_velocity[i] = forward[3:6]
        elif i == len(poses) - 1:
            angular_velocity[i] = -reverse[3:6]
        else:
            # Both differences start at pose i, so their tangent frames match.
            angular_velocity[i] = 0.5 * (forward[3:6] - reverse[3:6])
    return angular_velocity


def orientation_prediction_error(
    qpos: np.ndarray, angular_velocity: np.ndarray,
    frequency: float, model: mujoco.MjModel,
) -> dict[str, float]:
    """Independently check velocities by integrating to the next saved pose."""
    errors = np.empty(len(qpos) - 1)
    velocity, error = np.zeros(6), np.empty(3)
    for i in range(len(errors)):
        pose = np.array(qpos[i, :7], dtype=np.float64, copy=True)
        pose[3:7] /= np.linalg.norm(pose[3:7])
        target = np.asarray(qpos[i + 1, 3:7], dtype=np.float64)
        target = target / np.linalg.norm(target)
        velocity[3:6] = angular_velocity[i]
        mujoco.mj_integratePos(model, pose, velocity, 1.0 / frequency)
        mujoco.mju_subQuat(error, pose[3:7], target)
        errors[i] = np.rad2deg(np.linalg.norm(error))
    return {
        "rmse_deg": float(np.sqrt(np.mean(errors**2))),
        "median_deg": float(np.median(errors)),
        "p99_deg": float(np.quantile(errors, 0.99)),
    }


def write_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    mode = path.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
        try:
            np.savez(f, **arrays)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.chmod(temporary, mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "motion_files", nargs="*", type=Path,
        default=[ROOT / "robot/reference_data" / name
                 for name in ("walk1.npz", "walk1_mjcf.npz")],
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "logs/data_audit/walk1_root_angular_velocity.json",
    )
    args = parser.parse_args()
    model = root_model()
    prepared = []
    reports = []
    for path in args.motion_files:
        path = path.resolve()
        with np.load(path, allow_pickle=True) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if "loop" in arrays and bool(arrays["loop"].item()):
            raise ValueError(f"{path}: this exporter requires a non-looping trajectory")
        qpos, qvel = arrays["qpos"], arrays["qvel"]
        if qvel.ndim != 2 or qvel.shape[1] < 6 or len(qpos) != len(qvel):
            raise ValueError(f"{path}: qpos/qvel layout or frame counts do not match")
        if not np.issubdtype(qvel.dtype, np.floating) or not np.isfinite(qvel).all():
            raise ValueError(f"{path}: qvel must contain finite floating-point values")
        frequency = float(arrays["frequency"].item())
        angular_velocity = recompute_angular_velocity(qpos, frequency, model)
        updated = qvel.copy()
        updated[:, 3:6] = angular_velocity
        reports.append({
            "motion_file": str(path), "frames": len(qpos), "frequency": frequency,
            "angular_velocity_frame": "root_local",
            "method": "pose-centered bidirectional MuJoCo differences; one-sided endpoints",
            "before": orientation_prediction_error(qpos, qvel[:, 3:6], frequency, model),
            "after": orientation_prediction_error(qpos, updated[:, 3:6], frequency, model),
        })
        arrays["qvel"] = updated
        prepared.append((path, arrays))
    for path, arrays in prepared:
        write_archive(path, arrays)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        print(f"{Path(report['motion_file']).name}: orientation prediction RMSE "
              f"{report['before']['rmse_deg']:.6f} -> {report['after']['rmse_deg']:.6f} deg")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
