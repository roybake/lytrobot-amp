#!/usr/bin/env python3
"""Measure robot gait directly from a reference and its MJCF foot bodies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MODEL_XML = REPO_ROOT / "robot" / "MJCFmodelposition.xml"


def _heel_strikes(contact: np.ndarray, loop: bool) -> list[tuple[int, int]]:
    """Return (frame, side) rising contact edges, including cyclic seam."""
    edges: list[tuple[int, int]] = []
    for side in range(2):
        # A non-looping terrain clip may start mid-stance because it was
        # tiled from a cyclic walk.  That boundary is not a heel strike.
        previous = bool(contact[-1, side]) if loop else bool(contact[0, side])
        for frame, value in enumerate(contact[:, side].astype(bool)):
            if value and not previous:
                edges.append((frame, side))
            previous = value
    return sorted(edges)


def _sole_landmarks(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Return physical heel/toe sole points in foot_r/toe_r/foot_l/toe_l order."""
    body_ids: list[int] = []
    points: list[np.ndarray] = []
    for name in ("foot_r", "toe_r", "foot_l", "toe_l"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        mesh_id = int(model.geom_dataid[geom_id])
        begin = int(model.mesh_vertadr[mesh_id])
        vertices = np.asarray(model.mesh_vert[begin : begin + int(model.mesh_vertnum[mesh_id])], dtype=np.float64)
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        vertices = model.geom_pos[geom_id] + vertices @ rotation.reshape(3, 3).T
        candidates = vertices
        if name.startswith("foot"):
            xmin = float(vertices[:, 0].min())
            candidates = vertices[vertices[:, 0] <= xmin + 0.25 * float(np.ptp(vertices[:, 0]))]
        zmin = float(candidates[:, 2].min())
        point = np.mean(candidates[candidates[:, 2] <= zmin + 5.0e-4], axis=0)
        point[2] = zmin
        body_ids.append(int(model.geom_bodyid[geom_id]))
        points.append(point)
    return np.asarray(body_ids), np.asarray(points)


def analyze_reference(path: Path, model_xml: Path = MODEL_XML) -> dict[str, object]:
    """Use actual world foot-body positions; never infer steps from root travel."""
    with np.load(path) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        contact = np.asarray(archive["foot_contact"], dtype=bool)
        roll_mode = np.asarray(archive["foot_roll_mode"], dtype=np.int8)
        fps = float(archive["frequency"])
        loop = bool(np.asarray(archive["loop"]).item())
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    body_ids, local_points = _sole_landmarks(model)
    feet = np.empty((len(qpos), 2, 3), dtype=np.float64)
    landmarks_all = np.empty((len(qpos), 4, 3), dtype=np.float64)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        landmarks = np.asarray(
            [data.xpos[body] + data.xmat[body].reshape(3, 3) @ point for body, point in zip(body_ids, local_points, strict=True)]
        )
        landmarks_all[frame] = landmarks
        feet[frame] = landmarks[[0, 2]]  # physical heel locations
    duration = (len(qpos) - (0 if loop else 1)) / fps
    seam_x = float(qpos[-1, 0] - qpos[0, 0] + np.median(np.diff(qpos[:, 0]))) if loop else float(qpos[-1, 0] - qpos[0, 0])
    strikes = _heel_strikes(contact, loop)
    # Unwrap foot locations across a translating cyclic seam.
    unwrapped = [(frame, side, float(feet[frame, side, 0])) for frame, side in strikes]
    step_lengths: list[float] = []
    for (_, side_a, xa), (_, side_b, xb) in zip(unwrapped, unwrapped[1:]):
        if side_a != side_b:
            step_lengths.append(abs(xb - xa))
    if loop and len(unwrapped) >= 2 and unwrapped[-1][1] != unwrapped[0][1]:
        step_lengths.append(abs((unwrapped[0][2] + seam_x) - unwrapped[-1][2]))
    stance_durations: list[float] = []
    swing_durations: list[float] = []
    for side in range(2):
        values = contact[:, side]
        changes = np.flatnonzero(values[1:] != values[:-1]) + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, len(values)]
        for start, end in zip(starts, ends):
            (stance_durations if values[start] else swing_durations).append((end - start) / fps)
    # Different material points take over through heel -> sole -> toe.  Take
    # derivatives before selecting the active point, otherwise a legitimate
    # change of contact point becomes a false velocity spike.
    landmark_velocity = np.gradient(landmarks_all[:, :, :2], 1.0 / fps, axis=0)
    stance_speeds_by_side = np.empty((len(qpos), 2), dtype=np.float64)
    for side in range(2):
        heel_speed = np.linalg.norm(landmark_velocity[:, 2 * side], axis=1)
        toe_speed = np.linalg.norm(landmark_velocity[:, 2 * side + 1], axis=1)
        flat_speed = 0.5 * (heel_speed + toe_speed)
        stance_speeds_by_side[:, side] = np.where(
            roll_mode[:, side] == 1,
            heel_speed,
            np.where(roll_mode[:, side] == 3, toe_speed, flat_speed),
        )
    stance_speeds = stance_speeds_by_side[contact]
    metrics: dict[str, object] = {
        "mean_forward_speed_mps": seam_x / max(duration, 1.0e-9),
        "cadence_steps_per_min": len(strikes) * 60.0 / max(duration, 1.0e-9),
        "heel_strikes": [{"frame": frame, "side": "right" if side == 0 else "left"} for frame, side in strikes],
        "mean_step_length_m": float(np.mean(step_lengths)) if step_lengths else float("nan"),
        "alternating_step_lengths_m": step_lengths,
        "stride_length_m": float(2.0 * np.mean(step_lengths)) if step_lengths else float("nan"),
        "stance_duration_s": float(np.mean(stance_durations)),
        "swing_duration_s": float(np.mean(swing_durations)),
        "contact_duty_factor": float(np.mean(contact)),
        "stance_foot_horizontal_velocity_mean_mps": float(np.mean(stance_speeds)),
        "stance_slip_mean_mps": float(np.mean(stance_speeds)),
        "stance_slip_rms_mps": float(np.sqrt(np.mean(stance_speeds**2))),
        "stance_slip_p95_mps": float(np.percentile(stance_speeds, 95)),
        "stance_slip_max_mps": float(np.max(stance_speeds)),
        "qpos_finite": bool(np.isfinite(qpos).all()),
        "qvel_finite": bool(np.isfinite(qvel).all()),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--model-xml", type=Path, default=MODEL_XML)
    parser.add_argument("--json", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    metrics = analyze_reference(args.reference, args.model_xml)
    payload = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
