#!/usr/bin/env python3
"""Validate multi-motion leg clearance, turn rate and ordered foot contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lytrobot_amp.benchmark import (  # noqa: E402
    MIN_WALKING_SPEED_MPS,
    TRAINING_TARGET_SPEED_MPS,
)

MODEL_XML = REPO_ROOT / "robot" / "MJCFmodelposition.xml"
MOTION_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
DEFAULT_REPORT = (
    REPO_ROOT / "recording" / "reference" / "multimotion_validation.json"
)
CLIPS = (
    "walk",
    "turn_left",
    "turn_right",
    "rough_walk",
    "pit_walk",
    "gap_cross",
    "stair_up",
    "stair_down",
)
BODY_PAIRS = (
    ("femur_r", "femur_l"),
    ("tibia_r", "tibia_l"),
    ("foot_r", "foot_l"),
    ("foot_r", "toe_l"),
    ("toe_r", "foot_l"),
    ("toe_r", "toe_l"),
)


def geom_for_body(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    ids = np.flatnonzero(np.asarray(model.geom_bodyid) == body_id)
    if body_id < 0 or len(ids) == 0:
        raise ValueError(f"No geometry found for body {name!r}.")
    return int(ids[0])


def clip_metrics(model: mujoco.MjModel, motion_path: Path) -> dict[str, object]:
    with np.load(motion_path) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        frequency = float(archive["frequency"])
        loop = bool(np.asarray(archive["loop"]).item())
        metadata = json.loads(str(archive["metadata"]))

    data = mujoco.MjData(model)
    pair_ids = [
        (geom_for_body(model, first), geom_for_body(model, second))
        for first, second in BODY_PAIRS
    ]
    fromto = np.zeros(6, dtype=np.float64)
    distances = np.zeros((len(qpos), len(pair_ids)), dtype=np.float64)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        distances[frame] = [
            mujoco.mj_geomDistance(
                model,
                data,
                first,
                second,
                0.20,
                fromto,
            )
            for first, second in pair_ids
        ]

    quat = qpos[:, 3:7]
    yaw = np.unwrap(
        np.arctan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
    )
    duration = (len(qpos) - (0 if loop else 1)) / frequency
    forward_displacement = float(qpos[-1, 0] - qpos[0, 0])
    if loop:
        forward_displacement += float(np.median(np.diff(qpos[:, 0])))
    return {
        "frames": int(len(qpos)),
        "frequency_hz": frequency,
        "duration_s": duration,
        "mean_forward_speed_mps": forward_displacement / max(duration, 1.0e-6),
        "yaw_change_deg": float(np.rad2deg(yaw[-1] - yaw[0])),
        "maximum_yaw_rate_deg_s": float(np.rad2deg(np.max(np.abs(qvel[:, 5])))),
        "minimum_cross_leg_distance_m": float(distances.min()),
        "penetration_frames": int(np.count_nonzero(np.min(distances, axis=1) < -1.0e-5)),
        "heel_phase_toe_minus_heel_min_m": float(
            metadata["heel_phase_toe_minus_heel_min"]
        ),
        "toe_phase_toe_minus_heel_max_m": float(
            metadata["toe_phase_toe_minus_heel_max"]
        ),
        "maximum_roll_target_error_m": float(
            metadata["maximum_roll_target_error"]
        ),
        "terrain_kind": metadata.get("terrain_kind", "flat"),
        "support_clearance_min_m": metadata.get("support_clearance_min"),
        "support_clearance_max_m": metadata.get("support_clearance_max"),
        "unsupported_gap_contact_frames": metadata.get(
            "unsupported_gap_contact_frames", 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", type=Path, default=MODEL_XML)
    parser.add_argument("--motion-dir", type=Path, default=MOTION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    clips = {
        clip: clip_metrics(
            model,
            args.motion_dir / f"{clip}_omniretarget.npz",
        )
        for clip in CLIPS
    }
    failures: list[str] = []
    warnings: list[str] = []
    for clip, metrics in clips.items():
        if metrics["frequency_hz"] != 40.0:
            failures.append(f"{clip}: expected 40 Hz")
        if metrics["penetration_frames"] != 0:
            failures.append(
                f"{clip}: {metrics['penetration_frames']} cross-leg penetration frames"
            )
        if metrics["heel_phase_toe_minus_heel_min_m"] < -0.002:
            failures.append(f"{clip}: toe lands before heel during heel phase")
        if metrics["toe_phase_toe_minus_heel_max_m"] > 0.002:
            failures.append(f"{clip}: heel remains below toe during toe phase")
        if metrics["unsupported_gap_contact_frames"] != 0:
            failures.append(f"{clip}: a support foot lands in the gap")
        if (
            metrics["support_clearance_min_m"] is not None
            and metrics["support_clearance_min_m"] < -0.003
        ):
            failures.append(f"{clip}: support foot penetrates terrain by over 3 mm")
    for turn in ("turn_left", "turn_right"):
        if clips[turn]["duration_s"] < 3.5:
            failures.append(f"{turn}: duration remains too short")
        if clips[turn]["maximum_yaw_rate_deg_s"] > 50.0:
            failures.append(f"{turn}: yaw rate exceeds 50 deg/s")
        if abs(clips[turn]["yaw_change_deg"]) > 90.0 + 0.1:
            failures.append(f"{turn}: turn angle exceeds 90 degrees")
    if clips["turn_left"]["yaw_change_deg"] <= 0.0:
        failures.append("turn_left: expected positive Z-up yaw")
    if clips["turn_right"]["yaw_change_deg"] >= 0.0:
        failures.append("turn_right: expected negative Z-up yaw")
    for locomotion in ("walk", "rough_walk", "pit_walk", "gap_cross"):
        speed = float(clips[locomotion]["mean_forward_speed_mps"])
        if abs(speed - TRAINING_TARGET_SPEED_MPS) > 0.02:
            failures.append(
                f"{locomotion}: forward speed {speed:.3f} m/s does not match "
                f"{TRAINING_TARGET_SPEED_MPS:.3f} m/s"
            )
    if TRAINING_TARGET_SPEED_MPS < MIN_WALKING_SPEED_MPS:
        warnings.append(
            "Native reference speed "
            f"{TRAINING_TARGET_SPEED_MPS:.3f} m/s is below the external "
            f"acceptance minimum {MIN_WALKING_SPEED_MPS:.3f} m/s."
        )

    report = {
        "passed": not failures,
        "warnings": warnings,
        "criteria": {
            "cross_leg_penetration_frames": 0,
            "turn_maximum_yaw_rate_deg_s": 50.0,
            "turn_maximum_angle_deg": 90.0,
            "heel_then_toe_tolerance_m": 0.002,
            "output_frequency_hz": 40.0,
            "locomotion_target_speed_mps": TRAINING_TARGET_SPEED_MPS,
            "locomotion_speed_tolerance_mps": 0.02,
            "locomotion_minimum_speed_mps": MIN_WALKING_SPEED_MPS,
            "training_speed_meets_acceptance_minimum": (
                TRAINING_TARGET_SPEED_MPS >= MIN_WALKING_SPEED_MPS
            ),
        },
        "clips": clips,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
