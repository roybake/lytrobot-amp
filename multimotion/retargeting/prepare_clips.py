#!/usr/bin/env python3
"""Prepare CMU walk, left/right turns and stair clips for OmniRetarget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LEGACY = REPO_ROOT / "squat_training" / "retargeting"
sys.path.insert(0, str(LEGACY))

from prepare_cmu_squat import JOINT_NAMES, convert, parse_amc  # noqa: E402
from prepare_walk1 import foot_roll_schedule  # noqa: E402


CLIPS = {
    "walk": ("69", "69_01", 263, 407, 3, True),
    # Positive Z-up yaw is a left turn; negative yaw is a right turn.  These
    # source intervals finish at a stable heading and remain strictly below
    # the requested 90-degree limit after stride-6 sampling.
    "turn_left": ("69", "69_06", 2260, 2520, 6, False),
    "turn_right": ("69", "69_06", 3550, 3790, 6, False),
    "stair_up": ("114", "114_07", 300, 720, 6, False),
    "stair_down": ("114", "114_07", 900, 1370, 6, False),
}


def heading_from_hips(keypoints: np.ndarray) -> np.ndarray:
    right = keypoints[:, JOINT_NAMES.index("RightUpLeg"), :2]
    left = keypoints[:, JOINT_NAMES.index("LeftUpLeg"), :2]
    lateral = left - right
    # In the right-handed Z-up frame, the forward vector is the clockwise
    # perpendicular of the right-to-left hip vector: [dy, -dx].
    return np.unwrap(np.arctan2(-lateral[:, 0], lateral[:, 1]))


def canonicalize(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(keypoints, dtype=np.float64).copy()
    yaw = heading_from_hips(result)
    angle = -float(np.median(yaw[: min(10, len(yaw))]))
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    origin = result[0, 0, :2].copy()
    result[..., :2] = (result[..., :2] - origin) @ rotation.T
    yaw = np.unwrap(heading_from_hips(result))
    yaw -= yaw[0]
    return result, yaw


def infer_contacts(
    keypoints: np.ndarray,
    frequency: float,
    velocity_threshold: float,
) -> np.ndarray:
    toes = keypoints[
        :,
        [JOINT_NAMES.index("RightToeBase"), JOINT_NAMES.index("LeftToeBase")],
    ]
    speed = np.linalg.norm(np.gradient(toes, 1.0 / frequency, axis=0), axis=2)
    contacts = speed <= velocity_threshold
    for frame in range(len(contacts)):
        if not np.any(contacts[frame]):
            contacts[frame, int(np.argmin(speed[frame]))] = True
    return contacts


def prepare(name: str, source_dir: Path, output_dir: Path) -> Path:
    subject, trial, start, end, stride, loop = CLIPS[name]
    asf = source_dir / f"{subject}.asf"
    amc = source_dir / f"{trial}.amc"
    frame_count = len(parse_amc(amc))
    if end >= frame_count:
        raise ValueError(f"{trial}: requested end={end}, available frames={frame_count}.")
    full, _ = convert(asf, amc, 0, frame_count - 1, 1)
    indices = np.arange(start, end + 1, stride, dtype=np.int64)
    keypoints, yaw = canonicalize(np.asarray(full)[indices])

    source_frequency = 120.0
    frequency = source_frequency / stride
    contacts = infer_contacts(keypoints, frequency, velocity_threshold=0.22)
    if loop:
        keypoints[-1] = keypoints[0]
        contacts[-1] = contacts[0]
        yaw[-1] = yaw[0]
    roll_phase, roll_mode = foot_roll_schedule(contacts, cyclic=loop)

    toe_ids = [JOINT_NAMES.index("RightToeBase"), JOINT_NAMES.index("LeftToeBase")]
    ground = float(np.min(keypoints[:, toe_ids, 2]))
    initial = slice(0, min(12, len(keypoints)))
    human_pelvis_height = float(
        np.median(keypoints[initial, 0, 2] - np.min(keypoints[initial][:, toe_ids, 2]))
    )
    scale = 0.875 / human_pelvis_height
    root_xy = keypoints[:, 0, :2]
    root_velocity = np.gradient(root_xy, 1.0 / frequency, axis=0) * scale
    yaw_rate = np.gradient(yaw, 1.0 / frequency)  # radians/second

    output = output_dir / f"{name}_human.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        global_joint_positions=keypoints.astype(np.float32),
        fps=np.float32(frequency),
        joint_names=np.asarray(JOINT_NAMES),
        foot_contact=contacts.astype(np.float32),
        foot_roll_phase=roll_phase,
        foot_roll_mode=roll_mode,
        scale=np.float32(scale),
        target_root_velocity=np.column_stack(
            (root_velocity, np.zeros(len(keypoints), dtype=np.float32))
        ).astype(np.float32),
        target_yaw_rate=yaw_rate.astype(np.float32),
        target_yaw=yaw.astype(np.float32),
        source_frames=np.asarray([start, end, stride], dtype=np.int32),
        source=np.asarray(
            f"CMU MoCap subject {subject} trial {trial.split('_')[-1]}"
        ),
        source_url=np.asarray(
            f"https://mocap.cs.cmu.edu/search.php?subjectnumber={subject}"
        ),
        task=np.asarray(name),
        loop=np.asarray(loop),
        source_ground_z=np.float32(ground),
    )
    print(
        f"{name}: frames={len(keypoints)}, fps={frequency:g}, scale={scale:.4f}, "
        f"distance={np.linalg.norm(root_xy[-1] - root_xy[0]) * scale:.3f}m, "
        f"yaw={np.rad2deg(yaw[-1] - yaw[0]):.1f}deg -> {output}"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=HERE / "source_data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "prepared")
    parser.add_argument("--clips", nargs="+", choices=tuple(CLIPS), default=tuple(CLIPS))
    args = parser.parse_args()
    for name in args.clips:
        prepare(name, args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
