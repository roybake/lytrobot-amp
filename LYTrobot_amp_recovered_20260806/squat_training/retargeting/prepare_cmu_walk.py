#!/usr/bin/env python3
"""Prepare a stable CMU forward-walk cycle for OmniRetarget."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prepare_cmu_squat import JOINT_NAMES, convert, parse_amc
from prepare_walk1 import canonicalize_cycle, contact_windows, foot_roll_schedule

HERE = Path(__file__).resolve().parent
DEFAULT_ASF = HERE / "source_data" / "69.asf"
DEFAULT_AMC = HERE / "source_data" / "69_01.amc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asf", type=Path, default=DEFAULT_ASF)
    parser.add_argument("--amc", type=Path, default=DEFAULT_AMC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=263)
    parser.add_argument("--period", type=int, default=144)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--source-frequency", type=float, default=120.0)
    parser.add_argument("--target-pelvis-height", type=float, default=0.875)
    parser.add_argument("--contact-height", type=float, default=0.03)
    parser.add_argument("--contact-velocity", type=float, default=0.30)
    args = parser.parse_args()

    frame_count = len(parse_amc(args.amc))
    full_keypoints, _ = convert(args.asf, args.amc, 0, frame_count - 1, 1)
    full_keypoints = np.asarray(full_keypoints, dtype=np.float64)
    end = args.start + args.period
    if args.start < 0 or end >= frame_count:
        raise ValueError(f"Requested [{args.start}, {end}] from {frame_count} AMC frames.")
    if args.period % args.stride != 0:
        raise ValueError("--period must be divisible by --stride to preserve the exact endpoint.")

    sample_indices = np.arange(args.start, end + 1, args.stride, dtype=np.int64)
    keypoints = canonicalize_cycle(full_keypoints[sample_indices])
    right_hip = JOINT_NAMES.index("RightUpLeg")
    left_hip = JOINT_NAMES.index("LeftUpLeg")
    # The compact CMU converter swaps Y/Z without a sign change, which leaves
    # its Z-up coordinates mirrored.  The robot convention is +Y left, so the
    # right hip must have the smaller lateral coordinate.
    if np.median(keypoints[:, right_hip, 1] - keypoints[:, left_hip, 1]) > 0.0:
        keypoints[..., 1] *= -1.0
    displacement = full_keypoints[end, 0, :2] - full_keypoints[args.start, 0, :2]
    forward = displacement / np.linalg.norm(displacement)
    left = np.asarray([-forward[1], forward[0]])
    transformed_full = full_keypoints.copy()
    transformed_full[..., 0] = full_keypoints[..., :2] @ forward
    transformed_full[..., 1] = full_keypoints[..., :2] @ left

    toe_indices = (JOINT_NAMES.index("RightToeBase"), JOINT_NAMES.index("LeftToeBase"))
    toe_velocity = np.linalg.norm(
        np.gradient(
            transformed_full[:, toe_indices],
            1.0 / args.source_frequency,
            axis=0,
        ),
        axis=2,
    )[sample_indices]
    toe_height = keypoints[:, toe_indices, 2]
    toe_floor_per_side = np.min(toe_height, axis=0, keepdims=True)
    contacts = (toe_height <= toe_floor_per_side + args.contact_height) & (
        toe_velocity <= args.contact_velocity
    )
    for frame in range(len(contacts)):
        if not np.any(contacts[frame]):
            lower_side = int(np.argmin(toe_height[frame]))
            contacts[frame, lower_side] = True
    contacts[-1] = contacts[0]
    foot_roll_phase, foot_roll_mode = foot_roll_schedule(contacts)

    toe_floor = float(np.min(toe_height))
    pelvis_height = float(np.median(keypoints[:, 0, 2] - toe_floor))
    scale = args.target_pelvis_height / pelvis_height
    root_forward = transformed_full[:, 0, 0]
    forward_velocity = (
        np.gradient(root_forward, 1.0 / args.source_frequency)[sample_indices] * scale
    )
    pose_seam = float(
        np.mean(
            (
                (keypoints[-1] - keypoints[-1, 0])
                - (keypoints[0] - keypoints[0, 0])
            )
            ** 2
        )
    )
    output_frequency = args.source_frequency / args.stride

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        global_joint_positions=keypoints.astype(np.float32),
        fps=np.float32(output_frequency),
        joint_names=np.asarray(JOINT_NAMES),
        foot_contact=contacts.astype(np.float32),
        foot_roll_phase=foot_roll_phase,
        foot_roll_mode=foot_roll_mode,
        scale=np.float32(scale),
        target_forward_velocity=forward_velocity.astype(np.float32),
        source_frames=np.asarray([args.start, end, args.stride], dtype=np.int32),
        pose_seam_mse=np.float64(pose_seam),
        source=np.asarray("CMU MoCap subject 69 trial 01: walk forward"),
        source_url=np.asarray("https://mocap.cs.cmu.edu/search.php?subjectnumber=69"),
    )
    print(
        f"saved {args.output}: frames={len(keypoints)}, fps={output_frequency:g}, "
        f"scale={scale:.4f}, mean_vx={np.mean(forward_velocity):.3f}m/s, "
        f"seam_mse={pose_seam:.2e}, right_contact={contact_windows(contacts[:, 0])}, "
        f"left_contact={contact_windows(contacts[:, 1])}"
    )


if __name__ == "__main__":
    main()
