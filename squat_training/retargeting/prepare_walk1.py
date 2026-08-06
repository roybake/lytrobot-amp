#!/usr/bin/env python3
"""Prepare a closed LAFAN walk1 human-keypoint cycle for OmniRetarget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_GMR_ROOT = Path("/home/yeah/code/unitreeG1/GMR-master")
DEFAULT_BVH = DEFAULT_GMR_ROOT / "data" / "lafan1" / "walk1_subject1.bvh"

DEMO_JOINTS = (
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "Spine",
    "Spine1",
)
BVH_JOINTS = (
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "Spine",
    "Spine1",
)

ROLL_SWING = 0
ROLL_HEEL = 1
ROLL_FLAT = 2
ROLL_TOE = 3
HEEL_PHASE_END = 0.20
TOE_PHASE_START = 0.75


def contact_windows(contact: np.ndarray) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start: int | None = None
    for frame, active in enumerate(np.r_[contact, False]):
        if active and start is None:
            start = frame
        elif not active and start is not None:
            windows.append((start, frame - 1))
            start = None
    return windows


def foot_roll_schedule(
    contacts: np.ndarray,
    *,
    cyclic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign a heel-to-flat-to-toe phase to each stance interval.

    ``cyclic=False`` is required for finite clips.  Treating a finite stair or
    turn clip as circular can join the final stance to the initial stance and
    incorrectly label its first frames as toe-off instead of heel strike.
    """
    contacts = np.asarray(contacts, dtype=bool)
    if contacts.ndim != 2 or contacts.shape[1] != 2:
        raise ValueError(f"Expected contact shape (T, 2), got {contacts.shape}.")
    duplicate_endpoint = (
        cyclic
        and len(contacts) > 1
        and np.array_equal(contacts[0], contacts[-1])
    )
    unique = contacts[:-1] if duplicate_endpoint else contacts
    phases = np.full(unique.shape, -1.0, dtype=np.float32)
    modes = np.full(unique.shape, ROLL_SWING, dtype=np.int8)

    for side in range(2):
        active = unique[:, side]
        if cyclic:
            starts = np.flatnonzero(active & ~np.roll(active, 1))
        else:
            previous = np.r_[False, active[:-1]]
            starts = np.flatnonzero(active & ~previous)
        for start in starts:
            indices: list[int] = []
            frame = int(start)
            while (
                frame < len(active)
                and active[frame]
                and len(indices) < len(active)
            ):
                indices.append(frame)
                frame = (frame + 1) % len(active) if cyclic else frame + 1
            run_phase = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
            phases[indices, side] = run_phase
            modes[np.asarray(indices)[run_phase < HEEL_PHASE_END], side] = ROLL_HEEL
            modes[
                np.asarray(indices)[
                    (run_phase >= HEEL_PHASE_END) & (run_phase < TOE_PHASE_START)
                ],
                side,
            ] = ROLL_FLAT
            modes[np.asarray(indices)[run_phase >= TOE_PHASE_START], side] = ROLL_TOE

    if np.any(unique & (phases < 0.0)):
        raise ValueError("Could not assign a roll phase to every stance frame.")
    if duplicate_endpoint:
        phases = np.vstack((phases, phases[0]))
        modes = np.vstack((modes, modes[0]))
    return phases, modes


def load_lafan_keypoints(bvh: Path, gmr_root: Path) -> tuple[np.ndarray, list[str]]:
    vendor_root = gmr_root / "general_motion_retargeting" / "utils"
    if not vendor_root.exists():
        raise FileNotFoundError(f"GMR LAFAN parser not found: {vendor_root}")
    sys.path.insert(0, str(vendor_root))
    from lafan_vendor import utils  # type: ignore[import-not-found]
    from lafan_vendor.extract import read_bvh  # type: ignore[import-not-found]

    animation = read_bvh(str(bvh))
    _, global_positions = utils.quat_fk(animation.quats, animation.pos, animation.parents)
    # LAFAN is Y-up in centimetres. Rotate to right-handed Z-up metres using
    # the same convention as GMR's own LAFAN loader: [x, y, z] -> [x, -z, y].
    global_positions = np.asarray(global_positions, dtype=np.float64) / 100.0
    global_positions = global_positions[..., [0, 2, 1]]
    global_positions[..., 1] *= -1.0
    return global_positions, list(animation.bones)


def canonicalize_cycle(keypoints: np.ndarray) -> np.ndarray:
    result = keypoints.copy()
    displacement = result[-1, 0, :2] - result[0, 0, :2]
    distance = float(np.linalg.norm(displacement))
    if distance < 1.0e-6:
        raise ValueError("Selected cycle has no horizontal root displacement.")
    forward = displacement / distance
    left = np.asarray([-forward[1], forward[0]])
    centered_xy = result[..., :2] - result[0, 0, :2]
    result[..., 0] = centered_xy @ forward
    result[..., 1] = centered_xy @ left
    return result


def infer_contacts(
    full_keypoints: np.ndarray,
    start: int,
    stop: int,
    frequency: float,
    height_threshold: float,
    velocity_threshold: float,
) -> np.ndarray:
    toe_indices = (DEMO_JOINTS.index("RightToeBase"), DEMO_JOINTS.index("LeftToeBase"))
    toe_velocity = np.linalg.norm(
        np.gradient(full_keypoints[:, toe_indices], 1.0 / frequency, axis=0), axis=2
    )[start:stop]
    toe_height = full_keypoints[start:stop, toe_indices, 2]
    height_floor = np.min(toe_height, axis=0, keepdims=True)
    return (toe_height <= height_floor + height_threshold) & (
        toe_velocity <= velocity_threshold
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, default=DEFAULT_BVH)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=210)
    parser.add_argument("--period", type=int, default=44)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--target-pelvis-height", type=float, default=0.875)
    parser.add_argument("--contact-height", type=float, default=0.03)
    parser.add_argument("--contact-velocity", type=float, default=0.30)
    args = parser.parse_args()

    global_positions, bone_names = load_lafan_keypoints(args.bvh, args.gmr_root)
    missing = [name for name in BVH_JOINTS if name not in bone_names]
    if missing:
        raise ValueError(f"BVH joints missing: {missing}")
    full_keypoints = global_positions[:, [bone_names.index(name) for name in BVH_JOINTS]]
    stop = args.start + args.period + 1
    if args.start < 0 or stop > len(full_keypoints):
        raise ValueError(
            f"Requested [{args.start}, {stop}) from {len(full_keypoints)} BVH frames."
        )

    keypoints = canonicalize_cycle(full_keypoints[args.start:stop])
    # Apply the same heading transform to all frames before measuring velocity
    # and contacts, so saved speed is in the robot's +X direction.
    displacement = full_keypoints[stop - 1, 0, :2] - full_keypoints[args.start, 0, :2]
    forward = displacement / np.linalg.norm(displacement)
    left = np.asarray([-forward[1], forward[0]])
    transformed_full = full_keypoints.copy()
    transformed_full[..., 0] = full_keypoints[..., :2] @ forward
    transformed_full[..., 1] = full_keypoints[..., :2] @ left
    contacts = infer_contacts(
        transformed_full,
        args.start,
        stop,
        args.frequency,
        args.contact_height,
        args.contact_velocity,
    )
    # Preserve source double support: the outgoing toe and incoming heel are
    # both grounded briefly. Only fill genuinely unlabelled flight frames.
    toe_indices = (DEMO_JOINTS.index("RightToeBase"), DEMO_JOINTS.index("LeftToeBase"))
    toe_height = keypoints[:, toe_indices, 2]
    for frame in range(len(contacts)):
        if not np.any(contacts[frame]):
            lower_side = int(np.argmin(toe_height[frame]))
            contacts[frame, lower_side] = True
    contacts[-1] = contacts[0]
    foot_roll_phase, foot_roll_mode = foot_roll_schedule(contacts)

    toe_floor = float(np.min(keypoints[:, toe_indices, 2]))
    pelvis_height = float(np.median(keypoints[:, 0, 2] - toe_floor))
    scale = args.target_pelvis_height / pelvis_height
    root_x = transformed_full[:, 0, 0]
    forward_velocity = np.gradient(root_x, 1.0 / args.frequency)[args.start:stop] * scale
    pose_seam = float(
        np.mean(
            (
                (keypoints[-1] - keypoints[-1, 0])
                - (keypoints[0] - keypoints[0, 0])
            )
            ** 2
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        global_joint_positions=keypoints.astype(np.float32),
        fps=np.float32(args.frequency),
        joint_names=np.asarray(DEMO_JOINTS),
        foot_contact=contacts.astype(np.float32),
        foot_roll_phase=foot_roll_phase,
        foot_roll_mode=foot_roll_mode,
        scale=np.float32(scale),
        target_forward_velocity=forward_velocity.astype(np.float32),
        source_frames=np.asarray([args.start, args.start + args.period], dtype=np.int32),
        pose_seam_mse=np.float64(pose_seam),
        source=np.asarray(str(args.bvh)),
    )
    print(
        f"saved {args.output}: frames={len(keypoints)}, scale={scale:.4f}, "
        f"mean_vx={np.mean(forward_velocity):.3f}m/s, seam_mse={pose_seam:.2e}, "
        f"right_contact={contact_windows(contacts[:, 0])}, "
        f"left_contact={contact_windows(contacts[:, 1])}"
    )


if __name__ == "__main__":
    main()
