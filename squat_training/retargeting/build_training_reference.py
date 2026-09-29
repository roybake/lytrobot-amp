#!/usr/bin/env python3
"""Convert an OmniRetarget result into the project's cyclic 40 Hz motion format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

JOINT_NAMES = (
    "root",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
)


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _normalized_lerp(left: np.ndarray, right: np.ndarray, weight: float) -> np.ndarray:
    if np.dot(left, right) < 0.0:
        right = -right
    value = (1.0 - weight) * left + weight * right
    return value / max(float(np.linalg.norm(value)), 1e-12)


def resample_qpos_at_phase(qpos: np.ndarray, source_phase: np.ndarray) -> np.ndarray:
    """Interpolate positions at fractional source-frame phases."""
    source_phase = np.asarray(source_phase, dtype=np.float64)
    low = np.floor(source_phase).astype(np.int64)
    high = np.minimum(low + 1, len(qpos) - 1)
    weight = source_phase - low
    output = (1.0 - weight[:, None]) * qpos[low] + weight[:, None] * qpos[high]
    source_quat = _normalize_quaternions(qpos[:, 3:7])
    for index in range(len(output)):
        output[index, 3:7] = _normalized_lerp(source_quat[low[index]], source_quat[high[index]], weight[index])
    return output.astype(np.float64)


def resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Resample a uniformly sampled motion while preserving its source duration."""
    duration = (len(qpos) - 1) / source_fps
    count = int(round(duration * target_fps)) + 1
    target_times = np.linspace(0.0, duration, count)
    return resample_qpos_at_phase(qpos, target_times * source_fps)


def resample_squat_timing(
    qpos: np.ndarray,
    frequency: float,
    down_duration: float,
    up_duration: float,
) -> tuple[np.ndarray, int]:
    """Stretch the descending and ascending squat phases to exact durations."""
    if frequency <= 0.0 or down_duration <= 0.0 or up_duration <= 0.0:
        raise ValueError("Frequency and squat phase durations must be positive.")
    total_frames = int(round((down_duration + up_duration) * frequency))
    down_frame = int(round(down_duration * frequency))
    if total_frames < 4 or not 1 <= down_frame < total_frames - 1:
        raise ValueError("Requested squat timing produces too few frames.")

    lowest_source_frame = int(np.argmin(qpos[:, 2]))
    if not 1 <= lowest_source_frame < len(qpos) - 1:
        raise ValueError("The source motion must contain an interior squat minimum.")

    times = np.arange(total_frames, dtype=np.float64) / frequency
    phase = np.empty(total_frames, dtype=np.float64)
    descending = times <= down_duration
    phase[descending] = (
        times[descending] / down_duration * lowest_source_frame
    )
    phase[~descending] = lowest_source_frame + (
        (times[~descending] - down_duration)
        / up_duration
        * (len(qpos) - 1 - lowest_source_frame)
    )
    return resample_qpos_at_phase(qpos, phase), down_frame


def make_cyclic(qpos: np.ndarray, blend_frames: int, hold_frames: int) -> np.ndarray:
    blend_frames = min(max(blend_frames, 2), max(2, len(qpos) // 4))
    endpoint_samples = min(4, len(qpos) // 4)
    first = qpos[:endpoint_samples]
    last = qpos[-endpoint_samples:].copy()
    for index in range(endpoint_samples):
        if np.dot(first[index, 3:7], last[index, 3:7]) < 0.0:
            last[index, 3:7] *= -1.0
    stand = np.mean(np.vstack((first, last)), axis=0)
    stand[3:7] /= max(float(np.linalg.norm(stand[3:7])), 1e-12)

    result = qpos.copy()
    for index in range(blend_frames):
        phase = index / (blend_frames - 1)
        weight = phase * phase * (3.0 - 2.0 * phase)
        result[index] = (1.0 - weight) * stand + weight * qpos[index]
        result[index, 3:7] = _normalized_lerp(stand[3:7], qpos[index, 3:7], weight)
        tail = len(result) - 1 - index
        result[tail] = (1.0 - weight) * stand + weight * qpos[tail]
        result[tail, 3:7] = _normalized_lerp(stand[3:7], qpos[tail, 3:7], weight)
    result[0] = stand
    result[-1] = stand
    if hold_frames > 0:
        hold = np.repeat(stand[None, :], hold_frames, axis=0)
        result = np.vstack((hold, result, hold))
    return result


def clip_joint_limits(model: mujoco.MjModel, qpos: np.ndarray) -> None:
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE or not model.jnt_limited[joint_id]:
            continue
        address = int(model.jnt_qposadr[joint_id])
        qpos[:, address] = np.clip(qpos[:, address], *model.jnt_range[joint_id])


def cyclic_velocity(model: mujoco.MjModel, qpos: np.ndarray, frequency: float) -> np.ndarray:
    velocity = np.zeros((len(qpos), model.nv), dtype=np.float64)
    dt = 2.0 / frequency
    for index in range(len(qpos)):
        previous = qpos[(index - 1) % len(qpos)]
        following = qpos[(index + 1) % len(qpos)]
        mujoco.mj_differentiatePos(model, velocity[index], dt, previous, following)
    return velocity


def foot_height_statistics(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, float]:
    data = mujoco.MjData(model)
    body_names = ("right_foot", "right_toe", "left_foot", "left_toe")
    heights = np.zeros((len(qpos), len(body_names)), dtype=np.float64)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        heights[frame] = [data.body(name).xpos[2] for name in body_names]
    return {
        "foot_body_height_min": float(heights.min()),
        "foot_body_height_max": float(heights.max()),
        "foot_body_height_range": float(np.ptp(heights)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "lyt" / "lyt_14dof.xml",
    )
    parser.add_argument("--frequency", type=float, default=40.0)
    parser.add_argument("--blend-frames", type=int, default=8)
    parser.add_argument(
        "--down-duration",
        type=float,
        default=3.0,
        help="Target duration in seconds from standing to the deepest squat.",
    )
    parser.add_argument(
        "--up-duration",
        type=float,
        default=3.0,
        help="Target duration in seconds from the deepest squat back to standing.",
    )
    args = parser.parse_args()

    with np.load(args.input) as archive:
        raw_qpos = np.asarray(archive["qpos"], dtype=np.float64)
        source_fps = float(archive["fps"])
        cost = float(np.asarray(archive["cost"]).item())
        squat_depth_scale = float(archive["squat_depth_scale"]) if "squat_depth_scale" in archive else 1.0
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    if raw_qpos.shape[1] != model.nq:
        raise ValueError(f"qpos width {raw_qpos.shape[1]} does not match retarget model nq={model.nq}.")
    raw_qpos[:, 3:7] = _normalize_quaternions(raw_qpos[:, 3:7])
    clip_joint_limits(model, raw_qpos)
    # First make the source seam stationary.  The time warp below then keeps
    # the natural retargeted joint path but sets a deliberate 3 s down / 3 s
    # up rhythm without adding extra stationary frames to the six-second loop.
    source_qpos = make_cyclic(raw_qpos, args.blend_frames, hold_frames=0)
    qpos, lowest_frame = resample_squat_timing(
        source_qpos,
        args.frequency,
        args.down_duration,
        args.up_duration,
    )
    qpos[:, 3:7] = _normalize_quaternions(qpos[:, 3:7])
    clip_joint_limits(model, qpos)
    qvel = cyclic_velocity(model, qpos, args.frequency)
    foot_contact = np.ones((len(qpos), 2), dtype=np.float32)
    statistics = foot_height_statistics(model, qpos)
    metadata = {
        "task": "squat",
        "source": "CMU MoCap subject 22 trial 14, frames 340-620/stride 4",
        "source_url": "https://mocap.cs.cmu.edu/search.php?subjectnumber=22&trinum=14",
        "retargeter": "OmniRetarget/Holosoma interaction-mesh SQP",
        "retargeter_url": "https://github.com/amazon-far/holosoma",
        "retargeter_commit": "79d728c2c40e7f9ce8c28f9d9cab117ed2350bb5",
        "human_to_robot_mapping": {
            "Hips": "pelvis",
            "RightUpLeg": "right_hip",
            "RightLeg": "right_knee",
            "RightFoot": "right_foot",
            "RightToeBase": "right_toe",
            "LeftUpLeg": "left_hip",
            "LeftLeg": "left_knee",
            "LeftFoot": "left_foot",
            "LeftToeBase": "left_toe",
        },
        "source_fps": source_fps,
        "frequency": args.frequency,
        "down_duration_seconds": args.down_duration,
        "up_duration_seconds": args.up_duration,
        "cycle_duration_seconds": len(qpos) / args.frequency,
        "deepest_squat_frame": lowest_frame,
        "timing": "source phases time-warped at the pelvis-height minimum",
        "omniretarget_final_cost": cost,
        "squat_depth_scale": squat_depth_scale,
        **statistics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        joint_names=np.asarray(JOINT_NAMES),
        frequency=np.float64(args.frequency),
        foot_contact=foot_contact,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    print(
        f"saved {args.output}: frames={len(qpos)}, duration={len(qpos) / args.frequency:.3f}s, "
        f"down/up={args.down_duration:.3f}/{args.up_duration:.3f}s, frequency={args.frequency:g}Hz, "
        f"root_height=[{qpos[:, 2].min():.3f}, {qpos[:, 2].max():.3f}]m, "
        f"foot_height_range={statistics['foot_body_height_range']:.4f}m"
    )


if __name__ == "__main__":
    main()
