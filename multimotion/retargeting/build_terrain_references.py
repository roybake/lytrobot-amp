#!/usr/bin/env python3
"""Adapt the retargeted walking gait to MoRE gap, pit, and rough terrains."""

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
sys.path[:0] = [str(SRC_ROOT), str(HERE)]

from build_references import (  # noqa: E402
    CONTACT_CLEARANCE,
    JOINT_NAMES,
    MAIN_JOINT_QPOS,
    OUTPUT_FREQUENCY,
    ROLL_HEEL,
    clip_joint_limits,
    forward_speed,
    interpolate_qpos,
    landmark_positions,
    refine_contacts_and_collisions,
    retarget_terrain_model,
    sole_landmarks,
    velocities,
)
from lytrobot_amp.terrain import terrain_height_from_model  # noqa: E402


MODEL_XML = REPO_ROOT / "robot" / "MJCFmodelfinal.xml"
MOTION_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
TASK_CYCLES = {
    "rough_walk": 3,
    "pit_walk": 3,
    "gap_cross": 4,
    # A 3-cycle OmniRetarget walk covers the 3.10 m 6-degree assessment ramp.
    "slope_up": 3,
    "slope_down": 3,
}
SLOPE_OUTPUT_FREQUENCY = 120.0
SLOPE_FOOT_QPOS = MAIN_JOINT_QPOS
SLOPE_FOOT_STEP_RAD = np.deg2rad(8.0)


def tile_walk(
    qpos: np.ndarray,
    contact: np.ndarray,
    roll_phase: np.ndarray,
    roll_mode: np.ndarray,
    cycles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unroll a cyclic gait without a root-position teleport at cycle seams."""

    forward_step = float(
        qpos[-1, 0] + np.median(np.diff(qpos[:, 0]))
    )
    tiled_qpos: list[np.ndarray] = []
    for cycle in range(cycles):
        values = np.asarray(qpos, dtype=np.float64).copy()
        values[:, 0] += cycle * forward_step
        tiled_qpos.append(values)
    return (
        np.concatenate(tiled_qpos),
        np.tile(contact, (cycles, 1)),
        np.tile(roll_phase, (cycles, 1)),
        np.tile(roll_mode, (cycles, 1)),
    )


def audit_terrain_contacts(
    qpos: np.ndarray,
    contact: np.ndarray,
    task: str,
    model_xml: Path,
) -> dict[str, float | int]:
    """Measure support clearance and reject contacts placed inside a gap."""

    model, pose_offset, terrain_kind = retarget_terrain_model(task, model_xml)
    data = mujoco.MjData(model)
    _, landmark_body_ids, landmark_points = sole_landmarks(model)
    clearances: list[float] = []
    unsupported_gap_frames = 0
    for frame, local_pose in enumerate(qpos):
        pose = np.asarray(local_pose, dtype=np.float64).copy()
        pose[:3] += pose_offset
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        landmarks = landmark_positions(data, landmark_body_ids, landmark_points)
        ground = terrain_height_from_model(
            model,
            data,
            landmarks[:, 0],
            landmarks[:, 1],
        )
        for side in np.flatnonzero(contact[frame] > 0.5):
            support_slice = slice(2 * int(side), 2 * int(side) + 2)
            clearances.extend(
                landmarks[support_slice, 2] - ground[support_slice]
            )
            if terrain_kind == "gap" and np.min(ground[support_slice]) < 0.25:
                unsupported_gap_frames += 1
    if unsupported_gap_frames:
        raise RuntimeError(
            f"{task}: {unsupported_gap_frames} support frames land inside a gap."
        )
    values = np.asarray(clearances, dtype=np.float64)
    return {
        "support_clearance_min": float(values.min()),
        "support_clearance_max": float(values.max()),
        "support_clearance_rms": float(np.sqrt(np.mean(values**2))),
        "unsupported_gap_contact_frames": unsupported_gap_frames,
    }


def enforce_support_clearance(
    qpos: np.ndarray,
    contact: np.ndarray,
    task: str,
    model_xml: Path,
) -> dict[str, float]:
    """Lift the root per frame when terrain-edge IK leaves support penetration."""

    model, pose_offset, _ = retarget_terrain_model(task, model_xml)
    data = mujoco.MjData(model)
    _, landmark_body_ids, landmark_points = sole_landmarks(model)
    corrections = np.zeros(len(qpos), dtype=np.float64)
    for frame, local_pose in enumerate(qpos):
        support_sides = np.flatnonzero(contact[frame] > 0.5)
        if len(support_sides) == 0:
            continue
        pose = np.asarray(local_pose, dtype=np.float64).copy()
        pose[:3] += pose_offset
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        landmarks = landmark_positions(data, landmark_body_ids, landmark_points)
        ground = terrain_height_from_model(
            model,
            data,
            landmarks[:, 0],
            landmarks[:, 1],
        )
        support_indices = np.concatenate(
            [np.arange(2 * int(side), 2 * int(side) + 2) for side in support_sides]
        )
        minimum = float(np.min(landmarks[support_indices, 2] - ground[support_indices]))
        correction = max(CONTACT_CLEARANCE - minimum, 0.0)
        qpos[frame, 2] += correction
        corrections[frame] = correction
    return {
        "support_height_correction_max": float(corrections.max()),
        "support_height_correction_rms": float(np.sqrt(np.mean(corrections**2))),
    }


def align_slope_root_seed(
    qpos: np.ndarray,
    contact: np.ndarray,
    task: str,
    model_xml: Path,
) -> dict[str, float]:
    """Lift/lower the flat OmniRetarget seed onto a ramp before contact IK.

    Per-frame contact IK intentionally permits only a small root correction so
    it cannot create a pose jump.  A descending ramp can begin about 0.33 m
    above the flat floor, therefore its unshifted seed lies outside that local
    trust region.  This terrain-height seed is continuous in X and lets the
    subsequent full-foot solve make only local corrections.
    """

    if task not in {"slope_up", "slope_down"}:
        return {"slope_seed_root_shift_min": 0.0, "slope_seed_root_shift_max": 0.0}
    model, pose_offset, _ = retarget_terrain_model(task, model_xml)
    data = mujoco.MjData(model)
    _, landmark_body_ids, landmark_points = sole_landmarks(model)
    shifts = np.zeros(len(qpos), dtype=np.float64)
    for frame, pose in enumerate(qpos):
        world = np.asarray(pose, dtype=np.float64).copy()
        world[:3] += pose_offset
        data.qpos[:] = world
        mujoco.mj_forward(model, data)
        landmarks = landmark_positions(data, landmark_body_ids, landmark_points)
        support_sides = np.flatnonzero(contact[frame] > 0.5)
        if len(support_sides) == 0:
            shifts[frame] = shifts[frame - 1] if frame else 0.0
            continue
        indices = np.concatenate(
            [np.arange(2 * int(side), 2 * int(side) + 2) for side in support_sides]
        )
        ground = terrain_height_from_model(
            model, data, landmarks[indices, 0], landmarks[indices, 1]
        )
        shifts[frame] = float(np.mean(ground))
    qpos[:, 2] += shifts
    return {
        "slope_seed_root_shift_min": float(shifts.min()),
        "slope_seed_root_shift_max": float(shifts.max()),
    }


def regularize_slope_foot_seed(
    qpos: np.ndarray,
    contact: np.ndarray,
    roll_mode: np.ndarray,
) -> dict[str, float]:
    """Distribute retargeted leg transitions through adjacent swing frames.

    A contact-perfect preliminary solve can still contain an isolated joint
    limit flip at heel strike.  A forward/backward rate limiter turns it into
    a short, continuous preparation movement before the foot reaches ground.
    """

    values = qpos[:, SLOPE_FOOT_QPOS].copy()
    original_step = float(np.max(np.abs(np.diff(values, axis=0)), initial=0.0))
    side_columns = (tuple(range(0, 7)), tuple(range(7, 14)))
    for frame in range(1, len(values)):
        for side, columns in enumerate(side_columns):
            heel_strike = (
                contact[frame, side] > 0.5
                and int(roll_mode[frame, side]) == ROLL_HEEL
                and (
                    contact[frame - 1, side] <= 0.5
                    or int(roll_mode[frame - 1, side]) != ROLL_HEEL
                )
            )
            if not heel_strike:
                continue
            # This leg is airborne before heel strike, so move it toward the
            # already contact-valid target while it cannot disturb support.
            start = frame
            while start > 0 and contact[start - 1, side] <= 0.5:
                start -= 1
            start = max(start, frame - 16)
            initial = values[start, columns].copy()
            target = values[frame, columns].copy()
            span = frame - start + 1
            for index in range(start, frame):
                phase = (index - start + 1) / span
                phase = phase * phase * (3.0 - 2.0 * phase)
                values[index, columns] = (1.0 - phase) * initial + phase * target
    # A swing foot has no support constraint, so prevent the unconstrained IK
    # from using a one-frame subtalar/toe flip while it is in the air.  The
    # landing pose itself is untouched; it was prepared above before contact.
    for frame in range(1, len(values)):
        for side, columns in enumerate(side_columns):
            begins_heel = (
                contact[frame, side] > 0.5
                and int(roll_mode[frame, side]) == ROLL_HEEL
                and (
                    contact[frame - 1, side] <= 0.5
                    or int(roll_mode[frame - 1, side]) != ROLL_HEEL
                )
            )
            if begins_heel:
                continue
            values[frame, columns] = np.clip(
                values[frame, columns],
                values[frame - 1, columns] - SLOPE_FOOT_STEP_RAD,
                values[frame - 1, columns] + SLOPE_FOOT_STEP_RAD,
            )
    qpos[:, SLOPE_FOOT_QPOS] = values
    return {
        "slope_seed_foot_step_before_rad": original_step,
        "slope_seed_foot_step_after_rad": float(
            np.max(np.abs(np.diff(values, axis=0)), initial=0.0)
        ),
    }


def continuity_metrics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    frequency: float,
) -> dict[str, float]:
    """Quantify frame-to-frame continuity after terrain contact IK.

    These are diagnostics rather than a smoothing filter: preserving the
    OmniRetarget solution and detecting a discontinuity is safer than hiding
    one by post-filtering a support foot through the ramp.
    """

    del model
    values = np.asarray(qpos, dtype=np.float64)
    steps = np.diff(values, axis=0)
    joint_steps = np.abs(steps[:, 7:])
    root_steps = np.linalg.norm(steps[:, :3], axis=1)
    root_velocity = steps[:, :3] * float(frequency)
    root_acceleration = np.diff(root_velocity, axis=0) * float(frequency)
    return {
        "maximum_joint_step_rad": float(np.max(joint_steps, initial=0.0)),
        "maximum_root_step_m": float(np.max(root_steps, initial=0.0)),
        "maximum_root_acceleration_mps2": float(
            np.max(np.linalg.norm(root_acceleration, axis=1), initial=0.0)
        ),
    }


def build_task(
    task: str,
    source: Path,
    output: Path,
    model_xml: Path,
) -> None:
    with np.load(source) as archive:
        base_qpos = np.asarray(archive["qpos"], dtype=np.float64)
        contact = np.asarray(archive["foot_contact"], dtype=np.float32)
        roll_phase = np.asarray(archive["foot_roll_phase"], dtype=np.float32)
        roll_mode = np.asarray(archive["foot_roll_mode"], dtype=np.int8)
        source_metadata = json.loads(str(archive["metadata"]))

    qpos, contact, roll_phase, roll_mode = tile_walk(
        base_qpos,
        contact,
        roll_phase,
        roll_mode,
        TASK_CYCLES[task],
    )
    output_frequency = OUTPUT_FREQUENCY
    if task in {"slope_up", "slope_down"}:
        # Keep the same physical speed and duration, but give the contact IK
        # three samples around heel strike instead of asking it to execute the
        # full ankle/toe correction in one 40 Hz command frame.
        qpos, nearest = interpolate_qpos(
            qpos,
            source_frequency=OUTPUT_FREQUENCY,
            target_frequency=SLOPE_OUTPUT_FREQUENCY,
            time_scale=1.0,
            loop=False,
        )
        contact = contact[nearest]
        roll_phase = roll_phase[nearest]
        roll_mode = roll_mode[nearest]
        output_frequency = SLOPE_OUTPUT_FREQUENCY
    base_model = mujoco.MjModel.from_xml_path(str(model_xml))
    clip_joint_limits(base_model, qpos)
    slope_seed = align_slope_root_seed(qpos, contact, task, model_xml)
    slope_foot_seed = {
        "slope_seed_foot_step_before_rad": 0.0,
        "slope_seed_foot_step_after_rad": 0.0,
    }
    if task in {"slope_up", "slope_down"}:
        # Solve contact first, then use that physically valid motion to shape
        # the swing trajectory before the final bounded solve.  This prevents
        # a heel-strike correction from being requested in a single frame.
        qpos, refinement = refine_contacts_and_collisions(
            qpos,
            contact,
            roll_phase,
            roll_mode,
            task,
            model_xml,
            enforce_slope_continuity=False,
        )
        slope_foot_seed = regularize_slope_foot_seed(qpos, contact, roll_mode)
    else:
        qpos, refinement = refine_contacts_and_collisions(
            qpos,
            contact,
            roll_phase,
            roll_mode,
            task,
            model_xml,
        )
    clip_joint_limits(base_model, qpos)
    support_correction = enforce_support_clearance(qpos, contact, task, model_xml)
    audit = audit_terrain_contacts(qpos, contact, task, model_xml)
    qvel = velocities(base_model, qpos, output_frequency, loop=False)
    retargeted_forward_speed = forward_speed(qpos, output_frequency, loop=False)
    continuity = continuity_metrics(base_model, qpos, output_frequency)

    metadata = {
        "task": task,
        "loop": False,
        "frequency": output_frequency,
        "source": source_metadata.get("source"),
        "source_url": source_metadata.get("source_url"),
        "derived_from": source.name,
        "terrain_source": "MoRE-discoRL",
        "terrain_source_path": "robot/terrains",
        "terrain_adaptation": (
            "unrolled OmniRetarget gait plus full-foot terrain raycast IK; "
            "support constraints retain heel -> sole -> toe roll"
        ),
        "contact_sequence": "heel -> full sole -> toe",
        "cycles": TASK_CYCLES[task],
        "target_forward_speed_mps": source_metadata.get("target_forward_speed_mps"),
        "retargeted_forward_speed_mps": retargeted_forward_speed,
        **refinement,
        **slope_seed,
        **slope_foot_seed,
        **support_correction,
        **audit,
        **continuity,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        joint_names=np.asarray(JOINT_NAMES),
        frequency=np.float64(output_frequency),
        foot_contact=contact.astype(np.float32),
        foot_roll_phase=roll_phase.astype(np.float32),
        foot_roll_mode=roll_mode.astype(np.int8),
        task=np.asarray(task),
        terrain_kind=np.asarray(refinement["terrain_kind"]),
        loop=np.asarray(False),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    print(
        f"{task}: {len(qpos)} frames, root_z=[{qpos[:, 2].min():.3f}, "
        f"{qpos[:, 2].max():.3f}], support_clearance="
        f"[{audit['support_clearance_min']:.4f}, "
        f"{audit['support_clearance_max']:.4f}], "
        f"forward_speed={retargeted_forward_speed:.3f}m/s, "
        f"joint_step_max={continuity['maximum_joint_step_rad']:.4f}rad -> {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", type=Path, default=MODEL_XML)
    parser.add_argument(
        "--source",
        type=Path,
        default=MOTION_DIR / "walk_omniretarget.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=MOTION_DIR)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_CYCLES),
        default=tuple(TASK_CYCLES),
    )
    args = parser.parse_args()
    for task in args.tasks:
        build_task(
            task,
            args.source.expanduser().resolve(),
            args.output_dir.expanduser().resolve() / f"{task}_omniretarget.npz",
            args.model_xml.expanduser().resolve(),
        )


if __name__ == "__main__":
    main()
