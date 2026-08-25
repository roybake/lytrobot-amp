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
    OUTPUT_FREQUENCY,
    clip_joint_limits,
    forward_speed,
    landmark_positions,
    refine_contacts_and_collisions,
    retarget_terrain_model,
    sole_landmarks,
    velocities,
)
from lytrobot_amp.terrain import terrain_height_from_model  # noqa: E402


MODEL_XML = REPO_ROOT / "robot" / "MJCFmodelposition.xml"
MOTION_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
TASK_CYCLES = {
    "rough_walk": 3,
    "pit_walk": 3,
    "gap_cross": 4,
}


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
    base_model = mujoco.MjModel.from_xml_path(str(model_xml))
    clip_joint_limits(base_model, qpos)
    qpos, refinement = refine_contacts_and_collisions(
        qpos,
        contact,
        roll_phase,
        roll_mode,
        task,
        model_xml,
        False,
    )
    clip_joint_limits(base_model, qpos)
    support_correction = enforce_support_clearance(qpos, contact, task, model_xml)
    audit = audit_terrain_contacts(qpos, contact, task, model_xml)
    qvel = velocities(base_model, qpos, OUTPUT_FREQUENCY, loop=False)
    retargeted_forward_speed = forward_speed(qpos, OUTPUT_FREQUENCY, loop=False)

    metadata = {
        "task": task,
        "loop": False,
        "frequency": OUTPUT_FREQUENCY,
        "source": source_metadata.get("source"),
        "source_url": source_metadata.get("source_url"),
        "derived_from": source.name,
        "terrain_source": "MoRE-discoRL",
        "terrain_source_path": "robot/terrains/more_discolrl",
        "terrain_adaptation": (
            "unrolled OmniRetarget gait plus full-foot terrain raycast IK; "
            "support constraints retain heel -> sole -> toe roll"
        ),
        "contact_sequence": "heel -> full sole -> toe",
        "cycles": TASK_CYCLES[task],
        "target_forward_speed_mps": source_metadata.get("target_forward_speed_mps"),
        "retargeted_forward_speed_mps": retargeted_forward_speed,
        **refinement,
        **support_correction,
        **audit,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        joint_names=np.asarray(JOINT_NAMES),
        frequency=np.float64(OUTPUT_FREQUENCY),
        foot_contact=contact.astype(np.float32),
        foot_roll_phase=roll_phase.astype(np.float32),
        foot_roll_mode=roll_mode.astype(np.int8),
        task=np.asarray(task),
        terrain_kind=np.asarray(refinement["terrain_kind"]),
        loop=np.asarray(False),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    if source_metadata.get("profile") == "no_stairs_fast_small_step":
        from analyze_gait import analyze_reference

        gait = analyze_reference(output, model_xml)
        metadata.update(
            {
                "profile": "no_stairs_fast_small_step",
                "measured_forward_speed_mps": gait["mean_forward_speed_mps"],
                "mean_step_length_m": gait["mean_step_length_m"],
                "cadence_steps_per_min": gait["cadence_steps_per_min"],
                "stance_slip_mean_mps": gait["stance_slip_mean_mps"],
                "stance_slip_p95_mps": gait["stance_slip_p95_mps"],
            }
        )
        with np.load(output) as archive:
            payload = {key: archive[key] for key in archive.files if key != "metadata"}
        payload["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        np.savez_compressed(output, **payload)
    print(
        f"{task}: {len(qpos)} frames, root_z=[{qpos[:, 2].min():.3f}, "
        f"{qpos[:, 2].max():.3f}], support_clearance="
        f"[{audit['support_clearance_min']:.4f}, "
        f"{audit['support_clearance_max']:.4f}], "
        f"forward_speed={retargeted_forward_speed:.3f}m/s -> {output}"
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
