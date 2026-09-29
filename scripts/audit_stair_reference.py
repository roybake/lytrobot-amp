#!/usr/bin/env python3
"""Check a stair motion against the exact MuJoCo contact geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lytrobot_amp.motion import MotionReference  # noqa: E402


def audit_reference(
    model_xml: str | Path, motion_file: str | Path, *, root_x_offset: float = 0.0
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    reference = MotionReference(
        motion_file, model, zero_root_xy=False, loop_override=False
    )
    data = mujoco.MjData(model)
    foot_visual_geoms: dict[int, tuple[int, int]] = {}
    for side in ("r", "l"):
        collider_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (f"foot_{side}", f"toe_{side}")
        ]
        collider_ids = [
            geom_id for geom_id in collider_ids
            if geom_id >= 0 and (
                int(model.geom_contype[geom_id]) != 0
                or int(model.geom_conaffinity[geom_id]) != 0
            )
        ]
        # Backward compatibility for audit reports produced before the single
        # foot box was split into named foot/toe DeepMimic colliders.
        if min(collider_ids) < 0:
            collider_ids = [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_box_{side}"
                )
            ]
        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_geom_{side}")
        toe_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"toe_{side}")
        toe_ids = [
            geom_id for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == toe_body
            and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        ]
        if not collider_ids or min(*collider_ids, foot_id, toe_body) < 0 or len(toe_ids) != 1:
            raise ValueError(
                f"Stair audit requires foot/toe colliders, foot mesh and toe mesh for {side}."
            )
        for collider_id in collider_ids:
            foot_visual_geoms[collider_id] = (foot_id, toe_ids[0])
    foot_ids = set(foot_visual_geoms)
    visual_foot_ids = {
        geom_id for geom_ids in foot_visual_geoms.values() for geom_id in geom_ids
    }
    mesh_vertices = {}
    for geom_id in visual_foot_ids:
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        mesh_vertices[geom_id] = model.mesh_vert[
            start:start + int(model.mesh_vertnum[mesh_id])
        ]
    terrain_ids = {
        index for index in range(model.ngeom)
        if model.geom_bodyid[index] == 0 and model.geom_contype[index] != 0
    }
    tread_ids = {
        geom_id for geom_id in terrain_ids
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    }
    deepest = 0.0
    worst_frame = 0
    worst_pair = ("", "")
    deepest_visual = 0.0
    worst_visual_frame = 0
    worst_visual_pair = ("", "")
    penetrating_frames = 0
    unsupported_frames = 0
    longest_unsupported_run = 0
    longest_unsupported_start = 0
    unsupported_run = 0
    minimum_visual_tread_margin = float("inf")
    minimum_box_tread_margin = float("inf")
    tread_contact_frames: set[int] = set()
    tread_overhang_frames: set[int] = set()
    for frame in range(reference.frame_count):
        data.qpos[:] = reference.reference_qpos[frame]
        data.qpos[0] += float(root_x_offset)
        data.qvel[:] = reference.reference_qvel[frame]
        mujoco.mj_forward(model, data)
        for foot_id in visual_foot_ids:
            for terrain_id in terrain_ids:
                # The STL geoms are visual only (contype=0), so mj_forward
                # does not put these pairs into data.contact.  Query their
                # signed separation directly to distinguish an inaccurate
                # contact box from an incompatible foot pose.
                distance = float(mujoco.mj_geomDistance(
                    model, data, foot_id, terrain_id, 0.25, None
                ))
                if distance < deepest_visual:
                    deepest_visual = distance
                    worst_visual_frame = frame
                    worst_visual_pair = tuple(
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                        or str(geom)
                        for geom in (terrain_id, foot_id)
                    )
        frame_deepest = 0.0
        has_support = False
        top_contacts: set[tuple[int, int]] = set()
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not (pair & foot_ids and pair & terrain_ids):
                continue
            if (
                float(contact.dist) <= 0.005
                and abs(float(contact.frame[2])) >= 0.8
                and pair & tread_ids
            ):
                top_contacts.add((next(iter(pair & foot_ids)), next(iter(pair & tread_ids))))
            frame_deepest = min(frame_deepest, float(contact.dist))
            has_support |= float(contact.dist) <= 0.005
            if float(contact.dist) < deepest:
                deepest = float(contact.dist)
                worst_frame = frame
                worst_pair = tuple(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                    or str(geom)
                    for geom in (contact.geom1, contact.geom2)
                )
        for foot_id, tread_id in top_contacts:
            tread_contact_frames.add(frame)
            left = float(model.geom_pos[tread_id, 0] - model.geom_size[tread_id, 0])
            right = float(model.geom_pos[tread_id, 0] + model.geom_size[tread_id, 0])
            visual_limits = []
            for geom_id in foot_visual_geoms[foot_id]:
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                x = data.geom_xpos[geom_id, 0] + mesh_vertices[geom_id] @ rotation[0]
                visual_limits.append((float(x.min()), float(x.max())))
            visual_margin = min(
                min(limit[0] for limit in visual_limits) - left,
                right - max(limit[1] for limit in visual_limits),
            )
            box_rotation = data.geom_xmat[foot_id].reshape(3, 3)
            box_radius_x = float(np.dot(np.abs(box_rotation[0]), model.geom_size[foot_id]))
            box_margin = min(
                float(data.geom_xpos[foot_id, 0]) - box_radius_x - left,
                right - float(data.geom_xpos[foot_id, 0]) - box_radius_x,
            )
            minimum_visual_tread_margin = min(minimum_visual_tread_margin, visual_margin)
            minimum_box_tread_margin = min(minimum_box_tread_margin, box_margin)
            if min(visual_margin, box_margin) < -0.001:
                tread_overhang_frames.add(frame)
        penetrating_frames += frame_deepest < -0.02
        if has_support:
            if unsupported_run > longest_unsupported_run:
                longest_unsupported_run = unsupported_run
                longest_unsupported_start = frame - unsupported_run
            unsupported_run = 0
        else:
            unsupported_frames += 1
            unsupported_run += 1
    if unsupported_run > longest_unsupported_run:
        longest_unsupported_run = unsupported_run
        longest_unsupported_start = reference.frame_count - unsupported_run
    return {
        "model_xml": str(Path(model_xml).resolve()),
        "motion_file": str(Path(motion_file).resolve()),
        "root_x_offset_m": float(root_x_offset),
        "frame_count": reference.frame_count,
        "frequency_hz": reference.frequency,
        "maximum_foot_penetration_m": -deepest,
        "worst_penetration_frame": worst_frame,
        "worst_penetration_pair": worst_pair,
        "maximum_visual_foot_mesh_penetration_m": -deepest_visual,
        "worst_visual_mesh_frame": worst_visual_frame,
        "worst_visual_mesh_pair": worst_visual_pair,
        "frames_over_2cm_penetration": penetrating_frames,
        "tread_top_contact_frames": len(tread_contact_frames),
        "tread_overhang_frames": len(tread_overhang_frames),
        "minimum_visual_tread_edge_margin_m": (
            minimum_visual_tread_margin if tread_contact_frames else None
        ),
        "minimum_box_tread_edge_margin_m": (
            minimum_box_tread_margin if tread_contact_frames else None
        ),
        "unsupported_frames": unsupported_frames,
        "longest_unsupported_run_s": longest_unsupported_run / reference.frequency,
        "longest_unsupported_start_frame": longest_unsupported_start,
        "longest_unsupported_end_frame": longest_unsupported_start + longest_unsupported_run - 1,
        "alignment_pass": (
            deepest >= -0.02
            and deepest_visual >= -0.02
            and not tread_overhang_frames
        ),
        "contact_geometry_pass": (
            deepest >= -0.02
            and not tread_overhang_frames
            and longest_unsupported_run / reference.frequency <= 0.3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", default=str(ROOT / "robot/terrains/stairs.xml"))
    parser.add_argument("--motion-file", default=str(ROOT / "robot/reference_data/multimotion/stairs.npz"))
    parser.add_argument("--root-x-offset", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = audit_reference(
        args.model_xml, args.motion_file, root_x_offset=args.root_x_offset
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    if not report["contact_geometry_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
