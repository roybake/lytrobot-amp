#!/usr/bin/env python3
"""Correct small sole-height gaps without changing the human joint/XY motion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from lytrobot_amp.motion import MotionReference
from scripts.audit_stair_reference import audit_reference


def prepare(source: Path, output: Path, model_xml: Path, root_x_offset: float) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("The corrected motion must use a separate output file.")
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    ref = MotionReference(source, model, zero_root_xy=False, loop_override=False,
                          root_x_offset=root_x_offset)
    data = mujoco.MjData(model)
    feet = [
        [model.geom(f"{part}_{side}").id for part in ("foot", "toe")]
        for side in ("r", "l")
    ]
    mesh_vertices = {}
    for foot in (geom_id for side in feet for geom_id in side):
        mesh_id = int(model.geom_dataid[foot])
        start = int(model.mesh_vertadr[mesh_id])
        mesh_vertices[foot] = model.mesh_vert[start:start + int(model.mesh_vertnum[mesh_id])]
    boxes = [i for i in range(model.ngeom) if model.geom_bodyid[i] == 0
             and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
             and model.geom_contype[i] != 0]
    clearances = []
    side_clearances = []
    for qpos in ref.reference_qpos:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        side = []
        for foot_ids in feet:
            vertices = np.concatenate([
                data.geom_xpos[foot]
                + mesh_vertices[foot] @ data.geom_xmat[foot].reshape(3, 3).T
                for foot in foot_ids
            ])
            heights = np.zeros(len(vertices))
            for box in boxes:
                inside = np.all(np.abs(vertices[:, :2] - model.geom_pos[box, :2])
                                <= model.geom_size[box, :2] + 1e-8, axis=1)
                heights[inside] = np.maximum(heights[inside],
                    model.geom_pos[box, 2] + model.geom_size[box, 2])
            side.append(float(np.min(vertices[:, 2] - heights)))
        side_clearances.append(side)
        clearances.append(min(side))
    clearance = np.asarray(clearances)
    # Constrain the nearest sole to a small compliant penetration. The second
    # difference term smooths the correction while retaining contact each frame.
    desired = -0.0005 - clearance
    count = len(desired)
    second_difference = sparse.diags((np.ones(count-2), -2*np.ones(count-2),
                                      np.ones(count-2)), (0, 1, 2),
                                     shape=(count-2, count))
    matrix = sparse.vstack((sparse.eye(count), 10.0 * second_difference)).tocsr()
    target = np.r_[desired, np.zeros(count-2)]
    fit = lsq_linear(matrix, target, bounds=(desired - 0.0005, desired + 0.0005),
                     tol=1e-10, lsmr_tol=1e-10)
    if not fit.success:
        raise RuntimeError(f"Support correction failed: {fit.message}")
    shift = fit.x
    if np.max(np.abs(shift)) > 0.025:
        raise ValueError("More than 2.5 cm of height repair requires a new retargeting solve.")
    with np.load(source, allow_pickle=True) as archive:
        qpos = np.asarray(archive["qpos"]).copy()
        qvel = np.asarray(archive["qvel"]).copy()
        joint_names = archive["joint_names"].copy()
        split = archive["split_points"].copy() if "split_points" in archive else np.array([0, count])
    qpos[:, 2] += shift.astype(qpos.dtype)
    # Only root Z changed. Preserve the other source velocity channels.
    qvel[:, 2] = np.gradient(qpos[:, 2], 1.0 / ref.frequency)
    contact = (np.asarray(side_clearances) + shift[:, None] <= 0.003).astype(np.float32)
    metadata = {"source": str(source.resolve()),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "correction": "contact-constrained root height only",
                "max_root_height_change_m": float(np.max(np.abs(shift))),
                "joint_motion_and_root_xy_preserved": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, qpos=qpos, qvel=qvel, frequency=ref.frequency,
                        joint_names=joint_names, split_points=split, loop=False,
                        foot_contact=contact, root_height_correction=shift,
                        provenance_json=json.dumps(metadata))
    report = audit_reference(model_xml, output, root_x_offset=root_x_offset)
    report.update(metadata)
    output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not (report["alignment_pass"] and report["contact_geometry_pass"]):
        raise ValueError(f"Corrected reference failed its geometry audit: {report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "robot/reference_data/multimotion/stairs.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "robot/reference_data/multimotion/stairs_supported.npz")
    parser.add_argument("--model-xml", type=Path, default=ROOT / "robot/MJCFmodel_stair.xml")
    parser.add_argument("--root-x-offset", type=float, default=-0.39)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.model_xml, args.root_x_offset), indent=2))
