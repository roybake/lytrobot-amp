#!/usr/bin/env python3
"""Make the full-length walk1 human mocap consistent with the LYTrobot MJCF.

Differences removed (walk1_mjcf.npz -> walk1_mjcf_grounded.npz):
  * joint amplitudes exceeding the robot limits (hip flexion/rotation/adduction,
    ankle, knee) are rescaled about their centre instead of hard-clipped;
  * passive patella slide joints follow the model's polynomial equality;
  * pelvis heading is aligned with the walking direction (+X);
  * root height is solved per frame from the lowest vertex of the foot/toe STL
    meshes, so the support foot touches the floor (no penetration, no float);
  * qvel is re-differentiated from the corrected qpos; foot_contact is exported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
FOOT_GEOMS = ("foot_r", "toe_r", "foot_l", "toe_l")
CLEARANCE = 0.0005          # target gap of the lowest foot point (m)
MAX_PENETRATION = 0.0005    # allowed (m)


def passive(model, qpos):
    for eq in range(model.neq):
        if model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        a = int(model.jnt_qposadr[int(model.eq_obj1id[eq])])
        b = int(model.jnt_qposadr[int(model.eq_obj2id[eq])])
        x = qpos[:, b] - model.qpos0[b]
        qpos[:, a] = model.qpos0[a] + sum(model.eq_data[eq, k] * x**k for k in range(5))


def fit_limits(model, qpos):
    report = {}
    for j in range(1, model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[j]:
            continue
        name = model.joint(j).name
        if name.startswith("patella"):
            continue
        adr = int(model.jnt_qposadr[j])
        low, high = model.jnt_range[j] + np.array([0.002, -0.002])
        if name.startswith("knee_angle"):
            high = min(high, -0.02)
        v = qpos[:, adr]
        p_lo, p_hi = np.percentile(v, [0.1, 99.9])
        centre, half = 0.5 * (p_lo + p_hi), 0.5 * (p_hi - p_lo)
        if p_lo >= low and p_hi <= high:
            scale, new_centre = 1.0, centre
        else:
            new_half = min(half, 0.5 * (high - low))
            scale = new_half / max(half, 1e-9)
            new_centre = np.clip(centre, low + new_half, high - new_half)
        qpos[:, adr] = np.clip(new_centre + scale * (v - centre), low, high)
        report[name] = round(float(scale), 3)
    return report


def foot_vertices(model):
    out = []
    for name in FOOT_GEOMS:
        g = model.geom(name).id
        m = int(model.geom_dataid[g])
        a, n = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
        v = np.asarray(model.mesh_vert[a:a + n], dtype=np.float64)
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        out.append((int(model.geom_bodyid[g]), model.geom_pos[g] + v @ rot.reshape(3, 3).T))
    return out


def lowest_points(model, qpos, verts):
    """(N, 4) lowest world-z of foot_r, toe_r, foot_l, toe_l meshes."""
    data = mujoco.MjData(model)
    low = np.empty((len(qpos), 4))
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for k, (b, v) in enumerate(verts):
            low[i, k] = data.xpos[b, 2] + (v @ data.xmat[b].reshape(3, 3)[2]).min()
    return low


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "robot/reference_data/walk1_mjcf.npz")
    ap.add_argument("--output", type=Path, default=ROOT / "robot/reference_data/walk1_mjcf_grounded.npz")
    ap.add_argument("--model-xml", type=Path, default=ROOT / "robot/MJCFmodelposition.xml")
    ap.add_argument("--smooth-sigma", type=float, default=0.5, help="root-height smoothing (frames)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    src = np.load(args.input, allow_pickle=True)
    q = src["qpos"].astype(np.float64)
    hz = float(src["frequency"])
    names = [str(n) for n in src["joint_names"]]

    # 1) human root translation (source stores zeros) is taken from walk1.npz
    human = np.load(args.input.with_name("walk1.npz"), allow_pickle=True)["qpos"].astype(np.float64)
    q[:, :2] = human[:, :2] - human[0, :2]

    # 2) heading: remove the constant yaw offset so the subject walks along +X
    rot = Rotation.from_quat(q[:, [4, 5, 6, 3]])
    yaw = rot.as_euler("zyx")[:, 0]
    mean_yaw = float(np.angle(np.exp(1j * yaw).mean()))
    fix = Rotation.from_euler("z", -mean_yaw)
    q[:, 3:7] = (fix * rot).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(q)):
        if q[i, 3:7] @ q[i - 1, 3:7] < 0:
            q[i, 3:7] *= -1

    # 3) joint limits and passive joints
    scales = fit_limits(model, q)
    passive(model, q)

    # 4) per-frame root height from the foot meshes
    verts = foot_vertices(model)
    low = lowest_points(model, q, verts).min(1)
    z_raw = q[:, 2] + CLEARANCE - low
    z = np.maximum(gaussian_filter1d(z_raw, args.smooth_sigma, mode="nearest"),
                   z_raw - (CLEARANCE + MAX_PENETRATION))
    q[:, 2] = z

    # 5) contact labels, velocities
    low4 = lowest_points(model, q, verts)
    foot = np.column_stack((low4[:, :2].min(1), low4[:, 2:].min(1)))
    contact = (foot < 0.006).astype(np.float32)
    qvel = np.empty((len(q), model.nv))
    for i in range(len(q)):
        mujoco.mj_differentiatePos(model, qvel[i], 2.0 / hz, q[max(i - 1, 0)], q[min(i + 1, len(q) - 1)])
    qvel[0] *= 2.0
    qvel[-1] *= 2.0

    meta = {
        "task": "walk", "source": str(args.input), "model_xml": str(args.model_xml),
        "method": "limit-fit by centre-scaling, passive patella equality, heading aligned to +X, "
                  "per-frame root height from foot/toe STL lowest vertex",
        "enforce_passive_joint_constraints": True, "joint_amplitude_scale": scales,
        "removed_mean_yaw_deg": float(np.rad2deg(mean_yaw)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, qpos=q.astype(np.float32), qvel=qvel.astype(np.float32),
        joint_names=np.asarray(names[:0] or [str(n) for n in src["joint_names"]], dtype=object),
        frequency=hz, loop=False, foot_contact=contact, metadata=json.dumps(meta, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("lowest foot point mm: min %.3f  max %.3f | contact duty R %.2f L %.2f"
          % (low4.min() * 1e3, low4.min(1).max() * 1e3, contact[:, 0].mean(), contact[:, 1].mean()))
    print("saved", args.output)


if __name__ == "__main__":
    main()
