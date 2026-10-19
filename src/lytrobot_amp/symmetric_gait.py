"""Construct one mirrored walking cycle before deriving tracking/control targets."""

from __future__ import annotations

import mujoco
import numpy as np


def symmetric_walking_cycle(reference, forward_velocity):
    model = reference.model
    contacts = reference.reference_foot_contact > .5
    strikes = [np.flatnonzero(c[1:] & ~c[:-1]) + 1 for c in contacts.T]
    # Select a complete cycle with exactly one opposite-side landing.
    cycle = None
    for start, end in zip(strikes[0][:-1], strikes[0][1:]):
        opposite = strikes[1][(strikes[1] > start) & (strikes[1] < end)]
        if len(opposite) == 1:
            cycle = int(start), int(opposite[0]), int(end)
            break
    if cycle is None:
        raise ValueError("A symmetric walking reference requires a complete alternating gait cycle.")
    start, middle, end = cycle
    half = max(2, round((end - start) / 2))
    count = 2 * half
    phase = np.r_[np.linspace(start, middle, half, endpoint=False),
                  np.linspace(middle, end, half, endpoint=False)]
    low = np.floor(phase).astype(int)
    weight = (phase - low)[:, None]
    first = reference.reference_qpos[low].astype(np.float64)
    second = reference.reference_qpos[low + 1].astype(np.float64)
    second[np.sum(first[:, 3:7] * second[:, 3:7], axis=1) < 0, 3:7] *= -1
    poses = (1 - weight) * first + weight * second
    poses[:, 3:7] /= np.linalg.norm(poses[:, 3:7], axis=1, keepdims=True)
    opposite = np.roll(poses, half, axis=0)
    mirror_quat = opposite[:, 3:7] * [1, -1, 1, -1]
    mirror_quat[np.sum(poses[:, 3:7] * mirror_quat, axis=1) < 0] *= -1
    poses[:, 3:7] += mirror_quat
    poses[:, 3:7] /= np.linalg.norm(poses[:, 3:7], axis=1, keepdims=True)
    poses[:, 1] = .5 * (poses[:, 1] - opposite[:, 1])
    poses[:, 2] = .5 * (poses[:, 2] + opposite[:, 2])
    for right_name in reference.common_joint_names:
        if not right_name.endswith("_r"):
            continue
        right, left = model.joint(right_name).id, model.joint(right_name[:-2] + "_l").id
        r, l = int(model.jnt_qposadr[right]), int(model.jnt_qposadr[left])
        # A reflected hinge axis is an axial vector: det(S) S axis.
        reflected_axis = model.jnt_axis[right] * [-1, 1, -1]
        sign = 1. if np.dot(reflected_axis, model.jnt_axis[left]) >= 0 else -1.
        left_limits = np.sort(sign * model.jnt_range[left])
        lower = max(model.jnt_range[right, 0], left_limits[0])
        upper = min(model.jnt_range[right, 1], left_limits[1])
        if lower >= upper:
            raise ValueError(f"No common mirrored joint range for {right_name}.")
        right_pose = np.clip(.5 * (poses[:, r] + sign * opposite[:, l]), lower, upper)
        poses[:, r] = right_pose
        poses[:, l] = sign * np.roll(right_pose, half)
    # Passive joint equalities belong to the robot model as well.
    for eq in range(model.neq):
        if model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT or model.eq_obj2id[eq] < 0:
            continue
        a, b = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
        qa, qb = int(model.jnt_qposadr[a]), int(model.jnt_qposadr[b])
        x = poses[:, qb] - model.qpos0[qb]
        poses[:, qa] = model.qpos0[qa] + sum(model.eq_data[eq, k] * x**k for k in range(5))
    speed = float(forward_velocity)
    if speed <= 0 or not np.isfinite(speed):
        raise ValueError("The symmetric walking reference requires positive forward velocity.")
    # A mirrored pose must also place its physical foot colliders on the
    # canonical ground. Averaging joint poses alone can put a sole below it.
    data = mujoco.MjData(model)
    vertices = []
    for geom in reference.contact_geom_ids:
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Symmetric LYT gait construction requires mesh foot colliders.")
        mesh = int(model.geom_dataid[geom])
        begin, size = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices.append(model.mesh_vert[begin:begin + size])
    lift = np.empty(count)
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        bottom = min(float(np.min(points @ data.geom_xmat[geom].reshape(3, 3)[2]
                                  + data.geom_xpos[geom, 2]))
                     for geom, points in zip(reference.contact_geom_ids, vertices))
        lift[i] = -bottom
    poses[:, 2] += np.maximum(lift, np.roll(lift, half))
    # Keep the commanded speed, landing distances and cadence consistent:
    # one cycle advances by the sum of its left and right landing steps.
    steps = []
    for frame, side in ((0, 0), (half, 1)):
        data.qpos[:] = poses[frame]
        mujoco.mj_kinematics(model, data)
        foot_x = data.geom_xpos[reference.contact_geom_ids, 0].reshape(2, 2).mean(axis=1)
        steps.append(float(foot_x[side] - foot_x[1 - side]))
    if min(steps) <= 0:
        raise ValueError("The symmetric walking cycle must contain two forward landing steps.")
    period = sum(steps) / speed
    frequency = count / period
    translation = np.array([speed * period, 0.])
    poses[:, 0] = reference.reference_qpos[start, 0] + np.arange(count) * speed / frequency
    velocity = np.empty((count, model.nv))
    for i in range(count):
        previous, following = poses[(i - 1) % count].copy(), poses[(i + 1) % count].copy()
        if i == 0:
            previous[:2] -= translation
        if i == count - 1:
            following[:2] += translation
        mujoco.mj_differentiatePos(model, velocity[i], 2 / frequency, previous, following)
    right_contact = contacts[low, 0]
    contact = np.column_stack((right_contact, np.roll(right_contact, half))).astype(np.float32)
    return poses.astype(np.float32), velocity.astype(np.float32), contact, frequency
