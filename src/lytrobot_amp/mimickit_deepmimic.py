"""MimicKit DeepMimic reward port for the LYT MuJoCo character.

The five error terms and their scales follow MimicKit's
``mimickit/envs/deepmimic_env.py:compute_reward`` (Apache-2.0).  LYT has
one-axis joints, so wrapped hinge-angle differences replace joint-quaternion
angle differences. Four heel/toe contact geoms are the available key points.
The deployable sensor actor has no global target-position observation, hence
the original ``track_root=False`` branch is used: only root height is tracked.
"""

from __future__ import annotations

import torch


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    qvec = quat[..., 1:]
    while qvec.ndim < vector.ndim:
        qvec = qvec.unsqueeze(-2)
    cross = torch.cross(qvec.expand_as(vector), vector, dim=-1)
    return vector + 2.0 * (
        quat[..., :1].reshape(quat.shape[:-1] + (1,) * (vector.ndim - quat.ndim) + (1,))
        * cross + torch.cross(qvec.expand_as(vector), cross, dim=-1)
    )


def _heading_inverse(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = -0.5 * yaw
    zero = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)


def _quat_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(_normalize_quat(a) * _normalize_quat(b), dim=-1).abs()
    return 2.0 * torch.acos(dot.clamp(0.0, 1.0))


def deepmimic_reward(
    *,
    root_height: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    key_pos_root_local: torch.Tensor,
    target_root_height: torch.Tensor,
    target_root_quat: torch.Tensor,
    target_root_lin_vel: torch.Tensor,
    target_root_ang_vel: torch.Tensor,
    target_joint_pos: torch.Tensor,
    target_joint_vel: torch.Tensor,
    target_key_pos_root_local: torch.Tensor,
    joint_weights: torch.Tensor | None = None,
    root_xy: torch.Tensor | None = None,
    target_root_xy: torch.Tensor | None = None,
    root_xy_penalty: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the original five-term reward with LYT hinge/key-point mapping."""
    root_quat = _normalize_quat(root_quat)
    target_root_quat = _normalize_quat(target_root_quat)
    inv_heading = _heading_inverse(root_quat)
    inv_target_heading = _heading_inverse(target_root_quat)
    local_quat = _quat_mul(inv_heading, root_quat)
    local_target_quat = _quat_mul(inv_target_heading, target_root_quat)
    local_lin_vel = _quat_rotate(inv_heading, root_lin_vel)
    local_target_lin_vel = _quat_rotate(inv_target_heading, target_root_lin_vel)
    local_ang_vel = _quat_rotate(inv_heading, root_ang_vel)
    local_target_ang_vel = _quat_rotate(inv_target_heading, target_root_ang_vel)
    # MotionReference stores contact points in the full root frame. Convert
    # them back to world-relative positions, then to MimicKit's heading frame.
    key_pos = _quat_rotate(inv_heading, _quat_rotate(root_quat, key_pos_root_local))
    target_key_pos = _quat_rotate(
        inv_target_heading,
        _quat_rotate(target_root_quat, target_key_pos_root_local),
    )
    angle_diff = torch.remainder(joint_pos - target_joint_pos + torch.pi, 2.0 * torch.pi) - torch.pi
    velocity_diff = joint_vel - target_joint_vel
    weights = torch.ones_like(angle_diff) if joint_weights is None else joint_weights
    pose_err = (weights * angle_diff.square()).sum(dim=-1)
    vel_err = (weights * velocity_diff.square()).sum(dim=-1)
    root_pos_err = (root_height - target_root_height).square()
    xy_error = None
    if root_xy is not None or target_root_xy is not None:
        if root_xy is None or target_root_xy is None:
            raise ValueError("root_xy and target_root_xy must be provided together")
        xy_error = torch.linalg.vector_norm(root_xy - target_root_xy, dim=-1)
        root_pos_err = root_pos_err + xy_error.square()
    root_rot_err = _quat_angle(local_quat, local_target_quat).square()
    root_vel_err = (local_lin_vel - local_target_lin_vel).square().sum(dim=-1)
    root_ang_vel_err = (local_ang_vel - local_target_ang_vel).square().sum(dim=-1)
    key_pos_err = (key_pos - target_key_pos).square().sum(dim=(-2, -1))
    terms = {
        "pose": torch.exp(-0.25 * pose_err),
        "velocity": torch.exp(-0.01 * vel_err),
        "root_pose": torch.exp(-5.0 * (root_pos_err + 0.1 * root_rot_err)),
        "root_velocity": torch.exp(-(root_vel_err + 0.1 * root_ang_vel_err)),
        "key_position": torch.exp(-10.0 * key_pos_err),
    }
    reward = (
        0.50 * terms["pose"] + 0.10 * terms["velocity"]
        + 0.15 * terms["root_pose"] + 0.10 * terms["root_velocity"]
        + 0.15 * terms["key_position"]
    )
    if xy_error is not None:
        terms["root_xy_error"] = xy_error
        reward = reward - float(root_xy_penalty) * xy_error
    return reward, terms
