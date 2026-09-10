#!/usr/bin/env python3
"""Run official OmniRetarget interaction-mesh SQP for the LYT squat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from types import MethodType

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
OFFICIAL_SRC = HERE / "omniretarget_project" / "src" / "holosoma_retargeting"
if str(OFFICIAL_SRC) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SRC))

from holosoma_retargeting.config_types.retargeter import FootLockConfig  # noqa: E402
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter  # noqa: E402
from holosoma_retargeting.src.utils import preprocess_motion_data  # noqa: E402

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
JOINTS_MAPPING = {
    "Hips": "pelvis",
    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_foot",
    "RightToeBase": "right_toe",
    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_foot",
    "LeftToeBase": "left_toe",
}
ROBOT_JOINT_NAMES = (
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


def task_constants(model_urdf: Path) -> SimpleNamespace:
    return SimpleNamespace(
        ROBOT_DOF=14,
        ROBOT_HEIGHT=0.875,
        ROBOT_NAME="lyt_14dof",
        ROBOT_URDF_FILE=str(model_urdf.resolve()),
        OBJECT_NAME="ground",
        FOOT_STICKING_LINKS=("left_sole", "left_toe_sole", "right_sole", "right_toe_sole"),
        MANUAL_LB={"3": -1.0, "4": -1.0, "5": -1.0, "6": -1.0},
        MANUAL_UB={"3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0},
        MANUAL_COST={},
        NOMINAL_TRACKING_INDICES=np.asarray([2, 7, 10, 11, 14, 17, 18], dtype=np.int64),
        DEMO_JOINTS=list(DEMO_JOINTS),
        JOINTS_MAPPING=JOINTS_MAPPING,
    )


def ground_points(size: int = 7, extent: float = 0.65) -> np.ndarray:
    axis = np.linspace(-extent, extent, size)
    x, y = np.meshgrid(axis, axis)
    return np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)


def shallow_squat_motion(human: np.ndarray, depth_scale: float, endpoint_frames: int = 8) -> np.ndarray:
    """Scale a stand-squat-stand motion around its endpoint standing pose."""
    if not 0.0 < depth_scale <= 1.0:
        raise ValueError("Squat depth scale must be in (0, 1].")
    if depth_scale == 1.0:
        return human.copy()

    endpoint_frames = min(endpoint_frames, max(1, len(human) // 4))
    toe_indices = [DEMO_JOINTS.index("LeftToeBase"), DEMO_JOINTS.index("RightToeBase")]
    endpoint_indices = np.r_[0:endpoint_frames, len(human) - endpoint_frames : len(human)]
    endpoint_anchors = human[endpoint_indices][:, toe_indices].mean(axis=1, keepdims=True)
    standing_local = np.mean(human[endpoint_indices] - endpoint_anchors, axis=0)

    # Keep the per-frame support anchor, while reducing the full-body pose
    # displacement from standing. This halves both the pelvis descent and the
    # leg flexion presented to OmniRetarget and preserves the return-to-stand.
    anchors = human[:, toe_indices].mean(axis=1, keepdims=True)
    standing = standing_local[None, :, :] + anchors
    return standing + depth_scale * (human - standing)


def symmetric_squat_nominal(
    retargeter: InteractionMeshRetargeter,
    human: np.ndarray,
    depth_scale: float,
) -> np.ndarray:
    """Build a feasible symmetric seed so SQP does not enter a one-leg local minimum."""
    model = retargeter.robot_model
    data = mujoco.MjData(model)
    hip_height = human[:, DEMO_JOINTS.index("Hips"), 2]
    height_range = max(float(np.ptp(hip_height)), 1.0e-6)
    depth = np.clip((float(hip_height.max()) - hip_height) / height_range, 0.0, 1.0)
    nominal = np.repeat(np.asarray(model.qpos0)[None, :], len(human), axis=0)
    joint_targets = {
        "hip_flexion_r": 0.72,
        "knee_angle_r": -1.44,
        "ankle_angle_r": 0.70,
        "hip_flexion_l": 0.72,
        "knee_angle_l": -1.44,
        "ankle_angle_l": 0.70,
    }
    # Pelvis descent is nonlinear in the coupled hip/knee flexion. This
    # fourth-root mapping is calibrated for the LYT leg geometry so the SQP
    # solution's pelvis-height change follows the requested motion-depth scale.
    joint_depth_scale = float(np.sqrt(np.sqrt(depth_scale)))
    sole_names = ("right_sole", "right_toe_sole", "left_sole", "left_toe_sole")
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    sole_anchor_xy = np.mean([data.body(name).xpos[:2] for name in sole_names], axis=0)
    for frame, phase in enumerate(depth):
        for joint_name, maximum in joint_targets.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_address = int(model.jnt_qposadr[joint_id])
            nominal[frame, qpos_address] = maximum * joint_depth_scale * phase
        data.qpos[:] = nominal[frame]
        mujoco.mj_forward(model, data)
        soles = np.asarray([data.body(name).xpos.copy() for name in sole_names])
        nominal[frame, :2] += sole_anchor_xy - soles[:, :2].mean(axis=0)
        nominal[frame, 2] -= soles[:, 2].mean()
    return nominal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-urdf", type=Path, default=HERE / "models" / "lyt" / "lyt_14dof.urdf")
    parser.add_argument("--step-size", type=float, default=0.16)
    parser.add_argument("--foot-tolerance", type=float, default=0.0001)
    parser.add_argument("--nominal-weight", type=float, default=3.0)
    parser.add_argument(
        "--squat-depth-scale",
        type=float,
        default=0.5,
        help="Squat displacement relative to the source motion (default: half depth).",
    )
    args = parser.parse_args()

    with np.load(args.input) as archive:
        human = np.asarray(archive["global_joint_positions"], dtype=np.float64)
        human_height = float(archive["height"])
        fps = float(archive["fps"])
    if human.shape[1:] != (len(DEMO_JOINTS), 3):
        raise ValueError(f"Unexpected human keypoint shape: {human.shape}")
    human = shallow_squat_motion(human, args.squat_depth_scale)

    constants = task_constants(args.model_urdf)
    scale = constants.ROBOT_HEIGHT / human_height
    frame_count = len(human)
    foot_lock = FootLockConfig(
        enable=True,
        windows={"left": [(0, frame_count - 1)], "right": [(0, frame_count - 1)]},
        z_floor=0.0,
        tolerance=args.foot_tolerance,
    )
    retargeter = InteractionMeshRetargeter(
        task_constants=constants,
        object_urdf_path=None,
        q_a_init_idx=-7,
        activate_foot_sticking=True,
        activate_obj_non_penetration=True,
        activate_joint_limits=True,
        step_size=args.step_size,
        penetration_tolerance=0.001,
        foot_sticking_tolerance=args.foot_tolerance,
        foot_lock=foot_lock,
        visualize=False,
        debug=False,
        w_nominal_tracking_init=args.nominal_weight,
    )

    # MuJoCo 3.9's named geom accessor exposes ``bodyid`` as a length-one
    # array, while this official revision calls ``int(bodyid)`` directly.
    # Adapt the boundary locally without changing the OmniRetarget sources.
    original_contact_jacobian = retargeter._calc_contact_jacobian_from_point

    def contact_jacobian_compat(self, body_idx, *jacobian_args, **jacobian_kwargs):
        del self
        scalar_body_idx = int(np.asarray(body_idx).reshape(-1)[0])
        return original_contact_jacobian(scalar_body_idx, *jacobian_args, **jacobian_kwargs)

    retargeter._calc_contact_jacobian_from_point = MethodType(contact_jacobian_compat, retargeter)
    human = preprocess_motion_data(
        human,
        retargeter,
        ["LeftToeBase", "RightToeBase"],
        scale=scale,
    )
    nominal = symmetric_squat_nominal(retargeter, human, args.squat_depth_scale)
    q_init = nominal[0].copy()
    # The official ground-only path copies this seven-vector into the final
    # seven configuration coordinates.  Supplying the nominal left-leg slice
    # preserves the robot-only seed; the ground branch never treats it as SE(3).
    object_poses = nominal[:, -7:].copy()
    contacts = [{"LeftToeBase": True, "RightToeBase": True} for _ in range(frame_count)]
    points = ground_points()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    retargeter.retarget_motion(
        human_joint_motions=human,
        object_poses=object_poses,
        object_poses_augmented=object_poses,
        object_points_local_demo=points,
        object_points_local=points,
        foot_sticking_sequences=contacts,
        q_a_init=q_init,
        q_nominal_list=nominal,
        original=True,
        dest_res_path=str(args.output),
    )
    # Official writer uses 30 fps. Preserve the actual prepared rate explicitly.
    with np.load(args.output) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["fps"] = np.float32(fps)
    payload["joint_names"] = np.asarray(ROBOT_JOINT_NAMES)
    payload["squat_depth_scale"] = np.float32(args.squat_depth_scale)
    payload["mapping_human_to_robot"] = np.asarray(
        [f"{human}->{robot}" for human, robot in JOINTS_MAPPING.items()]
    )
    np.savez_compressed(args.output, **payload)
    print(f"saved OmniRetarget result: {args.output}")


if __name__ == "__main__":
    main()
