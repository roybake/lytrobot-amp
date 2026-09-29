#!/usr/bin/env python3
"""Retarget the prepared walk1 cycle with official OmniRetarget SQP."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import MethodType

import mujoco
import numpy as np
from scipy.optimize import least_squares

from prepare_walk1 import (
    ROLL_FLAT,
    ROLL_HEEL,
    ROLL_TOE,
    contact_windows,
    foot_roll_schedule,
)
from run_omniretarget import (
    DEMO_JOINTS,
    JOINTS_MAPPING,
    ROBOT_JOINT_NAMES,
    ground_points,
    task_constants,
)
from holosoma_retargeting.config_types.retargeter import FootLockConfig
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from holosoma_retargeting.src.utils import preprocess_motion_data

HERE = Path(__file__).resolve().parent


def walking_nominal(
    retargeter: InteractionMeshRetargeter,
    human: np.ndarray,
    contacts: np.ndarray,
    roll_mode: np.ndarray,
) -> np.ndarray:
    """Build a sequential lower-body IK seed from the actual LAFAN keypoints."""
    model = retargeter.robot_model
    data = mujoco.MjData(model)
    qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
    joint_qpos_addresses = np.asarray(
        [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ROBOT_JOINT_NAMES[1:]
        ],
        dtype=np.int64,
    )
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ROBOT_JOINT_NAMES[1:]
    ]
    lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in joint_ids])
    upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in joint_ids])
    target_pairs = (
        ("right_hip", "RightUpLeg", 0.5),
        ("right_knee", "RightLeg", 1.2),
        ("right_foot", "RightFoot", 1.5),
        ("right_toe", "RightToeBase", 1.0),
        ("left_hip", "LeftUpLeg", 0.5),
        ("left_knee", "LeftLeg", 1.2),
        ("left_foot", "LeftFoot", 1.5),
        ("left_toe", "LeftToeBase", 1.0),
    )
    body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) for body, _, _ in target_pairs]
    )
    human_ids = np.asarray([DEMO_JOINTS.index(name) for _, name, _ in target_pairs])
    weights = np.asarray([weight for _, _, weight in target_pairs])[:, None]
    previous = np.clip(qpos[joint_qpos_addresses], lower, upper)
    nominal = np.zeros((len(human), model.nq), dtype=np.float64)

    for frame, target in enumerate(human):
        qpos[:3] = target[DEMO_JOINTS.index("Hips")]
        qpos[3:7] = (1.0, 0.0, 0.0, 0.0)

        def residual(joints: np.ndarray) -> np.ndarray:
            qpos[joint_qpos_addresses] = joints
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            position_error = (np.asarray(data.xpos[body_ids]) - target[human_ids]) * weights
            smoothness = 0.08 * (joints - previous)
            return np.concatenate((position_error.ravel(), smoothness))

        solution = least_squares(
            residual,
            previous,
            bounds=(lower, upper),
            max_nfev=80,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        qpos[joint_qpos_addresses] = solution.x
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        active_soles = []
        for side, side_name in enumerate(("right", "left")):
            if contacts[frame, side]:
                mode = int(roll_mode[frame, side])
                if mode == ROLL_HEEL:
                    link_names = (f"{side_name}_heel_sole",)
                elif mode == ROLL_FLAT:
                    link_names = (
                        f"{side_name}_heel_sole",
                        f"{side_name}_sole",
                        f"{side_name}_toe_sole",
                    )
                elif mode == ROLL_TOE:
                    link_names = (f"{side_name}_toe_sole",)
                else:
                    raise ValueError(f"Invalid stance roll mode {mode} at frame {frame}.")
                active_soles.extend(data.body(name).xpos[2] for name in link_names)
        if active_soles:
            qpos[2] -= float(np.mean(active_soles))
        nominal[frame] = qpos
        previous = solution.x.copy()
    return nominal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-urdf",
        type=Path,
        default=HERE / "models" / "lyt" / "lyt_14dof.urdf",
    )
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--foot-tolerance", type=float, default=0.02)
    parser.add_argument("--nominal-weight", type=float, default=5.0)
    args = parser.parse_args()

    with np.load(args.input) as archive:
        human = np.asarray(archive["global_joint_positions"], dtype=np.float64)
        fps = float(archive["fps"])
        contacts = np.asarray(archive["foot_contact"], dtype=bool)
        roll_phase = (
            np.asarray(archive["foot_roll_phase"], dtype=np.float32)
            if "foot_roll_phase" in archive.files
            else foot_roll_schedule(contacts)[0]
        )
        roll_mode = (
            np.asarray(archive["foot_roll_mode"], dtype=np.int8)
            if "foot_roll_mode" in archive.files
            else foot_roll_schedule(contacts)[1]
        )
        nominal = (
            np.asarray(archive["q_nominal"], dtype=np.float64)
            if "q_nominal" in archive.files
            else None
        )
        scale = float(archive["scale"]) if "scale" in archive.files else 1.0
        target_forward_velocity = np.asarray(archive["target_forward_velocity"], dtype=np.float32)
        source_frames = np.asarray(archive["source_frames"], dtype=np.int32)
        source = str(archive["source"]) if "source" in archive.files else "unknown"
    expected_aux_shape = (len(human), 2)
    if (
        human.shape[1:] != (len(DEMO_JOINTS), 3)
        or contacts.shape != expected_aux_shape
        or roll_phase.shape != expected_aux_shape
        or roll_mode.shape != expected_aux_shape
    ):
        raise ValueError(
            "Unexpected prepared shapes: "
            f"human={human.shape}, contacts={contacts.shape}, "
            f"roll_phase={roll_phase.shape}, roll_mode={roll_mode.shape}."
        )

    constants = task_constants(args.model_urdf)
    # Track root height and all actuated joints.  The free-root orientation is
    # left to the interaction-mesh objective; constraining it here overpowers
    # the alternating stance-foot constraints at the cycle boundary.
    constants.NOMINAL_TRACKING_INDICES = np.asarray([2, *range(7, 21)], dtype=np.int64)
    constants.FOOT_STICKING_LINKS = (
        "right_heel_sole",
        "right_sole",
        "right_toe_sole",
        "left_heel_sole",
        "left_sole",
        "left_toe_sole",
    )
    foot_lock = FootLockConfig(
        enable=True,
        windows={
            "right": contact_windows(contacts[:, 0]),
            "left": contact_windows(contacts[:, 1]),
        },
        z_floor=0.0,
        tolerance=args.foot_tolerance,
    )
    retargeter = InteractionMeshRetargeter(
        task_constants=constants,
        object_urdf_path=None,
        q_a_init_idx=-7,
        # A rigid XY lock on every sole landmark prevents heel/toe pivoting.
        # The staged Z locks below preserve contact while allowing foot roll.
        activate_foot_sticking=False,
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
    original_contact_jacobian = retargeter._calc_contact_jacobian_from_point

    def contact_jacobian_compat(self, body_idx, *jacobian_args, **jacobian_kwargs):
        del self
        scalar_body_idx = int(np.asarray(body_idx).reshape(-1)[0])
        return original_contact_jacobian(scalar_body_idx, *jacobian_args, **jacobian_kwargs)

    retargeter._calc_contact_jacobian_from_point = MethodType(contact_jacobian_compat, retargeter)

    def staged_foot_lock(self, foot_link_key: str, frame_idx: int) -> bool:
        del self
        side = 0 if foot_link_key.startswith("right") else 1
        if not contacts[frame_idx, side]:
            return False
        mode = int(roll_mode[frame_idx, side])
        if mode == ROLL_HEEL:
            return "heel_sole" in foot_link_key
        if mode == ROLL_FLAT:
            return True
        if mode == ROLL_TOE:
            return "toe_sole" in foot_link_key
        return False

    retargeter._is_foot_locked_in_window = MethodType(staged_foot_lock, retargeter)
    human = preprocess_motion_data(
        human,
        retargeter,
        ["LeftToeBase", "RightToeBase"],
        scale=scale,
    )
    if nominal is None:
        nominal = walking_nominal(retargeter, human, contacts, roll_mode)
    object_poses = nominal[:, -7:].copy()
    contact_sequence = [
        {"RightToeBase": bool(frame[0]), "LeftToeBase": bool(frame[1])} for frame in contacts
    ]
    points = ground_points()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    retargeter.retarget_motion(
        human_joint_motions=human,
        object_poses=object_poses,
        object_poses_augmented=object_poses,
        object_points_local_demo=points,
        object_points_local=points,
        foot_sticking_sequences=contact_sequence,
        q_a_init=nominal[0],
        q_nominal_list=nominal,
        original=True,
        dest_res_path=str(args.output),
    )
    with np.load(args.output) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload.update(
        {
            "fps": np.float32(fps),
            "joint_names": np.asarray(ROBOT_JOINT_NAMES),
            "foot_contact": contacts.astype(np.float32),
            "foot_roll_phase": roll_phase.astype(np.float32),
            "foot_roll_mode": roll_mode.astype(np.int8),
            "target_forward_velocity": target_forward_velocity,
            "source_frames": source_frames,
            "source": np.asarray(source),
            "human_scale": np.float32(scale),
            "retarget_step_size": np.float32(args.step_size),
            "retarget_foot_tolerance": np.float32(args.foot_tolerance),
            "retarget_nominal_weight": np.float32(args.nominal_weight),
            "staged_foot_roll": np.bool_(True),
            "mapping_human_to_robot": np.asarray(
                [
                    f"{human_name}->{robot_name}"
                    for human_name, robot_name in JOINTS_MAPPING.items()
                ]
            ),
        }
    )
    np.savez_compressed(args.output, **payload)
    print(f"saved OmniRetarget walk result: {args.output}")


if __name__ == "__main__":
    main()
