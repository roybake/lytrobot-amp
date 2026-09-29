#!/usr/bin/env python3
"""Retarget one prepared locomotion clip with official OmniRetarget SQP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import MethodType

import mujoco
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LEGACY = REPO_ROOT / "squat_training" / "retargeting"
OFFICIAL_SRC = LEGACY / "omniretarget_project" / "src" / "holosoma_retargeting"
sys.path[:0] = [str(LEGACY), str(OFFICIAL_SRC)]

from holosoma_retargeting.config_types.retargeter import FootLockConfig  # noqa: E402
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,
)
from holosoma_retargeting.src.utils import preprocess_motion_data  # noqa: E402
from prepare_walk1 import ROLL_FLAT, ROLL_HEEL, ROLL_TOE  # noqa: E402
from run_omniretarget import (  # noqa: E402
    DEMO_JOINTS,
    ROBOT_JOINT_NAMES,
    task_constants,
)

MODEL_URDF = LEGACY / "models" / "lyt" / "lyt_14dof.urdf"


def kinematic_seed(
    retargeter: InteractionMeshRetargeter,
    human: np.ndarray,
    contacts: np.ndarray,
    roll_mode: np.ndarray,
    yaw: np.ndarray,
) -> np.ndarray:
    """Sequential lower-body IK with root yaw, support height and leg clearance."""

    model = retargeter.robot_model
    data = mujoco.MjData(model)
    qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
    joint_addresses = np.asarray(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
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
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            for body, _, _ in target_pairs
        ]
    )
    human_ids = np.asarray(
        [DEMO_JOINTS.index(name) for _, name, _ in target_pairs]
    )
    weights = np.asarray([weight for _, _, weight in target_pairs])[:, None]
    toe_ids = (
        DEMO_JOINTS.index("RightToeBase"),
        DEMO_JOINTS.index("LeftToeBase"),
    )
    previous = np.clip(qpos[joint_addresses], lower, upper)
    nominal = np.zeros((len(human), model.nq), dtype=np.float64)
    # The collision capsules have radii 45 mm (thigh) and 40 mm (shank).
    # Keep their centre lines separated before the SQP stage so the optimizer
    # never starts from the mirrored/cross-legged local minimum.
    minimum_lateral_separation = np.asarray([0.10, 0.09, 0.08])

    for frame, target in enumerate(human):
        qpos[:3] = target[DEMO_JOINTS.index("Hips")]
        half = 0.5 * float(yaw[frame])
        qpos[3:7] = (np.cos(half), 0.0, 0.0, np.sin(half))

        def residual(joints: np.ndarray) -> np.ndarray:
            qpos[joint_addresses] = joints
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            position_error = (
                np.asarray(data.xpos[body_ids]) - target[human_ids]
            ) * weights
            tracked_positions = np.asarray(data.xpos[body_ids])
            right_to_left = tracked_positions[4:, :2] - tracked_positions[:4, :2]
            heading = float(yaw[frame])
            lateral_separation = (
                -np.sin(heading) * right_to_left[:, 0]
                + np.cos(heading) * right_to_left[:, 1]
            )
            clearance_error = 5.0 * np.minimum(
                lateral_separation[1:] - minimum_lateral_separation,
                0.0,
            )
            return np.concatenate(
                (
                    position_error.ravel(),
                    clearance_error,
                    0.08 * (joints - previous),
                )
            )

        solution = least_squares(
            residual,
            previous,
            bounds=(lower, upper),
            max_nfev=80,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        qpos[joint_addresses] = solution.x
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        support_actual: list[float] = []
        support_target: list[float] = []
        for side, side_name in enumerate(("right", "left")):
            if not contacts[frame, side]:
                continue
            mode = int(roll_mode[frame, side])
            if mode == ROLL_HEEL:
                links = (f"{side_name}_heel_sole",)
            elif mode == ROLL_TOE:
                links = (f"{side_name}_toe_sole",)
            else:
                links = (f"{side_name}_sole", f"{side_name}_toe_sole")
            support_actual.extend(float(data.body(link).xpos[2]) for link in links)
            support_target.extend([float(target[toe_ids[side], 2])] * len(links))
        if support_actual:
            qpos[2] += float(np.mean(support_target) - np.mean(support_actual))
        nominal[frame] = qpos
        previous = solution.x.copy()
    return nominal


def terrain_points(human: np.ndarray, contacts: np.ndarray) -> np.ndarray:
    """Interaction-mesh samples following all demonstrated support levels."""

    toe_ids = (
        DEMO_JOINTS.index("RightToeBase"),
        DEMO_JOINTS.index("LeftToeBase"),
    )
    anchors = [
        human[frame, toe_ids[side], [0, 2]]
        for frame in range(len(human))
        for side in range(2)
        if contacts[frame, side]
    ]
    anchors_array = np.asarray(anchors)
    selected = anchors_array[:: max(1, len(anchors_array) // 24)]
    points = []
    for x, z in selected:
        for y in (-0.75, -0.25, 0.25, 0.75):
            points.append((x, y, z))
    return np.asarray(points, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-urdf", type=Path, default=MODEL_URDF)
    parser.add_argument("--step-size", type=float, default=0.20)
    parser.add_argument("--foot-tolerance", type=float, default=0.02)
    parser.add_argument("--nominal-weight", type=float, default=5.0)
    args = parser.parse_args()

    with np.load(args.input) as archive:
        human = np.asarray(archive["global_joint_positions"], dtype=np.float64)
        contacts = np.asarray(archive["foot_contact"], dtype=bool)
        roll_mode = np.asarray(archive["foot_roll_mode"], dtype=np.int8)
        yaw = np.asarray(archive["target_yaw"], dtype=np.float64)
        fps = float(archive["fps"])
        scale = float(archive["scale"])
        metadata = {key: archive[key] for key in archive.files if key != "global_joint_positions"}

    constants = task_constants(args.model_urdf)
    # Track the complete root orientation as well as the lower-body seed.
    # Leaving quaternion indices unconstrained lets the interaction mesh rotate
    # the pelvis far beyond the demonstrated stair heading.
    constants.NOMINAL_TRACKING_INDICES = np.asarray(
        [2, 3, 4, 5, 6, *range(7, 21)],
        dtype=np.int64,
    )
    constants.FOOT_STICKING_LINKS = (
        "right_heel_sole",
        "right_sole",
        "right_toe_sole",
        "left_heel_sole",
        "left_sole",
        "left_toe_sole",
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
        foot_lock=FootLockConfig(enable=False),
        visualize=False,
        debug=False,
        w_nominal_tracking_init=args.nominal_weight,
        nominal_tracking_tau=1.0e6,
    )
    original_contact_jacobian = retargeter._calc_contact_jacobian_from_point

    def contact_jacobian_compat(self, body_idx, *jacobian_args, **jacobian_kwargs):
        del self
        scalar_body_idx = int(np.asarray(body_idx).reshape(-1)[0])
        return original_contact_jacobian(
            scalar_body_idx, *jacobian_args, **jacobian_kwargs
        )

    retargeter._calc_contact_jacobian_from_point = MethodType(
        contact_jacobian_compat, retargeter
    )
    human = preprocess_motion_data(
        human, retargeter, ["LeftToeBase", "RightToeBase"], scale=scale
    )
    nominal = kinematic_seed(retargeter, human, contacts, roll_mode, yaw)
    points = terrain_points(human, contacts)
    object_poses = nominal[:, -7:].copy()
    contact_sequence = [
        {"RightToeBase": bool(row[0]), "LeftToeBase": bool(row[1])}
        for row in contacts
    ]
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
    payload.update(metadata)
    payload["fps"] = np.float32(fps)
    payload["joint_names"] = np.asarray(ROBOT_JOINT_NAMES)
    payload["retargeter"] = np.asarray("OmniRetarget/Holosoma interaction-mesh SQP")
    payload["retargeter_commit"] = np.asarray(
        "79d728c2c40e7f9ce8c28f9d9cab117ed2350bb5"
    )
    np.savez_compressed(args.output, **payload)
    print(f"saved OmniRetarget clip: {args.output}")


if __name__ == "__main__":
    main()
