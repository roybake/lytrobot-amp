#!/usr/bin/env python3
"""Load and visualize the local LYT robot MJCF with MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


ROBOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = ROBOT_DIR / "MJCFmodelposition.xml"
DEFAULT_REFERENCE = ROBOT_DIR / "reference_data" / "walk1.npz"
COMMON_JOINTS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize robot/MJCFmodelposition.xml in local MuJoCo.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--headless", action="store_true", help="Load and step the model without opening a viewer.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--pose", choices=("reference", "default"), default="reference")
    return parser.parse_args()


def set_reference_pose(model: mujoco.MjModel, data: mujoco.MjData, reference_path: Path) -> None:
    if not reference_path.exists():
        print(f"[WARN] reference not found, using MJCF default pose: {reference_path}")
        return

    archive = np.load(reference_path, allow_pickle=True)
    source_qpos = np.asarray(archive["qpos"], dtype=np.float64)
    source_qvel = np.asarray(archive["qvel"], dtype=np.float64)
    source_joint_names = [str(name) for name in archive["joint_names"]]
    source_joint_to_index = {name: i for i, name in enumerate(source_joint_names)}

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[:7] = source_qpos[0, :7]
    data.qpos[0:2] = 0.0
    data.qvel[:6] = source_qvel[0, :6]

    if data.qpos[3] < 0.0:
        data.qpos[3:7] *= -1.0

    for name in COMMON_JOINTS:
        source_index = source_joint_to_index.get(name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if source_index is None or joint_id < 0:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr] = source_qpos[0, 7 + source_index - 1]
        data.qvel[qvel_adr] = source_qvel[0, 6 + source_index - 1]

    mujoco.mj_forward(model, data)


def set_stable_tendon_controls(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_forward(model, data)
    ctrl_low = model.actuator_ctrlrange[:, 0]
    ctrl_high = model.actuator_ctrlrange[:, 1]
    tendon_ids = model.actuator_trnid[:, 0].astype(np.int32)
    data.ctrl[:] = np.clip(data.ten_length[tendon_ids], ctrl_low, ctrl_high)


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError("MuJoCo state became non-finite during headless model test.")
    print(
        "[MODEL] "
        f"loaded={model.nq=} {model.nv=} {model.nu=} {model.nbody=} "
        f"steps={steps} root_pos={np.round(data.qpos[:3], 4).tolist()}"
    )


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(steps):
            if not viewer.is_running():
                break
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if args.pose == "reference":
        set_reference_pose(model, data, reference_path)
    else:
        mujoco.mj_forward(model, data)
    set_stable_tendon_controls(model, data)

    if args.headless:
        run_headless(model, data, args.steps)
    else:
        run_viewer(model, data, args.steps)


if __name__ == "__main__":
    main()

