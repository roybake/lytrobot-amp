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
DEFAULT_REFERENCE = ROBOT_DIR / "reference_data" / "walk1_mjcf_repaired.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize robot/MJCFmodelposition.xml in local MuJoCo.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--headless", action="store_true", help="Load and step the model without opening a viewer.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--pose", choices=("reference", "default"), default="reference")
    return parser.parse_args()


def set_reference_pose(model: mujoco.MjModel, data: mujoco.MjData, reference_path: Path) -> None:
    """Load an actual model state; reject joint-order or dimension mismatches.

    The original walk1.npz is retained as source data.  The repaired archive
    carries the full MJCF joint layout, including passive patella joints, and
    has been fitted to the robot's mesh-based sole contact constraints.
    """
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference not found: {reference_path}")

    with np.load(reference_path, allow_pickle=False) as archive:
        source_qpos = np.asarray(archive["qpos"], dtype=np.float64)
        source_qvel = np.asarray(archive["qvel"], dtype=np.float64)
        source_names = [str(name) for name in archive["joint_names"]]

    expected_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
    ]
    if source_names != expected_names:
        raise ValueError(
            "Reference joint names/order do not match the MJCF model; "
            "use walk1_mjcf_repaired.npz or retarget the source first."
        )
    if source_qpos.ndim != 2 or source_qpos.shape[1] != model.nq:
        raise ValueError(f"Reference qpos must have shape (frames, {model.nq}).")
    if source_qvel.shape != (len(source_qpos), model.nv):
        raise ValueError(f"Reference qvel must have shape (frames, {model.nv}).")
    if not len(source_qpos) or not (
        np.isfinite(source_qpos).all() and np.isfinite(source_qvel).all()
    ):
        raise ValueError("Reference contains no frames or non-finite states.")

    data.qpos[:] = source_qpos[0]
    data.qvel[:] = source_qvel[0]
    data.qpos[:2] = 0.0
    mujoco.mj_forward(model, data)

    # Require physical ground support when loading the repaired reference.
    # This checks actual mesh contacts, not foot-body center heights.
    if reference_path.name == DEFAULT_REFERENCE.name:
        floor = model.geom("floor").id
        feet = {model.geom(name).id for name in ("foot_r", "toe_r", "foot_l", "toe_l")}
        signed = [
            float(contact.dist)
            for contact in data.contact
            if (contact.geom1 == floor and contact.geom2 in feet)
            or (contact.geom2 == floor and contact.geom1 in feet)
        ]
        if not signed or min(signed) < -0.0005 or min(signed) > 0.002:
            raise ValueError(
                f"Repaired reference has invalid initial ground contact: {signed}."
            )


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

