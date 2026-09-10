#!/usr/bin/env python3
"""Visualize the retargeted squat reference directly in MuJoCo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lytrobot_amp.motion import MotionReference

DEFAULT_MOTION = REPO_ROOT / "squat_training" / "reference_data" / "squat_cmu_22_14_omniretarget.npz"
DEFAULT_MODEL = REPO_ROOT / "robot" / "MJCFmodelposition.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of cycles to play; 0 keeps looping until the window closes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive.")

    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(model)
    motion = MotionReference(args.motion_file, model, zero_root_xy=False)
    qpos = np.asarray(motion.reference_qpos, dtype=np.float64)
    qvel = np.asarray(motion.reference_qvel, dtype=np.float64)
    frequency = motion.frequency

    frame_period = 1.0 / (frequency * args.speed)
    frame = 0
    completed_cycles = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.04, 0.0, 0.46)
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -8.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        deadline = time.monotonic()
        while viewer.is_running() and (args.cycles == 0 or completed_cycles < args.cycles):
            data.qpos[:] = qpos[frame]
            data.qvel[:] = qvel[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()

            frame += 1
            if frame == len(qpos):
                frame = 0
                completed_cycles += 1

            deadline += frame_period
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()


if __name__ == "__main__":
    main()
