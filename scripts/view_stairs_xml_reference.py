#!/usr/bin/env python3
"""Play the accepted stairs.xml OmniRetarget reference in the exact scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RETARGET = ROOT / "multimotion/retargeting"
sys.path.insert(0, str(RETARGET))

from stairs_xml_scene import load_stairs_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / (
        "robot/reference_data/multimotion/stairs_xml_up_down_omniretarget.npz"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    model = load_stairs_scene(ROOT / "robot/MJCFmodelposition.xml",
                              ROOT / "robot/terrains/more_discolrl/stairs.xml")
    with np.load(args.reference, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=float)
        fps = float(archive["fps"])
    if qpos.shape != (len(qpos), model.nq):
        raise ValueError(f"Reference shape {qpos.shape} does not match nq={model.nq}")
    print(f"{len(qpos)} frames, {(len(qpos)-1)/fps:.2f} s, {fps:g} Hz")
    if args.check_only:
        return
    data = mujoco.MjData(model)
    period = 1.0 / (fps * args.speed)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [1.6, 0.0, .55]
        viewer.cam.distance = 4.2
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -12
        while viewer.is_running():
            start = time.monotonic()
            for pose in qpos:
                if not viewer.is_running():
                    return
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0.0, period - (time.monotonic() - start)))
                start = time.monotonic()
            if args.once:
                return


if __name__ == "__main__":
    main()
