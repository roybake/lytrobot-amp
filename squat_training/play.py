#!/usr/bin/env python3
"""Play a squat sensor-policy checkpoint in CPU MuJoCo."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import play as base_play
from squat_training.envs import SquatSensorHistoryCpuAmpEnv


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


def main() -> None:
    defaults = {
        "--motion-file": str(REPO_ROOT / "squat_training" / "reference_data" / "squat_cmu_22_14_omniretarget.npz"),
        "--target-velocity": "0.0",
        "--stand-start-prob": "0.0",
        "--record-dir": str(REPO_ROOT / "recording" / "squat_sensor_policy"),
    }
    for name, value in defaults.items():
        if not _has_arg(name):
            sys.argv.extend([name, value])
    base_play.LytRobotAmpEnv = SquatSensorHistoryCpuAmpEnv
    base_play.main(sensor_observations=True)


if __name__ == "__main__":
    main()

