#!/usr/bin/env python3
"""Play a sensor_training checkpoint with the CPU MuJoCo environment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import play as base_play

from sensor_training.envs import SensorHistoryCpuAmpEnv


def main() -> None:
    base_play.LytRobotAmpEnv = SensorHistoryCpuAmpEnv
    base_play.main(sensor_observations=True)


if __name__ == "__main__":
    main()
