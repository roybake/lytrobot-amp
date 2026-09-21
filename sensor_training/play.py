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
    # Real deployment always begins from the commanded standing state.  The
    # training-only 15% random-reference resets must never leak into playback.
    if not any(
        arg == "--stand-start-prob" or arg.startswith("--stand-start-prob=")
        for arg in sys.argv[1:]
    ):
        sys.argv.extend(["--stand-start-prob", "1.0"])
    base_play.LytRobotAmpEnv = SensorHistoryCpuAmpEnv
    base_play.main()


if __name__ == "__main__":
    main()
