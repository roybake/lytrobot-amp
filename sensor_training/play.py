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
    # Stair trajectory tracking starts at its physical reference frame zero.
    # General sensor walking starts from the commanded standing state.
    if not any(
        arg == "--stand-start-prob" or arg.startswith("--stand-start-prob=")
        for arg in sys.argv[1:]
    ):
        checkpoint_arg = next(
            (sys.argv[index + 1] for index, value in enumerate(sys.argv[:-1])
             if value == "--checkpoint"),
            None,
        )
        stair_fixed_start = False
        if checkpoint_arg:
            checkpoint_path = Path(checkpoint_arg)
            if checkpoint_path.suffix.lower() == ".onnx":
                metadata = base_play.load_onnx_metadata(checkpoint_path)
            else:
                metadata = base_play.torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            stair_fixed_start = bool(metadata.get("env_config", {}).get("stair_fixed_start", False))
        sys.argv.extend(["--stand-start-prob", "0.0" if stair_fixed_start else "1.0"])
    base_play.LytRobotAmpEnv = SensorHistoryCpuAmpEnv
    base_play.main()


if __name__ == "__main__":
    main()
