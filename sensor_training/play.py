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
from sensor_training.experiment_options import observation_config_from_metadata
from functools import partial


def main() -> None:
    # Stair trajectory tracking starts at its physical reference frame zero.
    # General sensor walking starts from the commanded standing state.
    checkpoint_arg = next(
        (sys.argv[index + 1] for index, value in enumerate(sys.argv[:-1])
         if value == "--checkpoint"),
        None,
    )
    stair_fixed_start = False
    episode_steps = 2400
    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg)
        if checkpoint_path.suffix.lower() == ".onnx":
            metadata = base_play.load_onnx_metadata(checkpoint_path)
        else:
            metadata = base_play.torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        env_config = metadata.get("env_config", {})
        stair_fixed_start = bool(env_config.get("stair_fixed_start", False))
        episode_steps = int(env_config.get("episode_steps", episode_steps))
    if not any(
        arg == "--stand-start-prob" or arg.startswith("--stand-start-prob=")
        for arg in sys.argv[1:]
    ):
        sys.argv.extend(["--stand-start-prob", "0.0" if stair_fixed_start else "1.0"])
    if stair_fixed_start and not any(
        arg == "--steps" or arg.startswith("--steps=") for arg in sys.argv[1:]
    ):
        sys.argv.extend(["--steps", str(episode_steps)])
    base_play.LytRobotAmpEnv = partial(SensorHistoryCpuAmpEnv,
        observation_cfg=observation_config_from_metadata(metadata if checkpoint_arg else {}))
    base_play.main()


if __name__ == "__main__":
    main()
