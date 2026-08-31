#!/usr/bin/env python3
"""Train multimotion using only IMU and tendon-length actor observations."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_multimotion import default_train_args as base_multimotion_args
from lytrobot_amp.training_defaults import (
    ROUGH_TERRAIN_HEIGHT_M,
    SLOPE_ANGLE_DEG,
    TRAINING_TARGET_SPEED_MPS,
)


def _replace_option_values(
    args: list[str],
    option: str,
    values: list[str],
) -> None:
    if option not in args:
        args.extend([option, *values])
        return
    start = args.index(option)
    end = start + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    args[start:end] = [option, *values]


def default_train_args() -> list[str]:
    args = base_multimotion_args()
    _replace_option_values(
        args,
        "--rough-terrain-height",
        [f"{ROUGH_TERRAIN_HEIGHT_M:.2f}"],
    )
    _replace_option_values(
        args,
        "--target-velocity",
        [f"{TRAINING_TARGET_SPEED_MPS:.6f}"],
    )
    _replace_option_values(
        args,
        "--slope-angle-deg",
        [f"{SLOPE_ANGLE_DEG:.1f}"],
    )
    # The deployed control path must not receive the simulator-selected motion
    # reference.  With zero blend, actions remain 14 tendon target corrections
    # around measured tendon lengths and terrain adaptation comes from the GRU.
    _replace_option_values(args, "--reference-ctrl-blend", ["0.0"])
    _replace_option_values(args, "--motion-transition-blend-steps", ["0"])
    tag_index = args.index("--wandb-tags") + 1
    args[tag_index] = (
        "lytrobot,bpo,amp,multimotion,imu,tendon,recurrent,gru,implicit-terrain,"
        "asymmetric-critic,single-actor-mlp,sensor-only,"
        "0809-scene,no-stairs,more,serial-course,random-order,rough,slope-up,slope-down,"
        "sim2real"
    )
    # SensorHistoryEnv never exposes the skill one-hot, and persisting this as
    # false makes the checkpoint configuration accurately describe the actor.
    if "--skill-conditioning" in args:
        args[args.index("--skill-conditioning")] = "--no-skill-conditioning"
    elif "--no-skill-conditioning" not in args:
        args.append("--no-skill-conditioning")
    return args


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
