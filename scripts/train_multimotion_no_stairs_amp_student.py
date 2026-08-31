#!/usr/bin/env python3
"""Distil a frozen privileged Teacher into the IMU+tendon Student Actor."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_multimotion_no_stairs_amp_teacher import (
    default_train_args as teacher_args,
)


def _replace(args: list[str], option: str, values: list[str]) -> None:
    start = args.index(option)
    end = start + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    args[start:end] = [option, *values]


def default_train_args() -> list[str]:
    """Match the 100 Hz Teacher environment while hiding Teacher inputs."""

    args = teacher_args()
    _replace(args, "--training-task", ["multimotion_no_stairs_amp_student"])
    _replace(args, "--log-group", ["multimotion_no_stairs_amp_student"])
    _replace(args, "--reference-ctrl-blend", ["0.0"])
    _replace(
        args,
        "--wandb-tags",
        [
            "lytrobot,bpo,amp,multimotion,no-stairs,amp-student,0806,100hz,"
            "teacher-distillation,imu,tendon,recurrent,gru,sensor-only,"
            "rough,slope-up,slope-down"
        ],
    )
    args.remove("--legacy-reference-conditioned-actor")
    args.extend(
        [
            # The simulator retains labels solely to reconstruct the frozen
            # Teacher's four-motion 08-06 observation. SensorHistoryEnv never
            # exposes them to the Student actor.
            "--sensor-wrapper-base-skill-conditioning",
            "--distillation-coef",
            "1.0",
            # The Teacher action is a residual around the retargeted control;
            # convert it to the Student's measured-tendon control space.
            "--distill-teacher-control-target",
        ]
    )
    return args


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
