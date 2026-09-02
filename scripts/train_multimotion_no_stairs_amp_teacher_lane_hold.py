#!/usr/bin/env python3
"""Train a lane-stabilized privileged Teacher for the stair-free course.

This launcher deliberately has a new task identity.  Historical
``multimotion_no_stairs_amp_teacher`` checkpoints keep their original
observation contract and are not silently resumed with the two new lane
features.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_multimotion_no_stairs_amp_teacher import (
    _assert_reference_model_override_is_compatible,
    default_train_args as _historical_teacher_args,
)


TASK_NAME = "multimotion_no_stairs_amp_teacher_lane_hold"


def _replace(args: list[str], option: str, values: list[str]) -> None:
    start = args.index(option)
    end = start + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    args[start:end] = [option, *values]


def _remove(args: list[str], option: str) -> None:
    while option in args:
        args.remove(option)


def default_train_args() -> list[str]:
    """Return the historical Teacher setup with lane stabilization enabled."""

    args = _historical_teacher_args()
    _replace(args, "--training-task", [TASK_NAME])
    _replace(args, "--log-group", [TASK_NAME])
    _replace(
        args,
        "--wandb-tags",
        [
            "lytrobot,bpo,amp,multimotion,no-stairs,amp-teacher,0806,100hz,"
            "privileged,reference-conditioned,lane-hold,rough,slope-up,slope-down"
        ],
    )
    _replace(args, "--motion-reset-prob", ["0.50"])
    _replace(args, "--action-residual-scale", ["0.30"])
    _remove(args, "--no-serial-reference-state-init")
    _remove(args, "--serial-reference-state-init")
    args.append("--serial-reference-state-init")
    args.extend(
        [
            "--course-lane-hold",
            "--course-lane-position-reward-coef",
            "2.0",
            "--course-lane-velocity-reward-coef",
            "0.5",
            "--course-lane-position-sigma",
            "0.35",
            "--course-lane-velocity-sigma",
            "0.25",
            "--course-lane-termination-margin",
            "0.35",
        ]
    )
    return args


def main() -> None:
    import scripts.train as runner
    from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
    from lytrobot_amp.ppo_amp import PPOAMPConfig, PPOAMPTrainer

    _assert_reference_model_override_is_compatible(sys.argv[1:])
    runner.LytRobotWarpAmpEnv = LytRobotWarpAmpEnv
    runner.PPOAMPConfig = PPOAMPConfig
    runner.PPOAMPTrainer = PPOAMPTrainer
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    runner.main()


if __name__ == "__main__":
    main()
