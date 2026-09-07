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


EXPECTED_TEACHER_MOTION_NAMES = (
    "walk",
    "rough_walk",
    "slope_up",
    "slope_down",
)
EXPECTED_TEACHER_MOTION_FILES = (
    "walk_omniretarget.npz",
    "rough_walk_omniretarget.npz",
    "slope_up_omniretarget.npz",
    "slope_down_omniretarget.npz",
)


def _assert_teacher_checkpoint_matches_slope_distillation(
    argv: list[str],
) -> None:
    """Reject a Teacher checkpoint trained with the old stair clips."""

    checkpoint_value: str | None = None
    for index, arg in enumerate(argv):
        if arg == "--teacher-checkpoint":
            if index + 1 >= len(argv):
                raise SystemExit("--teacher-checkpoint requires a path.")
            checkpoint_value = argv[index + 1]
        elif arg.startswith("--teacher-checkpoint="):
            checkpoint_value = arg.split("=", 1)[1]
    if checkpoint_value is None:
        return

    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(
            "Student distillation Teacher checkpoint does not exist: "
            f"{checkpoint_path}"
        )
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    env_config = checkpoint.get("env_config", {})
    motion_names = tuple(env_config.get("motion_names", ()))
    motion_files = tuple(
        Path(value).name for value in env_config.get("motion_files", ())
    )
    motion_weights = tuple(
        float(value) for value in env_config.get("motion_weights", ())
    )
    motion_loop = tuple(bool(value) for value in env_config.get("motion_loop", ()))
    terrain_types = {
        str(value).lower() for value in env_config.get("terrain_types", ())
    }
    if (
        motion_names != EXPECTED_TEACHER_MOTION_NAMES
        or motion_files != EXPECTED_TEACHER_MOTION_FILES
        or motion_weights != (0.25, 0.25, 0.25, 0.25)
        or motion_loop != (True, False, False, False)
        or not {"rough", "slope_up", "slope_down"}.issubset(terrain_types)
        or "stairs" in terrain_types
    ):
        raise SystemExit(
            "Student distillation requires a slope-aware Teacher checkpoint "
            "with motion_names=(walk, rough_walk, slope_up, slope_down), "
            "equal weights, and no stairs; the supplied checkpoint records "
            f"motion_names={motion_names!r}, motion_files={motion_files!r}, "
            f"terrain_types={sorted(terrain_types)!r}."
        )


def _replace(args: list[str], option: str, values: list[str]) -> None:
    start = args.index(option)
    end = start + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    args[start:end] = [option, *values]


def _remove_flag(args: list[str], *options: str) -> None:
    args[:] = [arg for arg in args if arg not in options]


def default_train_args() -> list[str]:
    """Match the 100 Hz Teacher environment while hiding Teacher inputs."""

    args = teacher_args()
    _replace(args, "--training-task", ["multimotion_no_stairs_amp_student"])
    _replace(args, "--log-group", ["multimotion_no_stairs_amp_student"])
    _replace(args, "--reference-ctrl-blend", ["0.0"])
    _remove_flag(args, "--use-reference-velocity", "--no-use-reference-velocity")
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
            # Teacher's four-motion no-stairs observation. SensorHistoryEnv never
            # exposes them to the Student actor.
            "--sensor-wrapper-base-skill-conditioning",
            "--distillation-coef",
            "1.0",
            # The Teacher action is a residual around the retargeted control;
            # convert it to the Student's measured-tendon control space.
            "--distill-teacher-control-target",
            "--no-use-reference-velocity",
        ]
    )
    return args


def main() -> None:
    _assert_teacher_checkpoint_matches_slope_distillation(sys.argv[1:])
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
