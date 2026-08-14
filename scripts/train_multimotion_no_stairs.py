#!/usr/bin/env python3
"""Train ordinary sensor walking over a randomized stair-free serial course."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lytrobot_amp.benchmark import (
    MAX_ROUGH_HEIGHT_DIFFERENCE_M,
    MAX_SLOPE_DEG,
    TRAINING_TARGET_SPEED_MPS,
)


def default_train_args() -> list[str]:
    """Add only the terrain-course requirements to ordinary walking defaults."""

    return [
        "--training-task",
        "multimotion_no_stairs",
        "--motion-files",
        str(REFERENCE_DIR / "walk_omniretarget.npz"),
        str(REFERENCE_DIR / "rough_walk_omniretarget.npz"),
        "--motion-names",
        "walk",
        "rough_walk",
        "--motion-weights",
        "0.50",
        "0.50",
        "--motion-loop",
        "true",
        "false",
        "--no-skill-conditioning",
        "--reference-ctrl-blend",
        "0.0",
        "--target-velocity",
        f"{TRAINING_TARGET_SPEED_MPS:g}",
        "--forward-velocity-tracking-sigma",
        "0.20",
        "--forward-velocity-reward-coef",
        "2.0",
        "--forward-progress-reward-coef",
        "1.0",
        "--forward-velocity-deficit-penalty-coef",
        "1.5",
        "--forward-velocity-excess-penalty-coef",
        "3.0",
        "--min-forward-progress-speed",
        "0.10",
        "--fall-penalty",
        "10.0",
        "--terminate-incomplete-motion-end",
        "--terrain-types",
        "flat",
        "rough",
        "slope_up",
        "slope_down",
        "--serial-terrain-course",
        # flat、rough 与成对上下坡构成 3 个随机块，共 6 种唯一顺序。
        "--terrain-course-variants",
        "6",
        "--terrain-course-random-assignment",
        "--terrain-course-random-start-prob",
        "0.0",
        "--terrain-course-stage-reward",
        "5.0",
        "--terrain-course-run-out",
        "3.10",
        "--rough-terrain-height",
        f"{MAX_ROUGH_HEIGHT_DIFFERENCE_M:g}",
        "--slope-angle-deg",
        f"{MAX_SLOPE_DEG:g}",
        "--motion-transition-blend-steps",
        "15",
        "--episode-steps",
        "1800",
        "--log-root",
        str(REPO_ROOT / "logs" / "multimotion" / "no_stairs"),
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,walk,no-stairs,serial-course,random-order,"
            "flat,rough,slope-up,slope-down,imu,tendon,recurrent,gru,sensor-only"
        ),
    ]


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
