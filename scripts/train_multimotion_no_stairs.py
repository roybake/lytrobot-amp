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

from lytrobot_amp.training_defaults import (
    ROUGH_TERRAIN_HEIGHT_M,
    SLOPE_ANGLE_DEG,
    TRAINING_TARGET_SPEED_MPS,
)


# Keep most worlds on the deployable, full route.  Earlier runs started an
# 80% suffix curriculum at iteration 5k; the measured mean velocity then fell
# from about 0.30 m/s to zero and stage completion never recovered.  A late,
# low-probability suffix replay still gives every terrain direct samples while
# preventing those short resets from overwhelming entrance-to-exit walking.
COURSE_RANDOM_START_PROBABILITY = 0.25
COURSE_RANDOM_START_WARMUP_ITERATIONS = 18000
COURSE_RANDOM_START_RAMP_ITERATIONS = 8000


def default_stair_free_course_args() -> list[str]:
    """Return the shared walk/rough configuration required by the serial course."""

    return [
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
        "3.0",
        "--forward-progress-reward-coef",
        "2.0",
        "--forward-velocity-deficit-penalty-coef",
        "2.5",
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
        f"{COURSE_RANDOM_START_PROBABILITY:g}",
        "--terrain-course-random-start-warmup-iterations",
        str(COURSE_RANDOM_START_WARMUP_ITERATIONS),
        "--terrain-course-random-start-ramp-iterations",
        str(COURSE_RANDOM_START_RAMP_ITERATIONS),
        "--terrain-course-phase-sync",
        "--terrain-course-phase-lookahead",
        "0.10",
        "--terrain-course-stage-reward",
        "5.0",
        "--terrain-course-run-out",
        "3.10",
        "--rough-terrain-height",
        f"{ROUGH_TERRAIN_HEIGHT_M:g}",
        "--slope-angle-deg",
        f"{SLOPE_ANGLE_DEG:g}",
        "--motion-transition-blend-steps",
        "15",
        "--episode-steps",
        "1800",
    ]


def default_train_args() -> list[str]:
    """Add the stair-free task identity and rewards to the shared course."""

    return [
        "--training-task",
        "multimotion_no_stairs",
        *default_stair_free_course_args(),
        # Keep slip suppression, but do not let static stance/contact shaping
        # outscore sustained forward locomotion.
        "--stance-foot-reward-coef",
        "1.0",
        "--foot-slip-penalty-coef",
        "2.0",
        "--contact-reward-coef",
        "0.70",
        "--missed-contact-penalty-coef",
        "1.0",
        "--side-foot-pos-coef",
        "0.50",
        "--step-gap-coef",
        "0.50",
        "--swing-advance-coef",
        "0.75",
        "--symmetry-reward-coef",
        "0.50",
        "--specific-cot-penalty-coef",
        "0.0",
        "--task-reward-lerp",
        "0.65",
        "--min-action-std",
        "0.08",
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
