#!/usr/bin/env python3
"""Canonical sensor-only, terrain-aligned no-stairs AMP training task."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
ORDINARY_REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
SERIAL_TERRAIN_MODEL_XML = REPO_ROOT / "robot" / "MJCFmodelposition_0806.xml"
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lytrobot_amp.training_defaults import (
    ROUGH_TERRAIN_HEIGHT_M,
    SLOPE_ANGLE_DEG,
    TRAINING_TARGET_SPEED_MPS,
)


# Keep a substantial full-route stream while exposing later terrain early.
# At the final 0.70 random-start probability, stage zero still receives 30%
# of resets and each non-zero terrain receives roughly 23% across balanced
# course variants. Full-route success is counted only for stage-zero episodes.
COURSE_RANDOM_START_PROBABILITY = 0.70
COURSE_RANDOM_START_WARMUP_ITERATIONS = 750
COURSE_RANDOM_START_RAMP_ITERATIONS = 4000
COURSE_VARIANTS = 6


def default_no_stairs_course_args() -> list[str]:
    """Return the one canonical no-stairs sensor course contract.

    Stage launchers prepend this exact list and only append stage/runtime
    overrides. This is deliberately complete so parser tests can compare the
    effective Stage 1 and Stage 2 configurations without hand-copying fields.
    """

    return [
        "--model-xml", str(SERIAL_TERRAIN_MODEL_XML),
        "--motion-files",
        str(ORDINARY_REFERENCE_DIR / "walk_omniretarget.npz"),
        str(ORDINARY_REFERENCE_DIR / "rough_walk_omniretarget.npz"),
        str(ORDINARY_REFERENCE_DIR / "slope_up_omniretarget.npz"),
        str(ORDINARY_REFERENCE_DIR / "slope_down_omniretarget.npz"),
        "--motion-names", "walk", "rough_walk", "slope_up", "slope_down",
        "--motion-weights", "0.25", "0.25", "0.25", "0.25",
        "--motion-loop", "true", "false", "false", "false",
        "--no-skill-conditioning", "--no-use-reference-velocity",
        "--reference-ctrl-blend", "0.0",
        "--amp", "--amp-task-only",
        "--target-velocity", f"{TRAINING_TARGET_SPEED_MPS:g}",
        "--forward-velocity-tracking-sigma", "0.35",
        "--forward-velocity-reward-coef", "3.0",
        "--forward-progress-reward-coef", "1.75",
        "--forward-velocity-deficit-penalty-coef", "3.0",
        "--forward-velocity-excess-penalty-coef", "2.0",
        "--min-forward-progress-speed", "0.10",
        "--forward-progress-grace-steps", "100",
        "--forward-progress-window-steps", "100",
        "--forward-progress-termination-penalty", "4.0",
        "--fall-penalty", "10.0", "--terminate-incomplete-motion-end",
        "--terrain-types", "rough", "slope_up", "slope_down",
        "--no-stair-terrain", "--serial-terrain-course",
        "--no-serial-course-single-motion",
        "--terrain-course-variants", str(COURSE_VARIANTS),
        "--legacy-0806-terrain-model", "--terrain-course-random-assignment",
        "--terrain-course-random-start-prob", f"{COURSE_RANDOM_START_PROBABILITY:g}",
        "--terrain-course-random-start-warmup-iterations",
        str(COURSE_RANDOM_START_WARMUP_ITERATIONS),
        "--terrain-course-random-start-ramp-iterations",
        str(COURSE_RANDOM_START_RAMP_ITERATIONS),
        "--terrain-course-phase-sync", "--terrain-course-phase-lookahead", "0.10",
        # Boundary crossing is deliberately small; delayed survival pays the
        # meaningful bonus after the policy has stayed on the new terrain.
        "--terrain-course-stage-reward", "0.50",
        "--terrain-course-transition-survival-steps", "75",
        "--terrain-course-transition-success-reward", "2.0",
        "--terrain-course-run-out", "0.30",
        "--rough-terrain-height", f"{ROUGH_TERRAIN_HEIGHT_M:g}",
        "--slope-angle-deg", f"{SLOPE_ANGLE_DEG:g}",
        "--motion-transition-blend-steps", "0",
        "--serial-reference-state-init", "--motion-reset-prob", "0.50",
        "--stand-start-prob", "0.0", "--startup-action-ramp-steps", "0",
        "--foot-slip-velocity-threshold", "0.08",
        "--foot-slip-penalty-coef", "0.35",
        "--contact-reward-coef", "0.35",
        "--alternating-step-reward-coef", "1.50",
        "--terrain-penetration-tolerance", "0.01",
        "--terrain-penetration-penalty-coef", "20.0",
        "--task-reward-lerp", "0.70", "--amp-reward-coef", "1.5",
        "--min-action-std", "0.05",
        "--num-envs", "4096", "--num-steps", "24", "--epochs", "4",
        "--minibatches", "8", "--amp-minibatch-size", "4096",
        "--amp-disc-updates", "2", "--episode-metric-window", "4096",
        "--warp-nconmax", "128", "--warp-njmax", "512",
        "--episode-steps", "3600",
        "--log-root", str(REPO_ROOT / "logs" / "multimotion" / "no_stairs_amp_single"),
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,conditional-amp,no-stairs,serial-course,random-order,"
            "flat,rough,slope-up,slope-down,velocity-tracking,imu,tendon,recurrent,"
            "gru,sensor-only"
        ),
    ]


def default_ordinary_stair_free_course_args() -> list[str]:
    """Backward-compatible alias used by generic two-stage wrappers."""

    return default_no_stairs_course_args()


def default_stair_free_course_args() -> list[str]:
    """Backward-compatible alias for the canonical course."""

    return default_no_stairs_course_args()


def default_train_args() -> list[str]:
    """Return the canonical no-stairs task identity and course defaults."""

    return [
        "--training-task", "multimotion_no_stairs_amp_single",
        *default_no_stairs_course_args(),
    ]


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
