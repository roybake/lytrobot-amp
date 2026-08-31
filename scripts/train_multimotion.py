#!/usr/bin/env python3
"""Train the 08-09 visual, stair-free multimotion serial terrain course."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lytrobot_amp.training_defaults import (
    ROUGH_TERRAIN_HEIGHT_M,
    SLOPE_ANGLE_DEG,
    TRAINING_TARGET_SPEED_MPS,
)

REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
SERIAL_TERRAIN_MODEL_XML = REPO_ROOT / "robot" / "MJCFmodelposition_0806.xml"
PARALLEL_ENVS = 4096
ROLLOUT_STEPS = 24
PPO_MINIBATCHES = 8
AMP_MINIBATCH_SIZE = 4096
AMP_DISCRIMINATOR_UPDATES = 2
COURSE_RANDOM_START_WARMUP_ITERATIONS = 5000
COURSE_RANDOM_START_RAMP_ITERATIONS = 10000


def default_train_args() -> list[str]:
    return [
        "--training-task",
        "multimotion_0809_no_stairs",
        # The 08-09 replay scene has the near-black checker floor and bright
        # white terrain fixtures seen in the reference recording.  It is kept
        # separate from the default robot scene so unrelated tasks retain
        # their own appearance.
        "--model-xml",
        str(SERIAL_TERRAIN_MODEL_XML),
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
        # 4096 worlds x 24 steps = 98,304 transitions per rollout.  Eight PPO
        # minibatches keep each optimizer batch at 12,288 samples instead of
        # the overly coarse 24,576 produced by the generic four-way default.
        "--num-envs",
        str(PARALLEL_ENVS),
        "--num-steps",
        str(ROLLOUT_STEPS),
        "--epochs",
        "4",
        "--minibatches",
        str(PPO_MINIBATCHES),
        # Draw enough AMP examples to cover both terrain-aligned skills on
        # every update.
        "--amp-minibatch-size",
        str(AMP_MINIBATCH_SIZE),
        "--amp-disc-updates",
        str(AMP_DISCRIMINATOR_UPDATES),
        "--episode-metric-window",
        str(PARALLEL_ENVS),
        # Full-body terrain collision needs enough room for simultaneous body,
        # foot, and toe contacts after a fall or on complex terrain.
        "--warp-nconmax",
        "128",
        "--warp-njmax",
        "512",
        "--use-reference-velocity",
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
        "--forward-progress-grace-steps",
        "100",
        "--forward-progress-window-steps",
        "100",
        "--forward-progress-termination-penalty",
        "2.0",
        "--fall-penalty",
        "10.0",
        "--terminate-incomplete-motion-end",
        "--stance-foot-reward-coef",
        "2.0",
        "--foot-slip-penalty-coef",
        "2.0",
        "--contact-reward-coef",
        "1.0",
        "--missed-contact-penalty-coef",
        "1.2",
        "--side-foot-pos-coef",
        "0.50",
        "--step-gap-coef",
        "0.50",
        "--swing-advance-coef",
        "0.75",
        "--symmetry-reward-coef",
        "0.50",
        "--terrain-types",
        "rough",
        "slope_up",
        "slope_down",
        "--serial-terrain-course",
        "--terrain-course-variants",
        "6",
        "--terrain-course-run-in",
        "2.90",
        "--terrain-course-run-out",
        "0.30",
        "--terrain-course-base-height",
        "0.30",
        "--terrain-course-random-start-prob",
        "0.80",
        "--terrain-course-random-start-warmup-iterations",
        str(COURSE_RANDOM_START_WARMUP_ITERATIONS),
        "--terrain-course-random-start-ramp-iterations",
        str(COURSE_RANDOM_START_RAMP_ITERATIONS),
        "--terrain-course-phase-sync",
        "--terrain-course-phase-lookahead",
        "0.10",
        "--terrain-course-stage-reward",
        "5.0",
        "--rough-terrain-height",
        f"{ROUGH_TERRAIN_HEIGHT_M:g}",
        "--slope-angle-deg",
        f"{SLOPE_ANGLE_DEG:g}",
        "--task-reward-lerp",
        "0.60",
        "--min-action-std",
        "0.08",
        "--motion-reset-prob",
        "1.0",
        "--motion-transition-blend-steps",
        "30",
        "--startup-action-ramp-steps",
        "0",
        "--stand-start-prob",
        "0.0",
        "--episode-steps",
        "3600",
        "--log-group",
        "multimotion_0809_no_stairs",
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,multimotion,0809-scene,no-stairs,more,"
            "serial-course,random-order,rough,slope-up,slope-down"
        ),
    ]


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
