#!/usr/bin/env python3
"""Train one AMP+RL policy across six straight MoRE terrain actions."""

from __future__ import annotations

import sys
from pathlib import Path

from lytrobot_amp.benchmark import (
    MAX_ROUGH_HEIGHT_DIFFERENCE_M,
    MAX_SLOPE_DEG,
    MAX_SPECIFIC_COT_J_PER_KG_M,
    STAIR_HEIGHT_RANGE_M,
    STAIR_TREAD_RANGE_M,
    TRAINING_TARGET_SPEED_MPS,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
PARALLEL_ENVS = 4096
ROLLOUT_STEPS = 24
PPO_MINIBATCHES = 8
AMP_MINIBATCH_SIZE = 4096
AMP_DISCRIMINATOR_UPDATES = 2
COURSE_RANDOM_START_WARMUP_ITERATIONS = 5000
COURSE_RANDOM_START_RAMP_ITERATIONS = 10000


def default_train_args() -> list[str]:
    return [
        "--motion-files",
        str(REFERENCE_DIR / "walk_omniretarget.npz"),
        str(REFERENCE_DIR / "rough_walk_omniretarget.npz"),
        str(REFERENCE_DIR / "pit_walk_omniretarget.npz"),
        str(REFERENCE_DIR / "gap_cross_omniretarget.npz"),
        str(REFERENCE_DIR / "stair_up_omniretarget.npz"),
        str(REFERENCE_DIR / "stair_down_omniretarget.npz"),
        "--motion-names",
        "walk",
        "rough_walk",
        "pit_walk",
        "gap_cross",
        "stair_up",
        "stair_down",
        "--motion-weights",
        "0.20",
        "0.16",
        "0.16",
        "0.16",
        "0.16",
        "0.16",
        "--motion-loop",
        "true",
        "false",
        "false",
        "false",
        "false",
        "false",
        "--skill-conditioning",
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
        # Draw enough AMP examples to cover all six skills on every update.
        "--amp-minibatch-size",
        str(AMP_MINIBATCH_SIZE),
        "--amp-disc-updates",
        str(AMP_DISCRIMINATOR_UPDATES),
        "--episode-metric-window",
        str(PARALLEL_ENVS),
        # These tested per-world Warp capacities remain well within 12 GiB at
        # 4096 environments while covering the complex-terrain contacts.
        "--warp-nconmax",
        "32",
        "--warp-njmax",
        "128",
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
        "50",
        "--forward-progress-window-steps",
        "50",
        "--forward-progress-termination-penalty",
        "2.0",
        "--fall-penalty",
        "10.0",
        "--terminate-incomplete-motion-end",
        "--side-foot-pos-coef",
        "0.50",
        "--step-gap-coef",
        "0.50",
        "--swing-advance-coef",
        "0.75",
        "--symmetry-reward-coef",
        "0.50",
        "--terrain-types",
        "gap",
        "pit",
        "rough",
        "slope_up",
        "slope_down",
        "stairs",
        "--serial-terrain-course",
        "--terrain-course-variants",
        "8",
        "--terrain-course-random-start-prob",
        "0.80",
        "--terrain-course-random-start-warmup-iterations",
        str(COURSE_RANDOM_START_WARMUP_ITERATIONS),
        "--terrain-course-random-start-ramp-iterations",
        str(COURSE_RANDOM_START_RAMP_ITERATIONS),
        "--terrain-course-stage-reward",
        "5.0",
        "--gap-terrain-width",
        "0.20",
        "--rough-terrain-height",
        f"{MAX_ROUGH_HEIGHT_DIFFERENCE_M:g}",
        "--slope-angle-deg",
        f"{MAX_SLOPE_DEG:g}",
        "--stair-step-height",
        f"{STAIR_HEIGHT_RANGE_M[1]:g}",
        "--stair-step-depth",
        f"{STAIR_TREAD_RANGE_M[0]:g}",
        "--stair-step-count",
        "3",
        "--stair-start-x",
        "-0.95",
        "--stair-down-offset",
        "0.25",
        "--specific-cot-limit",
        f"{MAX_SPECIFIC_COT_J_PER_KG_M:g}",
        "--specific-cot-penalty-coef",
        "2.0",
        "--motion-reset-prob",
        "1.0",
        "--motion-transition-blend-steps",
        "15",
        "--startup-action-ramp-steps",
        "0",
        "--stand-start-prob",
        "0.0",
        "--episode-steps",
        "1800",
        "--log-group",
        "multimotion",
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,multimotion,more,serial-course,random-order,"
            "flat,gap,pit,rough,slope-up,slope-down,stairs,acceptance"
        ),
    ]


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
