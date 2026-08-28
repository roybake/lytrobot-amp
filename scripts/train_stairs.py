#!/usr/bin/env python3
"""Train one sensor policy on a continuous ground-up-down-ground stair route."""

from __future__ import annotations

import sys
from pathlib import Path

from lytrobot_amp.training_defaults import (
    STAIR_STAGE1_STEP_DEPTH_M,
    STAIR_STAGE1_STEP_HEIGHT_M,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
WALK_REFERENCE = REPO_ROOT / "robot" / "reference_data" / "walk_cmu_69_01_omniretarget.npz"
# Preserve the entrance gait before exposing the policy to isolated stair
# resets.  The former 70%/4k schedule made stair failures dominate the batch
# before flat walking was stable and the learned gait collapsed after one step.
COURSE_RANDOM_START_PROBABILITY = 0.35
COURSE_RANDOM_START_WARMUP_ITERATIONS = 15000
COURSE_RANDOM_START_RAMP_ITERATIONS = 7500
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def default_train_args() -> list[str]:
    """Return self-contained defaults for the dedicated up/down stair task."""

    return [
        "--training-task",
        "stairs_up_down",
        "--motion-files",
        str(WALK_REFERENCE),
        str(REFERENCE_DIR / "stair_up_omniretarget.npz"),
        str(REFERENCE_DIR / "stair_down_omniretarget.npz"),
        "--motion-names",
        "walk",
        "stair_up",
        "stair_down",
        "--motion-weights",
        "0.50",
        "0.25",
        "0.25",
        "--motion-loop",
        "true",
        "false",
        "false",
        "--no-skill-conditioning",
        "--num-envs",
        "4096",
        "--num-steps",
        "24",
        "--epochs",
        "4",
        "--minibatches",
        "8",
        "--amp-minibatch-size",
        "4096",
        "--amp-disc-updates",
        "2",
        "--episode-metric-window",
        "4096",
        "--warp-nconmax",
        "128",
        "--warp-njmax",
        "512",
        "--use-reference-velocity",
        "--reward-mode",
        "stairs",
        "--target-velocity",
        "0.55",
        "--forward-velocity-tracking-sigma",
        "0.20",
        "--forward-velocity-reward-coef",
        "1.50",
        "--forward-progress-reward-coef",
        "0.75",
        "--forward-velocity-deficit-penalty-coef",
        "1.0",
        "--forward-velocity-excess-penalty-coef",
        "1.0",
        "--min-forward-progress-speed",
        "0.05",
        "--forward-progress-grace-steps",
        "50",
        "--forward-progress-window-steps",
        "50",
        "--forward-progress-termination-penalty",
        "2.0",
        "--fall-penalty",
        "10.0",
        "--stance-foot-reward-coef",
        "1.0",
        "--foot-slip-penalty-coef",
        "0.5",
        "--contact-reward-coef",
        "0.7",
        "--missed-contact-penalty-coef",
        "0.8",
        "--stair-vertical-velocity-sigma",
        "0.25",
        "--stair-vertical-velocity-reward-coef",
        "1.0",
        "--stair-swing-clearance",
        "0.08",
        "--stair-swing-clearance-reward-coef",
        "0.75",
        "--stair-tread-support-reward-coef",
        "1.25",
        "--stair-min-root-clearance",
        "0.60",
        "--terrain-types",
        "stairs",
        "--serial-terrain-course",
        "--terrain-course-variants",
        "1",
        "--no-terrain-course-random-assignment",
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
        "--terrain-course-run-in",
        "3.10",
        "--terrain-course-run-out",
        "3.10",
        "--stair-step-height",
        f"{STAIR_STAGE1_STEP_HEIGHT_M:g}",
        "--stair-step-depth",
        f"{STAIR_STAGE1_STEP_DEPTH_M:g}",
        "--stair-step-count",
        "3",
        "--stair-start-x",
        "-0.95",
        "--stair-down-offset",
        "0.25",
        "--specific-cot-penalty-coef",
        "0.0",
        "--task-reward-lerp",
        "0.50",
        "--min-action-std",
        "0.05",
        "--motion-reset-prob",
        "1.0",
        "--terminate-incomplete-motion-end",
        "--reference-ctrl-blend",
        "0.0",
        "--motion-transition-blend-steps",
        "30",
        "--startup-action-ramp-steps",
        "0",
        "--stand-start-prob",
        "0.0",
        "--symmetry-reward-coef",
        "0.0",
        "--episode-steps",
        "2000",
        "--episode-timeout-jitter-steps",
        "0",
        "--log-group",
        "stairs_up_down",
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,stairs,walk-in,stair-up,stair-down,walk-out,continuous,"
            "terrain-aware-reward,imu,tendon,recurrent,gru,sensor-only,"
            "asymmetric-critic"
        ),
    ]


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
