#!/usr/bin/env python3
"""Train an independent multi-motion policy without staircase geometry/clips."""

from __future__ import annotations

import sys
from pathlib import Path

from lytrobot_amp.benchmark import (
    MAX_ROUGH_HEIGHT_DIFFERENCE_M,
    MAX_SLOPE_DEG,
    MAX_SPECIFIC_COT_J_PER_KG_M,
    TRAINING_TARGET_SPEED_MPS,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def default_train_args() -> list[str]:
    """Return self-contained defaults for the stair-free multi-motion task."""

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
        "32",
        "--warp-njmax",
        "128",
        # A fixed invisible command is learnable by the sensor-only actor and
        # avoids chasing the clip's 0.62--1.10 m/s within-cycle speed ripple.
        "--no-use-reference-velocity",
        "--target-velocity",
        f"{TRAINING_TARGET_SPEED_MPS:g}",
        "--forward-velocity-tracking-sigma",
        "0.35",
        "--forward-velocity-reward-coef",
        "4.0",
        "--forward-progress-reward-coef",
        "2.0",
        "--forward-velocity-deficit-penalty-coef",
        "4.0",
        "--forward-velocity-excess-penalty-coef",
        "5.0",
        "--min-forward-progress-speed",
        "0.25",
        "--forward-progress-grace-steps",
        "200",
        "--forward-progress-window-steps",
        "50",
        "--forward-progress-termination-penalty",
        "8.0",
        "--fall-penalty",
        "20.0",
        "--terminate-incomplete-motion-end",
        "--side-foot-pos-coef",
        "0.50",
        "--step-gap-coef",
        "0.50",
        "--swing-advance-coef",
        "1.0",
        "--stance-foot-vel-sigma",
        "0.12",
        "--stance-foot-reward-coef",
        "2.0",
        "--foot-slip-velocity-threshold",
        "0.08",
        "--foot-slip-penalty-coef",
        "2.0",
        "--contact-reward-coef",
        "1.0",
        "--missed-contact-penalty-coef",
        "1.2",
        "--symmetry-reward-coef",
        "0.50",
        "--terrain-types",
        "rough",
        "slope_up",
        "slope_down",
        "--serial-terrain-course",
        "--terrain-course-variants",
        "8",
        "--terrain-course-random-start-prob",
        "0.0",
        "--terrain-course-random-start-warmup-iterations",
        "5000",
        "--terrain-course-random-start-ramp-iterations",
        "10000",
        "--terrain-course-stage-reward",
        "10.0",
        "--rough-terrain-height",
        f"{MAX_ROUGH_HEIGHT_DIFFERENCE_M:g}",
        "--slope-angle-deg",
        f"{MAX_SLOPE_DEG:g}",
        "--specific-cot-limit",
        f"{MAX_SPECIFIC_COT_J_PER_KG_M:g}",
        "--specific-cot-penalty-coef",
        # Energy regularization is introduced only after a stable gait exists;
        # applying it from scratch rewards short, low-energy falling episodes.
        "0.0",
        "--motion-reset-prob",
        "1.0",
        # The actor receives no selected reference or skill command. Actions
        # are corrections around measured tendon lengths, so the control base
        # also cannot leak the simulator-selected motion into the policy path.
        "--reference-ctrl-blend",
        "0.0",
        "--motion-transition-blend-steps",
        "0",
        "--startup-action-ramp-steps",
        "25",
        "--stand-start-prob",
        "0.50",
        "--stand-start-phase-delay-steps",
        "20",
        "--stand-start-transition-min-steps",
        "80",
        "--stand-start-transition-max-steps",
        "120",
        "--episode-steps",
        "1400",
        "--log-group",
        "multimotion_no_stairs",
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,multimotion,independent,no-stair-terrain,"
            "imu,tendon,recurrent,gru,sensor-only,asymmetric-critic,"
            "flat,rough,slope-up,slope-down"
        ),
    ]


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
