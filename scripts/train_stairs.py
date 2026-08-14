#!/usr/bin/env python3
"""Train an independent conditioned policy for ascending and descending stairs."""

from __future__ import annotations

import sys
from pathlib import Path

from lytrobot_amp.benchmark import (
    MAX_SPECIFIC_COT_J_PER_KG_M,
    STAIR_HEIGHT_RANGE_M,
    STAIR_TREAD_RANGE_M,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def default_train_args() -> list[str]:
    """Return self-contained defaults for the dedicated up/down stair task."""

    return [
        "--training-task",
        "stairs_up_down",
        "--motion-files",
        str(REFERENCE_DIR / "stair_up_omniretarget.npz"),
        str(REFERENCE_DIR / "stair_down_omniretarget.npz"),
        "--motion-names",
        "stair_up",
        "stair_down",
        "--motion-weights",
        "0.50",
        "0.50",
        "--motion-loop",
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
        "32",
        "--warp-njmax",
        "128",
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
        "25",
        "--forward-progress-window-steps",
        "25",
        "--forward-progress-termination-penalty",
        "2.0",
        "--fall-penalty",
        "10.0",
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
        "--terrain-types",
        "stairs",
        "--stair-step-height",
        f"{STAIR_HEIGHT_RANGE_M[0]:g}",
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
        "1.0",
        "--motion-reset-prob",
        "1.0",
        "--reference-ctrl-blend",
        "0.0",
        "--startup-action-ramp-steps",
        "0",
        "--stand-start-prob",
        "0.0",
        "--symmetry-reward-coef",
        "0.0",
        "--episode-steps",
        "500",
        "--episode-timeout-jitter-steps",
        "0",
        "--log-group",
        "stairs_up_down",
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,stairs,stair-up,stair-down,independent,"
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
