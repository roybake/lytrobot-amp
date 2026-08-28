#!/usr/bin/env python3
"""Train the former 08-06 no-stairs setup with the unified sensor Actor."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"


def default_train_args() -> list[str]:
    """Return the 08-06 run settings with stair-only items removed.

    The two surviving motion weights are re-normalized from 0.25/0.25 to
    0.50/0.50; this is the only necessary numerical adjustment after removing
    ``stair_up`` and ``stair_down``.
    """

    return [
        "--training-task",
        "no_stairs_old_sensor_privileged_v2",
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
        "--use-reference-velocity",
        "--terrain-types",
        "rough",
        "slope_up",
        "slope_down",
        "--serial-terrain-course",
        "--terrain-course-variants",
        "8",
        "--terrain-course-random-start-prob",
        "0.80",
        "--terrain-course-random-start-warmup-iterations",
        "5000",
        "--terrain-course-random-start-ramp-iterations",
        "10000",
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
        "--target-velocity",
        "0.862295",
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
        "--specific-cot-limit",
        "0.8",
        "--specific-cot-penalty-coef",
        "2.0",
        "--reset-noise-scale",
        "0.01",
        "--action-residual-scale",
        "0.5",
        "--distal-action-scale",
        "1.0",
        "--reference-ctrl-blend",
        "0.0",
        "--motion-reset-prob",
        "1.0",
        "--motion-transition-blend-steps",
        "15",
        "--foot-vel-tracking-sigma",
        "1.0",
        "--stance-foot-vel-sigma",
        "0.15",
        "--foot-vel-reward-coef",
        "0.35",
        "--stance-foot-reward-coef",
        "1.0",
        "--contact-reward-coef",
        "0.7",
        "--missed-contact-penalty-coef",
        "0.8",
        "--side-foot-pos-sigma",
        "0.08",
        "--step-gap-sigma",
        "0.12",
        "--swing-advance-sigma",
        "0.10",
        "--side-foot-pos-coef",
        "0.5",
        "--step-gap-coef",
        "0.5",
        "--swing-advance-coef",
        "0.75",
        "--distal-joint-vel-sigma",
        "2.0",
        "--symmetry-reward-coef",
        "0.5",
        "--startup-action-ramp-steps",
        "0",
        "--stand-start-prob",
        "0.0",
        "--episode-steps",
        "1800",
        "--ctrl-penalty-coef",
        "0.04",
        "--action-rate-penalty-coef",
        "0.1",
        "--distal-action-penalty-coef",
        "0.03",
        "--distal-action-rate-penalty-coef",
        "0.12",
        "--learning-rate",
        "0.0003",
        "--reward-scale",
        "0.25",
        "--disc-learning-rate",
        "0.00003",
        "--amp-reward-coef",
        "1.2",
        "--task-reward-lerp",
        "0.5",
        "--min-action-std",
        "0.05",
        "--max-action-std",
        "0.6",
        "--log-group",
        "no_stairs_old_sensor_privileged_v2",
        "--wandb-tags",
        "lytrobot,bpo,amp,multimotion,no-stairs-old,sensor-only,asymmetric-critic,"
        "privileged-v2,serial-course,random-order,rough,slope-up,slope-down",
    ]


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
