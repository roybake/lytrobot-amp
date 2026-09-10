#!/usr/bin/env python3
"""Train the selectable two- or four-reference AMP terrain course."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lytrobot_amp.benchmark import (
    MAX_ROUGH_HEIGHT_DIFFERENCE_M,
    MAX_SLOPE_DEG,
)

REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
PARALLEL_ENVS = 4096
ROLLOUT_STEPS = 24
PPO_MINIBATCHES = 8
AMP_MINIBATCH_SIZE = 4096
AMP_DISCRIMINATOR_UPDATES = 2
COURSE_RANDOM_START_WARMUP_ITERATIONS = 5000
COURSE_RANDOM_START_RAMP_ITERATIONS = 10000
REFERENCE_SET_CHOICES = ("two", "four")

# ``two`` intentionally preserves the original experiment: slopes are still
# present in the physical course, but use the rough-walk expert.  ``four`` is
# the new terrain-specialized curriculum with dedicated slope experts.
REFERENCE_SET_CONFIGS = {
    "two": {
        "files": ("walk_omniretarget.npz", "rough_walk_omniretarget.npz"),
        "names": ("walk", "rough_walk"),
        "weights": ("0.50", "0.50"),
        "loops": ("true", "false"),
        "log_group": "multimotion_two_ref",
        "tags": "two-ref,flat,rough,slope-fallback",
    },
    "four": {
        "files": (
            "walk_omniretarget.npz",
            "rough_walk_omniretarget.npz",
            "slope_up_omniretarget.npz",
            "slope_down_omniretarget.npz",
        ),
        "names": ("walk", "rough_walk", "slope_up", "slope_down"),
        "weights": ("0.35", "0.25", "0.20", "0.20"),
        "loops": ("true", "false", "false", "false"),
        "log_group": "multimotion_four_ref",
        "tags": "four-ref,flat,rough,slope-up,slope-down",
    },
}


def default_train_args(reference_set: str = "four") -> list[str]:
    """Return defaults for the requested reference layout.

    A policy checkpoint can only resume into the same layout: skill one-hot
    dimensions are 2 and 4 respectively.
    """

    if reference_set not in REFERENCE_SET_CONFIGS:
        raise ValueError(
            f"reference_set must be one of {REFERENCE_SET_CHOICES}, got {reference_set!r}."
        )
    reference = REFERENCE_SET_CONFIGS[reference_set]
    return [
        "--motion-files",
        *(str(REFERENCE_DIR / filename) for filename in reference["files"]),
        "--motion-names",
        *reference["names"],
        "--motion-weights",
        *reference["weights"],
        "--motion-loop",
        *reference["loops"],
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
        # Draw enough AMP examples to cover all selected skills on every update.
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
        # Keep the Stage1 flat-walking reward unchanged.  The only terrain
        # reward shaping below is the course-stage and course-completion bonus.
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
        str(COURSE_RANDOM_START_WARMUP_ITERATIONS),
        "--terrain-course-random-start-ramp-iterations",
        str(COURSE_RANDOM_START_RAMP_ITERATIONS),
        "--terrain-course-stage-reward",
        "5.0",
        "--rough-terrain-height",
        f"{MAX_ROUGH_HEIGHT_DIFFERENCE_M:g}",
        "--slope-angle-deg",
        f"{MAX_SLOPE_DEG:g}",
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
        reference["log_group"],
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,multimotion,more,serial-course,random-order,"
            f"{reference['tags']},acceptance"
        ),
    ]


def split_reference_set_args(argv: list[str]) -> tuple[str, list[str]]:
    """Consume this wrapper's selector before forwarding arguments to train.py."""

    reference_set = "four"
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--reference-set":
            if index + 1 >= len(argv):
                raise SystemExit("--reference-set requires 'two' or 'four'.")
            reference_set = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--reference-set="):
            reference_set = argument.partition("=")[2]
            index += 1
            continue
        remaining.append(argument)
        index += 1
    if reference_set not in REFERENCE_SET_CONFIGS:
        raise SystemExit(
            f"--reference-set must be one of: {', '.join(REFERENCE_SET_CHOICES)}."
        )
    return reference_set, remaining


def main() -> None:
    reference_set, user_args = split_reference_set_args(sys.argv[1:])
    sys.argv = [str(HERE / "train.py"), *default_train_args(reference_set), *user_args]
    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
