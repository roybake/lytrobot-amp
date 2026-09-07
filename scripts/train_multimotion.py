#!/usr/bin/env python3
"""Compatibility name for the canonical no-stairs sensor course launcher."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_multimotion_no_stairs import (
    COURSE_RANDOM_START_RAMP_ITERATIONS,
    COURSE_RANDOM_START_WARMUP_ITERATIONS,
    COURSE_VARIANTS,
    default_no_stairs_course_args,
    default_train_args as _canonical_train_args,
)


# These names remain import-compatible for config tests and downstream scripts.
PARALLEL_ENVS = 4096
ROLLOUT_STEPS = 24
PPO_MINIBATCHES = 8
AMP_MINIBATCH_SIZE = 4096
AMP_DISCRIMINATOR_UPDATES = 2


def default_train_args() -> list[str]:
    """Return the canonical no-stairs configuration, without duplication."""

    return _canonical_train_args()


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
