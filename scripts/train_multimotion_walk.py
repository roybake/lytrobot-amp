#!/usr/bin/env python3
"""Pretrain stable flat walking with the multimotion-compatible policy layout."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_multimotion import default_train_args as multimotion_train_args


def _replace_option_values(
    args: list[str],
    option: str,
    values: list[str],
) -> None:
    if option not in args:
        args.extend([option, *values])
        return
    start = args.index(option)
    end = start + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    args[start:end] = [option, *values]


def default_train_args() -> list[str]:
    """Keep the stair-free terrain-aligned layout but sample only walk."""

    args = multimotion_train_args()
    _replace_option_values(
        args,
        "--motion-weights",
        ["1.0", "0.0", "0.0", "0.0"],
    )
    _replace_option_values(args, "--episode-steps", ["2000"])
    _replace_option_values(args, "--log-group", ["multimotion_walk_warmup"])
    _replace_option_values(
        args,
        "--wandb-tags",
        ["lytrobot,bpo,amp,multimotion,walk,flat,stability-warmup"],
    )
    args.remove("--serial-terrain-course")
    return args


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
