#!/usr/bin/env python3
"""Pretrain stable flat walking with the multimotion-compatible policy layout."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_multimotion import (
    default_train_args as multimotion_train_args,
    split_reference_set_args,
)


def _replace_option_values(
    args: list[str],
    option: str,
    values: list[str],
) -> None:
    start = args.index(option)
    end = start + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    args[start:end] = [option, *values]


def _option_values(args: list[str], option: str) -> list[str]:
    start = args.index(option) + 1
    end = start
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return args[start:end]


def default_train_args(reference_set: str = "four") -> list[str]:
    """Keep the straight-terrain skills in the layout but sample only walk."""

    args = multimotion_train_args(reference_set)
    _replace_option_values(
        args,
        "--motion-weights",
        ["1.0", *("0.0" for _ in _option_values(args, "--motion-files")[1:])],
    )
    _replace_option_values(args, "--episode-steps", ["1000"])
    _replace_option_values(args, "--log-group", ["multimotion_walk_warmup"])
    _replace_option_values(
        args,
        "--wandb-tags",
        ["lytrobot,bpo,amp,multimotion,walk,flat,stability-warmup"],
    )
    args.remove("--serial-terrain-course")
    return args


def main() -> None:
    reference_set, user_args = split_reference_set_args(sys.argv[1:])
    sys.argv = [
        str(HERE / "train.py"),
        *default_train_args(reference_set),
        *user_args,
    ]
    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
