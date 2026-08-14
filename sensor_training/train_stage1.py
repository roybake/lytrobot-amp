#!/usr/bin/env python3
"""Sensor-only stage1 training with the same defaults as scripts/train_stage1.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


if __name__ == "__main__":
    if not _has_arg("--stage"):
        sys.argv.extend(["--stage", "stage1"])
    if not _has_arg("--training-task"):
        sys.argv.extend(["--training-task", "walk"])
    domain_flags = (
        "--domain-randomization",
        "--no-domain-randomization",
        "--warp-global-domain-randomization",
        "--no-warp-global-domain-randomization",
    )
    sys.argv[:] = [
        arg
        for arg in sys.argv
        if arg not in domain_flags
        and not any(arg.startswith(f"{name}=") for name in domain_flags)
    ]
    sys.argv.extend(
        ["--no-domain-randomization", "--no-warp-global-domain-randomization"]
    )
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    if not _has_arg("--device"):
        sys.argv.extend(["--device", "cuda"])
    if not _has_arg("--sim-backend"):
        sys.argv.extend(["--sim-backend", "warp"])

    # Keep sensor stage1 on the same reward/training defaults as
    # scripts/train_stage1.py.  The previous foot-dominant symmetry shaping
    # produced a high per-step reward while forward motion, posture, and
    # episode length collapsed late in training.
    stage1_defaults = {
        "--num-envs": "4096",
        "--symmetry-reward-coef": "1.2",
        "--symmetry-foot-pos-sigma": "0.08",
        "--side-foot-pos-coef": "0.0",
        "--step-gap-coef": "0.0",
        "--swing-advance-coef": "0.0",
        "--save-interval": "500",
    }
    for name, value in stage1_defaults.items():
        if not _has_arg(name):
            sys.argv.extend([name, value])
    runpy.run_path(str(Path(__file__).resolve().parent / "train.py"), run_name="__main__")
