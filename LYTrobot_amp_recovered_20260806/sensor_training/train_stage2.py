#!/usr/bin/env python3
"""Sensor-only stage2 robustness training with domain randomization."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


def _take_option(name: str) -> str | None:
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == name:
            if index + 1 >= len(sys.argv):
                raise SystemExit(f"{name} requires a checkpoint path.")
            value = sys.argv[index + 1]
            del sys.argv[index : index + 2]
            return value
        prefix = f"{name}="
        if arg.startswith(prefix):
            value = arg[len(prefix) :]
            del sys.argv[index]
            return value
    return None


if __name__ == "__main__":
    stage1_checkpoint = _take_option("--stage1-checkpoint") or _take_option("--stage1-model")
    if stage1_checkpoint is not None:
        if _has_arg("--resume-checkpoint"):
            raise SystemExit("Use only one of --stage1-checkpoint/--stage1-model or --resume-checkpoint.")
        sys.argv.extend(["--resume-checkpoint", stage1_checkpoint, "--resume-reset-progress"])

    defaults = {
        "--stage": "stage2",
        "--device": "cuda",
        "--sim-backend": "warp",
        "--num-envs": "4096",
        "--max-iterations": "30000",
        "--save-interval": "500",
        # Keep stage2 compatible with the deployable sensor-only stage1 policy:
        # commands are residuals around measured tendon lengths, not around
        # simulator reference controls.  The task reward itself remains the
        # stage1 reward; stage2 changes robustness only through domain randomization.
        "--reference-ctrl-blend": "0.0",
    }
    for name, value in defaults.items():
        if not _has_arg(name):
            sys.argv.extend([name, value])
    if not (_has_arg("--domain-randomization") or _has_arg("--no-domain-randomization")):
        sys.argv.append("--domain-randomization")
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    runpy.run_path(str(Path(__file__).resolve().parent / "train.py"), run_name="__main__")
