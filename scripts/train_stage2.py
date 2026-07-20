#!/usr/bin/env python3
"""Train stage2 robustness policy on GPU/Warp with domain randomization."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _take_option(argv: list[str], name: str) -> str | None:
    for index, arg in enumerate(argv[1:], start=1):
        if arg == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a checkpoint path.")
            value = argv[index + 1]
            del argv[index : index + 2]
            return value
        prefix = f"{name}="
        if arg.startswith(prefix):
            value = arg[len(prefix) :]
            del argv[index]
            return value
    return None


if __name__ == "__main__":
    stage1_checkpoint = _take_option(sys.argv, "--stage1-checkpoint")
    stage1_checkpoint = stage1_checkpoint or _take_option(sys.argv, "--stage1-model")
    if stage1_checkpoint is not None:
        if any(arg == "--resume-checkpoint" or arg.startswith("--resume-checkpoint=") for arg in sys.argv):
            raise SystemExit("Use only one of --stage1-checkpoint/--stage1-model or --resume-checkpoint.")
        sys.argv.extend(["--resume-checkpoint", stage1_checkpoint, "--resume-reset-progress"])

    if not any(arg == "--stage" or arg.startswith("--stage=") for arg in sys.argv):
        sys.argv.extend(["--stage", "stage2"])
    if not any(
        arg == "--domain-randomization"
        or arg.startswith("--domain-randomization=")
        or arg == "--no-domain-randomization"
        or arg.startswith("--no-domain-randomization=")
        for arg in sys.argv
    ):
        sys.argv.append("--domain-randomization")
    if not any(
        arg == "--wandb"
        or arg.startswith("--wandb=")
        or arg == "--no-wandb"
        or arg.startswith("--no-wandb=")
        for arg in sys.argv
    ):
        sys.argv.append("--wandb")
    if not any(arg == "--device" or arg.startswith("--device=") for arg in sys.argv):
        sys.argv.extend(["--device", "cuda"])
    if not any(arg == "--sim-backend" or arg.startswith("--sim-backend=") for arg in sys.argv):
        sys.argv.extend(["--sim-backend", "warp"])
    # 4096 envs generate four times as many samples per iteration as stage1.
    # 7500 iterations therefore preserve the stage1 sample budget instead of
    # accidentally running a 4x larger 2.95-billion-transition job.
    if not any(arg == "--num-envs" or arg.startswith("--num-envs=") for arg in sys.argv):
        sys.argv.extend(["--num-envs", "4096"])
    if not any(arg == "--max-iterations" or arg.startswith("--max-iterations=") for arg in sys.argv):
        sys.argv.extend(["--max-iterations", "7500"])
    if not any(arg == "--save-interval" or arg.startswith("--save-interval=") for arg in sys.argv):
        sys.argv.extend(["--save-interval", "500"])
    runpy.run_path(str(Path(__file__).resolve().parent / "train.py"), run_name="__main__")
