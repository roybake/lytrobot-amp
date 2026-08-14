#!/usr/bin/env python3
"""Stage 2 squat robustness training with domain randomization."""

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
        sys.argv.extend(
            [
                "--resume-checkpoint",
                stage1_checkpoint,
                "--resume-reset-progress",
                "--resume-reset-optimizer",
            ]
        )
    elif not _has_arg("--resume-checkpoint"):
        raise SystemExit(
            "Stage 2 requires --stage1-checkpoint <path>; use "
            "--resume-checkpoint only to continue Stage 2."
        )

    defaults = {
        "--stage": "stage2",
        "--device": "cuda",
        "--sim-backend": "warp",
        "--num-envs": "4096",
        "--max-iterations": "30000",
        "--save-interval": "500",
        "--learning-rate": "1e-4",
        "--min-learning-rate": "1e-5",
        "--max-learning-rate": "1e-4",
        "--dr-curriculum-mode": "adaptive",
        "--dr-curriculum-initial-scale": "0.0",
        "--dr-curriculum-update-interval": "250",
        "--dr-curriculum-step": "0.025",
        "--push-interval-min-steps": "75",
        "--push-interval-max-steps": "200",
        # Robustness randomizes dynamics and disturbances, not the deployment
        # contract that squat control starts from stable standing.
        "--stand-start-prob": "1.0",
    }
    for name, value in defaults.items():
        if not _has_arg(name):
            sys.argv.extend([name, value])
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
    sys.argv.extend(["--domain-randomization", "--warp-global-domain-randomization"])
    if not _has_arg("--global-dr-resample-interval"):
        sys.argv.extend(["--global-dr-resample-interval", "250"])
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    runpy.run_path(str(Path(__file__).resolve().parent / "train.py"), run_name="__main__")
