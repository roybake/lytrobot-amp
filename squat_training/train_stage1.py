#!/usr/bin/env python3
"""Stage 1 squat imitation training without domain randomization."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


if __name__ == "__main__":
    defaults = {
        "--stage": "stage1",
        "--device": "cuda",
        "--sim-backend": "warp",
        "--num-envs": "4096",
        "--max-iterations": "30000",
        "--save-interval": "500",
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
    sys.argv.extend(
        ["--no-domain-randomization", "--no-warp-global-domain-randomization"]
    )
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    runpy.run_path(str(Path(__file__).resolve().parent / "train.py"), run_name="__main__")
