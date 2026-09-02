#!/usr/bin/env python3
"""Stage 1 for the terrain-aligned, sensor-only stair-free AMP task."""

from __future__ import annotations

import sys
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_multimotion_no_stairs import default_train_args


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


def run_stage1() -> None:
    """Start an AMP-only gait stage without inheriting the old two-clip setup."""

    if not _has_arg("--stage"):
        sys.argv.extend(["--stage", "stage1"])

    # Stage 1 establishes the unperturbed alternating gait.  Stage 2 remains
    # responsible for adding the existing domain-randomization curriculum.
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
    sys.argv.extend(["--no-domain-randomization", "--no-warp-global-domain-randomization"])
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    if not _has_arg("--device"):
        sys.argv.extend(["--device", "cuda"])
    if not _has_arg("--sim-backend"):
        sys.argv.extend(["--sim-backend", "warp"])
    if not _has_arg("--save-interval"):
        sys.argv.extend(["--save-interval", "500"])

    runpy.run_path(str(ROOT / "scripts" / "train.py"), run_name="__main__")


def main() -> None:
    sys.argv = [
        str(ROOT / "scripts" / "train.py"),
        *default_train_args(),
        *sys.argv[1:],
    ]
    run_stage1()


if __name__ == "__main__":
    main()
