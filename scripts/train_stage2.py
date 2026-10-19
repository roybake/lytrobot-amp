#!/usr/bin/env python3
"""Train stage2 robustness policy on GPU/Warp with domain randomization."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAIT_REFERENCE = REPO_ROOT / "robot" / "reference_data" / "walk1_mjcf.npz"


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
    if not any(arg == "--motion-file" or arg.startswith("--motion-file=") for arg in sys.argv):
        sys.argv.extend(["--motion-file", str(DEFAULT_GAIT_REFERENCE)])
    if not any(arg == "--target-velocity" or arg.startswith("--target-velocity=") for arg in sys.argv):
        sys.argv.extend(["--target-velocity", "1.28"])
    if not any(arg == "--forward-velocity-reward-coef" or arg.startswith("--forward-velocity-reward-coef=") for arg in sys.argv):
        sys.argv.extend(["--forward-velocity-reward-coef", "2.0"])
    if not any(arg == "--swing-advance-coef" or arg.startswith("--swing-advance-coef=") for arg in sys.argv):
        sys.argv.extend(["--swing-advance-coef", "0.0"])
    if not any(arg == "--stand-start-motion-phase" or arg.startswith("--stand-start-motion-phase=") for arg in sys.argv):
        sys.argv.extend(["--stand-start-motion-phase", "31"])
    if not any(arg == "--stand-start-prob" or arg.startswith("--stand-start-prob=") for arg in sys.argv):
        sys.argv.extend(["--stand-start-prob", "1.0"])
    if not any(arg == "--motion-reset-prob" or arg.startswith("--motion-reset-prob=") for arg in sys.argv):
        sys.argv.extend(["--motion-reset-prob", "0.0"])
    if not any(arg == "--motion-loop" or arg.startswith("--motion-loop=") for arg in sys.argv):
        sys.argv.extend(["--motion-loop", "true"])
    if not any(arg == "--termination-mode" or arg.startswith("--termination-mode=") for arg in sys.argv):
        sys.argv.extend(["--termination-mode", "fall"])
    if not any(arg == "--episode-steps" or arg.startswith("--episode-steps=") for arg in sys.argv):
        sys.argv.extend(["--episode-steps", "0"])
    if not any(arg == "--episode-timeout-jitter-steps" or arg.startswith("--episode-timeout-jitter-steps=") for arg in sys.argv):
        sys.argv.extend(["--episode-timeout-jitter-steps", "0"])
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
