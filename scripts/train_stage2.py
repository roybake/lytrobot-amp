#!/usr/bin/env python3
"""Train stage2 robustness policy on GPU/Warp with domain randomization."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_multimotion_no_stairs import (
    default_ordinary_stair_free_course_args,
)


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


def run_stage2() -> None:
    """Run the canonical ordinary-walking Stage 2 configuration."""

    # Keep explicit CLI overrides last while making the ordinary walking task
    # use the same randomized stair-free course as Stage 1.
    user_args = sys.argv[1:]
    sys.argv[1:] = [*default_ordinary_stair_free_course_args(), *user_args]

    stage1_checkpoint = _take_option(sys.argv, "--stage1-checkpoint")
    stage1_checkpoint = stage1_checkpoint or _take_option(sys.argv, "--stage1-model")
    if stage1_checkpoint is not None:
        if any(arg == "--resume-checkpoint" or arg.startswith("--resume-checkpoint=") for arg in sys.argv):
            raise SystemExit("Use only one of --stage1-checkpoint/--stage1-model or --resume-checkpoint.")
        sys.argv.extend(
            [
                "--resume-checkpoint",
                stage1_checkpoint,
                "--resume-reset-progress",
                "--resume-reset-optimizer",
            ]
        )
    elif not any(
        arg == "--resume-checkpoint" or arg.startswith("--resume-checkpoint=")
        for arg in sys.argv
    ):
        raise SystemExit(
            "Stage 2 requires --stage1-checkpoint <path>; use "
            "--resume-checkpoint only to continue Stage 2."
        )

    if not any(arg == "--stage" or arg.startswith("--stage=") for arg in sys.argv):
        sys.argv.extend(["--stage", "stage2"])
    if not any(
        arg == "--training-task" or arg.startswith("--training-task=")
        for arg in sys.argv
    ):
        sys.argv.extend(["--training-task", "multimotion_no_stairs_amp_single"])
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
    if not any(
        arg == "--global-dr-resample-interval"
        or arg.startswith("--global-dr-resample-interval=")
        for arg in sys.argv
    ):
        sys.argv.extend(["--global-dr-resample-interval", "250"])
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


if __name__ == "__main__":
    run_stage2()
