#!/usr/bin/env python3
"""Train stage1 base policy on GPU/Warp without domain randomization."""

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


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


def run_stage1() -> None:
    """Run the canonical ordinary-walking Stage 1 configuration."""

    # Keep explicit CLI overrides last while making the ordinary walking task
    # use the same randomized stair-free course as the dedicated launcher.
    user_args = sys.argv[1:]
    sys.argv[1:] = [*default_ordinary_stair_free_course_args(), *user_args]

    if not _has_arg("--stage"):
        sys.argv.extend(["--stage", "stage1"])
    if not _has_arg("--training-task"):
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
    sys.argv.extend(
        ["--no-domain-randomization", "--no-warp-global-domain-randomization"]
    )
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    if not _has_arg("--device"):
        sys.argv.extend(["--device", "cuda"])
    if not _has_arg("--sim-backend"):
        sys.argv.extend(["--sim-backend", "warp"])

    # 复现 06-11 优质策略的训练口径。新增的逐脚参考奖励会学习参考数据本身
    # 的左右不对称，并在当前机器人上表现为右脚前跳，因此 stage1 默认关闭。
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


if __name__ == "__main__":
    run_stage1()
