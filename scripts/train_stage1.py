#!/usr/bin/env python3
"""Train stage1 base policy on GPU/Warp without domain randomization."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


if __name__ == "__main__":
    if not _has_arg("--stage"):
        sys.argv.extend(["--stage", "stage1"])
    if not (_has_arg("--domain-randomization") or _has_arg("--no-domain-randomization")):
        sys.argv.append("--no-domain-randomization")
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    if not _has_arg("--device"):
        sys.argv.extend(["--device", "cuda"])
    if not _has_arg("--sim-backend"):
        sys.argv.extend(["--sim-backend", "warp"])

    # 复现 06-11 优质策略的训练口径。新增的逐脚参考奖励会学习参考数据本身
    # 的左右不对称，并在当前机器人上表现为右脚前跳，因此 stage1 默认关闭。
    stage1_defaults = {
        "--num-envs": "1024",
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
