#!/usr/bin/env python3
"""Train stage1 base policy on GPU/Warp without domain randomization."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAIT_REFERENCE = REPO_ROOT / "robot" / "reference_data" / "walk1_mjcf.npz"


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

    command_walking = "--command-walking" in sys.argv

    # Use the 10-06 Teacher recipe with standing starts and continuous walking.
    stage1_defaults = {
        "--log-root": str(REPO_ROOT / "logs" / "sensor_policy" / "distillation"),
        "--log-role": "teacher",
        "--num-envs": "1024",
        "--max-iterations": "30000",
        "--num-steps": "24",
        "--decimation": "2",
        "--task-reward-mode": "stage1_reference",
        "--motion-file": str(DEFAULT_GAIT_REFERENCE),
        "--termination-mode": "fall",
        "--motion-loop": "true",
        "--episode-steps": "0",
        "--episode-timeout-jitter-steps": "0",
        "--target-velocity": "1.28",
        "--forward-velocity-reward-coef": "2.0",
        "--forward-progress-reward-coef": "0.0",
        "--forward-velocity-deficit-penalty-coef": "0.0",
        "--forward-velocity-tracking-sigma": "0.25",
        "--fall-penalty": "2.0",
        "--task-reward-lerp": "0.5",
        "--motion-reset-prob": "0.0",
        "--stand-start-prob": "1.0",
        "--stand-start-motion-phase": "0",
        "--stand-start-phase-delay-steps": "25",
        "--stand-start-transition-min-steps": "80",
        "--stand-start-transition-max-steps": "160",
        "--symmetry-reward-coef": "1.2",
        "--symmetry-foot-pos-sigma": "0.08",
        "--side-foot-pos-coef": "0.0",
        "--step-gap-coef": "0.0",
        "--swing-advance-coef": "0.0",
        "--save-interval": "500",
        "--gait-evaluation-seconds": "30",
    }
    if command_walking:
        stage1_defaults["--log-root"] = str(REPO_ROOT / "logs" / "command_walking")
        # The fixed-speed gait evaluator measures world-X progress and is not
        # a checkpoint selector for a policy that is commanded to turn.
        for name in ("--evaluate-gait", "--gait-fidelity-selection"):
            if not (_has_arg(name) or _has_arg(f"--no-{name[2:]}")):
                sys.argv.append(f"--no-{name[2:]}")
        if not _has_arg("--velocity-range"):
            sys.argv.extend(["--velocity-range", "0.6", "1.28"])
    for name, value in stage1_defaults.items():
        if not _has_arg(name):
            sys.argv.extend([name, value])

    for name in ("--evaluate-gait", "--gait-fidelity-selection", "--symmetric-gait-reference"):
        if not (_has_arg(name) or _has_arg(f"--no-{name[2:]}")):
            sys.argv.append(name)
    if not (_has_arg("--stop-on-gait-acceptance") or _has_arg("--no-stop-on-gait-acceptance")):
        sys.argv.append("--no-stop-on-gait-acceptance")

    runpy.run_path(str(Path(__file__).resolve().parent / "train.py"), run_name="__main__")
