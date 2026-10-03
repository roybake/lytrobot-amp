#!/usr/bin/env python3
"""Sensor-only stage2 robustness training with domain randomization."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensor_training.training_defaults import (
    recurrent_cli_defaults,
    stage_policy_rollout_envs,
)


# Must match sensor_training/train_stage1.py: a Stage2 run resumes a Stage1
# checkpoint, so the reference clip, control rate and action interface may not
# change between the two stages.
DEFAULT_GAIT_REFERENCE = (
    REPO_ROOT / "robot" / "reference_data" / "walk1_mjcf_repaired.npz"
)


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
    default_decimation, default_num_steps, _default_envs = recurrent_cli_defaults(
        sys.argv[1:]
    )
    default_num_envs = stage_policy_rollout_envs(default_num_steps, default_decimation)
    stage1_checkpoint = _take_option("--stage1-checkpoint") or _take_option("--stage1-model")
    if stage1_checkpoint is not None:
        if _has_arg("--resume-checkpoint"):
            raise SystemExit("Use only one of --stage1-checkpoint/--stage1-model or --resume-checkpoint.")
        sys.argv.extend(["--resume-checkpoint", stage1_checkpoint, "--resume-reset-progress"])

    # Same deployable interface, reward and PPO budget as Stage1; Stage2 only
    # adds domain randomization and a longer schedule.
    defaults = {
        "--stage": "stage2",
        "--device": "cuda",
        "--sim-backend": "warp",
        "--num-envs": str(default_num_envs),
        "--num-steps": str(default_num_steps),
        "--max-iterations": "30000",
        "--motion-file": str(DEFAULT_GAIT_REFERENCE),
        "--motion-loop": "true",
        "--target-velocity": "0.87",
        # Must match Stage1: the residual is added on top of the onboard
        # reference tendon targets, exactly like the accepted privileged gait.
        "--reference-ctrl-blend": "1.0",
        "--action-residual-scale": "0.5",
        "--decimation": str(default_decimation),
        "--startup-action-ramp-steps": "50",
        "--save-interval": "500",
        "--fall-penalty": "2.0",
        "--stand-start-prob": "0.5",
        "--motion-reset-prob": "0.5",
        "--stand-start-phase-delay-steps": "25",
        "--stand-start-transition-min-steps": "100",
        "--stand-start-transition-max-steps": "200",
        "--episode-steps": "1200",
        "--learning-rate": "3.0e-4",
        "--min-learning-rate": "1.0e-5",
        "--max-learning-rate": "3.0e-4",
        "--clip-coef": "0.2",
        "--target-kl": "0.025",
        "--target-kl-stop-multiplier": "2.0",
        "--min-action-std": "0.05",
        "--max-action-std": "0.6",
        "--task-reward-lerp": "0.5",
        "--amp-reward-coef": "1.2",
        "--min-forward-progress-speed": "0.1",
        "--forward-progress-grace-steps": "400",
        "--forward-progress-window-steps": "100",
        "--forward-progress-termination-penalty": "2.0",
    }
    for name, value in defaults.items():
        if not _has_arg(name):
            sys.argv.extend([name, value])
    if not (_has_arg("--sensor-stability-reward") or _has_arg("--no-sensor-stability-reward")):
        sys.argv.append("--no-sensor-stability-reward")
    if not (_has_arg("--domain-randomization") or _has_arg("--no-domain-randomization")):
        sys.argv.append("--domain-randomization")
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    runpy.run_path(str(Path(__file__).resolve().parent / "train.py"), run_name="__main__")
