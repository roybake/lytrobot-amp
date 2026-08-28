#!/usr/bin/env python3
"""Train the frozen privileged Teacher from the 08-06 no-stairs baseline."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_no_stairs_old import default_train_args as baseline_args


def _replace(args: list[str], option: str, values: list[str]) -> None:
    start = args.index(option)
    end = start + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    args[start:end] = [option, *values]


def default_train_args() -> list[str]:
    """Current-safe approximation of 08-06, with stairs already removed."""

    args = baseline_args()
    _replace(args, "--training-task", ["multimotion_no_stairs_amp_teacher"])
    _replace(args, "--log-group", ["multimotion_no_stairs_amp_teacher"])
    args.extend(
        [
            "--legacy-reference-conditioned-actor",
            "--log-root",
            str(ROOT / "logs" / "multimotion"),
            "--wandb-tags",
            "lytrobot,bpo,amp,no-stairs,amp-teacher,privileged,reference-conditioned,"
            "rough,slope-up,slope-down",
        ]
    )
    return args


def main() -> None:
    import scripts.train as runner
    from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
    from lytrobot_amp.ppo_amp import PPOAMPConfig, PPOAMPTrainer

    runner.LytRobotWarpAmpEnv = LytRobotWarpAmpEnv
    runner.PPOAMPConfig = PPOAMPConfig
    runner.PPOAMPTrainer = PPOAMPTrainer
    runner.validate_locomotion_actor_input_contract = lambda _env: None
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    runner.main()


if __name__ == "__main__":
    main()
