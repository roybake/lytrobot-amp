#!/usr/bin/env python3
"""Run the BPO-AMP training loop with the isolated sensor environment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import train as base_train

from sensor_training.envs import SensorHistoryWarpAmpEnv
from sensor_training.ppo_config import SensorPPOAMPConfig
from sensor_training.trainer import SensorHistoryPPOAMPTrainer


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


def main() -> None:
    if not _has_arg("--log-root"):
        sys.argv.extend(["--log-root", str(REPO_ROOT / "logs" / "sensor_policy")])
    if not _has_arg("--wandb-tags"):
        sys.argv.extend(["--wandb-tags", "lytrobot,bpo,amp,imu,tendon,history6"])

    base_train.LytRobotWarpAmpEnv = SensorHistoryWarpAmpEnv
    base_train.PPOAMPConfig = SensorPPOAMPConfig
    base_train.PPOAMPTrainer = SensorHistoryPPOAMPTrainer
    base_train.main()


if __name__ == "__main__":
    main()
