#!/usr/bin/env python3
"""Run the isolated privileged-Teacher sensor-distillation loop."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import train as base_train

from sensor_training.distillation_config import SensorDistillationPPOAMPConfig
from sensor_training.distillation_env import (
    SensorDistillationMjlabAmpEnv,
    SensorDistillationWarpAmpEnv,
)
from sensor_training.distillation_trainer import SensorDistillationPPOAMPTrainer
from sensor_training.training_defaults import (
    DEFAULT_SENSOR_DECIMATION,
    cli_int,
    validate_sensor_decimation,
)


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


def main() -> None:
    if not _has_arg("--teacher-checkpoint"):
        raise SystemExit("Distillation requires --teacher-checkpoint PATH.")
    validate_sensor_decimation(
        cli_int(sys.argv[1:], "--decimation", DEFAULT_SENSOR_DECIMATION)
    )
    if not _has_arg("--log-root"):
        sys.argv.extend(
            ["--log-root", str(REPO_ROOT / "logs" / "sensor_policy" / "distillation")]
        )
    if not _has_arg("--wandb-tags"):
        sys.argv.extend(
            ["--wandb-tags", "lytrobot,sensor-only,teacher-distillation"]
        )

    base_train.LytRobotWarpAmpEnv = SensorDistillationWarpAmpEnv
    base_train.LytRobotMjlabAmpEnv = SensorDistillationMjlabAmpEnv
    base_train.PPOAMPConfig = SensorDistillationPPOAMPConfig
    base_train.PPOAMPTrainer = SensorDistillationPPOAMPTrainer
    base_train.DISTILLATION_CLI_ENABLED = True
    base_train.main()


if __name__ == "__main__":
    main()
