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

from sensor_training.envs import (
    SensorHistoryCpuAmpEnv,
    SensorHistoryMjlabAmpEnv,
    SensorHistoryWarpAmpEnv,
)
from sensor_training.ppo_config import SensorPPOAMPConfig
from sensor_training.trainer import SensorHistoryPPOAMPTrainer
from sensor_training.training_defaults import (
    DEFAULT_SENSOR_DECIMATION,
    cli_int,
    validate_sensor_decimation,
)


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


def main() -> None:
    validate_sensor_decimation(
        cli_int(sys.argv[1:], "--decimation", DEFAULT_SENSOR_DECIMATION)
    )
    if not _has_arg("--log-root"):
        sys.argv.extend(["--log-root", str(REPO_ROOT / "logs" / "sensor_policy")])
    if not _has_arg("--wandb-tags"):
        sys.argv.extend(
            [
                "--wandb-tags",
                "lytrobot,bpo,amp,imu,tendon,recurrent,gru,residual-moe",
            ]
        )

    # CPU MuJoCo rollouts have a small batch and spend most time stepping
    # physics; compiling many episode-start graph variants delays fine-tuning.
    cpu_backend = "--sim-backend=cpu" in sys.argv or (
        "--sim-backend" in sys.argv
        and sys.argv.index("--sim-backend") + 1 < len(sys.argv)
        and sys.argv[sys.argv.index("--sim-backend") + 1] == "cpu"
    )
    if cpu_backend and not (
        _has_arg("--compile-rollout") or _has_arg("--no-compile-rollout")
    ):
        sys.argv.append("--no-compile-rollout")

    base_train.LytRobotWarpAmpEnv = SensorHistoryWarpAmpEnv
    base_train.LytRobotMjlabAmpEnv = SensorHistoryMjlabAmpEnv
    base_train.LytRobotAmpEnv = SensorHistoryCpuAmpEnv
    base_train.PPOAMPConfig = SensorPPOAMPConfig
    base_train.PPOAMPTrainer = SensorHistoryPPOAMPTrainer
    from sensor_training.experiment_options import install_training_options
    install_training_options(base_train)
    base_train.main()


if __name__ == "__main__":
    main()
