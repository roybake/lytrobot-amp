#!/usr/bin/env python3
"""Run isolated sensor-only BPO-AMP squat training."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import train as base_train
from sensor_training.ppo_config import SensorPPOAMPConfig
from sensor_training.trainer import SensorHistoryPPOAMPTrainer
from squat_training.envs import SquatSensorHistoryWarpAmpEnv


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


def main() -> None:
    defaults = {
        "--motion-file": str(REPO_ROOT / "squat_training" / "reference_data" / "squat_cmu_22_14_omniretarget.npz"),
        "--log-root": str(REPO_ROOT / "logs" / "squat_sensor_policy"),
        "--wandb-project": "lytrobot-squat-amp",
        "--wandb-tags": "lytrobot,bpo,amp,squat,half-depth,imu,tendon,history6,omniretarget",
        "--target-velocity": "0.0",
        "--forward-velocity-reward-coef": "0.0",
        "--forward-progress-reward-coef": "0.0",
        "--forward-velocity-deficit-penalty-coef": "0.0",
        "--forward-velocity-excess-penalty-coef": "0.0",
        "--foot-vel-reward-coef": "0.0",
        "--symmetry-reward-coef": "0.75",
        "--missed-contact-penalty-coef": "1.5",
        "--action-residual-scale": "0.25",
        "--stand-start-prob": "0.0",
        "--motion-reset-prob": "1.0",
    }
    for name, value in defaults.items():
        if not _has_arg(name):
            sys.argv.extend([name, value])

    base_train.LytRobotWarpAmpEnv = SquatSensorHistoryWarpAmpEnv
    base_train.PPOAMPConfig = SensorPPOAMPConfig
    base_train.PPOAMPTrainer = SensorHistoryPPOAMPTrainer
    base_train.main()


if __name__ == "__main__":
    main()
