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


def squat_training_defaults() -> dict[str, str]:
    """Return task defaults shared by both squat training stages."""

    return {
        "--motion-file": str(
            REPO_ROOT
            / "squat_training"
            / "reference_data"
            / "squat_cmu_22_14_omniretarget.npz"
        ),
        "--log-root": str(REPO_ROOT / "logs" / "squat_sensor_policy"),
        "--training-task": "squat",
        "--wandb-project": "lytrobot-squat-amp",
        "--wandb-tags": (
            "lytrobot,bpo,amp,squat,half-depth,imu,tendon,history6,omniretarget"
        ),
        "--target-velocity": "0.0",
        "--forward-velocity-reward-coef": "0.0",
        "--forward-progress-reward-coef": "0.0",
        "--forward-velocity-deficit-penalty-coef": "0.0",
        "--forward-velocity-excess-penalty-coef": "0.0",
        "--foot-vel-reward-coef": "0.0",
        "--symmetry-reward-coef": "0.75",
        "--missed-contact-penalty-coef": "1.5",
        "--action-residual-scale": "0.25",
        # Stable standing is the task's initial condition.  Stage 1 must learn
        # stand -> squat -> stand before any random-phase recovery curriculum.
        "--stand-start-prob": "1.0",
        "--motion-reset-prob": "0.0",
        "--reference-ctrl-blend": "0.0",
        "--min-upright-z": "0.65",
        # Squat is a slow 300-step recurrent task.  The generic sensor-policy
        # defaults allow updates that are too large for this long horizon and
        # repeatedly destroy an already successful stand-up policy.
        "--learning-rate": "1e-4",
        "--min-learning-rate": "2e-5",
        "--max-learning-rate": "1e-4",
        "--target-kl": "0.01",
        "--target-kl-stop-multiplier": "1.5",
        "--kl-adapt-factor": "1.25",
        "--clip-coef": "0.10",
        "--epochs": "2",
        # Keep completion subordinate to survival.  A policy must remain alive
        # for the full episode instead of cashing out after one risky cycle.
        "--squat-cycle-completion-bonus": "5.0",
    }


def main() -> None:
    if _has_arg("--skill-conditioning"):
        raise ValueError(
            "Squat actor inputs are limited to IMU, tendon length, and the "
            "feedback-corrected controller phase."
        )
    for name, value in squat_training_defaults().items():
        if not _has_arg(name):
            sys.argv.extend([name, value])
    if not _has_arg("--no-skill-conditioning"):
        sys.argv.append("--no-skill-conditioning")
    if not (_has_arg("--squat-phase-feedback") or _has_arg("--no-squat-phase-feedback")):
        sys.argv.append("--squat-phase-feedback")
    if not (
        _has_arg("--squat-terminate-on-cycle-completion")
        or _has_arg("--no-squat-terminate-on-cycle-completion")
    ):
        sys.argv.append("--no-squat-terminate-on-cycle-completion")

    original_components = (
        base_train.LytRobotWarpAmpEnv,
        base_train.PPOAMPConfig,
        base_train.PPOAMPTrainer,
    )
    try:
        base_train.LytRobotWarpAmpEnv = SquatSensorHistoryWarpAmpEnv
        base_train.PPOAMPConfig = SensorPPOAMPConfig
        base_train.PPOAMPTrainer = SensorHistoryPPOAMPTrainer
        base_train.main()
    finally:
        (
            base_train.LytRobotWarpAmpEnv,
            base_train.PPOAMPConfig,
            base_train.PPOAMPTrainer,
        ) = original_components


if __name__ == "__main__":
    main()
