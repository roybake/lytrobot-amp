#!/usr/bin/env python3
"""Compatibility Stage 2 launcher for sensor multimotion training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.two_stage_task import run_task_stage
from scripts.train_multimotion_no_stairs import default_train_args


if __name__ == "__main__":
    run_task_stage(
        ROOT / "sensor_training" / "train_multimotion.py",
        stage="stage2",
        log_root=ROOT / "logs" / "multimotion" / "no_stairs_amp_single",
        task_defaults=default_train_args(),
    )
