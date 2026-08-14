#!/usr/bin/env python3
"""Stage 2 multimotion-layout flat-walk pretraining with randomization."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.two_stage_task import run_task_stage


if __name__ == "__main__":
    run_task_stage(
        ROOT / "scripts" / "train_multimotion_walk.py",
        stage="stage2",
        log_root=ROOT / "logs" / "sensor_policy" / "multimotion_walk_warmup",
    )
