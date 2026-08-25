#!/usr/bin/env python3
"""Stage 2 independent up/down-stair training with domain randomization."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lytrobot_amp.training_defaults import (
    STAIR_STAGE2_STEP_DEPTH_M,
    STAIR_STAGE2_STEP_HEIGHT_M,
)
from scripts.two_stage_task import run_task_stage


if __name__ == "__main__":
    run_task_stage(
        ROOT / "scripts" / "train_stairs.py",
        stage="stage2",
        log_root=ROOT / "logs" / "stairs",
        task_defaults=[
            "--stair-step-height",
            f"{STAIR_STAGE2_STEP_HEIGHT_M:g}",
            "--stair-step-depth",
            f"{STAIR_STAGE2_STEP_DEPTH_M:g}",
        ],
    )
