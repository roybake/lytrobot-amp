#!/usr/bin/env python3
"""Phase 1: distill a natural stage1 gait into the sensor-only Actor."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensor_training.distillation_entrypoints import run_stage1


if __name__ == "__main__":
    run_stage1()
