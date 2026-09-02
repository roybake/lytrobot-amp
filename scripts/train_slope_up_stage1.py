#!/usr/bin/env python3
"""Stage 1 independent uphill training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.independent_terrain_stage import run


if __name__ == "__main__":
    run("slope_up", "stage1")
