#!/usr/bin/env python3
"""Stage 2 independent uphill training with randomization."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.independent_terrain_stage import run


if __name__ == "__main__":
    run("slope_up", "stage2")
