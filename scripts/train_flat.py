#!/usr/bin/env python3
"""Train the independent flat-ground sensor-only AMP task."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.independent_terrain_task import default_train_args as _default_train_args
from scripts.independent_terrain_task import launch


def default_train_args() -> list[str]:
    return _default_train_args("flat")


def main() -> None:
    launch("flat")


if __name__ == "__main__":
    main()
