#!/usr/bin/env python3
"""Ordinary-walking Stage 2 under the stair-free task identity/log tree."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_multimotion_no_stairs import default_train_args
from scripts.train_stage2 import run_stage2


def main() -> None:
    sys.argv = [
        str(ROOT / "scripts" / "train_stage2.py"),
        *default_train_args(),
        *sys.argv[1:],
    ]
    run_stage2()


if __name__ == "__main__":
    main()
