#!/usr/bin/env python3
"""Explicit alias for the 58D IMU+tendon+deployable-phase baseline."""
from pathlib import Path
import runpy


if __name__ == '__main__':
    runpy.run_path(
        str(Path(__file__).with_name('train_phase_stage1.py')),
        run_name='__main__',
    )
