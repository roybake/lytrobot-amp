#!/usr/bin/env python3
"""Recommended Teacher → sensor Stage1 → sensor Stage2 DR training entry."""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from sensor_training import train_serial_distillation


def main():
    # This is a thin preset over the existing serial trainer, not a new trainer.
    argv = list(sys.argv)
    for old, new in (('--teacher-checkpoint', '--teacher-stage1-checkpoint'),
                     ('--student-checkpoint', '--student-stage1-checkpoint')):
        argv = [new+value[len(old):] if value == old or value.startswith(old+'=') else value
                for value in argv]
    if '--final-sensor-gait' not in argv:
        argv.append('--final-sensor-gait')
    sys.argv[:] = argv
    train_serial_distillation.main()


if __name__ == '__main__':
    main()
