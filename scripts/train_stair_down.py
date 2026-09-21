#!/usr/bin/env python3
"""训练固定 0.13 m 五级楼梯下楼动作。"""

from __future__ import annotations

try:  # package import (tests) vs direct ``python scripts/...`` execution
    from scripts.train_stairs import default_train_args as _default_train_args, main as _main
except ModuleNotFoundError:  # pragma: no cover - direct script path
    from train_stairs import default_train_args as _default_train_args, main as _main


def default_train_args() -> list[str]:
    return _default_train_args("down")


def main() -> None:
    _main("down")


if __name__ == "__main__":
    main()
