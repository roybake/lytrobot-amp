from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import sys

from scripts import train
from scripts.train_no_stairs_old import default_train_args


ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = ROOT / "robot" / "reference_data" / "multimotion"


def _values(args: list[str], option: str) -> list[str]:
    start = args.index(option) + 1
    end = start
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return args[start:end]


def test_no_stairs_old_preserves_the_reference_conditioned_task_without_stairs() -> None:
    args = default_train_args()

    assert _values(args, "--training-task") == ["no_stairs_old"]
    assert [Path(value) for value in _values(args, "--motion-files")] == [
        MOTION_DIR / "walk_omniretarget.npz",
        MOTION_DIR / "rough_walk_omniretarget.npz",
    ]
    assert _values(args, "--motion-names") == ["walk", "rough_walk"]
    assert _values(args, "--motion-weights") == ["0.50", "0.50"]
    assert _values(args, "--motion-loop") == ["true", "false"]
    assert "--skill-conditioning" in args
    assert set(_values(args, "--terrain-types")) == {"rough", "slope_up", "slope_down"}
    assert "stairs" not in args
    assert not any(value.startswith("--stair-") for value in args)
    assert not any("stair_" in value for value in _values(args, "--motion-files"))
    assert _values(args, "--log-group") == ["no_stairs_old"]

    with patch.object(sys, "argv", ["train.py", *args]):
        parsed = train.parse_args()
    assert parsed.training_task == "no_stairs_old"
    assert parsed.skill_conditioning is True
    assert parsed.use_reference_velocity is True
