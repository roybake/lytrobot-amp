from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lytrobot_amp.training_defaults import (
    NO_STAIRS_FAST_TARGET_SPEED_MPS,
    TRAINING_TARGET_SPEED_MPS,
)
from multimotion.retargeting.analyze_gait import analyze_reference
from scripts.train_multimotion_no_stairs import default_train_args


ROOT = Path(__file__).resolve().parents[1]
FAST_DIR = ROOT / "robot" / "reference_data" / "no_stairs_fast"
MOTION_DIR = ROOT / "robot" / "reference_data" / "multimotion"


def _values(args: list[str], option: str) -> list[str]:
    start = args.index(option) + 1
    end = start
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return args[start:end]


def test_no_stairs_entrypoint_prioritizes_stable_sensor_only_completion() -> None:
    args = default_train_args()
    assert [Path(value) for value in _values(args, "--motion-files")] == [
        MOTION_DIR / "walk_omniretarget.npz",
        MOTION_DIR / "rough_walk_omniretarget.npz",
    ]
    assert np.isclose(
        float(_values(args, "--target-velocity")[0]), TRAINING_TARGET_SPEED_MPS
    )
    assert _values(args, "--terrain-types") == ["flat", "rough", "slope_up", "slope_down"]
    assert not any(arg.startswith("--stair-") for arg in args)
    assert "--no-skill-conditioning" in args
    assert "--serial-reference-state-init" in args
    assert _values(args, "--motion-loop") == ["true", "false"]
    assert _values(args, "--motion-reset-prob") == ["0.50"]
    assert "--use-reference-velocity" in args
    assert _values(args, "--terrain-course-random-start-prob") == ["0.8"]
    assert _values(args, "--warp-nconmax") == ["128"]
    assert _values(args, "--warp-njmax") == ["512"]


@pytest.mark.skipif(
    not (FAST_DIR / "walk_omniretarget.npz").exists(),
    reason="fast reference must be generated with OmniRetarget before reference validation",
)
def test_fast_walk_meets_small_step_profile() -> None:
    gait = analyze_reference(FAST_DIR / "walk_omniretarget.npz")
    assert gait["qpos_finite"] and gait["qvel_finite"]
    assert abs(gait["mean_forward_speed_mps"] - NO_STAIRS_FAST_TARGET_SPEED_MPS) <= 0.02
    assert abs(gait["mean_step_length_m"] - 0.48) <= 0.02
    assert 170.0 <= gait["cadence_steps_per_min"] <= 180.0


@pytest.mark.skipif(
    not (FAST_DIR / "rough_walk_omniretarget.npz").exists(),
    reason="fast rough reference must be generated from the fast walk",
)
def test_fast_rough_reference_is_measurable() -> None:
    gait = analyze_reference(FAST_DIR / "rough_walk_omniretarget.npz")
    assert gait["qpos_finite"] and gait["qvel_finite"]
    assert abs(gait["mean_forward_speed_mps"] - NO_STAIRS_FAST_TARGET_SPEED_MPS) <= 0.02
