from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lytrobot_amp.training_defaults import (
    NO_STAIRS_FAST_TARGET_SPEED_MPS,
    TRAINING_TARGET_SPEED_MPS,
)
from multimotion.retargeting.analyze_gait import analyze_reference
from lytrobot_amp.env import locomotion_task_reward_contract
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


def test_no_stairs_entrypoint_uses_terrain_aligned_references() -> None:
    args = default_train_args()
    assert [Path(value) for value in _values(args, "--motion-files")] == [
        MOTION_DIR / "walk_omniretarget.npz",
        MOTION_DIR / "rough_walk_omniretarget.npz",
        MOTION_DIR / "slope_up_omniretarget.npz",
        MOTION_DIR / "slope_down_omniretarget.npz",
    ]
    assert np.isclose(
        float(_values(args, "--target-velocity")[0]), TRAINING_TARGET_SPEED_MPS
    )
    assert _values(args, "--motion-names") == [
        "walk",
        "rough_walk",
        "slope_up",
        "slope_down",
    ]
    assert _values(args, "--motion-weights") == ["0.25"] * 4
    assert _values(args, "--terrain-types") == ["rough", "slope_up", "slope_down"]
    assert not any(arg.startswith("--stair-") for arg in args)
    assert "--no-skill-conditioning" in args
    assert "--serial-reference-state-init" in args
    assert _values(args, "--motion-loop") == ["true", "false", "false", "false"]
    assert _values(args, "--motion-reset-prob") == ["0.50"]
    assert "--serial-course-single-motion" not in args
    assert "--no-use-reference-velocity" in args
    assert _values(args, "--motion-transition-blend-steps") == ["0"]
    assert "--amp-task-only" in args
    # AMP constrains gait style while task reward keeps forward progress in
    # the PPO objective and prevents an in-place leg-swing local optimum.
    assert float(_values(args, "--task-reward-lerp")[0]) == 0.70
    assert float(_values(args, "--amp-reward-coef")[0]) > 0.0
    assert _values(args, "--terrain-course-random-start-prob") == ["0.7"]
    assert _values(args, "--warp-nconmax") == ["128"]
    assert _values(args, "--warp-njmax") == ["512"]


def test_amp_task_reward_rejects_standing_local_optimum() -> None:
    common = dict(
        target_velocity=TRAINING_TARGET_SPEED_MPS,
        upright=1.0,
        physical_height=1.0,
        support=1.0,
        forward_velocity_tracking_sigma=0.35,
        forward_velocity_reward_coef=3.0,
        forward_progress_reward_coef=1.75,
        forward_velocity_deficit_penalty_coef=3.0,
        forward_velocity_excess_penalty_coef=2.0,
        contact_reward_coef=0.35,
        foot_slip_penalty_coef=0.35,
    )
    standing = locomotion_task_reward_contract(forward_velocity=0.0, **common)
    target = locomotion_task_reward_contract(
        forward_velocity=TRAINING_TARGET_SPEED_MPS,
        **common,
    )
    assert standing < 0.0
    assert target > standing + 7.5


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
