"""Behavioral checks for phase-free contact order and mirrored stepping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from lytrobot_amp.gait_constraints import sensor_gait_scores, sensor_gait_scores_numpy
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv


def _state(torch_backend: bool):
    xp = torch if torch_backend else np
    zeros = lambda dtype: xp.zeros((1, 2), dtype=dtype)
    integer = torch.long if torch_backend else np.int32
    boolean = torch.bool if torch_backend else bool
    floating = torch.float32 if torch_backend else np.float32
    return SimpleNamespace(
        dt=0.01,
        sensor_last_foot_contact=zeros(boolean),
        sensor_foot_roll_stage=zeros(integer),
        sensor_toe_only_steps=zeros(integer),
        sensor_last_touchdown_side=xp.full((1,), -1, dtype=integer),
        sensor_liftoff_x=zeros(floating),
        sensor_had_liftoff=zeros(boolean),
        sensor_stride_ema=zeros(floating),
        sensor_stride_seen=zeros(boolean),
        sensor_contact_duty_ema=zeros(floating),
        sensor_single_support_ema=xp.zeros((1,), dtype=floating),
        sensor_time_since_touchdown=xp.zeros((1,), dtype=floating),
        sensor_step_interval_ema=xp.zeros((1,), dtype=floating),
        sensor_touchdown_seen=xp.zeros((1,), dtype=boolean),
        sensor_cadence_seen=xp.zeros((1,), dtype=boolean),
        sensor_swing_steps=zeros(integer),
    )


def _step(state, zones, foot_x=(0.0, 0.0), *, torch_backend=True):
    if torch_backend:
        return sensor_gait_scores(
            state,
            torch.tensor([zones], dtype=torch.bool),
            torch.tensor([foot_x], dtype=torch.float32),
        )
    return sensor_gait_scores_numpy(
        state, 0, np.asarray(zones, dtype=bool), np.asarray(foot_x, dtype=np.float32)
    )


def test_toe_first_is_worse_than_ordered_heel_sole_toe_roll():
    toe_first = _state(True)
    bad = _step(toe_first, [[0, 0, 1], [0, 0, 0]])
    assert float(bad["sensor_roll_violation"][0]) > 0.0
    assert float(bad["sensor_tiptoe"][0]) > 0.0

    ordered = _state(True)
    _step(ordered, [[1, 0, 0], [0, 0, 0]])
    _step(ordered, [[1, 1, 0], [0, 0, 0]])
    good = _step(ordered, [[0, 0, 1], [0, 0, 0]])
    assert int(ordered.sensor_foot_roll_stage[0, 0]) == 3
    assert float(good["sensor_roll_violation"][0]) == 0.0
    assert float(good["sensor_tiptoe"][0]) == 0.0
    assert float(good["sensor_foot_roll"][0]) > float(bad["sensor_foot_roll"][0])
    for _ in range(16):
        good = _step(ordered, [[0, 0, 1], [0, 0, 0]])
    assert float(good["sensor_tiptoe"][0]) > 0.0


def test_synchronous_hop_and_touchdown_are_penalized():
    state = _state(True)
    _step(state, [[1, 0, 0], [1, 0, 0]])
    flight = _step(state, [[0, 0, 0], [0, 0, 0]])
    assert float(flight["sensor_hop"][0]) == 1.0
    simultaneous = _step(state, [[1, 0, 0], [1, 0, 0]])
    assert float(simultaneous["sensor_alternation"][0]) < 0.6


def test_equal_left_right_strides_outscore_unequal_strides():
    def score(right_stride: float):
        state = _state(True)
        _step(state, [[1, 0, 0], [1, 0, 0]])
        for _ in range(5):
            _step(state, [[0, 0, 0], [0, 0, 0]])
        _step(state, [[1, 0, 0], [0, 0, 0]], (0.2, 0.0))
        for _ in range(5):
            _step(state, [[1, 0, 0], [0, 0, 0]], (0.2, 0.0))
        result = _step(state, [[1, 0, 0], [1, 0, 0]], (0.2, right_stride))
        return float(result["sensor_mirror"][0])

    assert score(0.2) > score(0.5)


def test_stride_score_requires_both_sides_and_rewards_length():
    zones = torch.tensor([[[1, 0, 0], [1, 0, 0]]], dtype=torch.bool)
    foot_x = torch.zeros((1, 2))
    state = _state(True)
    state.sensor_stride_ema[0] = torch.tensor([0.17, 0.17])
    neutral = sensor_gait_scores(state, zones, foot_x)
    assert float(neutral["sensor_stride_length"][0]) == 0.5
    state.sensor_stride_seen[0] = True
    short = sensor_gait_scores(state, zones, foot_x)
    state.sensor_stride_ema[0] = torch.tensor([0.45, 0.45])
    full = sensor_gait_scores(state, zones, foot_x)
    assert float(short["sensor_stride_length"][0]) < 0.2
    assert float(full["sensor_stride_length"][0]) == 1.0


def test_reference_cadence_outscores_rapid_shuffling():
    def cadence_score(wait_steps: int) -> float:
        state = _state(True)
        for _ in range(5):
            _step(state, [[0, 0, 0], [0, 0, 0]])
        _step(state, [[1, 0, 0], [0, 0, 0]])
        for _ in range(wait_steps - 1):
            _step(state, [[0, 0, 0], [0, 0, 0]])
        result = _step(state, [[0, 0, 0], [1, 0, 0]])
        return float(result["sensor_step_cadence"][0])

    assert cadence_score(75) > 0.99
    assert cadence_score(15) < 0.10


def test_swing_clearance_tracks_reference_height() -> None:
    state = _state(True)
    zones = torch.tensor([[[1, 0, 0], [0, 0, 0]]], dtype=torch.bool)
    foot_x = torch.zeros((1, 2))
    good = sensor_gait_scores(
        state, zones, foot_x, foot_height=torch.tensor([[0.0, 0.03]])
    )
    low = sensor_gait_scores(
        _state(True), zones, foot_x, foot_height=torch.tensor([[0.0, 0.0]])
    )
    assert float(good["sensor_swing_clearance"][0]) > 0.99
    assert float(low["sensor_swing_clearance"][0]) < 0.4


def test_lateral_foot_positions_are_mirrored_about_pelvis():
    zones = torch.tensor([[[1, 0, 0], [1, 0, 0]]], dtype=torch.bool)
    foot_x = torch.zeros((1, 2))
    centered = sensor_gait_scores(_state(True), zones, foot_x, torch.tensor([0.0]))
    skewed = sensor_gait_scores(_state(True), zones, foot_x, torch.tensor([0.16]))
    assert float(centered["sensor_mirror"][0]) > float(skewed["sensor_mirror"][0])


def test_stance_sliding_is_penalized_and_double_support_is_not_a_step():
    zones = torch.tensor([[[1, 0, 0], [1, 0, 0]]], dtype=torch.bool)
    foot_x = torch.zeros((1, 2))
    static = sensor_gait_scores(
        _state(True), zones, foot_x, stance_speed=torch.tensor([[0.0, 0.0]])
    )
    sliding = sensor_gait_scores(
        _state(True), zones, foot_x, stance_speed=torch.tensor([[0.8, 0.8]])
    )
    assert float(static["sensor_stance_slip"][0]) == 0.0
    assert float(sliding["sensor_stance_slip"][0]) > 1.0
    assert float(sliding["sensor_alternation"][0]) <= 0.25


def test_synchronous_forward_motion_is_worse_than_opposite_leg_motion():
    zones = torch.tensor([[[1, 0, 0], [1, 0, 0]]], dtype=torch.bool)
    foot_x = torch.zeros((1, 2))
    together = sensor_gait_scores(
        _state(True), zones, foot_x, foot_forward_speed=torch.tensor([[0.8, 0.8]])
    )
    opposed = sensor_gait_scores(
        _state(True), zones, foot_x, foot_forward_speed=torch.tensor([[0.8, -0.8]])
    )
    assert float(together["sensor_synchronous_forward"][0]) > 1.0
    assert float(opposed["sensor_synchronous_forward"][0]) == 0.0
    assert float(opposed["sensor_mirror"][0]) > float(together["sensor_mirror"][0])


def test_cpu_and_warp_gait_scores_agree_on_same_contact_sequence():
    cpu, warp = _state(False), _state(True)
    sequence = (
        ([[1, 0, 0], [1, 0, 0]], (0.0, 0.0)),
        ([[0, 0, 0], [0, 0, 0]], (0.0, 0.0)),
        ([[1, 0, 0], [0, 0, 0]], (0.2, 0.0)),
        ([[1, 1, 0], [1, 0, 0]], (0.2, 0.2)),
        ([[0, 0, 1], [1, 1, 0]], (0.2, 0.2)),
    )
    for zones, foot_x in sequence:
        a = _step(cpu, zones, foot_x, torch_backend=False)
        b = _step(warp, zones, foot_x)
        for key in ("sensor_mirror", "sensor_stride_length", "sensor_foot_roll", "sensor_roll_violation",
                    "sensor_alternation", "sensor_hop", "sensor_tiptoe"):
            np.testing.assert_allclose(a[key], float(b[key][0]), atol=1e-6)


def test_warp_contact_classifier_ignores_stale_slots_after_nacon():
    env = object.__new__(LytRobotWarpAmpEnv)
    env.device = torch.device("cpu")
    env.num_envs = 1
    env.contact_geom_id_values = (1, 2, 3, 4)
    env.terrain_geom_ids = torch.tensor([0])
    env.contact_geom = torch.tensor([[0, 1], [0, 3]])
    env.contact_pos = torch.zeros((2, 3))
    env.contact_worldid = torch.zeros(2, dtype=torch.long)
    env.contact_dim = torch.tensor([3, 3], dtype=torch.long)
    env.contact_nacon = torch.tensor([1], dtype=torch.long)
    env.contact_slot_ids = torch.arange(2)
    env.sensor_foot_body_ids = (1, 2)
    env.xpos = torch.zeros((1, 3, 3))
    env.xmat = torch.eye(3).expand(1, 3, 3, 3).clone()

    assert env._foot_geom_contacts().tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert env._sensor_foot_roll_contacts().tolist() == [[[True, False, False], [False, False, False]]]
    env.contact_nacon[0] = 0
    assert not env._foot_geom_contacts().any()
    assert not env._sensor_foot_roll_contacts().any()
    env.contact_geom[1, 1] = 2
    env.contact_nacon[0] = 2
    assert env._sensor_foot_roll_contacts().tolist() == [
        [[True, True, True], [False, False, False]]
    ]
