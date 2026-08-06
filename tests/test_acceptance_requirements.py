from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from lytrobot_amp.benchmark import (
    AcceptanceResult,
    MAX_ROUGH_HEIGHT_DIFFERENCE_M,
    MAX_SLOPE_DEG,
    MAX_SPECIFIC_COT_J_PER_KG_M,
    MIN_WALKING_SPEED_MPS,
    ORIGINAL_RETARGETED_WALK_SPEED_MPS,
    REQUIRED_SCENARIOS,
    STAIR_ASSESSMENT_CASES,
    STAIR_HEIGHT_RANGE_M,
    STAIR_TREAD_RANGE_M,
    TRAINING_TARGET_SPEED_MPS,
    specific_cost_of_transport,
)
from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.terrain import (
    MoreTerrainCfg,
    StairTerrainCfg,
    course_segments,
    load_model,
    randomized_course_orders,
    serial_course_boxes,
    terrain_height_from_model,
)
from scripts.evaluate_acceptance import assessment_course_cfg


ROOT = Path(__file__).resolve().parents[1]


def test_acceptance_constants_match_contract() -> None:
    assert np.isclose(MIN_WALKING_SPEED_MPS, 5.0 / 3.6)
    assert np.isclose(ORIGINAL_RETARGETED_WALK_SPEED_MPS, 0.8622949262039437)
    assert TRAINING_TARGET_SPEED_MPS == ORIGINAL_RETARGETED_WALK_SPEED_MPS
    assert TRAINING_TARGET_SPEED_MPS < MIN_WALKING_SPEED_MPS
    assert MAX_SPECIFIC_COT_J_PER_KG_M == 0.8
    assert MAX_ROUGH_HEIGHT_DIFFERENCE_M == 0.05
    assert MAX_SLOPE_DEG == 6.0
    assert STAIR_HEIGHT_RANGE_M == (0.13, 0.20)
    assert STAIR_TREAD_RANGE_M == (0.22, 0.32)
    assert STAIR_ASSESSMENT_CASES == (
        (0.13, 0.22),
        (0.13, 0.32),
        (0.20, 0.22),
        (0.20, 0.32),
    )
    assert len(REQUIRED_SCENARIOS) == 6


def test_specific_cot_and_acceptance_result_are_fail_closed() -> None:
    assert specific_cost_of_transport(8.0, 10.0, 1.0) == 0.8
    assert np.isinf(specific_cost_of_transport(0.0, 10.0, 0.0))
    passed = AcceptanceResult(
        mean_speed_mps=MIN_WALKING_SPEED_MPS,
        flat_specific_cot_j_per_kg_m=MAX_SPECIFIC_COT_J_PER_KG_M,
        stable_scenarios=REQUIRED_SCENARIOS,
        course_successes=1,
    )
    assert passed.passed
    assert not AcceptanceResult(0.0, 1.0, (), 0).passed
    assert not AcceptanceResult(float("nan"), float("nan"), REQUIRED_SCENARIOS, 1).passed


def test_slope_pair_stays_adjacent_in_randomized_courses() -> None:
    terrain_types = ("rough", "slope_up", "slope_down", "stairs")
    orders = randomized_course_orders(terrain_types, 8, 42)
    assert all(
        order.index("slope_down") == order.index("slope_up") + 1
        for order in orders
    )


def test_assessment_course_geometry_matches_limits() -> None:
    order = ("rough", "slope_up", "slope_down", "stairs")
    more = MoreTerrainCfg(
        enabled=True,
        terrain_types=order,
        serial_course=True,
        course_orders=(order,),
        rough_height=MAX_ROUGH_HEIGHT_DIFFERENCE_M,
        slope_angle_deg=MAX_SLOPE_DEG,
    )
    stairs = StairTerrainCfg(
        enabled=True,
        step_height=STAIR_HEIGHT_RANGE_M[1],
        step_depth=STAIR_TREAD_RANGE_M[0],
        step_count=3,
    )
    model = load_model(ROOT / "robot" / "MJCFmodelposition.xml", stairs, more)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    segments = {segment.kind: segment for segment in course_segments(more, 0)}
    y = more.course_lane_ys[0]

    for kind, expected_sign in (("slope_up", 1.0), ("slope_down", -1.0)):
        segment = segments[kind]
        x = np.asarray([segment.start_x + 0.02, segment.end_x - 0.02])
        heights = terrain_height_from_model(model, data, x, np.full(2, y))
        measured_angle = np.rad2deg(
            np.arctan2(heights[1] - heights[0], x[1] - x[0])
        )
        assert np.isclose(measured_angle, expected_sign * MAX_SLOPE_DEG, atol=0.1)

    rough = segments["rough"]
    rough_x = np.linspace(rough.start_x + 0.05, rough.end_x - 0.05, 64)
    rough_height = terrain_height_from_model(model, data, rough_x, np.full(64, y))
    assert np.ptp(rough_height) <= MAX_ROUGH_HEIGHT_DIFFERENCE_M + 1.0e-6

    boxes = serial_course_boxes(more, stairs)
    stair_steps = [
        (name, center, half_size)
        for name, center, half_size in boxes
        if "stair_up_" in name or "stair_down_" in name
    ]
    assert len(stair_steps) == 6
    assert all(
        np.isclose(2.0 * half_size[0], STAIR_TREAD_RANGE_M[0])
        for _name, _center, half_size in stair_steps
    )
    up_tops = [
        center[2] + half_size[2]
        for name, center, half_size in stair_steps
        if "stair_up_" in name
    ]
    assert np.allclose(np.diff(up_tops), STAIR_HEIGHT_RANGE_M[1])


def test_evaluator_forces_worst_case_contract_geometry() -> None:
    cfg = assessment_course_cfg(LytRobotEnvCfg())
    assert cfg.terrain_types == ("rough", "slope_up", "slope_down", "stairs")
    assert cfg.rough_terrain_height == MAX_ROUGH_HEIGHT_DIFFERENCE_M
    assert cfg.slope_angle_deg == MAX_SLOPE_DEG
    assert cfg.stair_step_height == STAIR_HEIGHT_RANGE_M[1]
    assert cfg.stair_step_depth == STAIR_TREAD_RANGE_M[0]
    assert cfg.serial_terrain_course

    for height, tread in STAIR_ASSESSMENT_CASES:
        case_cfg = assessment_course_cfg(
            LytRobotEnvCfg(),
            stair_height=height,
            stair_tread=tread,
        )
        assert case_cfg.stair_step_height == height
        assert case_cfg.stair_step_depth == tread


def test_all_stair_assessment_boundaries_build_exact_geometry() -> None:
    order = ("rough", "slope_up", "slope_down", "stairs")
    more = MoreTerrainCfg(
        enabled=True,
        terrain_types=order,
        serial_course=True,
        course_orders=(order,),
    )
    for height, tread in STAIR_ASSESSMENT_CASES:
        stairs = StairTerrainCfg(
            enabled=True,
            step_height=height,
            step_depth=tread,
            step_count=3,
        )
        steps = [
            (name, center, half_size)
            for name, center, half_size in serial_course_boxes(more, stairs)
            if "stair_up_" in name or "stair_down_" in name
        ]
        assert len(steps) == 6
        assert all(np.isclose(2.0 * half_size[0], tread) for _, _, half_size in steps)
        up_tops = [
            center[2] + half_size[2]
            for name, center, half_size in steps
            if "stair_up_" in name
        ]
        np.testing.assert_allclose(np.diff(up_tops), height)


def test_cpu_step_reports_finite_mechanical_energy_and_cot() -> None:
    env = LytRobotAmpEnv(
        LytRobotEnvCfg(
            model_xml=str(ROOT / "robot" / "MJCFmodelposition.xml"),
            motion_file=str(ROOT / "robot" / "reference_data" / "walk1_mjcf.npz"),
            num_envs=1,
            reset_noise_scale=0.0,
            motion_reset_prob=1.0,
            stand_start_prob=0.0,
            domain_randomization=False,
            episode_timeout_jitter_steps=0,
            specific_cot_limit=MAX_SPECIFIC_COT_J_PER_KG_M,
            specific_cot_penalty_coef=2.0,
        )
    )
    try:
        _observation, reward, _done, info = env.step(
            np.zeros((1, env.action_dim), dtype=np.float32)
        )
        for key in (
            "actuator_power",
            "mechanical_energy",
            "specific_cot",
            "specific_cot_excess",
        ):
            assert key in info
            assert np.isfinite(info[key]).all()
            assert np.all(info[key] >= 0.0)
        assert np.isfinite(reward).all()
    finally:
        env.close()
