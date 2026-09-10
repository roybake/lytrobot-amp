from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

import lytrobot_amp.env_warp as env_warp_module

from lytrobot_amp.env import (
    LytRobotAmpEnv,
    LytRobotEnvCfg,
    _forward_progress_guard as numpy_forward_progress_guard,
    _stair_root_height_transform,
)
from lytrobot_amp.env_warp import (
    LytRobotWarpAmpEnv,
    _forward_progress_guard as torch_forward_progress_guard,
)
from lytrobot_amp.motion import MotionLibrary, _repair_loop_root_translation_velocity
from lytrobot_amp.terrain import (
    MORE_TERRAIN_TYPES,
    MoreTerrainCfg,
    SERIAL_ROUGH_TRANSITION_LENGTH,
    StairTerrainCfg,
    course_motion_sequence,
    course_stage_end_positions,
    course_segments,
    load_model,
    load_multi_robot_model,
    randomized_course_orders,
    sample_course_start_stages,
    serial_course_boxes,
    terrain_height_from_model,
)
from scripts.play import playback_pelvis_body_ids, sync_multi_robot_state


ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = ROOT / "robot" / "reference_data" / "multimotion"
NAMES = (
    "walk",
    "rough_walk",
    "slope_up",
    "slope_down",
    "pit_walk",
    "gap_cross",
    "stair_up",
    "stair_down",
)


def motion_paths() -> tuple[str, ...]:
    return tuple(str(MOTION_DIR / f"{name}_omniretarget.npz") for name in NAMES)


def course_cfg(variants: int = 3) -> tuple[MoreTerrainCfg, StairTerrainCfg]:
    orders = randomized_course_orders(MORE_TERRAIN_TYPES, variants, 42)
    return (
        MoreTerrainCfg(
            enabled=True,
            terrain_types=MORE_TERRAIN_TYPES,
            serial_course=True,
            course_orders=orders,
        ),
        StairTerrainCfg(
            enabled=True,
            step_height=0.13,
            step_depth=0.58,
            step_count=3,
        ),
    )


def test_course_variants_are_reproducible_full_random_permutations() -> None:
    orders = randomized_course_orders(MORE_TERRAIN_TYPES, 8, 123)
    assert orders == randomized_course_orders(MORE_TERRAIN_TYPES, 8, 123)
    assert len(set(orders)) == 8
    assert all(set(order) == set(MORE_TERRAIN_TYPES) for order in orders)
    assert len({order[0] for order in orders}) > 1


def test_straight_course_excludes_turns_and_uses_geometry_boundaries() -> None:
    more, _stairs = course_cfg(variants=3)
    for variant, order in enumerate(more.normalized_course_orders):
        sequence = course_motion_sequence(order)
        boundaries = course_stage_end_positions(more, variant)
        assert sequence[0] == "walk"
        assert "turn_left" not in sequence
        assert "turn_right" not in sequence
        assert len(boundaries) == len(sequence)
        assert boundaries[0] == more.course_run_in
        assert boundaries[-1] == course_segments(more, variant)[-1].end_x


def test_serial_stair_descent_begins_at_ascent_descent_stage_split() -> None:
    more, _ = course_cfg(variants=1)
    stairs = StairTerrainCfg(
        enabled=True,
        step_height=0.20,
        step_depth=0.22,
        step_count=3,
        start_x=-0.95,
    )
    stair_segment = next(
        segment for segment in course_segments(more, 0) if segment.kind == "stairs"
    )
    boxes = serial_course_boxes(more, stairs)
    first_descent = next(
        (center, size)
        for name, center, size in boxes
        if name.endswith("stair_down_00")
    )
    descent_start = float(first_descent[0][0] - first_descent[1][0])
    assert descent_start == pytest.approx(stair_segment.center_x)


def test_course_curriculum_starts_preserve_full_route_population() -> None:
    rng = np.random.default_rng(7)
    stages = sample_course_start_stages(
        rng,
        batch_size=4096,
        stage_count=8,
        random_start_prob=0.80,
    )
    assert np.any(stages == 0)
    assert set(np.unique(stages)) == set(range(8))
    assert 0.15 < float(np.mean(stages == 0)) < 0.25


def test_course_random_start_probability_setters_validate_bounds() -> None:
    cpu_env = object.__new__(LytRobotAmpEnv)
    cpu_env.cfg = LytRobotEnvCfg()
    warp_env = object.__new__(LytRobotWarpAmpEnv)
    warp_env.cfg = LytRobotEnvCfg()

    for env in (cpu_env, warp_env):
        env.set_terrain_course_random_start_prob(0.4)
        assert env.cfg.terrain_course_random_start_prob == pytest.approx(0.4)
        with pytest.raises(ValueError):
            env.set_terrain_course_random_start_prob(-0.01)
        with pytest.raises(ValueError):
            env.set_terrain_course_random_start_prob(1.01)


def test_two_reference_course_reuses_rough_expert_for_slope_stages() -> None:
    names = ("walk", "rough_walk")
    env = LytRobotAmpEnv(
        LytRobotEnvCfg(
            motion_files=tuple(
                str(MOTION_DIR / f"{name}_omniretarget.npz") for name in names
            ),
            motion_names=names,
            motion_loop=(True, False),
            terrain_types=("rough", "slope_up", "slope_down"),
            serial_terrain_course=True,
            terrain_course_variants=1,
            num_envs=1,
            domain_randomization=False,
        )
    )
    try:
        sequence = course_motion_sequence(env.more_terrain_cfg.normalized_course_orders[0])
        expected = [0 if name == "walk" else 1 for name in sequence]
        assert env.course_motion_sequences[0].tolist() == expected
    finally:
        env.close()


def test_warp_partial_reset_uses_authoritative_per_world_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = object.__new__(LytRobotWarpAmpEnv)
    env.num_envs = 4
    env.wp_device = object()
    env.wp_model = object()
    env.wp_data = object()
    env._warp_reset_mask = torch.zeros(4, dtype=torch.bool)
    calls: list[tuple[object, object, torch.Tensor]] = []

    monkeypatch.setattr(
        env_warp_module.wp,
        "ScopedDevice",
        lambda _device: nullcontext(),
    )
    monkeypatch.setattr(
        env_warp_module.wp,
        "from_torch",
        lambda tensor: tensor.clone(),
    )
    monkeypatch.setattr(
        env_warp_module.mjwarp,
        "reset_data",
        lambda model, data, *, reset: calls.append((model, data, reset)),
    )

    env._reset_warp_worlds(torch.tensor([1, 3], dtype=torch.long))

    assert len(calls) == 1
    assert calls[0][0] is env.wp_model
    assert calls[0][1] is env.wp_data
    torch.testing.assert_close(
        calls[0][2], torch.tensor([False, True, False, True])
    )


def test_stair_reference_height_matches_compiled_hard_stair_rise() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "robot" / "MJCFmodelposition.xml")
    )
    library = MotionLibrary(
        motion_paths(),
        model,
        motion_names=NAMES,
        motion_loop=(True, False, False, False, False, False, False, False),
    )
    for name in ("stair_up", "stair_down"):
        motion = library.motions[NAMES.index(name)]
        anchor, scale = _stair_root_height_transform(
            motion,
            step_height=0.20,
            step_count=3,
            descending=name == "stair_down",
        )
        transformed = anchor + (motion.reference_root_height - anchor) * scale
        assert abs(float(transformed[-1] - transformed[0])) == pytest.approx(0.60)


def test_hard_stair_down_start_is_not_misclassified_as_too_tall() -> None:
    env = LytRobotAmpEnv(
        LytRobotEnvCfg(
            motion_files=motion_paths(),
            motion_names=NAMES,
            motion_loop=(True, False, False, False, False, False, False, False),
            terrain_types=MORE_TERRAIN_TYPES,
            serial_terrain_course=True,
            terrain_course_variants=8,
            terrain_course_random_start_prob=1.0,
            num_envs=32,
            stair_step_height=0.20,
            stair_step_depth=0.22,
            stair_step_count=3,
            stair_down_offset=0.25,
            stand_start_prob=0.0,
            motion_reset_prob=1.0,
            reset_noise_scale=0.0,
            domain_randomization=False,
            episode_timeout_jitter_steps=0,
        )
    )
    try:
        stair_down_id = NAMES.index("stair_down")
        stair_down_envs = np.flatnonzero(env.motion_ids == stair_down_id)
        assert len(stair_down_envs) > 0
        root_height = np.asarray(
            [env.datas[env_id].qpos[2] for env_id in stair_down_envs]
        )
        assert np.all(root_height > env.cfg.max_height)

        amp_observation = env.amp_observe()
        _, _, done, info = env.step(
            np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
        )
        assert np.isfinite(amp_observation).all()
        np.testing.assert_array_equal(done[stair_down_envs], 0.0)
        np.testing.assert_array_equal(info["termination"][stair_down_envs], 0.0)
    finally:
        env.close()


def test_forward_progress_guard_terminates_only_slow_completed_windows() -> None:
    position_x = np.asarray([0.105, 0.12, 0.0], dtype=np.float32)
    episode_step = np.asarray([15, 15, 9], dtype=np.int32)
    anchor_x = np.asarray([0.10, 0.10, 0.0], dtype=np.float32)
    anchor_step = np.asarray([5, 5, 5], dtype=np.int32)
    kwargs = {
        "dt": 0.02,
        "min_speed": 0.05,
        "grace_steps": 5,
        "window_steps": 10,
    }

    speed, stalled, next_x, next_step = numpy_forward_progress_guard(
        position_x, episode_step, anchor_x, anchor_step, **kwargs
    )
    np.testing.assert_allclose(speed, [0.025, 0.10, 0.0], atol=1.0e-6)
    np.testing.assert_array_equal(stalled, [True, False, False])
    np.testing.assert_allclose(next_x, [0.105, 0.12, 0.0])
    np.testing.assert_array_equal(next_step, [15, 15, 5])

    torch_result = torch_forward_progress_guard(
        torch.as_tensor(position_x),
        torch.as_tensor(episode_step),
        torch.as_tensor(anchor_x),
        torch.as_tensor(anchor_step),
        **kwargs,
    )
    torch.testing.assert_close(torch_result[0], torch.as_tensor(speed))
    torch.testing.assert_close(torch_result[1], torch.as_tensor(stalled))
    torch.testing.assert_close(torch_result[2], torch.as_tensor(next_x))
    torch.testing.assert_close(torch_result[3], torch.as_tensor(next_step))


def test_warp_motion_transition_blends_and_counts_down() -> None:
    env = object.__new__(LytRobotWarpAmpEnv)
    env.device = torch.device("cpu")
    env.cfg = LytRobotEnvCfg(motion_transition_blend_steps=4)
    env.motion_transition_from_ctrl = torch.zeros((2, 14), dtype=torch.float32)
    env.last_reference_ctrl = torch.zeros((2, 14), dtype=torch.float32)
    env.motion_transition_steps_left = torch.tensor([4, 0], dtype=torch.long)
    target = torch.ones((2, 14), dtype=torch.float32)

    transitioned = env._transition_reference_ctrl(target)

    torch.testing.assert_close(transitioned[0], torch.full((14,), 0.25))
    torch.testing.assert_close(transitioned[1], torch.ones(14))
    torch.testing.assert_close(
        env.motion_transition_steps_left, torch.tensor([3, 0], dtype=torch.long)
    )
    torch.testing.assert_close(env.last_reference_ctrl, transitioned)


def test_looping_walk_reference_has_no_backwards_seam_impulse() -> None:
    path = MOTION_DIR / "walk_omniretarget.npz"
    with np.load(path) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float32)
        qvel = np.asarray(archive["qvel"], dtype=np.float32)
        frequency = float(np.asarray(archive["frequency"]).item())

    # The checked-in generated artifact must also be correct, while the runtime
    # repair below remains a compatibility guard for older local references.
    assert qvel[0, 0] > 0.0
    assert qvel[-1, 0] > 0.0
    repaired = _repair_loop_root_translation_velocity(qpos, qvel, frequency)
    assert repaired[0, 0] > 0.0
    assert repaired[-1, 0] > 0.0
    assert np.max(np.abs(repaired[:, :2])) < 3.0


def test_each_lane_contains_all_terrains_in_its_own_order() -> None:
    more, stairs = course_cfg()
    model = load_model(ROOT / "robot" / "MJCFmodelposition.xml", stairs, more)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    course_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
        and name.startswith("course_")
    ]
    np.testing.assert_allclose(
        model.geom_rgba[course_geom_ids],
        np.tile([0.5, 0.5, 0.5, 1.0], (len(course_geom_ids), 1)),
    )

    for variant, order in enumerate(more.normalized_course_orders):
        segments = course_segments(more, variant)
        assert tuple(segment.kind for segment in segments) == order
        lane_y = more.course_lane_ys[variant]
        heights = terrain_height_from_model(
            model,
            data,
            np.asarray([segment.center_x for segment in segments]),
            np.full(len(segments), lane_y),
        )
        assert np.all(heights >= more.course_base_height - 1.0e-6)


def test_serial_rough_heightfield_is_flush_with_flat_course_seams() -> None:
    """The nonzero source-map edge must not form a step onto rough terrain."""

    more = MoreTerrainCfg(
        enabled=True,
        terrain_types=("rough",),
        serial_course=True,
        course_orders=(("rough",),),
    )
    model = load_model(ROOT / "robot" / "MJCFmodelposition.xml", None, more)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    segment = course_segments(more, 0)[0]

    seam_x = np.asarray(
        [
            segment.start_x - 1.0e-3,
            segment.start_x,
            segment.end_x,
            segment.end_x + 1.0e-3,
        ]
    )
    seam_heights = terrain_height_from_model(
        model, data, seam_x, np.zeros_like(seam_x)
    )
    np.testing.assert_allclose(seam_heights, more.course_base_height, atol=1.0e-6)

    # The first part of the rough field can rise only through the smooth
    # envelope; it cannot retain the old full-height edge protrusion.
    probe_x = segment.start_x + 0.5 * SERIAL_ROUGH_TRANSITION_LENGTH
    probe_height = terrain_height_from_model(model, data, probe_x, 0.0)
    assert more.course_base_height <= probe_height < (
        more.course_base_height + more.rough_height
    )


def test_three_robot_display_model_has_three_independent_robot_states() -> None:
    more, stairs = course_cfg()
    model = load_multi_robot_model(
        ROOT / "robot" / "MJCFmodelposition.xml", 3, stairs, more
    )
    assert model.nq == 3 * 23
    assert len(playback_pelvis_body_ids(model)) == 3
    active_lights = np.flatnonzero(model.light_active)
    np.testing.assert_array_equal(active_lights, [0])
    np.testing.assert_allclose(model.light_diffuse[active_lights[0]], [0.7, 0.7, 0.7])


def test_environment_assigns_three_robots_to_three_randomized_courses() -> None:
    env = LytRobotAmpEnv(
        LytRobotEnvCfg(
            motion_files=motion_paths(),
            motion_names=NAMES,
            motion_loop=(True, False, False, False, False, False, False, False),
            terrain_types=MORE_TERRAIN_TYPES,
            serial_terrain_course=True,
            terrain_course_variants=3,
            terrain_course_random_assignment=False,
            num_envs=3,
            stair_step_height=0.13,
            stair_step_depth=0.58,
            stair_step_count=3,
            stand_start_prob=0.0,
            motion_reset_prob=1.0,
            domain_randomization=False,
            episode_timeout_jitter_steps=0,
            terminate_incomplete_motion_end=True,
            motion_transition_blend_steps=4,
        )
    )
    try:
        np.testing.assert_array_equal(env.course_variant_ids, [0, 1, 2])
        np.testing.assert_allclose(
            [data.qpos[1] for data in env.datas], env.more_terrain_cfg.course_lane_ys
        )
        np.testing.assert_allclose(
            [data.qvel[:6] for data in env.datas],
            np.zeros((env.num_envs, 6)),
            atol=1.0e-7,
        )
        walk = env.motion.motions[0]
        assert walk.reference_root_lin_vel[0, 0] > 0.0
        assert np.max(np.abs(walk.reference_root_lin_vel[:, :2])) < 3.0
        for variant, order in enumerate(env.more_terrain_cfg.normalized_course_orders):
            expected = course_motion_sequence(order)
            actual = tuple(
                NAMES[motion_id] for motion_id in env.course_motion_sequences[variant]
            )
            assert actual == expected

        env.cfg.terrain_course_random_start_prob = 1.0
        env.reset()
        assert np.all(env.course_start_stage_ids > 0)
        np.testing.assert_array_equal(
            env.course_stage_ids, env.course_start_stage_ids
        )
        np.testing.assert_array_equal(
            env.motion_ids,
            env.course_motion_sequences[
                env.course_variant_ids, env.course_start_stage_ids
            ],
        )
        assert all(data.qpos[0] >= env.more_terrain_cfg.course_run_in for data in env.datas)
        env.cfg.terrain_course_random_start_prob = 0.0
        env.reset()
        np.testing.assert_array_equal(env.course_start_stage_ids, [0, 0, 0])

        env.motion_phase[:] = env.motion.frame_counts[env.motion_ids] - 0.1
        _, _, done, _ = env.step(
            np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
        )
        np.testing.assert_array_equal(env.course_stage_ids, [0, 0, 0])
        np.testing.assert_array_equal(done, [0.0, 0.0, 0.0])
        assert np.all(env.motion_phase < 1.0)

        for env_id, data in enumerate(env.datas):
            data.qpos[0] = env.course_stage_end_x[env.course_variant_ids[env_id], 0]
            mujoco.mj_forward(env.model, data)
        previous_reference_ctrl = env.last_reference_ctrl.copy()
        env.motion_phase[:] = env.motion.frame_counts[env.motion_ids] - 1.1
        _, _, done, info = env.step(
            np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
        )
        np.testing.assert_array_equal(env.course_stage_ids, [1, 1, 1])
        np.testing.assert_array_equal(
            env.motion_ids,
            env.course_motion_sequences[env.course_variant_ids, 1],
        )
        np.testing.assert_array_equal(env.motion_transition_steps_left, [3, 3, 3])
        new_reference_ctrl = env._targets_at_current_phase().tendon_ctrl
        np.testing.assert_allclose(
            env.last_reference_ctrl,
            0.75 * previous_reference_ctrl + 0.25 * new_reference_ctrl,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(done, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(info["course_complete"], [0.0, 0.0, 0.0])

        last_stage = env.course_motion_sequences.shape[1] - 1
        env.course_stage_ids[:] = last_stage
        env.motion_ids[:] = env.course_motion_sequences[
            env.course_variant_ids, last_stage
        ]
        env.motion_phase[:] = env.motion.frame_counts[env.motion_ids] - 1.1
        _, _, done, info = env.step(
            np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
        )
        np.testing.assert_array_equal(done, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(info["course_complete"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(info["incomplete_motion_end"], [1.0, 1.0, 1.0])

        env.course_stage_ids[:] = last_stage
        env.motion_ids[:] = env.course_motion_sequences[
            env.course_variant_ids, last_stage
        ]
        for data in env.datas:
            data.qpos[0] = env.course_finish_x
            mujoco.mj_forward(env.model, data)
        env.motion_phase[:] = env.motion.frame_counts[env.motion_ids] - 1.1
        _, _, done, info = env.step(
            np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
        )
        np.testing.assert_array_equal(done, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(info["course_complete"], [1.0, 1.0, 1.0])

        # Curriculum suffix completions end the sampled episode and receive
        # completion reward, but must not inflate strict full-route success.
        env.cfg.terrain_course_random_start_prob = 1.0
        env.reset()
        assert np.all(env.course_start_stage_ids > 0)
        env.course_stage_ids[:] = last_stage
        env.motion_ids[:] = env.course_motion_sequences[
            env.course_variant_ids, last_stage
        ]
        for data in env.datas:
            data.qpos[0] = env.course_finish_x
            mujoco.mj_forward(env.model, data)
        env.motion_phase[:] = env.motion.frame_counts[env.motion_ids] - 1.1
        _, _, done, info = env.step(
            np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
        )
        np.testing.assert_array_equal(done, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(info["course_suffix_complete"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(info["course_complete"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(info["successes"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(info["course_stage"], [last_stage] * 3)

        display_model = load_multi_robot_model(
            env.cfg.model_xml, 3, env.stair_cfg, env.more_terrain_cfg
        )
        display_data = mujoco.MjData(display_model)
        sync_multi_robot_state(display_model, display_data, env)
        for index, source in enumerate(env.datas):
            np.testing.assert_allclose(
                display_data.qpos[index * env.model.nq : (index + 1) * env.model.nq],
                source.qpos,
            )
    finally:
        env.close()
