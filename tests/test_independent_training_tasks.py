from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import mujoco
import numpy as np
import torch

from scripts import train
from scripts.train_multimotion import default_train_args as original_multimotion_args
from scripts.train_multimotion_no_stairs import (
    COURSE_RANDOM_START_PROBABILITY as NO_STAIRS_RANDOM_START_PROBABILITY,
    COURSE_RANDOM_START_RAMP_ITERATIONS as NO_STAIRS_RANDOM_START_RAMP_ITERATIONS,
    COURSE_RANDOM_START_WARMUP_ITERATIONS as NO_STAIRS_RANDOM_START_WARMUP_ITERATIONS,
    default_train_args as no_stairs_args,
)
from scripts.train_stairs import (
    COURSE_RANDOM_START_PROBABILITY as STAIRS_RANDOM_START_PROBABILITY,
    COURSE_RANDOM_START_RAMP_ITERATIONS as STAIRS_RANDOM_START_RAMP_ITERATIONS,
    COURSE_RANDOM_START_WARMUP_ITERATIONS as STAIRS_RANDOM_START_WARMUP_ITERATIONS,
    WALK_REFERENCE as STAIRS_WALK_REFERENCE,
    default_train_args as stairs_args,
)
from scripts.train_multimotion_no_stairs import main as no_stairs_main
from scripts.train_stairs import main as stairs_main
from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
from lytrobot_amp.terrain import StairTerrainCfg
from sensor_training.envs import SensorHistoryCpuAmpEnv


ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = ROOT / "robot" / "reference_data" / "multimotion"
NO_STAIRS_MOTION_DIR = ROOT / "robot" / "reference_data" / "no_stairs_fast"


def _option_values(args: list[str], option: str) -> list[str]:
    start = args.index(option) + 1
    end = start
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return args[start:end]


def test_stair_free_task_reuses_walking_policy_on_random_serial_course() -> None:
    args = no_stairs_args()

    assert _option_values(args, "--training-task") == ["multimotion_no_stairs_amp_single"]
    assert _option_values(args, "--motion-names") == ["walk"]
    assert [Path(value) for value in _option_values(args, "--motion-files")] == [
        NO_STAIRS_MOTION_DIR / "walk_omniretarget.npz",
    ]
    assert _option_values(args, "--motion-weights") == ["1.0"]
    assert _option_values(args, "--motion-loop") == ["true"]
    assert "--serial-course-single-motion" in args
    assert "--amp-task-only" in args
    assert set(_option_values(args, "--terrain-types")) == {
        "flat",
        "rough",
        "slope_up",
        "slope_down",
    }
    assert "--serial-terrain-course" in args
    assert _option_values(args, "--terrain-course-variants") == ["6"]
    assert "--terrain-course-random-assignment" in args
    assert float(_option_values(args, "--terrain-course-random-start-prob")[0]) == (
        NO_STAIRS_RANDOM_START_PROBABILITY
    )
    assert NO_STAIRS_RANDOM_START_PROBABILITY >= 0.80
    assert float(
        _option_values(args, "--terrain-course-random-start-warmup-iterations")[0]
    ) == NO_STAIRS_RANDOM_START_WARMUP_ITERATIONS
    assert NO_STAIRS_RANDOM_START_WARMUP_ITERATIONS <= 5000
    assert float(
        _option_values(args, "--terrain-course-random-start-ramp-iterations")[0]
    ) == NO_STAIRS_RANDOM_START_RAMP_ITERATIONS
    assert "--terrain-course-phase-sync" in args
    assert float(_option_values(args, "--foot-slip-penalty-coef")[0]) > 0.0
    assert _option_values(args, "--specific-cot-penalty-coef") == ["0.0"]
    assert float(_option_values(args, "--forward-velocity-reward-coef")[0]) == 3.0
    assert float(_option_values(args, "--forward-progress-reward-coef")[0]) == 1.0
    assert float(_option_values(args, "--forward-velocity-deficit-penalty-coef")[0]) == 1.0
    assert float(_option_values(args, "--forward-velocity-excess-penalty-coef")[0]) == 0.0
    assert float(_option_values(args, "--contact-reward-coef")[0]) == 0.35
    assert float(_option_values(args, "--alternating-step-reward-coef")[0]) == 1.50
    assert "--reward-mode" not in args
    assert not any(value.startswith("--stair-") for value in args)
    assert "--no-skill-conditioning" in args
    assert _option_values(args, "--reference-ctrl-blend") == ["0.0"]
    assert "--serial-reference-state-init" in args
    assert _option_values(args, "--motion-reset-prob") == ["0.50"]
    assert _option_values(args, "--stand-start-prob") == ["0.25"]
    assert _option_values(args, "--startup-action-ramp-steps") == ["40"]

    with patch.object(sys, "argv", ["train.py", *args]):
        stair_free = vars(train.parse_args())
    # Match the stable 08-06 multi-motion sampling regime instead of the
    # under-sampled 08-25 run (4 PPO minibatches, 1024 AMP samples, one
    # discriminator update).
    assert stair_free["num_envs"] == 4096
    assert stair_free["num_steps"] == 24
    assert stair_free["epochs"] == 4
    assert stair_free["minibatches"] == 8
    assert stair_free["amp_minibatch_size"] == 4096
    assert stair_free["amp_disc_updates"] == 2
    assert stair_free["warp_nconmax"] == 128
    assert stair_free["warp_njmax"] == 512

    # The original all-terrain entry point remains a separate four-clip task.
    original = original_multimotion_args()
    assert _option_values(original, "--training-task") == ["multimotion"]
    assert _option_values(original, "--motion-names") == [
        "walk",
        "rough_walk",
        "stair_up",
        "stair_down",
    ]
    assert "stairs" in _option_values(original, "--terrain-types")
    assert _option_values(original, "--log-group") == ["multimotion"]
    assert "--no-skill-conditioning" in original
    assert "--skill-conditioning" not in original


def test_dedicated_stair_task_uses_one_continuous_route_and_reward() -> None:
    args = stairs_args()

    assert _option_values(args, "--training-task") == ["stairs_up_down"]
    assert _option_values(args, "--motion-names") == ["walk", "stair_up", "stair_down"]
    assert Path(_option_values(args, "--motion-files")[0]) == STAIRS_WALK_REFERENCE
    assert _option_values(args, "--motion-weights") == ["0.50", "0.25", "0.25"]
    assert _option_values(args, "--terrain-types") == ["stairs"]
    assert float(_option_values(args, "--stair-step-height")[0]) == 0.13
    assert float(_option_values(args, "--stair-step-depth")[0]) == 0.32
    assert _option_values(args, "--reward-mode") == ["stairs"]
    assert _option_values(args, "--log-group") == ["stairs_up_down"]
    assert "--serial-terrain-course" in args
    assert float(_option_values(args, "--terrain-course-random-start-prob")[0]) == (
        STAIRS_RANDOM_START_PROBABILITY
    )
    assert STAIRS_RANDOM_START_PROBABILITY <= 0.35
    assert float(
        _option_values(args, "--terrain-course-random-start-warmup-iterations")[0]
    ) == STAIRS_RANDOM_START_WARMUP_ITERATIONS
    assert STAIRS_RANDOM_START_WARMUP_ITERATIONS >= 15000
    assert float(
        _option_values(args, "--terrain-course-random-start-ramp-iterations")[0]
    ) == STAIRS_RANDOM_START_RAMP_ITERATIONS
    assert "--terrain-course-phase-sync" in args
    assert float(_option_values(args, "--foot-slip-penalty-coef")[0]) > 0.0
    assert _option_values(args, "--specific-cot-penalty-coef") == ["0.0"]
    assert float(_option_values(args, "--forward-velocity-reward-coef")[0]) == 1.5
    assert float(_option_values(args, "--forward-progress-reward-coef")[0]) == 0.75
    assert float(_option_values(args, "--forward-velocity-deficit-penalty-coef")[0]) == 1.0
    assert float(_option_values(args, "--foot-slip-penalty-coef")[0]) == 0.5
    assert _option_values(args, "--terrain-course-run-in") == ["3.10"]
    assert _option_values(args, "--terrain-course-run-out") == ["3.10"]
    assert "--terminate-incomplete-motion-end" in args
    assert "--no-skill-conditioning" in args
    assert _option_values(args, "--reference-ctrl-blend") == ["0.0"]
    assert float(_option_values(args, "--stair-swing-clearance-reward-coef")[0]) == 0.75
    assert float(_option_values(args, "--stair-tread-support-reward-coef")[0]) == 1.25
    assert float(_option_values(args, "--stair-vertical-velocity-reward-coef")[0]) == 1.0
    assert float(_option_values(args, "--task-reward-lerp")[0]) == 0.5
    assert float(_option_values(args, "--min-action-std")[0]) == 0.05


def test_new_task_entrypoints_delegate_to_sensor_training_stack() -> None:
    for entrypoint in (no_stairs_main, stairs_main):
        with patch("sensor_training.train.main") as sensor_main:
            with patch.object(sys, "argv", ["task.py", "--max-iterations", "7"]):
                entrypoint()
                delegated_argv = list(sys.argv)
            sensor_main.assert_called_once_with()
            assert "--no-skill-conditioning" in delegated_argv
            assert _option_values(delegated_argv, "--reference-ctrl-blend") == ["0.0"]


def test_stair_reward_tracks_tread_support_clearance_and_vertical_motion() -> None:
    cfg = LytRobotEnvCfg(
        reward_mode="stairs",
        stair_swing_clearance=0.08,
        stair_vertical_velocity_sigma=0.25,
    )
    stair_cfg = StairTerrainCfg(
        enabled=True,
        step_height=0.13,
        step_depth=0.58,
        step_count=3,
        start_x=-0.95,
    )
    x = stair_cfg.ascent_start + 0.1
    surface = stair_cfg.step_height
    clear_foot = np.asarray(
        [
            [x, 0.0, surface],
            [x, 0.0, surface],
            [x, 0.0, surface + 0.08],
            [x, 0.0, surface + 0.08],
        ],
        dtype=np.float32,
    )
    low_foot = clear_foot.copy()
    low_foot[2:, 2] = surface
    fake_cpu = SimpleNamespace(cfg=cfg, stair_cfg=stair_cfg)

    matched = LytRobotAmpEnv._stair_terrain_reward_terms(
        fake_cpu,
        clear_foot,
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        vertical_velocity=0.30,
        reference_vertical_velocity=0.30,
    )
    missed = LytRobotAmpEnv._stair_terrain_reward_terms(
        fake_cpu,
        low_foot,
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        vertical_velocity=-0.30,
        reference_vertical_velocity=0.30,
    )

    assert matched["stair_vertical_velocity_track"] == 1.0
    assert np.isclose(matched["stair_swing_clearance"], 1.0)
    assert np.isclose(matched["stair_tread_support"], 1.0)
    assert missed["stair_vertical_velocity_track"] < matched["stair_vertical_velocity_track"]
    assert missed["stair_swing_clearance"] < matched["stair_swing_clearance"]
    assert missed["stair_tread_support"] < matched["stair_tread_support"]

    # Keep CPU validation and the production Torch/Warp reward numerically aligned.
    fake_warp = SimpleNamespace(
        cfg=cfg,
        stair_cfg=stair_cfg,
        num_envs=1,
        device=torch.device("cpu"),
    )
    torch_terms = LytRobotWarpAmpEnv._stair_terrain_reward_terms(
        fake_warp,
        torch.from_numpy(clear_foot[None, ...]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        vertical_velocity=torch.tensor([0.30]),
        reference_vertical_velocity=torch.tensor([0.30]),
    )
    for key, expected in matched.items():
        np.testing.assert_allclose(torch_terms[key].numpy(), [expected], atol=1.0e-6)


def test_stair_free_task_actor_observation_is_strictly_sensor_only() -> None:
    env = SensorHistoryCpuAmpEnv(
        LytRobotEnvCfg(
            model_xml=str(ROOT / "robot" / "MJCFmodelposition.xml"),
            training_task="multimotion_no_stairs_amp_single",
            motion_files=(
                str(NO_STAIRS_MOTION_DIR / "walk_omniretarget.npz"),
            ),
            motion_names=("walk",),
            motion_weights=(1.0,),
            motion_loop=(True,),
            skill_conditioning=False,
            terrain_types=("flat", "rough", "slope_up", "slope_down"),
            serial_terrain_course=True,
            serial_course_single_motion=True,
            amp_task_only=True,
            serial_reference_state_init=True,
            terrain_course_variants=6,
            num_envs=1,
            stand_start_prob=0.0,
            reset_noise_scale=0.0,
            reference_ctrl_blend=0.0,
            domain_randomization=False,
            episode_timeout_jitter_steps=0,
        )
    )
    try:
        observation = env.reset()
        assert env.action_dim == 14
        assert env.sensor_frame_dim == 42 + 14
        assert observation.shape == (1, 56)
        assert env.observation_metadata["actor_input_fields"] == ("imu", "tendon_length")
        assert env.observation_metadata["command_dim"] == 0
        assert env.stair_cfg.enabled is False
        assert env.more_terrain_cfg.serial_course is True
        assert env.base_env.motion_count == 1
    finally:
        env.close()


def test_stair_task_keeps_terrain_reward_outside_sensor_actor() -> None:
    env = SensorHistoryCpuAmpEnv(
        LytRobotEnvCfg(
            model_xml=str(ROOT / "robot" / "MJCFmodelposition.xml"),
            training_task="stairs_up_down",
            motion_files=(
                str(MOTION_DIR / "walk_omniretarget.npz"),
                str(MOTION_DIR / "stair_up_omniretarget.npz"),
                str(MOTION_DIR / "stair_down_omniretarget.npz"),
            ),
            motion_names=("walk", "stair_up", "stair_down"),
            motion_weights=(0.5, 0.25, 0.25),
            motion_loop=(True, False, False),
            skill_conditioning=False,
            terrain_types=("stairs",),
            serial_terrain_course=True,
            terrain_course_variants=1,
            reward_mode="stairs",
            stair_step_height=0.13,
            stair_step_depth=0.58,
            stair_step_count=3,
            stair_start_x=-0.95,
            stair_down_offset=0.25,
            num_envs=1,
            stand_start_prob=0.0,
            motion_reset_prob=1.0,
            reset_noise_scale=0.0,
            reference_ctrl_blend=0.0,
            use_reference_velocity=True,
            domain_randomization=False,
            episode_timeout_jitter_steps=0,
        )
    )
    try:
        observation = env.reset()
        assert env.action_dim == 14
        assert observation.shape == (1, 56)
        assert env.observation_metadata["actor_input_fields"] == ("imu", "tendon_length")
        assert env.observation_metadata["command_dim"] == 0
        observation, reward, done, info = env.step(
            np.zeros((1, env.action_dim), dtype=np.float32)
        )
        assert observation.shape == (1, 56)
        assert np.isfinite(reward).all()
        assert np.isfinite(done).all()
        for key in (
            "stair_vertical_velocity_track",
            "stair_swing_clearance",
            "stair_tread_support",
        ):
            assert key in info
            assert np.isfinite(info[key]).all()
    finally:
        env.close()


def test_serial_stair_stage_terminates_on_deep_geometry_penetration() -> None:
    env = LytRobotAmpEnv(
        LytRobotEnvCfg(
            model_xml=str(ROOT / "robot" / "MJCFmodelposition.xml"),
            motion_files=(
                str(MOTION_DIR / "walk_omniretarget.npz"),
                str(MOTION_DIR / "stair_up_omniretarget.npz"),
                str(MOTION_DIR / "stair_down_omniretarget.npz"),
            ),
            motion_names=("walk", "stair_up", "stair_down"),
            motion_weights=(0.5, 0.25, 0.25),
            motion_loop=(True, False, False),
            skill_conditioning=False,
            terrain_types=("stairs",),
            serial_terrain_course=True,
            terrain_course_variants=1,
            reward_mode="stairs",
            stair_step_height=0.13,
            stair_step_depth=0.22,
            stair_step_count=3,
            stair_start_x=-0.95,
            stair_min_root_clearance=0.60,
            num_envs=1,
            stand_start_prob=0.0,
            reset_noise_scale=0.0,
            reference_ctrl_blend=0.0,
            domain_randomization=False,
            episode_timeout_jitter_steps=0,
        )
    )
    try:
        env.course_stage_ids[0] = 1
        env.motion_ids[0] = 1
        env.motion_phase[0] = 0.0
        stairs = env.course_stair_cfgs[0]
        root_x = stairs.top_start + 0.5 * stairs.top_length
        surface = float(
            env._stair_surface_height(np.asarray([root_x], dtype=np.float32), 0)[0]
        )
        data = env.datas[0]
        data.qpos[0] = root_x
        data.qpos[1] = env.more_terrain_cfg.course_lane_ys[0]
        data.qpos[2] = surface + env.cfg.stair_min_root_clearance - 0.08
        data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, data)
        env.last_foot_world_pos[0] = env._foot_positions_world(data)

        _, _, done, info = env.step(
            np.zeros((1, env.action_dim), dtype=np.float32)
        )

        np.testing.assert_array_equal(done, [1.0])
        np.testing.assert_array_equal(info["termination"], [1.0])
        np.testing.assert_array_equal(info["terrain_penetration"], [1.0])
        assert (
            info["terrain_penetration_depth"][0]
            > env.cfg.terrain_penetration_tolerance
        )
    finally:
        env.close()
