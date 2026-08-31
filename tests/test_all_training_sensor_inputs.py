from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import train
from scripts.train_multimotion import default_train_args as multimotion_args
from scripts.train_multimotion_no_stairs import default_train_args as no_stairs_args
from scripts.train_multimotion_walk import default_train_args as walk_only_args
from scripts.train_multimotion_no_stairs_amp_teacher import (
    default_train_args as sensor_teacher_args,
)
from scripts.train_multimotion_no_stairs_amp_student import (
    default_train_args as sensor_student_args,
)
from scripts.train_stairs import default_train_args as stairs_args
from sensor_training.config import SensorObservationConfig
from sensor_training.envs import SensorHistoryEnv, SensorHistoryWarpAmpEnv
from sensor_training.ppo_config import SensorPPOAMPConfig
from sensor_training.trainer import SensorHistoryPPOAMPTrainer


def _option_values(args: list[str], option: str) -> list[str]:
    start = args.index(option) + 1
    end = start
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return args[start:end]


def test_generic_training_engine_always_builds_sensor_actor() -> None:
    assert train.LytRobotWarpAmpEnv is SensorHistoryWarpAmpEnv
    assert train.PPOAMPConfig is SensorPPOAMPConfig
    assert train.PPOAMPTrainer is SensorHistoryPPOAMPTrainer

    with patch.object(sys, "argv", ["train.py"]):
        args = train.parse_args()
    assert args.skill_conditioning is False
    assert args.reference_ctrl_blend == 0.0
    assert args.seed is None
    assert args.episode_random_seed is True
    assert args.log_root.endswith("logs/sensor_policy")
    # Every task launcher ultimately uses this parser. Full-body collision and
    # its solver capacity are shared defaults for every training task.
    assert args.robot_collision_profile == "full_body"
    assert args.solver_iterations == 20
    assert args.solver_ls_iterations == 20
    assert args.warp_nconmax == 128
    assert args.warp_njmax == 512
    assert args.warp_cuda_graph is True
    assert {"imu", "tendon", "recurrent", "sensor-only"}.issubset(
        set(args.wandb_tags.split(","))
    )


def test_generic_training_rejects_actor_commands_and_reference_leakage() -> None:
    invalid_argv = (
        ["train.py", "--skill-conditioning"],
        ["train.py", "--reference-ctrl-blend", "0.1"],
        ["train.py", "--legacy-reference-conditioned-actor"],
    )
    for argv in invalid_argv:
        with patch.object(sys, "argv", argv):
            try:
                train.main()
            except ValueError as exc:
                assert (
                    "sensor-only" in str(exc)
                    or "reference-ctrl-blend 0" in str(exc)
                    or "reserved for explicit no-stairs Teacher tasks" in str(exc)
                )
            else:  # pragma: no cover - failure branch for a regressed guard.
                raise AssertionError(f"Training unexpectedly accepted {argv[1:]}")


def test_legacy_teacher_is_the_only_reference_conditioned_training_exception() -> None:
    args = sensor_teacher_args()

    assert "--legacy-reference-conditioned-actor" in args
    assert "--skill-conditioning" in args
    assert _option_values(args, "--reference-ctrl-blend") == ["1.0"]
    assert _option_values(args, "--decimation") == ["2"]
    assert _option_values(args, "--solver-iterations") == ["10"]
    assert _option_values(args, "--warp-nconmax") == ["32"]
    assert _option_values(args, "--warp-njmax") == ["128"]
    assert "stairs" not in _option_values(args, "--terrain-types")


def test_every_multimotion_task_disables_commands_and_reference_control() -> None:
    for args in (
        multimotion_args(),
        no_stairs_args(),
        stairs_args(),
    ):
        assert "--no-skill-conditioning" in args
        assert "--skill-conditioning" not in args
        assert "--legacy-reference-conditioned-actor" not in args
        if "--reference-ctrl-blend" in args:
            assert _option_values(args, "--reference-ctrl-blend") == ["0.0"]


def test_student_distillation_keeps_teacher_labels_out_of_actor_inputs() -> None:
    args = sensor_student_args()

    assert "--legacy-reference-conditioned-actor" not in args
    assert "--skill-conditioning" in args
    assert "--sensor-wrapper-base-skill-conditioning" in args
    assert _option_values(args, "--reference-ctrl-blend") == ["0.0"]
    assert _option_values(args, "--distillation-coef") == ["1.0"]
    assert _option_values(args, "--decimation") == ["2"]
    assert _option_values(args, "--warp-nconmax") == ["32"]
    assert _option_values(args, "--warp-njmax") == ["128"]
    assert "stairs" not in _option_values(args, "--terrain-types")


def test_every_training_task_resolves_to_full_body_collision_defaults() -> None:
    task_defaults = (
        [],
        multimotion_args(),
        no_stairs_args(),
        walk_only_args(),
        stairs_args(),
    )
    for defaults in task_defaults:
        with patch.object(sys, "argv", ["train.py", *defaults]):
            args = train.parse_args()
        assert args.robot_collision_profile == "full_body"
        assert args.solver_iterations == 20
        assert args.solver_ls_iterations == 20
        assert args.warp_nconmax == 128
        assert args.warp_njmax == 512
        assert args.warp_cuda_graph is True


def test_all_training_task_launchers_use_100hz_control() -> None:
    task_defaults = (
        [],
        multimotion_args(),
        no_stairs_args(),
        walk_only_args(),
        stairs_args(),
        sensor_teacher_args(),
        sensor_student_args(),
    )
    for defaults in task_defaults:
        with patch.object(sys, "argv", ["train.py", *defaults]):
            args = train.parse_args()
        assert args.decimation == 2


def test_sensor_observation_contract_is_exactly_imu_plus_tendon_length() -> None:
    # This metadata is emitted into every checkpoint and ONNX file by the
    # generic sensor trainer. Keep it as an explicit deployment contract.
    fake_env = SimpleNamespace(
        sensor_observation_cfg=SensorObservationConfig(),
        imu_layout=SimpleNamespace(names=tuple(f"imu_{index}" for index in range(14)), dim=42),
        base_env=SimpleNamespace(position_action_dim=14),
        base_motion=SimpleNamespace(feature_dim=45),
        sensor_frame_dim=56,
        obs_dim=56,
        sensor_history_obs_dim=336,
        actor_history_enabled=False,
        amp_obs_dim=270,
        privileged_context_metadata={
            "version": 2,
            "actor_input": False,
            "training_only": True,
        },
        estimator_context_metadata={
            "dim": 11,
            "actor_input": False,
            "training_target_only": True,
        },
    )
    metadata_getter = SensorHistoryEnv.observation_metadata.fget
    assert metadata_getter is not None
    metadata = metadata_getter(fake_env)
    assert metadata["actor_input_fields"] == ("imu", "tendon_length")
    assert metadata["command_fields"] == ()
    assert metadata["command_dim"] == 0
    assert metadata["strict_sensor_only"] is True
    assert metadata["actor_frame_dim"] == 42 + 14
    assert metadata["actor_obs_dim"] == 42 + 14
    assert metadata["sensor_history_obs_dim"] == (42 + 14) * 6
    assert metadata["actor_uses_stacked_history"] is False
    assert metadata["privileged_training_context"]["actor_input"] is False


def test_locomotion_training_fails_closed_on_non_sensor_actor_input() -> None:
    metadata = {
        "imu_dim_per_frame": 42,
        "tendon_length_dim_per_frame": 14,
        "actor_frame_dim": 56,
        "actor_input_fields": ("imu", "tendon_length"),
        "command_fields": (),
        "command_dim": 0,
        "strict_sensor_only": True,
        "privileged_training_context": {
            "version": 2,
            "dim": 229,
            "actor_input": False,
        },
        "estimator_training_target": {
            "dim": 11,
            "actor_input": False,
        },
    }
    env = SimpleNamespace(
        cfg=SimpleNamespace(training_task="walk"),
        observation_metadata=metadata,
        sensor_frame_dim=56,
        privileged_context_dim=229,
        estimator_context_dim=11,
    )
    train.validate_locomotion_actor_input_contract(env)

    env.observation_metadata = {
        **metadata,
        "actor_input_fields": ("imu", "tendon_length", "terrain_id"),
    }
    try:
        train.validate_locomotion_actor_input_contract(env)
    except ValueError as exc:
        assert "exactly IMU and tendon length" in str(exc)
    else:  # pragma: no cover - failure branch for a regressed guard.
        raise AssertionError("Terrain state unexpectedly reached the locomotion Actor.")


def test_sensor_step_reuses_current_frames_when_refilling_reset_history() -> None:
    env = SensorHistoryEnv.__new__(SensorHistoryEnv)
    env.is_torch_env = True
    env.actor_history_enabled = False
    env.num_envs = 3
    env.sensor_observation_cfg = SimpleNamespace(frame_count=2)
    env._actor_frame = torch.zeros((3, 2), dtype=torch.float32)
    env._amp_history = torch.tensor(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
        ]
    )
    calls = {"sensor": 0, "amp": 0}
    actor_frame = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    amp_frame = torch.tensor([[12.0], [22.0], [32.0]])

    def normalized_sensor_frame() -> torch.Tensor:
        calls["sensor"] += 1
        return actor_frame

    def current_amp_frame() -> torch.Tensor:
        calls["amp"] += 1
        return amp_frame

    done = torch.tensor([0.0, 1.0, 0.0])
    env._normalized_sensor_frame = normalized_sensor_frame
    env._current_amp_frame = current_amp_frame
    env.base_env = SimpleNamespace(
        step=lambda _action: (None, torch.zeros(3), done, {})
    )

    observation, _reward, returned_done, _info = env.step(torch.zeros((3, 1)))

    assert calls == {"sensor": 1, "amp": 1}
    torch.testing.assert_close(observation, actor_frame)
    torch.testing.assert_close(returned_done, done)
    torch.testing.assert_close(env._amp_history[0, :, 0], torch.tensor([2.0, 12.0]))
    torch.testing.assert_close(env._amp_history[1, :, 0], torch.tensor([22.0, 22.0]))
    torch.testing.assert_close(env._amp_history[2, :, 0], torch.tensor([6.0, 32.0]))
