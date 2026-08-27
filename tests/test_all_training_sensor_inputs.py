from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

from scripts import train
from scripts.train_multimotion import default_train_args as multimotion_args
from scripts.train_multimotion_no_stairs import default_train_args as no_stairs_args
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
    assert {"imu", "tendon", "recurrent", "sensor-only"}.issubset(
        set(args.wandb_tags.split(","))
    )


def test_generic_training_rejects_actor_commands_and_reference_leakage() -> None:
    invalid_argv = (
        ["train.py", "--skill-conditioning"],
        ["train.py", "--reference-ctrl-blend", "0.1"],
    )
    for argv in invalid_argv:
        with patch.object(sys, "argv", argv):
            try:
                train.main()
            except ValueError as exc:
                assert "sensor-only" in str(exc) or "reference-ctrl-blend 0" in str(exc)
            else:  # pragma: no cover - failure branch for a regressed guard.
                raise AssertionError(f"Training unexpectedly accepted {argv[1:]}")


def test_every_multimotion_task_disables_commands_and_reference_control() -> None:
    for args in (multimotion_args(), no_stairs_args(), stairs_args()):
        assert "--no-skill-conditioning" in args
        assert "--skill-conditioning" not in args
        if "--reference-ctrl-blend" in args:
            assert _option_values(args, "--reference-ctrl-blend") == ["0.0"]


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
        privileged_context_metadata={"actor_input": False, "training_only": True},
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
        "privileged_training_context": {"actor_input": False},
    }
    env = SimpleNamespace(
        cfg=SimpleNamespace(training_task="walk"),
        observation_metadata=metadata,
        sensor_frame_dim=56,
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
