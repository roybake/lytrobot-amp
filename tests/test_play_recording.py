from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.play import (
    checkpoint_recording_path,
    infer_stage,
    make_env_cfg,
    resolve_step_limit,
)


def test_multimotion_recording_group_is_inferred_from_checkpoint_path() -> None:
    checkpoint_path = Path("logs/multimotion/07-30-11:52:12/model_26000.pt")

    assert infer_stage(checkpoint_path, {}, "auto") == "multimotion"


def test_multimotion_recording_group_is_inferred_from_checkpoint_metadata() -> None:
    checkpoint = {
        "env_config": {
            "motion_files": ("walk.npz", "turn_left.npz"),
            "domain_randomization": True,
        }
    }

    assert infer_stage("relocated/model_final.pt", checkpoint, "auto") == "multimotion"


def test_no_stairs_task_metadata_overrides_legacy_stage1_log_path() -> None:
    checkpoint = {
        "training_task": "multimotion_no_stairs",
        "env_config": {"training_task": "multimotion_no_stairs"},
    }

    assert (
        infer_stage(
            "logs/sensor_policy/multimotion_no_stairs/stage1/run/model_final.pt",
            checkpoint,
            "auto",
        )
        == "multimotion"
    )


def test_multimotion_recording_path_uses_portable_run_name(tmp_path: Path) -> None:
    checkpoint_path = Path("logs/multimotion/07-30-11:52:12/model_26000.pt")

    output = checkpoint_recording_path(
        checkpoint_path,
        {},
        tmp_path,
        infer_stage(checkpoint_path, {}, "auto"),
    )

    assert output == tmp_path / "multimotion" / "07-30-11-52-12-26000.mp4"
    assert output.parent.is_dir()


def test_play_can_disable_checkpoint_serial_course() -> None:
    args = SimpleNamespace(
        model_xml="robot.xml",
        motion_file="walk.npz",
        num_robots=1,
        seed=42,
        target_velocity=1.0,
        serial_terrain_course=False,
        stand_start_prob=None,
        stand_start_phase_delay_steps=None,
        stand_start_transition_min_steps=None,
        stand_start_transition_max_steps=None,
        stand_start_push_delay_steps=None,
        domain_randomization=None,
        episode_random_seed=None,
        seed_was_provided=True,
        checkpoint="logs/multimotion/run/model.pt",
        stage="multimotion",
    )
    checkpoint = {"env_config": {"serial_terrain_course": True}}

    cfg = make_env_cfg(args, checkpoint, "multimotion")

    assert cfg.serial_terrain_course is False


def test_all_playback_uses_deterministic_phase_zero_and_no_timeout() -> None:
    args = SimpleNamespace(
        model_xml="robot.xml",
        motion_file="motion.npz",
        num_robots=1,
        seed=42,
        target_velocity=0.0,
        serial_terrain_course=False,
        stand_start_prob=None,
        stand_start_phase_delay_steps=None,
        stand_start_transition_min_steps=None,
        stand_start_transition_max_steps=None,
        stand_start_push_delay_steps=None,
        domain_randomization=None,
        episode_random_seed=None,
        seed_was_provided=True,
        checkpoint="logs/stage1/run/model.pt",
        stage="stage1",
    )
    checkpoint = {
        "env_config": {
            "motion_reset_prob": 1.0,
            "episode_steps": 1000,
            "episode_timeout_jitter_steps": 240,
        }
    }

    cfg = make_env_cfg(args, checkpoint, "stage1")

    assert cfg.motion_reset_prob == 0.0
    assert cfg.episode_steps == 2_147_483_647
    assert cfg.episode_timeout_jitter_steps == 0
    assert cfg.stand_start_prob == 1.0


def test_playback_preserves_checkpoint_target_velocity_by_default() -> None:
    args = SimpleNamespace(
        model_xml="robot.xml",
        motion_file="motion.npz",
        num_robots=1,
        seed=42,
        target_velocity=None,
        serial_terrain_course=False,
        stand_start_prob=None,
        stand_start_phase_delay_steps=None,
        stand_start_transition_min_steps=None,
        stand_start_transition_max_steps=None,
        stand_start_push_delay_steps=None,
        domain_randomization=None,
        episode_random_seed=None,
        seed_was_provided=True,
        checkpoint="logs/stage1/run/model.pt",
        stage="stage1",
    )
    checkpoint = {"env_config": {"target_velocity": 0.862295}}

    cfg = make_env_cfg(args, checkpoint, "stage1")

    assert cfg.target_velocity == 0.862295


def test_viewer_defaults_to_unlimited_repetition() -> None:
    assert resolve_step_limit(None, viewer=True) == 0
    assert resolve_step_limit(None, viewer=False) == 1000
    assert resolve_step_limit(250, viewer=True) == 250


def test_playback_randomizes_every_episode_when_seed_is_omitted() -> None:
    args = SimpleNamespace(
        model_xml="robot.xml",
        motion_file="motion.npz",
        num_robots=1,
        seed=123456,
        target_velocity=None,
        serial_terrain_course=False,
        stand_start_prob=None,
        stand_start_phase_delay_steps=None,
        stand_start_transition_min_steps=None,
        stand_start_transition_max_steps=None,
        stand_start_push_delay_steps=None,
        domain_randomization=None,
        episode_random_seed=None,
        seed_was_provided=False,
        checkpoint="logs/stage2/run/model.pt",
        stage="stage2",
    )
    checkpoint = {"env_config": {"randomize_episode_seed": False}}

    cfg = make_env_cfg(args, checkpoint, "stage2")

    assert cfg.randomize_episode_seed is True


def test_fixed_playback_seed_is_reproducible_by_default() -> None:
    args = SimpleNamespace(
        model_xml="robot.xml",
        motion_file="motion.npz",
        num_robots=1,
        seed=42,
        target_velocity=None,
        serial_terrain_course=False,
        stand_start_prob=None,
        stand_start_phase_delay_steps=None,
        stand_start_transition_min_steps=None,
        stand_start_transition_max_steps=None,
        stand_start_push_delay_steps=None,
        domain_randomization=None,
        episode_random_seed=None,
        seed_was_provided=True,
        checkpoint="logs/stage2/run/model.pt",
        stage="stage2",
    )
    checkpoint = {"env_config": {"randomize_episode_seed": True}}

    cfg = make_env_cfg(args, checkpoint, "stage2")

    assert cfg.randomize_episode_seed is False
