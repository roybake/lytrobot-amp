from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.play import (
    RealTimePacer,
    SimulationFrameScheduler,
    best_checkpoint_quality_warning,
    checkpoint_recording_path,
    infer_stage,
    make_env_cfg,
    resolve_step_limit,
    validate_recurrent_sensor_architecture,
)


def test_incomplete_best_checkpoint_reports_that_it_is_not_ready(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_best.onnx"
    checkpoint.touch()
    (tmp_path / "model_best.json").write_text(
        '{"selection_metrics": {'
        '"checkpoint_score_full_success": 0.0, '
        '"checkpoint_score_stage_completion": 0.0, '
        '"checkpoint_score_stage_progress": 0.14}}',
        encoding="utf-8",
    )

    warning = best_checkpoint_quality_warning(checkpoint)

    assert warning is not None
    assert "best-so-far" in warning
    assert "not deployment-ready" in warning


def test_stage_completing_best_checkpoint_has_no_quality_warning(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.touch()
    (tmp_path / "model_best.json").write_text(
        '{"selection_metrics": {'
        '"checkpoint_score_full_success": 0.0, '
        '"checkpoint_score_stage_completion": 0.001}}',
        encoding="utf-8",
    )

    assert best_checkpoint_quality_warning(checkpoint) is None


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


def test_no_stairs_serial_metadata_keeps_two_stage_group() -> None:
    checkpoint = {
        "training_task": "multimotion_no_stairs",
        "env_config": {
            "training_task": "multimotion_no_stairs",
            "motion_files": ("walk.npz", "rough_walk.npz"),
            "serial_terrain_course": True,
        },
    }

    assert (
        infer_stage(
            "logs/sensor_policy/multimotion_no_stairs/stage1/run/model_final.pt",
            checkpoint,
            "auto",
        )
        == "stage1"
    )


def test_no_stairs_policy_uses_checkpoint_two_stage_group() -> None:
    checkpoint = {
        "training_task": "multimotion_no_stairs",
        "env_config": {
            "training_task": "multimotion_no_stairs",
            "motion_files": None,
            "serial_terrain_course": True,
            "domain_randomization": False,
        },
    }

    assert (
        infer_stage(
            "logs/multimotion/no_stairs/stage1/run/model_final.pt",
            checkpoint,
            "auto",
        )
        == "stage1"
    )
    checkpoint["env_config"]["domain_randomization"] = True
    assert (
        infer_stage(
            "logs/multimotion/no_stairs/stage2/run/model_final.pt",
            checkpoint,
            "auto",
        )
        == "stage2"
    )


def test_multimotion_recording_path_uses_portable_run_name(tmp_path: Path) -> None:
    checkpoint_path = Path("logs/multimotion/07-30-11:52:12/model_26000.pt")

    output = checkpoint_recording_path(
        checkpoint_path,
        {},
        tmp_path,
        infer_stage(checkpoint_path, {}, "auto"),
    )

    assert output == (
        tmp_path / "multimotion" / "recordings" / "07-30-11-52-12-26000.mp4"
    )
    assert output.parent.is_dir()


def test_default_recording_path_lives_in_task_stage_logs(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "logs" / "stairs" / "stage2" / "run" / "model_final.pt"
    checkpoint_path.parent.mkdir(parents=True)

    output = checkpoint_recording_path(
        checkpoint_path,
        {"iteration": 7500},
        None,
        "stage2",
    )

    assert output == (
        tmp_path
        / "logs"
        / "stairs"
        / "stage2"
        / "recordings"
        / "run-7500.mp4"
    )


def test_best_checkpoint_recording_uses_selected_iteration(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "logs" / "stairs" / "stage1" / "run" / "model_best.pt"
    checkpoint_path.parent.mkdir(parents=True)

    output = checkpoint_recording_path(
        checkpoint_path,
        {"iteration": 12500},
        None,
        "stage1",
    )

    assert output.name == "run-12500.mp4"


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


def test_stair_playback_preserves_full_route_and_disables_stage_jumps() -> None:
    args = SimpleNamespace(
        model_xml="robot.xml",
        motion_file="walk.npz",
        num_robots=1,
        seed=42,
        target_velocity=None,
        serial_terrain_course=None,
        stand_start_prob=None,
        stand_start_phase_delay_steps=None,
        stand_start_transition_min_steps=None,
        stand_start_transition_max_steps=None,
        stand_start_push_delay_steps=None,
        domain_randomization=None,
        episode_random_seed=None,
        seed_was_provided=True,
        checkpoint="logs/stairs/stage1/run/model.pt",
        stage="stage1",
    )
    checkpoint = {
        "training_task": "stairs_up_down",
        "env_config": {
            "training_task": "stairs_up_down",
            "serial_terrain_course": True,
            "terrain_course_random_start_prob": 0.8,
            "stand_start_prob": 0.5,
        },
    }

    cfg = make_env_cfg(args, checkpoint, "stage1")

    assert cfg.serial_terrain_course is True
    assert cfg.terrain_course_random_start_prob == 0.0
    assert cfg.stand_start_prob == 0.0
    assert cfg.motion_reset_prob == 0.0


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


def test_playback_uses_checkpoint_100hz_control_timing() -> None:
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
    checkpoint = {"env_config": {"timestep": 0.005, "decimation": 2}}

    cfg = make_env_cfg(args, checkpoint, "stage1")

    assert cfg.timestep * cfg.decimation == pytest.approx(0.01)


def test_playback_rejects_old_50hz_checkpoint() -> None:
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

    with pytest.raises(ValueError, match="100 Hz"):
        make_env_cfg(
            args,
            {"env_config": {"timestep": 0.005, "decimation": 4}},
            "stage1",
        )


def test_playback_rejects_incompatible_recurrent_checkpoint() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        validate_recurrent_sensor_architecture(
            {
                "sensor_policy_architecture": {
                    "name": "legacy_recurrent_sensor_policy",
                    "recurrent": True,
                }
            }
        )


def test_recording_frame_scheduler_preserves_one_second_duration_at_100hz() -> None:
    scheduler = SimulationFrameScheduler(30.0)

    frame_count = sum(scheduler.frames_due((step + 1) * 0.01) for step in range(100))

    assert frame_count == 30
    assert scheduler.frames_emitted / scheduler.fps == pytest.approx(1.0)


def test_realtime_pacer_never_catches_up_with_accelerated_steps() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleeper(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    pacer = RealTimePacer(clock=clock, sleeper=sleeper)
    now[0] += 0.002
    assert pacer.wait(0.01) == pytest.approx(0.008)
    now[0] += 0.020
    assert pacer.wait(0.01) == 0.0
    now[0] += 0.002
    assert pacer.wait(0.01) == pytest.approx(0.008)
    assert sleeps == pytest.approx([0.008, 0.008])


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
