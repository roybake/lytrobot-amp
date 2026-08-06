from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.play import checkpoint_recording_path, infer_stage, make_env_cfg


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
