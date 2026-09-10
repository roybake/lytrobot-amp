"""Verify the historical argument and asset contracts without starting training."""

import copy
import sys
from pathlib import Path

import pytest

from recovery import run
from scripts import play, train, train_multimotion


def test_recorded_launcher_and_all_effective_arguments(monkeypatch):
    recorded = run.recorded_config()
    argv = run.relocate(recorded["argv"][1:])
    assert "stairs" in argv
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    actual = vars(train.parse_args())
    # The historical main resolves this tri-state after parsing.
    if actual["domain_randomization"] is None:
        actual["domain_randomization"] = actual["stage"] == "stage2"
    assert actual == run.relocate(recorded["args"])
    assert Path(train.__file__).resolve().is_relative_to(run.ROOT)
    assert Path(sys.modules["lytrobot_amp.env"].__file__).resolve().is_relative_to(run.ROOT)


def test_recovery_training_uses_current_no_stair_defaults():
    argv = run.training_argv([])
    assert train_multimotion.default_train_args() + ["--max-iterations", "30000"] == argv[:-4]
    assert "stairs" not in argv
    assert "stair_up" not in argv
    assert "stair_down" not in argv


def test_playback_uses_historical_assets_without_mutating_metadata(monkeypatch):
    checkpoint = {"env_config": run.recorded_config()["env"]}
    before = copy.deepcopy(checkpoint)
    monkeypatch.setattr(sys, "argv", ["play.py", "--checkpoint", str(run.RUN / "model_final.pt"),
                                     "--seed", "42", "--num-robots", "3"])
    args = play.parse_args()
    cfg = run.local_play_cfg(play.make_env_cfg, args, checkpoint, "multimotion")
    for name in (cfg.model_xml, cfg.motion_file, cfg.terrain_asset_dir, *cfg.motion_files):
        assert Path(name).is_relative_to(run.ROOT)
        assert Path(name).exists()
    assert checkpoint == before
    assert cfg.skill_conditioning
    assert cfg.motion_names == ["walk", "rough_walk", "stair_up", "stair_down"]
    assert cfg.motion_weights == [0.25] * 4
    assert cfg.motion_loop == [True, False, False, False]
    assert cfg.serial_terrain_course
    assert cfg.terrain_course_random_start_prob == 0.0
    assert cfg.num_envs == 3


@pytest.mark.parametrize("extra", [
    ["--log-dir", str(run.RUN)],
    ["--resume-checkpoint", str(run.RUN / "model_final.pt")],
    ["--run-name", "existing-run"],
    ["--wandb-run-id", "4ltcse1e"],
])
def test_retraining_cannot_reuse_saved_run(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["train.py", *run.training_argv(extra)])
    with pytest.raises(ValueError):
        run.validate_training_args(train.parse_args())


def test_retraining_has_separate_log_destination(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", *run.training_argv([])])
    args = train.parse_args()
    run.validate_training_args(args)
    assert args.log_root == str(run.ROOT / "logs")
    assert args.log_group == "multimotion_recovered_20260806"
    assert args.log_dir is None and args.run_name is None
