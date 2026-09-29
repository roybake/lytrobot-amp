"""Verify the historical argument and asset contracts without starting training."""

import copy
import sys
from pathlib import Path

from recovery import run
from scripts import play, train


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

