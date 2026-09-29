#!/usr/bin/env python3
"""Launch the recovered 2026-08-06 run with local historical assets."""

from __future__ import annotations

import argparse
import copy
from functools import partial
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = "/home/yeah/code/LYTrobot_amp"
RUN = ROOT / "logs" / "multimotion" / "08-06-10:56:09"
LOG_GROUP = "multimotion_recovered_20260806"
for directory in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(directory))


def relocate(value):
    """Rebase saved project paths without touching checkpoint bytes."""
    if isinstance(value, str) and value.startswith(ORIGINAL_ROOT + "/"):
        return str(ROOT / value[len(ORIGINAL_ROOT) + 1 :])
    if isinstance(value, dict):
        return {key: relocate(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(relocate(item) for item in value)
    if isinstance(value, list):
        return [relocate(item) for item in value]
    return value


def recorded_config() -> dict:
    return json.loads((ROOT / "recovery" / "original_config.json").read_text())


def training_argv(extra: list[str]) -> list[str]:
    from scripts import train_multimotion

    args = [*train_multimotion.default_train_args(), "--max-iterations", "30000"]
    return [*args, *extra, "--log-root", str(ROOT / "logs"),
            "--log-group", LOG_GROUP]


def validate_training_args(args) -> None:
    # A replay experiment always creates a new run; never resume into the archive.
    if args.log_dir is not None or args.resume_checkpoint is not None:
        raise ValueError("恢复入口只创建新实验；不接受 --log-dir 或 --resume-checkpoint。")
    if args.run_name is not None:
        raise ValueError("请使用自动时间戳运行名，避免覆盖已有实验。")
    if args.wandb_run_id is not None or args.wandb_resume_from is not None:
        raise ValueError("恢复入口不复用已有 W&B run。")


def local_play_cfg(original, args, checkpoint, stage=None):
    saved = copy.deepcopy(checkpoint)
    saved["env_config"] = relocate(saved.get("env_config", {}))
    return original(args, saved, stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "play"))
    parser.add_argument("--dry-run", action="store_true",
                        help="仅解析参数并显示配置，不创建环境、日志或 W&B run")
    wrapper, extra = parser.parse_known_args()
    os.chdir(ROOT)
    if wrapper.mode == "train":
        from scripts import train

        sys.argv = [str(ROOT / "scripts" / "train.py"), *training_argv(extra)]
        args = train.parse_args()
        validate_training_args(args)
        if wrapper.dry_run:
            print(shlex.join([sys.executable, *sys.argv]))
            print(json.dumps({key: getattr(args, key) for key in (
                "stage", "seed", "num_envs", "max_iterations", "sim_backend",
                "motion_files", "motion_names", "motion_weights", "motion_loop",
                "skill_conditioning", "terrain_types", "serial_terrain_course",
                "log_root", "log_group")}, indent=2))
            return
        train.main()
        return

    from scripts import play

    sys.argv = [str(ROOT / "scripts" / "play.py"),
                "--checkpoint", str(RUN / "model_final.pt"),
                "--target-velocity", str(recorded_config()["args"]["target_velocity"]),
                "--stage", "multimotion", *extra]
    play.make_env_cfg = partial(local_play_cfg, play.make_env_cfg)
    if wrapper.dry_run:
        args = play.parse_args()
        args.seed_was_provided = args.seed is not None
        args.seed = 42 if args.seed is None else args.seed
        import torch

        checkpoint = (play.load_onnx_metadata(args.checkpoint)
                      if Path(args.checkpoint).suffix == ".onnx"
                      else torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        cfg = play.make_env_cfg(args, checkpoint, "multimotion")
        print(json.dumps({"checkpoint": args.checkpoint, "model_xml": cfg.model_xml,
                          "motion_files": cfg.motion_files,
                          "terrain_asset_dir": cfg.terrain_asset_dir,
                          "terrain_types": cfg.terrain_types,
                          "num_envs": cfg.num_envs, "skill_conditioning": cfg.skill_conditioning,
                          "target_velocity": cfg.target_velocity}, indent=2))
        return
    play.main()


if __name__ == "__main__":
    main()
