#!/usr/bin/env python3
"""Train the LYT robot with BPO-AMP."""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from lytrobot_amp import __version__
from lytrobot_amp.env import LytRobotEnvCfg
from lytrobot_amp.ppo_amp import PPOAMPConfig, PPOAMPTrainer

try:
    from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
except Exception as exc:  # pragma: no cover - optional backend import guard.
    LytRobotWarpAmpEnv = None
    WARP_IMPORT_ERROR = exc
else:
    WARP_IMPORT_ERROR = None


RESOURCE_METRIC_KEYS = (
    "cuda_mem_allocated_mb",
    "cuda_mem_reserved_mb",
    "cuda_mem_peak_allocated_mb",
    "cuda_mem_peak_reserved_mb",
    "gpu_total_used_mb",
    "gpu_temp_c",
    "gpu_power_w",
    "process_rss_mb",
    "system_mem_available_mb",
)

TRAIN_LOG_PROGRESS_RE = re.compile(r"\[TRAIN\]\s+iter=(\d+)\s+step=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LYT biped locomotion with BPO-AMP.")
    parser.add_argument("--model-xml", default=str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"))
    parser.add_argument("--motion-file", default=str(REPO_ROOT / "robot" / "reference_data" / "walk1_mjcf.npz"))
    parser.add_argument("--log-root", default=str(REPO_ROOT / "logs"))
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Existing run directory to append logs/checkpoints into when resuming.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stage", default="stage1", choices=("stage1", "stage2"))
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--resume-reset-optimizer", action="store_true")
    parser.add_argument(
        "--resume-reset-progress",
        action="store_true",
        help="Load checkpoint weights but restart iteration/global_step from zero.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-random-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--sim-backend", default="warp", choices=("warp",))
    parser.add_argument("--torch-threads", type=int, default=0)

    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--episode-timeout-jitter-steps", type=int, default=240)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--solver-ls-iterations", type=int, default=20)
    parser.add_argument("--warp-nconmax", type=int, default=32)
    parser.add_argument("--warp-njmax", type=int, default=128)
    parser.add_argument("--warp-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-velocity", type=float, default=1.0)
    parser.add_argument("--forward-velocity-tracking-sigma", type=float, default=0.25)
    parser.add_argument("--forward-velocity-reward-coef", type=float, default=0.0)
    parser.add_argument("--forward-progress-reward-coef", type=float, default=0.0)
    parser.add_argument("--forward-velocity-deficit-penalty-coef", type=float, default=0.0)
    parser.add_argument("--forward-velocity-excess-penalty-coef", type=float, default=0.0)
    parser.add_argument("--reset-noise-scale", type=float, default=0.01)
    parser.add_argument("--motion-reset-prob", type=float, default=0.5)
    parser.add_argument("--action-residual-scale", type=float, default=0.5)
    parser.add_argument("--distal-action-scale", type=float, default=1.0)
    parser.add_argument("--reference-ctrl-blend", type=float, default=1.0)
    parser.add_argument("--foot-vel-tracking-sigma", type=float, default=1.0)
    parser.add_argument("--stance-foot-vel-sigma", type=float, default=0.15)
    parser.add_argument("--foot-vel-reward-coef", type=float, default=0.35)
    parser.add_argument("--stance-foot-reward-coef", type=float, default=1.00)
    parser.add_argument("--contact-reward-coef", type=float, default=0.70)
    parser.add_argument("--missed-contact-penalty-coef", type=float, default=0.80)
    parser.add_argument("--side-foot-pos-sigma", type=float, default=0.08)
    parser.add_argument("--step-gap-sigma", type=float, default=0.12)
    parser.add_argument("--swing-advance-sigma", type=float, default=0.10)
    parser.add_argument("--side-foot-pos-coef", type=float, default=0.0)
    parser.add_argument("--step-gap-coef", type=float, default=0.0)
    parser.add_argument("--swing-advance-coef", type=float, default=0.0)
    parser.add_argument("--distal-joint-vel-sigma", type=float, default=2.0)
    parser.add_argument("--symmetry-reward-coef", type=float, default=1.20)
    parser.add_argument("--symmetry-foot-pos-sigma", type=float, default=0.08)
    parser.add_argument("--symmetry-balance-sigma", type=float, default=0.25)
    parser.add_argument("--symmetry-ema-decay", type=float, default=0.99)
    parser.add_argument("--symmetry-foot-weight", type=float, default=0.60)
    parser.add_argument("--symmetry-contact-weight", type=float, default=0.25)
    parser.add_argument("--symmetry-action-weight", type=float, default=0.15)
    parser.add_argument("--startup-action-ramp-steps", type=int, default=20)
    parser.add_argument("--stand-start-prob", type=float, default=0.5)
    parser.add_argument("--stand-start-phase-delay-steps", type=int, default=25)
    parser.add_argument("--stand-start-transition-min-steps", type=int, default=80)
    parser.add_argument("--stand-start-transition-max-steps", type=int, default=160)
    parser.add_argument("--stand-start-push-delay-steps", type=int, default=100)
    parser.add_argument("--ctrl-penalty-coef", type=float, default=0.040)
    parser.add_argument("--action-rate-penalty-coef", type=float, default=0.100)
    parser.add_argument("--distal-action-penalty-coef", type=float, default=0.030)
    parser.add_argument("--distal-action-rate-penalty-coef", type=float, default=0.120)
    parser.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--payload-mass-min", type=float, default=0.0)
    parser.add_argument("--payload-mass-max", type=float, default=3.0)
    parser.add_argument("--body-mass-scale-min", type=float, default=0.85)
    parser.add_argument("--body-mass-scale-max", type=float, default=1.15)
    parser.add_argument("--tendon-length-scale-min", type=float, default=0.95)
    parser.add_argument("--tendon-length-scale-max", type=float, default=1.05)
    parser.add_argument("--ground-friction-min", type=float, default=0.55)
    parser.add_argument("--ground-friction-max", type=float, default=1.40)
    parser.add_argument("--gravity-min", type=float, default=9.75)
    parser.add_argument("--gravity-max", type=float, default=9.85)
    parser.add_argument("--push-force-min", type=float, default=0.0)
    parser.add_argument("--push-force-max", type=float, default=80.0)
    parser.add_argument("--push-interval-min-steps", type=int, default=25)
    parser.add_argument("--push-interval-max-steps", type=int, default=100)
    parser.add_argument("--push-duration-min-steps", type=int, default=2)
    parser.add_argument("--push-duration-max-steps", type=int, default=8)
    parser.add_argument("--obs-noise-std", type=float, default=0.02)
    parser.add_argument("--action-noise-std", type=float, default=0.01)
    parser.add_argument("--action-delay-min-ms", type=float, default=2.0)
    parser.add_argument("--action-delay-max-ms", type=float, default=5.0)

    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--reward-scale", type=float, default=0.25)
    parser.add_argument("--clip-coef", type=float, default=None)
    parser.add_argument("--bpo-lambda", type=float, default=None)
    parser.add_argument("--bpo-tv-coef", type=float, default=None)
    parser.add_argument("--value-clip-coef", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--median-value-coef", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--target-kl-stop-multiplier", type=float, default=None)
    parser.add_argument("--adaptive-lr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--max-learning-rate", type=float, default=None)
    parser.add_argument("--kl-adapt-factor", type=float, default=None)
    parser.add_argument("--disc-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--amp-minibatch-size", type=int, default=1024)
    parser.add_argument("--amp-disc-updates", type=int, default=1)
    parser.add_argument("--amp-grad-penalty-coef", type=float, default=2.0)
    parser.add_argument("--amp-label-smoothing", type=float, default=0.20)
    parser.add_argument("--amp-obs-noise-std", type=float, default=0.05)
    parser.add_argument("--amp-mirror-ref-prob", type=float, default=0.50)
    parser.add_argument("--amp-reward-coef", type=float, default=1.20)
    parser.add_argument("--task-reward-lerp", type=float, default=0.50)
    parser.add_argument("--min-action-std", type=float, default=0.05)
    parser.add_argument("--max-action-std", type=float, default=0.60)
    parser.add_argument("--episode-metric-window", type=int, default=4096)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--resource-log-interval", type=int, default=50)
    parser.add_argument("--wandb-log-interval", type=int, default=10)
    parser.add_argument(
        "--max-process-rss-mb",
        type=float,
        default=32.0 * 1024.0,
        help="Save and stop before the training process exhausts system RAM; set 0 to disable.",
    )
    parser.add_argument(
        "--memory-cleanup-interval",
        type=int,
        default=0,
        help="Run Python GC and return free glibc heap pages to the OS every N iterations; set 0 to disable.",
    )
    parser.add_argument(
        "--iteration-sleep-ms",
        type=float,
        default=0.0,
        help="Yield the shared display GPU/desktop briefly after each training iteration; set 0 to disable.",
    )

    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-watch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--wandb-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log an explicit zero bootstrap point at run start.",
    )
    parser.add_argument("--wandb-project", default="lytrobot-amp")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="Existing wandb run id to resume; inferred from --log-dir/wandb when omitted.",
    )
    parser.add_argument(
        "--wandb-resume-from",
        default=None,
        help="W&B rewind target, e.g. '<run_id>?_step=<step>'; cannot be combined with normal resume.",
    )
    parser.add_argument(
        "--resume-wandb-warmup-iterations",
        type=int,
        default=100,
        help=(
            "When appending to an existing run, skip reward/episode/task metrics in wandb "
            "for this many resumed iterations so reset episode windows do not create spikes."
        ),
    )
    parser.add_argument(
        "--resume-wandb-min-episode-reward",
        type=float,
        default=None,
        help="Keep reward/episode/task metrics out of wandb until mean_episode_reward reaches this value.",
    )
    parser.add_argument(
        "--resume-wandb-min-episode-length",
        type=float,
        default=None,
        help="Keep reward/episode/task metrics out of wandb until mean_episode_length reaches this value.",
    )
    parser.add_argument("--wandb-tags", default="lytrobot,bpo,amp,walk")
    return parser.parse_args()


def select_device(device_arg: str) -> str:
    if device_arg != "cuda":
        raise RuntimeError("Training only supports --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA, but torch.cuda.is_available() is false.")
    return "cuda"


def select_sim_backend(backend_arg: str, device: str) -> str:
    if backend_arg != "warp":
        raise RuntimeError("Training only supports --sim-backend warp.")
    if device != "cuda":
        raise RuntimeError("--sim-backend warp requires --device cuda.")
    if LytRobotWarpAmpEnv is None:
        raise RuntimeError("The warp backend is unavailable; install mujoco-warp and warp-lang.") from WARP_IMPORT_ERROR
    return "warp"


def make_log_dir(log_root: str, stage: str, run_name: str | None) -> Path:
    root = Path(log_root).expanduser().resolve() / stage
    stem = run_name or datetime.now().strftime("%m-%d-%H:%M:%S")
    log_dir = root / stem
    suffix = 1
    while log_dir.exists():
        log_dir = root / f"{stem}_{suffix}"
        suffix += 1
    log_dir.mkdir(parents=True, exist_ok=False)
    return log_dir


def resolve_log_dir(args: argparse.Namespace) -> Path:
    if args.log_dir is None:
        return make_log_dir(args.log_root, args.stage, args.run_name)

    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.exists():
        raise FileNotFoundError(f"--log-dir does not exist: {log_dir}")
    if not log_dir.is_dir():
        raise NotADirectoryError(f"--log-dir is not a directory: {log_dir}")
    return log_dir


def resolve_resume_checkpoint(path: str | None) -> str | None:
    """Validate a resume checkpoint before allocating logs, W&B, or CUDA state."""

    if path is None:
        return None
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.is_file():
        return str(checkpoint)

    fallback = checkpoint.parent / "model_final.pt"
    fallback_hint = (
        f" The completed run has {fallback}; use that file instead."
        if fallback.is_file()
        else ""
    )
    raise FileNotFoundError(f"--resume-checkpoint does not exist: {checkpoint}.{fallback_hint}")


def infer_wandb_run_id(log_dir: Path) -> str | None:
    wandb_dir = log_dir / "wandb"
    if not wandb_dir.is_dir():
        return None
    run_dirs = sorted(
        (path for path in wandb_dir.iterdir() if path.is_dir() and path.name.startswith("run-")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        for artifact in run_dir.glob("run-*.wandb"):
            stem = artifact.stem
            if stem.startswith("run-"):
                return stem.removeprefix("run-")
        parts = run_dir.name.split("-")
        if parts:
            return parts[-1]
    return None


def read_logged_training_progress(log_dir: Path) -> tuple[int, int] | None:
    train_log = log_dir / "train.log"
    if not train_log.exists():
        return None
    best: tuple[int, int] | None = None
    with train_log.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            match = TRAIN_LOG_PROGRESS_RE.search(line)
            if match is None:
                continue
            iteration = int(match.group(1))
            global_step = int(match.group(2))
            if best is None or iteration > best[0] or (iteration == best[0] and global_step > best[1]):
                best = (iteration, global_step)
    return best


def maybe_init_wandb(args: argparse.Namespace, log_dir: Path):
    mode = os.environ.get("WANDB_MODE", "").lower()
    if not args.wandb or mode == "disabled":
        print("[INFO] wandb disabled for this run.")
        return None
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb is not installed; continuing without online logging.")
        return None

    tags = tuple(tag.strip() for tag in args.wandb_tags.split(",") if tag.strip())
    run_id = args.wandb_run_id or infer_wandb_run_id(log_dir)
    resume_from = args.wandb_resume_from
    if resume_from is not None:
        print(f"[INFO] rewinding/resuming wandb from {resume_from}")
    if run_id is not None:
        print(f"[INFO] resuming wandb run id={run_id}")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=None if resume_from is not None else run_id,
        resume=None if resume_from is not None else ("allow" if run_id is not None else None),
        resume_from=resume_from,
        name=log_dir.name,
        tags=tags,
        dir=str(log_dir),
        config=vars(args),
        settings=wandb.Settings(console="off"),
    )
    _setup_wandb_metrics()
    return run


def _setup_wandb_metrics() -> None:
    """Define how every training metric appears in the wandb UI.

    The wildcard ``"*"`` definition tells wandb to use ``global_step`` as
    the x-axis for every chart.  We then explicitly clear ``step_metric``
    on ``global_step`` and ``iteration`` so the wildcard doesn't make them
    self-referencing (which would break all chart axes).
    """
    import wandb

    # -- Axes ---------------------------------------------------------------
    # All metrics use global_step as x-axis.
    wandb.define_metric("*", step_metric="global_step")
    # Prevent the wildcard from making global_step / iteration
    # self-referencing.  overwrite=True clears the step_metric the
    # wildcard just set on these two.
    wandb.define_metric("global_step", summary="last", step_metric=None, overwrite=True)
    wandb.define_metric("iteration", summary="last", step_metric=None, overwrite=True)

    # -- Episode rewards (headline numbers) ---------------------------------
    for m in (
        "mean_episode_reward", "mean_reward",
        "mean_task_reward", "mean_amp_reward",
        "rollout_episode_reward", "rollout_mean_reward",
        "rollout_mean_task_reward", "rollout_mean_amp_reward",
    ):
        wandb.define_metric(m, summary="last")

    # -- Episode statistics -------------------------------------------------
    for m in (
        "mean_episode_length", "mean_length",
        "rollout_episode_length",
        "mean_episode_progress", "done_fraction",
    ):
        wandb.define_metric(m, summary="last")

    # -- BPO training losses ------------------------------------------------
    for m in (
        "policy_loss",
        "value_loss",
        "median_value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "bpo_ratio_error",
        "bpo_target_ratio",
        "bpo_policy_weight",
        "bpo_median_advantage_abs",
        "learning_rate",
    ):
        wandb.define_metric(m, summary="last")

    # -- AMP discriminator --------------------------------------------------
    for m in ("amp_loss", "amp_grad_penalty", "amp_agent_prob", "amp_ref_prob"):
        wandb.define_metric(m, summary="last")


def _log_wandb_bootstrap(run, *, start_iteration: int, global_step: int) -> None:
    bootstrap_metrics = {
        "global_step": float(global_step),
        "iteration": float(start_iteration),
        "mean_reward": 0.0,
        "mean_episode_reward": 0.0,
        "mean_task_reward": 0.0,
        "mean_amp_reward": 0.0,
        "mean_length": 0.0,
        "mean_episode_length": 0.0,
        "rollout_mean_reward": 0.0,
        "rollout_mean_task_reward": 0.0,
        "rollout_mean_amp_reward": 0.0,
        "run_bootstrap": 1.0,
    }
    run.log(bootstrap_metrics, step=int(global_step))


def filter_resume_wandb_warmup_metrics(metrics: dict[str, float]) -> dict[str, float]:
    filtered: dict[str, float] = {}
    for key, value in metrics.items():
        if (
            "reward" in key
            or "length" in key
            or key.startswith("task_")
            or key in {"mean_episode_progress", "done_fraction"}
        ):
            continue
        filtered[key] = value
    filtered["resume_wandb_warmup"] = 1.0
    return filtered


def should_filter_resume_wandb_metrics(
    metrics: dict[str, float],
    *,
    iteration: int,
    warmup_until_iteration: int,
    min_episode_reward: float | None,
    min_episode_length: float | None,
) -> bool:
    if iteration < warmup_until_iteration:
        return True
    if min_episode_reward is not None and metrics.get("mean_episode_reward", 0.0) < min_episode_reward:
        return True
    if min_episode_length is not None and metrics.get("mean_episode_length", 0.0) < min_episode_length:
        return True
    return False


def save_config(log_dir: Path, args: argparse.Namespace, env_cfg: LytRobotEnvCfg, ppo_cfg: PPOAMPConfig) -> None:
    payload = {
        "project_version": __version__,
        "argv": sys.argv,
        "args": vars(args),
        "env": asdict(env_cfg),
        "ppo_amp": asdict(ppo_cfg),
    }
    (log_dir / "params").mkdir(parents=True, exist_ok=True)
    config_path = log_dir / "params" / "config.json"
    if args.log_dir is not None and config_path.exists():
        config_path = log_dir / "params" / f"resume_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _running_in_vscode_terminal() -> bool:
    return os.environ.get("TERM_PROGRAM") == "vscode" or "VSCODE_PID" in os.environ


def _effective_console_log_interval(log_interval: int) -> int:
    env_value = os.environ.get("LYTROBOT_CONSOLE_LOG_INTERVAL")
    if env_value:
        try:
            return max(int(env_value), 1)
        except ValueError:
            print(f"[WARN] ignoring invalid LYTROBOT_CONSOLE_LOG_INTERVAL={env_value!r}")
    interval = max(int(log_interval), 1)
    if _running_in_vscode_terminal():
        return max(interval, 10)
    return interval


class TrainingRunLogger:
    """Keep run logging file-oriented so VSCode terminals and watchers stay responsive."""

    def __init__(self, metrics_csv_path: Path, train_log_path: Path, *, log_interval: int) -> None:
        self.metrics_csv_path = metrics_csv_path
        self.train_log_path = train_log_path
        self.console_log_interval = _effective_console_log_interval(log_interval)
        self._csv_file = None
        self._csv_writer: csv.DictWriter | None = None
        self._log_file = train_log_path.open("a", encoding="utf-8", buffering=64 * 1024)
        if self.console_log_interval > max(int(log_interval), 1):
            print(
                "[INFO] VSCode terminal detected; "
                f"console training logs every {self.console_log_interval} iterations, "
                f"full logs at {self.train_log_path}"
            )

    def _open_csv_writer(self, metrics: dict[str, float]) -> None:
        requested_fieldnames = list(metrics.keys())
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.metrics_csv_path.exists() or self.metrics_csv_path.stat().st_size == 0
        fieldnames = requested_fieldnames
        if not write_header:
            with self.metrics_csv_path.open(newline="", encoding="utf-8") as existing_file:
                reader = csv.DictReader(existing_file)
                existing_fieldnames = list(reader.fieldnames or [])
                fieldnames = existing_fieldnames + [
                    field for field in requested_fieldnames if field not in existing_fieldnames
                ]
                if fieldnames != existing_fieldnames:
                    rows = list(reader)
                    temp_path = self.metrics_csv_path.with_suffix(self.metrics_csv_path.suffix + ".tmp")
                    with temp_path.open("w", newline="", encoding="utf-8") as temp_file:
                        writer = csv.DictWriter(temp_file, fieldnames=fieldnames, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(rows)
                    temp_path.replace(self.metrics_csv_path)
        self._csv_file = self.metrics_csv_path.open("a", newline="", encoding="utf-8", buffering=1024 * 1024)
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            self._csv_writer.writeheader()

    def write_metrics(self, metrics: dict[str, float]) -> None:
        if self._csv_writer is None:
            self._open_csv_writer(metrics)
        self._csv_writer.writerow(metrics)

    def emit(self, message: str, *, iteration: int | None = None, force: bool = False) -> None:
        self._log_file.write(message + "\n")
        if force or iteration is None or iteration % 10 == 0:
            self._log_file.flush()
            if self._csv_file is not None:
                self._csv_file.flush()
        should_print = force or iteration is None or iteration % self.console_log_interval == 0
        if should_print:
            print(message)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        self._log_file.close()


def raise_keyboard_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def release_host_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def read_gpu_snapshot(device: str) -> dict[str, float]:
    if device != "cuda" or not torch.cuda.is_available():
        return {}

    snapshot = {
        "cuda_mem_allocated_mb": torch.cuda.memory_allocated() / (1024.0 * 1024.0),
        "cuda_mem_reserved_mb": torch.cuda.memory_reserved() / (1024.0 * 1024.0),
        "cuda_mem_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "cuda_mem_peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024.0 * 1024.0),
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        first_line = result.stdout.strip().splitlines()[0]
        used_mb, temp_c, power_w = [float(value.strip()) for value in first_line.split(",")[:3]]
        snapshot.update(
            {
                "gpu_total_used_mb": used_mb,
                "gpu_temp_c": temp_c,
                "gpu_power_w": power_w,
            }
        )
    except Exception:
        pass
    try:
        with Path("/proc/self/status").open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    snapshot["process_rss_mb"] = float(line.split()[1]) / 1024.0
                    break
        with Path("/proc/meminfo").open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    snapshot["system_mem_available_mb"] = float(line.split()[1]) / 1024.0
                    break
    except (OSError, ValueError, IndexError):
        pass
    return snapshot


def main() -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)

    args = parse_args()
    # Fail before creating a log directory, starting W&B, or initializing the
    # simulator. Long runs commonly replace numbered snapshots with
    # model_final.pt during graceful shutdown.
    args.resume_checkpoint = resolve_resume_checkpoint(args.resume_checkpoint)
    if args.domain_randomization is None:
        args.domain_randomization = args.stage == "stage2"
    if args.max_iterations is None:
        args.max_iterations = 7500 if args.stage == "stage2" else 30000
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = select_device(args.device)
    sim_backend = select_sim_backend(args.sim_backend, device)
    log_dir = resolve_log_dir(args)
    print(f"[INFO] log_dir={log_dir}")
    print(
        f"[INFO] stage={args.stage}, device={device}, sim_backend={sim_backend}, "
        f"seed={args.seed}, control=position, "
        f"domain_randomization={args.domain_randomization}, "
        f"episode_random_seed={args.episode_random_seed}, wandb_default={args.wandb}"
    )

    env_cfg = LytRobotEnvCfg(
        model_xml=args.model_xml,
        motion_file=args.motion_file,
        num_envs=args.num_envs,
        seed=args.seed,
        randomize_episode_seed=bool(args.episode_random_seed),
        decimation=args.decimation,
        episode_steps=args.episode_steps,
        episode_timeout_jitter_steps=args.episode_timeout_jitter_steps,
        solver_iterations=args.solver_iterations,
        solver_ls_iterations=args.solver_ls_iterations,
        warp_nconmax=args.warp_nconmax,
        warp_njmax=args.warp_njmax,
        warp_cuda_graph=args.warp_cuda_graph,
        target_velocity=args.target_velocity,
        forward_velocity_tracking_sigma=args.forward_velocity_tracking_sigma,
        forward_velocity_reward_coef=args.forward_velocity_reward_coef,
        forward_progress_reward_coef=args.forward_progress_reward_coef,
        forward_velocity_deficit_penalty_coef=args.forward_velocity_deficit_penalty_coef,
        forward_velocity_excess_penalty_coef=args.forward_velocity_excess_penalty_coef,
        reset_noise_scale=args.reset_noise_scale,
        motion_reset_prob=args.motion_reset_prob,
        action_residual_scale=args.action_residual_scale,
        distal_action_scale=args.distal_action_scale,
        reference_ctrl_blend=args.reference_ctrl_blend,
        foot_vel_tracking_sigma=args.foot_vel_tracking_sigma,
        stance_foot_vel_sigma=args.stance_foot_vel_sigma,
        foot_vel_reward_coef=args.foot_vel_reward_coef,
        stance_foot_reward_coef=args.stance_foot_reward_coef,
        contact_reward_coef=args.contact_reward_coef,
        missed_contact_penalty_coef=args.missed_contact_penalty_coef,
        side_foot_pos_sigma=args.side_foot_pos_sigma,
        step_gap_sigma=args.step_gap_sigma,
        swing_advance_sigma=args.swing_advance_sigma,
        side_foot_pos_coef=args.side_foot_pos_coef,
        step_gap_coef=args.step_gap_coef,
        swing_advance_coef=args.swing_advance_coef,
        distal_joint_vel_sigma=args.distal_joint_vel_sigma,
        symmetry_reward_coef=args.symmetry_reward_coef,
        symmetry_foot_pos_sigma=args.symmetry_foot_pos_sigma,
        symmetry_balance_sigma=args.symmetry_balance_sigma,
        symmetry_ema_decay=args.symmetry_ema_decay,
        symmetry_foot_weight=args.symmetry_foot_weight,
        symmetry_contact_weight=args.symmetry_contact_weight,
        symmetry_action_weight=args.symmetry_action_weight,
        startup_action_ramp_steps=args.startup_action_ramp_steps,
        stand_start_prob=args.stand_start_prob,
        stand_start_phase_delay_steps=args.stand_start_phase_delay_steps,
        stand_start_transition_min_steps=args.stand_start_transition_min_steps,
        stand_start_transition_max_steps=args.stand_start_transition_max_steps,
        stand_start_push_delay_steps=args.stand_start_push_delay_steps,
        ctrl_penalty_coef=args.ctrl_penalty_coef,
        action_rate_penalty_coef=args.action_rate_penalty_coef,
        distal_action_penalty_coef=args.distal_action_penalty_coef,
        distal_action_rate_penalty_coef=args.distal_action_rate_penalty_coef,
        domain_randomization=bool(args.domain_randomization),
        payload_mass_min=args.payload_mass_min,
        payload_mass_max=args.payload_mass_max,
        body_mass_scale_min=args.body_mass_scale_min,
        body_mass_scale_max=args.body_mass_scale_max,
        tendon_length_scale_min=args.tendon_length_scale_min,
        tendon_length_scale_max=args.tendon_length_scale_max,
        ground_friction_min=args.ground_friction_min,
        ground_friction_max=args.ground_friction_max,
        gravity_min=args.gravity_min,
        gravity_max=args.gravity_max,
        push_force_min=args.push_force_min,
        push_force_max=args.push_force_max,
        push_interval_min_steps=args.push_interval_min_steps,
        push_interval_max_steps=args.push_interval_max_steps,
        push_duration_min_steps=args.push_duration_min_steps,
        push_duration_max_steps=args.push_duration_max_steps,
        obs_noise_std=args.obs_noise_std,
        action_noise_std=args.action_noise_std,
        action_delay_min_ms=args.action_delay_min_ms,
        action_delay_max_ms=args.action_delay_max_ms,
    )
    ppo_defaults = PPOAMPConfig()
    ppo_cfg = PPOAMPConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        reward_scale=args.reward_scale,
        clip_coef=args.clip_coef if args.clip_coef is not None else ppo_defaults.clip_coef,
        bpo_lambda=args.bpo_lambda if args.bpo_lambda is not None else ppo_defaults.bpo_lambda,
        bpo_tv_coef=args.bpo_tv_coef if args.bpo_tv_coef is not None else ppo_defaults.bpo_tv_coef,
        value_clip_coef=args.value_clip_coef,
        value_coef=args.value_coef,
        median_value_coef=(
            args.median_value_coef if args.median_value_coef is not None else ppo_defaults.median_value_coef
        ),
        target_kl=args.target_kl if args.target_kl is not None else ppo_defaults.target_kl,
        target_kl_stop_multiplier=(
            args.target_kl_stop_multiplier
            if args.target_kl_stop_multiplier is not None
            else ppo_defaults.target_kl_stop_multiplier
        ),
        adaptive_lr=args.adaptive_lr if args.adaptive_lr is not None else ppo_defaults.adaptive_lr,
        min_learning_rate=(
            args.min_learning_rate if args.min_learning_rate is not None else ppo_defaults.min_learning_rate
        ),
        max_learning_rate=(
            args.max_learning_rate if args.max_learning_rate is not None else ppo_defaults.max_learning_rate
        ),
        kl_adapt_factor=args.kl_adapt_factor if args.kl_adapt_factor is not None else ppo_defaults.kl_adapt_factor,
        discriminator_learning_rate=args.disc_learning_rate,
        num_steps=args.num_steps,
        update_epochs=args.epochs,
        num_minibatches=args.minibatches,
        amp_enabled=args.amp,
        amp_reward_coef=args.amp_reward_coef,
        task_reward_lerp=args.task_reward_lerp,
        amp_discriminator_updates=args.amp_disc_updates,
        amp_minibatch_size=args.amp_minibatch_size,
        amp_grad_penalty_coef=args.amp_grad_penalty_coef,
        amp_label_smoothing=args.amp_label_smoothing,
        amp_obs_noise_std=args.amp_obs_noise_std,
        amp_mirror_ref_prob=args.amp_mirror_ref_prob,
        min_action_std=args.min_action_std,
        max_action_std=args.max_action_std,
        episode_metric_window=args.episode_metric_window,
    )
    save_config(log_dir, args, env_cfg, ppo_cfg)

    run = maybe_init_wandb(args, log_dir)
    metrics_csv_path = log_dir / "metrics.csv"
    assert sim_backend == "warp"
    assert LytRobotWarpAmpEnv is not None
    env = LytRobotWarpAmpEnv(env_cfg, device=device)
    trainer = PPOAMPTrainer(env, ppo_cfg, device=device, log_dir=log_dir)

    # Optional: useful for debugging gradients, but expensive during long GPU/Warp runs.
    if run is not None and args.wandb_watch:
        import wandb as _wandb

        _wandb.watch(trainer.actor_critic, log="gradients", log_freq=500)
        if trainer.cfg.amp_enabled:
            _wandb.watch(trainer.discriminator, log="gradients", log_freq=500)

    start_iteration = 0
    if args.resume_checkpoint is not None:
        checkpoint_start_iteration = trainer.load(
            args.resume_checkpoint,
            load_optimizers=not args.resume_reset_optimizer,
        )
        checkpoint_global_step = int(trainer.global_step)
        if args.resume_reset_progress:
            start_iteration = 0
            trainer.global_step = 0
        else:
            start_iteration = checkpoint_start_iteration
            logged_progress = read_logged_training_progress(log_dir) if args.log_dir is not None else None
            if logged_progress is not None:
                logged_iteration, logged_global_step = logged_progress
                logged_start_iteration = logged_iteration + 1
                if logged_start_iteration > start_iteration or logged_global_step > int(trainer.global_step):
                    start_iteration = max(start_iteration, logged_start_iteration)
                    trainer.global_step = max(int(trainer.global_step), logged_global_step)
                    print(
                        "[INFO] fast-forwarded resume progress from existing train.log "
                        f"to iteration={start_iteration} global_step={trainer.global_step}"
                    )
        optimizer_state = "reset" if args.resume_reset_optimizer else "loaded"
        progress_state = (
            f"reset from iteration={checkpoint_start_iteration} global_step={checkpoint_global_step}"
            if args.resume_reset_progress
            else f"iteration={start_iteration} global_step={trainer.global_step}"
        )
        print(
            f"[INFO] resumed {args.resume_checkpoint} "
            f"at {progress_state} optimizer={optimizer_state}"
        )
    if start_iteration >= args.max_iterations:
        raise RuntimeError(
            f"Nothing to train: start_iteration={start_iteration} >= max_iterations={args.max_iterations}."
        )

    resuming_existing_log_dir = args.log_dir is not None and args.resume_checkpoint is not None
    if run is not None and args.wandb_bootstrap:
        _log_wandb_bootstrap(run, start_iteration=start_iteration, global_step=int(trainer.global_step))

    wandb_warmup_until_iteration = start_iteration
    if run is not None and resuming_existing_log_dir and not args.resume_reset_progress:
        warmup_iterations = max(int(args.resume_wandb_warmup_iterations), 0)
        wandb_warmup_until_iteration = start_iteration + warmup_iterations
        if warmup_iterations > 0:
            print(
                "[INFO] wandb resume warmup enabled: "
                f"skipping reward/episode/task metrics until iteration {wandb_warmup_until_iteration}"
            )

    last_iteration = start_iteration - 1
    last_global_step = trainer.global_step
    run_logger = TrainingRunLogger(
        metrics_csv_path,
        log_dir / "train.log",
        log_interval=args.log_interval,
    )
    try:
        for iteration in range(start_iteration, args.max_iterations):
            metrics = trainer.train_iteration()
            last_iteration = iteration
            last_global_step = int(metrics["global_step"])
            if args.memory_cleanup_interval > 0 and (iteration + 1) % args.memory_cleanup_interval == 0:
                release_host_memory()
            metrics["iteration"] = float(iteration)
            metrics.update({key: float("nan") for key in RESOURCE_METRIC_KEYS})
            resource_metrics = {}
            if args.resource_log_interval > 0 and iteration % args.resource_log_interval == 0:
                resource_metrics = read_gpu_snapshot(device)
                metrics.update(resource_metrics)
            run_logger.write_metrics(metrics)
            if iteration % max(args.log_interval, 1) == 0:
                run_logger.emit(
                    "[TRAIN] "
                    f"iter={iteration:05d} "
                    f"step={int(metrics['global_step'])} "
                    f"ep_reward={metrics['mean_episode_reward']:.4f} "
                    f"task={metrics['mean_task_reward']:.4f} "
                    f"amp={metrics['mean_amp_reward']:.4f} "
                    f"ep_len={metrics['mean_episode_length']:.2f} "
                    f"done={metrics['done_fraction']:.3f} "
                    f"pi={metrics['policy_loss']:.4f} "
                    f"v={metrics['value_loss']:.4f} "
                    f"mv={metrics['median_value_loss']:.4f} "
                    f"br={metrics['bpo_ratio_error']:.4f} "
                    f"kl={metrics['approx_kl']:.4f} "
                    f"lr={metrics['learning_rate']:.2e} "
                    f"d={metrics['amp_loss']:.4f} "
                    f"da={metrics['amp_agent_prob']:.3f} "
                    f"dr={metrics['amp_ref_prob']:.3f} "
                    f"fps={metrics['throughput_env_steps_s']:.0f} "
                    f"time={metrics['timing_iteration_s']:.2f}s",
                    iteration=iteration,
                )
            if resource_metrics:
                run_logger.emit(
                    "[RESOURCE] "
                    f"iter={iteration:05d} "
                    f"cuda_alloc={resource_metrics['cuda_mem_allocated_mb']:.0f}MB "
                    f"cuda_reserved={resource_metrics['cuda_mem_reserved_mb']:.0f}MB "
                    f"cuda_peak={resource_metrics['cuda_mem_peak_allocated_mb']:.0f}MB "
                    f"gpu_used={resource_metrics.get('gpu_total_used_mb', 0.0):.0f}MB "
                    f"rss={resource_metrics.get('process_rss_mb', 0.0):.0f}MB "
                    f"mem_avail={resource_metrics.get('system_mem_available_mb', 0.0):.0f}MB "
                    f"temp={resource_metrics.get('gpu_temp_c', 0.0):.0f}C "
                    f"power={resource_metrics.get('gpu_power_w', 0.0):.0f}W",
                    iteration=iteration,
                    force=True,
                )
                torch.cuda.reset_peak_memory_stats()
                process_rss_mb = resource_metrics.get("process_rss_mb", 0.0)
                if args.max_process_rss_mb > 0.0 and process_rss_mb >= args.max_process_rss_mb:
                    emergency_path = log_dir / f"model_memory_guard_{iteration + 1}.pt"
                    emergency_pt, emergency_onnx = trainer.save(emergency_path, iteration=iteration)
                    run_logger.emit(
                        "[MEMORY_GUARD] "
                        f"rss={process_rss_mb:.0f}MB reached limit={args.max_process_rss_mb:.0f}MB; "
                        f"saved {emergency_pt} and {emergency_onnx}; stopping before system OOM.",
                        force=True,
                    )
                    break
            if run is not None and iteration % max(args.wandb_log_interval, 1) == 0:
                # 兼容旧口径：若历史使用 `mean_reward`，把它同步为
                # `mean_episode_reward`，以便旧 run 与新面板都能显示。
                if "mean_episode_reward" not in metrics and "mean_reward" in metrics:
                    metrics["mean_episode_reward"] = float(metrics["mean_reward"])
                # Keep logging global_step explicitly because all charts use
                # step_metric="global_step".
                if should_filter_resume_wandb_metrics(
                    metrics,
                    iteration=iteration,
                    warmup_until_iteration=wandb_warmup_until_iteration,
                    min_episode_reward=args.resume_wandb_min_episode_reward,
                    min_episode_length=args.resume_wandb_min_episode_length,
                ):
                    run.log(filter_resume_wandb_warmup_metrics(metrics), step=int(metrics["global_step"]))
                else:
                    metrics["resume_wandb_warmup"] = 0.0
                    run.log(metrics, step=int(metrics["global_step"]))
            if (iteration + 1) % max(args.save_interval, 1) == 0:
                ckpt_step = iteration + 1
                ckpt_path = log_dir / f"model_{ckpt_step}.pt"
                saved_pt, saved_onnx = trainer.save(ckpt_path, iteration=iteration)
                run_logger.emit(f"[INFO] saved latest models: {saved_pt}, {saved_onnx}", force=True)
            if args.iteration_sleep_ms > 0.0:
                torch.cuda.synchronize()
                time.sleep(args.iteration_sleep_ms / 1000.0)
    finally:
        trainer.global_step = last_global_step
        final_path = log_dir / "model_final.pt"
        saved_pt, saved_onnx = trainer.save(final_path, iteration=last_iteration)
        run_logger.emit(f"[INFO] saved latest models: {saved_pt}, {saved_onnx}", force=True)
        run_logger.close()
        env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
