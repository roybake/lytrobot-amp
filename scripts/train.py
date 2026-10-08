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
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from lytrobot_amp.training_paths import new_run_dir, validate_run_name

from lytrobot_amp import __version__
from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.ppo_amp import PPOAMPConfig, PPOAMPTrainer

try:
    from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
except Exception as exc:  # pragma: no cover - optional backend import guard.
    LytRobotWarpAmpEnv = None
    WARP_IMPORT_ERROR = exc
else:
    WARP_IMPORT_ERROR = None

try:
    from lytrobot_amp.env_mjlab import LytRobotMjlabAmpEnv
except Exception as exc:  # pragma: no cover - optional backend import guard.
    LytRobotMjlabAmpEnv = None
    MJLAB_IMPORT_ERROR = exc
else:
    MJLAB_IMPORT_ERROR = None


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

WANDB_EPISODE_TOTAL_REWARD_KEY = "episode/total_reward"
WANDB_EPISODE_LENGTH_KEY = "episode/length"
# Enabled only by sensor_training/train_distill.py before main() is called.
DISTILLATION_CLI_ENABLED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LYT biped locomotion with BPO-AMP.")
    parser.add_argument("--model-xml", default=str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"))
    parser.add_argument("--motion-file", default=str(REPO_ROOT / "robot" / "reference_data" / "walk1_mjcf_repaired.npz"))
    parser.add_argument(
        "--motion-files",
        nargs="+",
        default=None,
        help="Retargeted clips for multi-action AMP; overrides --motion-file.",
    )
    parser.add_argument("--motion-names", nargs="+", default=None)
    parser.add_argument("--motion-weights", nargs="+", type=float, default=None)
    parser.add_argument(
        "--motion-loop",
        nargs="+",
        choices=("true", "false"),
        default=None,
        help="One loop flag per --motion-files clip.",
    )
    parser.add_argument(
        "--skill-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-reference-velocity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Track each clip's retargeted linear/angular root velocity.",
    )
    parser.add_argument(
        "--terrain-types",
        nargs="+",
        choices=("gap", "pit", "rough", "slope_up", "slope_down", "stairs"),
        default=None,
        help="MoRE-discoRL terrain lanes to attach to every batched world.",
    )
    parser.add_argument(
        "--terrain-asset-dir",
        default=str(REPO_ROOT / "robot" / "terrains" / "more_discolrl"),
    )
    parser.add_argument(
        "--terrain-embedded-in-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use terrain already authored in --model-xml instead of attaching another copy.",
    )
    parser.add_argument(
        "--preserve-motion-world-xy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep reference root XY in the MJCF world frame (for contact-solved terrain clips).",
    )
    parser.add_argument("--deepmimic-track-root-xy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deepmimic-root-xy-penalty", type=float, default=0.0)
    parser.add_argument("--deepmimic-route-tracking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deepmimic-route-xy-sigma", type=float, default=0.35)
    parser.add_argument("--deepmimic-route-velocity-sigma", type=float, default=0.25)
    parser.add_argument("--deepmimic-route-max-xy-error", type=float, default=0.60)
    parser.add_argument("--trajectory-feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sensor-gait-command", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sensor-phase-observation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sensor-root-imu-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reference-root-x-offset", type=float, default=0.0)
    parser.add_argument("--stair-terrain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--stair-step-height", type=float, default=0.13,
        help="楼梯立板高度（专项课程固定为 0.13 m）。",
    )
    parser.add_argument(
        "--stair-step-depth", type=float, default=0.32,
        help="前进方向踏步宽度（专项课程固定为 0.32 m）。",
    )
    parser.add_argument(
        "--stair-tread-width",
        dest="stair_step_depth",
        type=float,
        default=argparse.SUPPRESS,
        help="--stair-step-depth 的语义别名（前进方向踏步宽度）。",
    )
    parser.add_argument("--stair-step-count", type=int, default=5)
    parser.add_argument("--stair-width", type=float, default=2.0)
    parser.add_argument("--stair-run-in", type=float, default=1.5)
    parser.add_argument("--stair-top-length", type=float, default=0.64)
    parser.add_argument("--stair-run-out", type=float, default=1.5)
    parser.add_argument("--stair-start-x", type=float, default=-0.95)
    parser.add_argument("--stair-down-offset", type=float, default=0.25)
    parser.add_argument(
        "--stair-fixed-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="每个楼梯专项 episode 固定从参考动作第 0 帧开始。",
    )
    parser.add_argument("--stair-phase-reset-prob", type=float, default=0.0)
    parser.add_argument(
        "--stair-distance-tolerance", type=float, default=0.05,
        help="楼梯参考根轨迹末端距离容差（米）。",
    )
    parser.add_argument(
        "--stair-distance-failure-penalty", type=float, default=5.0,
        help="未达到固定楼梯距离时的终止惩罚。",
    )
    parser.add_argument(
        "--stair-lateral-tolerance", type=float, default=0.30,
        help="楼梯中心线横向偏差上限（米），用于阻止绕开台阶。",
    )
    parser.add_argument(
        "--stair-lateral-failure-penalty", type=float, default=5.0,
        help="横向绕开楼梯时的终止惩罚。",
    )
    parser.add_argument(
        "--stair-terminal-hold-steps", type=int, default=30,
        help="专项楼梯参考末帧的稳定保持控制步数。",
    )
    parser.add_argument("--flat-lane-y", type=float, default=3.0)
    parser.add_argument("--rough-lane-y", type=float, default=6.0)
    parser.add_argument("--pit-lane-y", type=float, default=9.0)
    parser.add_argument("--gap-lane-y", type=float, default=12.0)
    parser.add_argument("--terrain-lane-width", type=float, default=2.0)
    parser.add_argument("--rough-terrain-height", type=float, default=0.05)
    parser.add_argument("--slope-angle-deg", type=float, default=6.0)
    parser.add_argument("--gap-terrain-width", type=float, default=0.20)
    parser.add_argument(
        "--serial-terrain-course",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Place every enabled terrain in a randomized serial course in front of each robot.",
    )
    parser.add_argument("--terrain-course-variants", type=int, default=8)
    parser.add_argument("--terrain-course-lane-spacing", type=float, default=3.0)
    parser.add_argument("--terrain-course-base-height", type=float, default=0.30)
    parser.add_argument("--terrain-course-completion-reward", type=float, default=25.0)
    parser.add_argument(
        "--terrain-course-stage-reward",
        type=float,
        default=0.0,
        help="Dense reward paid whenever a serial-course stage is completed.",
    )
    parser.add_argument("--terrain-course-stage-tolerance", type=float, default=0.35)
    parser.add_argument(
        "--terrain-course-random-start-prob",
        type=float,
        default=0.0,
        help=(
            "Probability that a training reset starts at a random nonzero course "
            "stage; full-route success is still counted only for stage-zero starts."
        ),
    )
    parser.add_argument(
        "--terrain-course-random-start-warmup-iterations",
        type=int,
        default=0,
        help=(
            "Keep every serial-course reset at stage zero for this many iterations "
            "before exposing later terrain stages."
        ),
    )
    parser.add_argument(
        "--terrain-course-random-start-ramp-iterations",
        type=int,
        default=0,
        help=(
            "Linearly ramp random-stage resets from zero to "
            "--terrain-course-random-start-prob over this many iterations."
        ),
    )
    parser.add_argument(
        "--terrain-course-random-assignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-root", default=str(REPO_ROOT / "logs"))
    parser.add_argument(
        "--log-group",
        default=None,
        help="Optional category below --log-root; defaults to the training stage.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Training log/checkpoint directory; created automatically if missing.",
    )
    parser.add_argument(
        "--log-role",
        choices=("teacher", "student"),
        default=None,
        help="Save under a teacher/student subdirectory of the timestamped run.",
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
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--sim-backend", default="warp", choices=("warp", "mjlab", "cpu"))
    parser.add_argument("--quiet-startup", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=0)

    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--episode-timeout-jitter-steps", type=int, default=240)
    parser.add_argument(
        "--decimation",
        type=int,
        default=2,
        help=(
            "Physics steps per policy action (physics timestep: 5 ms). "
            "Default 2 = 100 Hz control; use 4 for 50 Hz compatibility."
        ),
    )
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--solver-ls-iterations", type=int, default=20)
    parser.add_argument("--warp-nconmax", type=int, default=32)
    parser.add_argument("--warp-njmax", type=int, default=128)
    parser.add_argument("--warp-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-velocity", type=float, default=0.87)
    parser.add_argument("--forward-velocity-tracking-sigma", type=float, default=0.25)
    parser.add_argument("--forward-velocity-reward-coef", type=float, default=0.0)
    parser.add_argument("--forward-progress-reward-coef", type=float, default=0.0)
    parser.add_argument("--forward-velocity-deficit-penalty-coef", type=float, default=0.0)
    parser.add_argument("--forward-velocity-excess-penalty-coef", type=float, default=0.0)
    parser.add_argument(
        "--min-forward-progress-speed",
        type=float,
        default=0.0,
        help="Terminate policies whose net world-X speed stays below this threshold; 0 disables.",
    )
    parser.add_argument("--forward-progress-grace-steps", type=int, default=50)
    parser.add_argument("--forward-progress-window-steps", type=int, default=50)
    parser.add_argument("--forward-progress-termination-penalty", type=float, default=2.0)
    parser.add_argument("--fall-penalty", type=float, default=2.0)
    parser.add_argument(
        "--task-reward-mode",
        choices=("stage1_reference", "sensor_stability"),
        default=None,
        help="Explicit task objective; supersedes the legacy sensor-stability boolean.",
    )
    parser.add_argument(
        "--sensor-stability-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the phase-independent stability/velocity task reward in the "
            "sensor-only environment."
        ),
    )
    parser.add_argument("--sensor-fall-penalty", type=float, default=25.0)
    parser.add_argument("--sensor-knee-flexion-target", type=float, default=0.25)
    parser.add_argument("--sensor-knee-gait-reward-coef", type=float, default=0.0)
    parser.add_argument("--sensor-natural-gait-reward-coef", type=float, default=0.0)
    parser.add_argument("--reference-gait-style-reward-coef", type=float, default=0.0)
    parser.add_argument('--reference-joint-position-reward-coef', type=float, default=1.4)
    parser.add_argument('--reference-joint-velocity-reward-coef', type=float, default=.45)
    parser.add_argument('--reference-root-velocity-reward-coef', type=float, default=.9)
    parser.add_argument('--reference-foot-position-reward-coef', type=float, default=1.25)
    parser.add_argument("--sensor-gait-height-target", type=float, default=0.845)
    parser.add_argument("--sensor-gait-height-sigma", type=float, default=0.07)
    parser.add_argument("--sensor-upright-sigma", type=float, default=0.12)
    parser.add_argument("--sensor-mirror-reward-coef", type=float, default=0.0)
    parser.add_argument("--sensor-stride-reward-coef", type=float, default=0.0)
    parser.add_argument("--sensor-foot-roll-reward-coef", type=float, default=0.0)
    parser.add_argument("--sensor-alternation-reward-coef", type=float, default=0.0)
    parser.add_argument("--sensor-hop-penalty-coef", type=float, default=0.0)
    parser.add_argument("--sensor-tiptoe-penalty-coef", type=float, default=0.0)
    parser.add_argument("--sensor-stance-slip-penalty-coef", type=float, default=0.0)
    parser.add_argument("--sensor-sync-forward-penalty-coef", type=float, default=0.0)
    parser.add_argument("--sensor-velocity-excess-penalty-coef", type=float, default=0.0)
    parser.add_argument(
        "--terminate-incomplete-motion-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Terminate a non-looping serial-course stage that ends before its spatial boundary.",
    )
    parser.add_argument("--reset-noise-scale", type=float, default=0.01)
    parser.add_argument("--motion-reset-prob", type=float, default=0.5)
    parser.add_argument("--action-residual-scale", type=float, default=0.5)
    parser.add_argument("--distal-action-scale", type=float, default=1.0)
    parser.add_argument("--reference-ctrl-blend", type=float, default=1.0)
    parser.add_argument(
        "--motion-transition-blend-steps",
        type=int,
        default=0,
        help="Control steps used to blend the reference target after a serial-course motion switch.",
    )
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
    parser.add_argument(
        "--stand-start-motion-phase",
        type=float,
        default=0.0,
        help=(
            "Reference frame used when blending from the standing reset into motion; "
            "choose a double-support frame for deployable walking starts."
        ),
    )
    parser.add_argument("--stand-start-phase-delay-steps", type=int, default=25)
    parser.add_argument("--stand-start-transition-min-steps", type=int, default=80)
    parser.add_argument("--stand-start-transition-max-steps", type=int, default=160)
    parser.add_argument("--stand-start-push-delay-steps", type=int, default=100)
    parser.add_argument("--ctrl-penalty-coef", type=float, default=0.040)
    parser.add_argument("--action-rate-penalty-coef", type=float, default=0.100)
    parser.add_argument("--distal-action-penalty-coef", type=float, default=0.030)
    parser.add_argument("--distal-action-rate-penalty-coef", type=float, default=0.120)
    parser.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--dr-curriculum-mode",
        choices=("linear", "adaptive"),
        default="linear",
        help="Use a time-based linear ramp or performance-gated adaptive domain randomization.",
    )
    parser.add_argument(
        "--dr-curriculum-iterations",
        type=int,
        default=0,
        help="Linearly ramp domain-randomization strength from 0 to 1 over this many iterations.",
    )
    parser.add_argument("--dr-curriculum-initial-scale", type=float, default=0.0)
    parser.add_argument("--dr-curriculum-update-interval", type=int, default=250)
    parser.add_argument("--dr-curriculum-step", type=float, default=0.025)
    parser.add_argument("--dr-promote-success-rate", type=float, default=0.50)
    parser.add_argument("--dr-promote-depth-success-rate", type=float, default=0.30)
    parser.add_argument("--dr-promote-max-termination", type=float, default=0.001)
    parser.add_argument("--dr-backoff-success-rate", type=float, default=0.30)
    parser.add_argument("--dr-backoff-depth-success-rate", type=float, default=0.15)
    parser.add_argument("--dr-backoff-termination", type=float, default=0.002)
    parser.add_argument(
        "--warp-global-domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to Warp's shared mass/inertia/friction/gravity draw. This is one draw for the "
            "whole batch, not per-environment randomization."
        ),
    )
    parser.add_argument(
        "--global-dr-resample-interval",
        type=int,
        default=0,
        help=(
            "When shared Warp randomization is enabled, synchronously reset all worlds and resample it every "
            "N iterations. Set 0 to disable; synchronized resets perturb training curves."
        ),
    )
    parser.add_argument("--payload-mass-min", type=float, default=0.0)
    parser.add_argument("--payload-mass-max", type=float, default=3.0)
    parser.add_argument("--body-mass-scale-min", type=float, default=0.85)
    parser.add_argument("--body-mass-scale-max", type=float, default=1.15)
    parser.add_argument("--tendon-length-scale-min", type=float, default=0.95)
    parser.add_argument("--tendon-length-scale-max", type=float, default=1.05)
    parser.add_argument('--actuator-strength-scale-min', type=float, default=1.0)
    parser.add_argument('--actuator-strength-scale-max', type=float, default=1.0)
    parser.add_argument('--gait-fidelity-selection', action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument(
        "--use-moe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable residual MoE for the serial multi-terrain sensor curriculum.",
    )
    parser.add_argument(
        "--compile-policy-update",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compile the fixed-shape recurrent policy backward update with "
            "torch.compile. Enabled by default; first compilation can take a few minutes."
        ),
    )
    parser.add_argument(
        "--compile-rollout",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compile the no-grad recurrent rollout and AMP reward graphs. "
            "Enabled by default for the sensor trainer."
        ),
    )
    parser.add_argument(
        "--compile-policy-chunk-steps",
        type=int,
        default=None,
        help=(
            "BPTT steps per compiled recurrent-policy graph. "
            "The sensor-policy default 16 compiles and caches a bounded recurrent chunk, avoiding "
            "both per-step Python calls and an entire rollout unroll."
        ),
    )
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
    if DISTILLATION_CLI_ENABLED:
        parser.add_argument(
            "--teacher-checkpoint",
            required=True,
            help=(
                "Privileged full-state Teacher checkpoint used only by the "
                "dedicated sensor-distillation entry point."
            ),
        )
        parser.add_argument("--teacher-action-loss-coef", type=float, default=0.0)
        parser.add_argument("--teacher-action-loss-final-coef", type=float, default=0.0)
        parser.add_argument("--teacher-action-loss-decay-iterations", type=int, default=0)
        parser.add_argument("--teacher-action-loss-scale", type=float, default=1.0)
        parser.add_argument("--teacher-action-probability", type=float, default=0.0)
        parser.add_argument("--teacher-action-probability-final", type=float, default=0.0)
        parser.add_argument("--teacher-action-probability-decay-iterations", type=int, default=5000)
        parser.add_argument(
            "--distillation-reset-schedule",
            action="store_true",
            help="Restart the Teacher-action loss schedule after loading a Student checkpoint.",
        )
    parser.add_argument("--min-action-std", type=float, default=0.05)
    parser.add_argument("--max-action-std", type=float, default=0.60)
    parser.add_argument("--episode-metric-window", type=int, default=4096)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--evaluate-gait", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gait-evaluation-seconds", type=float, default=20.0)
    parser.add_argument("--gait-evaluation-seeds", type=int, nargs='+', default=[42, 123, 2026])
    parser.add_argument("--stop-on-gait-acceptance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--save-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve model_best.pt/.onnx when rollout stability improves.",
    )
    parser.add_argument(
        "--best-checkpoint-min-completed",
        type=int,
        default=32,
        help="Minimum completed episodes before model_best can be updated.",
    )
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
    parser.add_argument("--deepmimic-reward", action="store_true", help="Use five-term DeepMimic tracking reward instead of the task reward.")
    parser.add_argument("--lcp-weight", type=float, default=0.0, help="Sensor-actor Lipschitz penalty weight (sensor stair trainer only).")
    parser.add_argument("--deepmimic-ppo", action="store_true", help="Use clipped PPO objective in the recurrent sensor trainer.")
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
    if device_arg == "cpu":
        return "cpu"
    if device_arg != "cuda":
        raise RuntimeError(f"Unsupported training device: {device_arg}.")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA, but torch.cuda.is_available() is false.")
    return "cuda"


def select_sim_backend(backend_arg: str, device: str) -> str:
    if backend_arg == "cpu":
        return "cpu"
    if backend_arg not in ("warp", "mjlab"):
        raise RuntimeError(f"Unsupported simulation backend: {backend_arg}.")
    if device != "cuda":
        raise RuntimeError(f"--sim-backend {backend_arg} requires --device cuda.")
    if backend_arg == "warp" and LytRobotWarpAmpEnv is None:
        raise RuntimeError("The warp backend is unavailable; install mujoco-warp and warp-lang.") from WARP_IMPORT_ERROR
    if backend_arg == "mjlab" and LytRobotMjlabAmpEnv is None:
        raise RuntimeError("The mjlab backend is unavailable; install mjlab.") from MJLAB_IMPORT_ERROR
    return backend_arg


def make_log_dir(
    log_root: str,
    stage: str,
    run_name: str | None,
    log_group: str | None = None,
    log_role: str | None = None,
) -> Path:
    if log_role is not None:
        directory = new_run_dir(Path(log_root), run_name) / log_role
        directory.mkdir()
        return directory
    if DISTILLATION_CLI_ENABLED:
        directory = new_run_dir(Path(log_root), run_name) / stage / "student"
        directory.mkdir(parents=True)
        return directory
    group = log_group or stage
    if Path(group).name != group or group in ("", ".", ".."):
        raise ValueError(f"--log-group must be one directory name, got: {group!r}")
    root = Path(log_root).expanduser().resolve() / group
    return new_run_dir(root, run_name)


def resolve_log_dir(args: argparse.Namespace) -> Path:
    if args.log_dir is None:
        if DISTILLATION_CLI_ENABLED and args.log_role == "student":
            teacher_dir = Path(args.teacher_checkpoint).expanduser().resolve().parent
            if args.run_name is None and teacher_dir.name != "teacher":
                raise ValueError(
                    "Default Student logs require a checkpoint in <run>/teacher; "
                    "specify --log-dir for a different checkpoint layout."
                )
            run_name = validate_run_name(args.run_name or teacher_dir.parent.name)
            log_dir = Path(args.log_root).expanduser().resolve() / run_name / "student"
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
            log_dir = make_log_dir(
                args.log_root, args.stage, args.run_name, args.log_group, args.log_role
            )
    else:
        log_dir = Path(args.log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
    if DISTILLATION_CLI_ENABLED:
        from lytrobot_amp.training_paths import snapshot_teacher
        snapshot_teacher(args.teacher_checkpoint, log_dir.parent / "teacher")
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
    run_name = log_dir.name
    if run_name in ("teacher", "student"):
        run_dir = log_dir.parent
        if run_dir.name in ("stage1", "stage2"):
            run_dir = run_dir.parent
        run_name = f"{run_dir.name}-{run_name}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=None if resume_from is not None else run_id,
        resume=None if resume_from is not None else ("allow" if run_id is not None else None),
        resume_from=resume_from,
        name=run_name,
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
        WANDB_EPISODE_TOTAL_REWARD_KEY,
        "mean_episode_reward", "mean_reward",
        "mean_task_reward", "mean_amp_reward",
        "rollout_episode_reward", "rollout_mean_reward",
        "rollout_mean_task_reward", "rollout_mean_amp_reward",
    ):
        wandb.define_metric(m, summary="last")

    # -- Episode statistics -------------------------------------------------
    for m in (
        WANDB_EPISODE_LENGTH_KEY,
        "mean_episode_length", "mean_length",
        "rollout_episode_length",
        "mean_episode_progress", "done_fraction",
        "timeout_fraction", "termination_fraction",
        "termination_episode_step_mean", "termination_episode_step_min",
        "termination_episode_step_max",
        "episode_success_rate", "mean_episode_success_rate", "rollout_episode_success_rate",
        "completed_episode_count", "successful_episode_count",
        "bottom_depth_success_rate",
        "task_invalid_state",
        "nonfinite_task_reward_count", "nonfinite_task_reward_fraction",
        "domain_randomization_scale", "domain_randomization_scale_next",
        "dr_window_success_rate", "dr_window_depth_success_rate",
        "dr_window_termination_fraction", "dr_curriculum_decision",
        "terrain_course_random_start_probability",
        "terrain_course_random_start_target_probability",
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
        "gate_change_loss",
        "expert_load_balance_loss",
        "residual_l2_loss",
        "privileged_context_loss",
        "policy_updates_skipped_nonfinite",
        "policy_samples_dropped_nonfinite",
        "importance_ratio_log_clip_fraction",
        "learning_rate",
    ):
        wandb.define_metric(m, summary="last")

    # -- AMP discriminator --------------------------------------------------
    for m in (
        "amp_loss",
        "amp_grad_penalty",
        "amp_agent_prob",
        "amp_ref_prob",
        "amp_updates_skipped_nonfinite",
    ):
        wandb.define_metric(m, summary="last")


def prepare_wandb_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Add stable W&B episode curves without changing CSV/log metric names.

    ``episode/total_reward`` is the mean cumulative reward of completed
    episodes, not the per-step rollout reward.  Keep the historical flat
    aliases so old dashboards and checkpoints remain comparable.
    """
    prepared = dict(metrics)

    episode_reward = prepared.get("mean_episode_reward", prepared.get("mean_reward"))
    if episode_reward is not None:
        episode_reward = float(episode_reward)
        prepared["mean_episode_reward"] = episode_reward
        prepared["mean_reward"] = episode_reward
        prepared[WANDB_EPISODE_TOTAL_REWARD_KEY] = episode_reward

    episode_length = prepared.get("mean_episode_length", prepared.get("mean_length"))
    if episode_length is not None:
        episode_length = float(episode_length)
        prepared["mean_episode_length"] = episode_length
        prepared["mean_length"] = episode_length
        prepared[WANDB_EPISODE_LENGTH_KEY] = episode_length

    return prepared


def domain_randomization_scale_at_iteration(iteration: int, curriculum_iterations: int) -> float:
    """Return the linear domain-randomization curriculum strength in [0, 1]."""

    if curriculum_iterations <= 0:
        return 1.0
    return float(np.clip(float(iteration) / float(curriculum_iterations), 0.0, 1.0))


def terrain_course_random_start_prob_at_iteration(
    iteration: int,
    target_probability: float,
    warmup_iterations: int,
    ramp_iterations: int,
) -> float:
    """Return a stage-zero warmup followed by a bounded linear exposure ramp."""

    target = float(np.clip(target_probability, 0.0, 1.0))
    warmup = max(int(warmup_iterations), 0)
    ramp = max(int(ramp_iterations), 0)
    if int(iteration) < warmup:
        return 0.0
    if ramp == 0:
        return target
    progress = (float(iteration) - float(warmup)) / float(ramp)
    return target * float(np.clip(progress, 0.0, 1.0))


def adaptive_domain_randomization_step(
    current_scale: float,
    *,
    success_rate: float,
    depth_success_rate: float,
    termination_fraction: float,
    step: float,
    promote_success_rate: float,
    promote_depth_success_rate: float,
    promote_max_termination: float,
    backoff_success_rate: float,
    backoff_depth_success_rate: float,
    backoff_termination: float,
) -> tuple[float, int]:
    """Return next DR scale and decision (-1 backoff, 0 hold, +1 promote)."""

    current_scale = float(np.clip(current_scale, 0.0, 1.0))
    step = max(float(step), 0.0)
    if (
        success_rate >= promote_success_rate
        and depth_success_rate >= promote_depth_success_rate
        and termination_fraction <= promote_max_termination
    ):
        next_scale = float(min(1.0, current_scale + step))
        return next_scale, int(next_scale > current_scale)
    if (
        success_rate < backoff_success_rate
        or depth_success_rate < backoff_depth_success_rate
        or termination_fraction > backoff_termination
    ):
        next_scale = float(max(0.0, current_scale - step))
        return next_scale, -int(next_scale < current_scale)
    return current_scale, 0


def _log_wandb_bootstrap(run, *, start_iteration: int, global_step: int) -> None:
    bootstrap_metrics = prepare_wandb_metrics({
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
    })
    run.log(bootstrap_metrics, step=int(global_step))


def filter_resume_wandb_warmup_metrics(metrics: dict[str, float]) -> dict[str, float]:
    filtered: dict[str, float] = {}
    for key, value in metrics.items():
        if (
            "reward" in key
            or "length" in key
            or "success_rate" in key
            or key.startswith("task_")
            or key in {
                "mean_episode_progress",
                "done_fraction",
                "timeout_fraction",
                "termination_fraction",
            }
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


def load_best_checkpoint_quality(
    metrics_csv_path: Path,
    *,
    min_completed: int,
    deepmimic_ppo: bool = False,
    stair_route: bool = False,
) -> tuple[float, float]:
    """Recover the best checkpoint score when a run is resumed.

    A fixed stair route must rank actual full-route completions ahead of
    imitation reward.  RSI-only DeepMimic tasks retain reward ranking.
    """

    if not metrics_csv_path.is_file():
        return (-1.0, -1.0)
    best = (-1.0, -1.0)
    try:
        with metrics_csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                completed = float(row.get("completed_episode_count", "0") or 0.0)
                primary_key = (
                    "rollout_episode_success_rate"
                    if stair_route or not deepmimic_ppo
                    else "rollout_mean_task_reward"
                )
                secondary_key = (
                    "rollout_mean_task_reward" if stair_route
                    else "mean_episode_length"
                )
                primary = float(row.get(primary_key, "nan"))
                secondary = float(row.get(secondary_key, "nan"))
                if (
                    completed >= max(int(min_completed), 1)
                    and np.isfinite(primary)
                    and np.isfinite(secondary)
                ):
                    best = max(best, (primary, secondary))
    except (OSError, ValueError):
        return best
    return best


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
    if args.deepmimic_ppo and not hasattr(PPOAMPTrainer, "_evaluate_policy_sequence"):
        raise ValueError("--deepmimic-ppo requires the recurrent sensor training entry point.")
    if args.deepmimic_ppo and (not args.deepmimic_reward or args.amp):
        raise ValueError("--deepmimic-ppo requires --deepmimic-reward and --no-amp.")
    teacher_checkpoint = getattr(args, "teacher_checkpoint", None)
    teacher_action_loss_coef = float(getattr(args, "teacher_action_loss_coef", 0.0))
    teacher_action_loss_final_coef = float(
        getattr(args, "teacher_action_loss_final_coef", 0.0)
    )
    teacher_action_loss_decay_iterations = int(
        getattr(args, "teacher_action_loss_decay_iterations", 0)
    )
    distillation_reset_schedule = bool(
        getattr(args, "distillation_reset_schedule", False)
    )
    if args.min_forward_progress_speed < 0.0:
        raise ValueError("--min-forward-progress-speed must be non-negative.")
    if not 0.0 <= args.terrain_course_random_start_prob <= 1.0:
        raise ValueError("--terrain-course-random-start-prob must be in [0, 1].")
    if args.terrain_course_random_start_warmup_iterations < 0:
        raise ValueError(
            "--terrain-course-random-start-warmup-iterations must be non-negative."
        )
    if args.terrain_course_random_start_ramp_iterations < 0:
        raise ValueError(
            "--terrain-course-random-start-ramp-iterations must be non-negative."
        )
    if args.terrain_course_stage_reward < 0.0:
        raise ValueError("--terrain-course-stage-reward must be non-negative.")
    if args.use_moe and not args.serial_terrain_course:
        raise ValueError(
            "--use-moe is reserved for the serial terrain course; use "
            "sensor_training/train_multimotion.py."
        )
    if args.min_forward_progress_speed > 0.0 and args.forward_progress_window_steps <= 0:
        raise ValueError(
            "--forward-progress-window-steps must be positive when the forward-progress guard is enabled."
        )
    if args.forward_progress_grace_steps < 0:
        raise ValueError("--forward-progress-grace-steps must be non-negative.")
    if args.fall_penalty < 0.0:
        raise ValueError("--fall-penalty must be non-negative.")
    if args.stair_distance_tolerance < 0.0:
        raise ValueError("--stair-distance-tolerance must be non-negative.")
    if not 0.0 <= args.stair_phase_reset_prob <= 1.0:
        raise ValueError("--stair-phase-reset-prob must be in [0, 1].")
    if args.deepmimic_root_xy_penalty < 0.0:
        raise ValueError("--deepmimic-root-xy-penalty must be non-negative.")
    if args.deepmimic_track_root_xy and not args.deepmimic_reward:
        raise ValueError("--deepmimic-track-root-xy requires --deepmimic-reward.")
    if args.deepmimic_route_tracking:
        if not args.deepmimic_reward or not args.deepmimic_track_root_xy or args.sensor_stability_reward:
            raise ValueError("Route tracking requires DeepMimic world-XY tracking without the sensor stability reward.")
        if min(args.deepmimic_route_xy_sigma, args.deepmimic_route_velocity_sigma,
               args.deepmimic_route_max_xy_error) <= 0:
            raise ValueError("DeepMimic route tolerances must be positive.")
    if args.lcp_weight < 0.0:
        raise ValueError("--lcp-weight must be non-negative.")
    if args.stair_distance_failure_penalty < 0.0:
        raise ValueError("--stair-distance-failure-penalty must be non-negative.")
    if args.stair_lateral_tolerance < 0.0:
        raise ValueError("--stair-lateral-tolerance must be non-negative.")
    if args.stair_lateral_failure_penalty < 0.0:
        raise ValueError("--stair-lateral-failure-penalty must be non-negative.")
    if args.stair_terminal_hold_steps < 0:
        raise ValueError("--stair-terminal-hold-steps must be non-negative.")
    if args.sensor_knee_flexion_target <= 0.0:
        raise ValueError("--sensor-knee-flexion-target must be positive.")
    if args.sensor_knee_gait_reward_coef < 0.0:
        raise ValueError("--sensor-knee-gait-reward-coef must be non-negative.")
    if args.sensor_natural_gait_reward_coef < 0.0:
        raise ValueError("--sensor-natural-gait-reward-coef must be non-negative.")
    if args.sensor_gait_height_sigma <= 0.0:
        raise ValueError("--sensor-gait-height-sigma must be positive.")
    if args.sensor_upright_sigma <= 0.0:
        raise ValueError("--sensor-upright-sigma must be positive.")
    for name in (
        "sensor_mirror_reward_coef",
        "sensor_stride_reward_coef",
        "sensor_foot_roll_reward_coef",
        "sensor_alternation_reward_coef",
        "sensor_hop_penalty_coef",
        "sensor_tiptoe_penalty_coef",
        "sensor_stance_slip_penalty_coef",
        "sensor_sync_forward_penalty_coef",
        "sensor_velocity_excess_penalty_coef",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if teacher_action_loss_coef < 0.0 or teacher_action_loss_final_coef < 0.0:
        raise ValueError("Teacher action-loss coefficients must be non-negative.")
    if teacher_action_loss_decay_iterations < 0:
        raise ValueError("--teacher-action-loss-decay-iterations must be non-negative.")
    if args.reference_gait_style_reward_coef < 0:
        raise ValueError('Reference gait style coefficient must be non-negative.')
    if args.reference_gait_style_reward_coef > 0 and args.sensor_stability_reward:
        raise ValueError('Phase-aligned gait style requires the reference task reward.')
    if args.gait_evaluation_seconds <= 0:
        raise ValueError('Gait evaluation duration must be positive.')
    if args.stop_on_gait_acceptance and not args.evaluate_gait:
        raise ValueError('Stopping on gait acceptance requires --evaluate-gait.')
    if DISTILLATION_CLI_ENABLED:
        if args.teacher_action_loss_scale <= 0:
            raise ValueError('Teacher action loss scale must be positive.')
        if not all(0 <= p <= 1 for p in (args.teacher_action_probability, args.teacher_action_probability_final)):
            raise ValueError('Teacher behavior probabilities must be in [0, 1].')
        if args.teacher_action_probability_decay_iterations < 0:
            raise ValueError('Teacher probability decay iterations must be non-negative.')
    if args.compile_policy_chunk_steps is not None and args.compile_policy_chunk_steps < 1:
        raise ValueError("--compile-policy-chunk-steps must be positive.")
    if teacher_checkpoint is None and (
        teacher_action_loss_coef > 0.0 or teacher_action_loss_final_coef > 0.0
    ):
        raise ValueError("Teacher action loss requires --teacher-checkpoint.")
    if args.motion_files is not None:
        clip_count = len(args.motion_files)
        for option, values in (
            ("--motion-names", args.motion_names),
            ("--motion-weights", args.motion_weights),
            ("--motion-loop", args.motion_loop),
        ):
            if values is not None and len(values) != clip_count:
                raise ValueError(
                    f"{option} requires {clip_count} values to match --motion-files, "
                    f"got {len(values)}."
                )
    # Fail before creating a log directory, starting W&B, or initializing the
    # simulator. Long runs commonly replace numbered snapshots with
    # model_final.pt during graceful shutdown.
    args.resume_checkpoint = resolve_resume_checkpoint(args.resume_checkpoint)
    teacher_checkpoint = resolve_resume_checkpoint(teacher_checkpoint)
    if DISTILLATION_CLI_ENABLED:
        args.teacher_checkpoint = teacher_checkpoint
    if args.domain_randomization is None:
        args.domain_randomization = args.stage == "stage2"
    if args.max_iterations is None:
        args.max_iterations = 7500 if args.stage == "stage2" else 30000
    if not args.domain_randomization:
        initial_dr_scale = 0.0
    elif args.dr_curriculum_mode == "adaptive":
        initial_dr_scale = float(np.clip(args.dr_curriculum_initial_scale, 0.0, 1.0))
    else:
        initial_dr_scale = domain_randomization_scale_at_iteration(0, args.dr_curriculum_iterations)
    initial_course_random_start_prob = terrain_course_random_start_prob_at_iteration(
        0,
        args.terrain_course_random_start_prob,
        args.terrain_course_random_start_warmup_iterations,
        args.terrain_course_random_start_ramp_iterations,
    )
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = select_device(args.device)
    sim_backend = select_sim_backend(args.sim_backend, device)
    log_dir = resolve_log_dir(args)
    if not args.quiet_startup:
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
        motion_files=tuple(args.motion_files or ()),
        motion_names=tuple(args.motion_names or ()),
        motion_weights=tuple(args.motion_weights or ()),
        motion_loop=tuple(value == "true" for value in (args.motion_loop or ())),
        skill_conditioning=bool(args.skill_conditioning),
        use_reference_velocity=bool(args.use_reference_velocity),
        deepmimic_reward=bool(args.deepmimic_reward),
        deepmimic_track_root_xy=bool(args.deepmimic_track_root_xy),
        deepmimic_root_xy_penalty=args.deepmimic_root_xy_penalty,
        deepmimic_route_tracking=args.deepmimic_route_tracking,
        deepmimic_route_xy_sigma=args.deepmimic_route_xy_sigma,
        deepmimic_route_velocity_sigma=args.deepmimic_route_velocity_sigma,
        deepmimic_route_max_xy_error=args.deepmimic_route_max_xy_error,
        trajectory_feedback=bool(args.trajectory_feedback),
        sensor_gait_command_enabled=bool(args.sensor_gait_command),
        sensor_phase_observation=bool(args.sensor_phase_observation),
        sensor_root_imu_only=bool(args.sensor_root_imu_only),
        reference_root_x_offset=args.reference_root_x_offset,
        terrain_types=tuple(args.terrain_types or ()),
        terrain_asset_dir=args.terrain_asset_dir,
        terrain_embedded_in_model=bool(args.terrain_embedded_in_model),
        preserve_motion_world_xy=bool(args.preserve_motion_world_xy),
        stair_terrain=bool(args.stair_terrain),
        stair_step_height=args.stair_step_height,
        stair_step_depth=args.stair_step_depth,
        stair_step_count=args.stair_step_count,
        stair_width=args.stair_width,
        stair_run_in=args.stair_run_in,
        stair_top_length=args.stair_top_length,
        stair_run_out=args.stair_run_out,
        stair_start_x=args.stair_start_x,
        stair_down_offset=args.stair_down_offset,
        stair_fixed_start=bool(args.stair_fixed_start),
        stair_phase_reset_prob=args.stair_phase_reset_prob,
        stair_distance_tolerance=args.stair_distance_tolerance,
        stair_distance_failure_penalty=args.stair_distance_failure_penalty,
        stair_lateral_tolerance=args.stair_lateral_tolerance,
        stair_lateral_failure_penalty=args.stair_lateral_failure_penalty,
        stair_terminal_hold_steps=args.stair_terminal_hold_steps,
        flat_lane_y=args.flat_lane_y,
        rough_lane_y=args.rough_lane_y,
        pit_lane_y=args.pit_lane_y,
        gap_lane_y=args.gap_lane_y,
        terrain_lane_width=args.terrain_lane_width,
        rough_terrain_height=args.rough_terrain_height,
        slope_angle_deg=args.slope_angle_deg,
        gap_terrain_width=args.gap_terrain_width,
        serial_terrain_course=bool(args.serial_terrain_course),
        terrain_course_variants=args.terrain_course_variants,
        terrain_course_lane_spacing=args.terrain_course_lane_spacing,
        terrain_course_base_height=args.terrain_course_base_height,
        terrain_course_random_assignment=bool(args.terrain_course_random_assignment),
        terrain_course_completion_reward=args.terrain_course_completion_reward,
        terrain_course_stage_reward=args.terrain_course_stage_reward,
        terrain_course_stage_tolerance=args.terrain_course_stage_tolerance,
        terrain_course_random_start_prob=initial_course_random_start_prob,
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
        min_forward_progress_speed=args.min_forward_progress_speed,
        forward_progress_grace_steps=args.forward_progress_grace_steps,
        forward_progress_window_steps=args.forward_progress_window_steps,
        forward_progress_termination_penalty=args.forward_progress_termination_penalty,
        fall_penalty=args.fall_penalty,
        task_reward_mode=args.task_reward_mode,
        sensor_stability_reward=bool(args.sensor_stability_reward),
        sensor_fall_penalty=args.sensor_fall_penalty,
        sensor_knee_flexion_target=args.sensor_knee_flexion_target,
        sensor_knee_gait_reward_coef=args.sensor_knee_gait_reward_coef,
        sensor_natural_gait_reward_coef=args.sensor_natural_gait_reward_coef,
        reference_gait_style_reward_coef=args.reference_gait_style_reward_coef,
        reference_joint_position_reward_coef=args.reference_joint_position_reward_coef,
        reference_joint_velocity_reward_coef=args.reference_joint_velocity_reward_coef,
        reference_root_velocity_reward_coef=args.reference_root_velocity_reward_coef,
        reference_foot_position_reward_coef=args.reference_foot_position_reward_coef,
        sensor_gait_height_target=args.sensor_gait_height_target,
        sensor_gait_height_sigma=args.sensor_gait_height_sigma,
        sensor_upright_sigma=args.sensor_upright_sigma,
        sensor_mirror_reward_coef=args.sensor_mirror_reward_coef,
        sensor_stride_reward_coef=args.sensor_stride_reward_coef,
        sensor_foot_roll_reward_coef=args.sensor_foot_roll_reward_coef,
        sensor_alternation_reward_coef=args.sensor_alternation_reward_coef,
        sensor_hop_penalty_coef=args.sensor_hop_penalty_coef,
        sensor_tiptoe_penalty_coef=args.sensor_tiptoe_penalty_coef,
        sensor_stance_slip_penalty_coef=args.sensor_stance_slip_penalty_coef,
        sensor_sync_forward_penalty_coef=args.sensor_sync_forward_penalty_coef,
        sensor_velocity_excess_penalty_coef=args.sensor_velocity_excess_penalty_coef,
        terminate_incomplete_motion_end=bool(args.terminate_incomplete_motion_end),
        reset_noise_scale=args.reset_noise_scale,
        motion_reset_prob=args.motion_reset_prob,
        action_residual_scale=args.action_residual_scale,
        distal_action_scale=args.distal_action_scale,
        reference_ctrl_blend=args.reference_ctrl_blend,
        motion_transition_blend_steps=args.motion_transition_blend_steps,
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
        stand_start_motion_phase=args.stand_start_motion_phase,
        stand_start_phase_delay_steps=args.stand_start_phase_delay_steps,
        stand_start_transition_min_steps=args.stand_start_transition_min_steps,
        stand_start_transition_max_steps=args.stand_start_transition_max_steps,
        stand_start_push_delay_steps=args.stand_start_push_delay_steps,
        ctrl_penalty_coef=args.ctrl_penalty_coef,
        action_rate_penalty_coef=args.action_rate_penalty_coef,
        distal_action_penalty_coef=args.distal_action_penalty_coef,
        distal_action_rate_penalty_coef=args.distal_action_rate_penalty_coef,
        domain_randomization=bool(args.domain_randomization),
        domain_randomization_scale=initial_dr_scale,
        warp_global_domain_randomization=bool(args.warp_global_domain_randomization),
        payload_mass_min=args.payload_mass_min,
        payload_mass_max=args.payload_mass_max,
        body_mass_scale_min=args.body_mass_scale_min,
        body_mass_scale_max=args.body_mass_scale_max,
        tendon_length_scale_min=args.tendon_length_scale_min,
        tendon_length_scale_max=args.tendon_length_scale_max,
        actuator_strength_scale_min=args.actuator_strength_scale_min,
        actuator_strength_scale_max=args.actuator_strength_scale_max,
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
    if not 0 < args.actuator_strength_scale_min <= args.actuator_strength_scale_max:
        raise ValueError('Actuator strength scales require 0 < min <= max.')
    ppo_defaults = PPOAMPConfig()
    supports_distillation = hasattr(ppo_defaults, "teacher_checkpoint")
    if teacher_checkpoint is not None and not supports_distillation:
        raise ValueError(
            "Teacher distillation is isolated from ordinary training; use "
            "sensor_training/train_distill_stage1.py or train_distill_stage2.py."
        )
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
        compile_policy_update=(
            args.compile_policy_update
            if args.compile_policy_update is not None
            else ppo_defaults.compile_policy_update
        ),
        compile_rollout=(
            args.compile_rollout
            if args.compile_rollout is not None
            else ppo_defaults.compile_rollout
        ),
        compile_policy_chunk_steps=(
            args.compile_policy_chunk_steps
            if args.compile_policy_chunk_steps is not None
            else ppo_defaults.compile_policy_chunk_steps
        ),
        discriminator_learning_rate=args.disc_learning_rate,
        num_steps=args.num_steps,
        update_epochs=args.epochs,
        num_minibatches=args.minibatches,
        amp_enabled=args.amp,
        lcp_weight=args.lcp_weight,
        deepmimic_ppo=args.deepmimic_ppo,
        amp_reward_coef=args.amp_reward_coef,
        task_reward_lerp=args.task_reward_lerp,
        **(
            {
                "teacher_checkpoint": teacher_checkpoint,
                "teacher_action_loss_coef": teacher_action_loss_coef,
                "teacher_action_loss_final_coef": teacher_action_loss_final_coef,
                "teacher_action_loss_decay_iterations": teacher_action_loss_decay_iterations,
                "distillation_reset_schedule": distillation_reset_schedule,
                "teacher_action_loss_scale": args.teacher_action_loss_scale,
                "teacher_action_probability": args.teacher_action_probability,
                "teacher_action_probability_final": args.teacher_action_probability_final,
                "teacher_action_probability_decay_iterations": args.teacher_action_probability_decay_iterations,
            }
            if supports_distillation
            else {}
        ),
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
    if args.use_moe:
        if not hasattr(ppo_cfg, "use_moe"):
            raise ValueError("--use-moe is only available for the recurrent sensor policy.")
        ppo_cfg.use_moe = True
    save_config(log_dir, args, env_cfg, ppo_cfg)

    run = maybe_init_wandb(args, log_dir)
    metrics_csv_path = log_dir / "metrics.csv"
    if sim_backend == "warp":
        assert LytRobotWarpAmpEnv is not None
        env = LytRobotWarpAmpEnv(env_cfg, device=device)
    elif sim_backend == "mjlab":
        assert LytRobotMjlabAmpEnv is not None
        env = LytRobotMjlabAmpEnv(env_cfg, device=device)
    else:
        env = LytRobotAmpEnv(env_cfg)
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

    current_course_random_start_prob = terrain_course_random_start_prob_at_iteration(
        start_iteration,
        args.terrain_course_random_start_prob,
        args.terrain_course_random_start_warmup_iterations,
        args.terrain_course_random_start_ramp_iterations,
    )
    previous_course_random_start_prob = float(
        env.cfg.terrain_course_random_start_prob
    )
    env.set_terrain_course_random_start_prob(current_course_random_start_prob)
    if abs(
        previous_course_random_start_prob - current_course_random_start_prob
    ) > 1.0e-12:
        trainer.reset_training_env()
    if args.serial_terrain_course:
        print(
            "[INFO] terrain-course random-start curriculum "
            f"prob={current_course_random_start_prob:.4f} "
            f"target={args.terrain_course_random_start_prob:.4f} "
            f"warmup={args.terrain_course_random_start_warmup_iterations} "
            f"ramp={args.terrain_course_random_start_ramp_iterations}"
        )

    if not args.domain_randomization:
        current_dr_scale = 0.0
    elif args.dr_curriculum_mode == "adaptive":
        should_restore_dr_scale = (
            args.resume_checkpoint is not None
            and not args.resume_reset_progress
            and trainer.loaded_domain_randomization_scale is not None
        )
        current_dr_scale = (
            float(np.clip(trainer.loaded_domain_randomization_scale, 0.0, 1.0))
            if should_restore_dr_scale
            else initial_dr_scale
        )
    else:
        current_dr_scale = domain_randomization_scale_at_iteration(
            start_iteration, args.dr_curriculum_iterations
        )
    if args.domain_randomization:
        previous_dr_scale = float(getattr(env, "domain_randomization_scale", 1.0))
        env.set_domain_randomization_scale(current_dr_scale)
        if abs(previous_dr_scale - current_dr_scale) > 1.0e-12:
            trainer.reset_training_env()
        print(
            f"[INFO] domain-randomization curriculum mode={args.dr_curriculum_mode} "
            f"scale={current_dr_scale:.4f} "
            f"warp_shared_global={bool(args.warp_global_domain_randomization)} "
            f"global_resample_interval={max(args.global_dr_resample_interval, 0)}"
        )
        if args.global_dr_resample_interval > 0 and not args.warp_global_domain_randomization:
            print("[WARN] --global-dr-resample-interval is ignored because shared Warp DR is disabled.")

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
    dr_window_completed = 0.0
    dr_window_successful = 0.0
    dr_window_bottom_gate = 0.0
    dr_window_bottom_success = 0.0
    dr_window_termination_sum = 0.0
    dr_window_count = 0
    last_dr_window_success = float("nan")
    last_dr_window_depth_success = float("nan")
    last_dr_window_termination = float("nan")
    best_checkpoint_quality = load_best_checkpoint_quality(
        metrics_csv_path,
        min_completed=args.best_checkpoint_min_completed,
        deepmimic_ppo=bool(args.deepmimic_ppo),
        stair_route=bool(args.deepmimic_ppo and args.stair_fixed_start),
    )
    fixed_route_evaluation = bool(
        args.deepmimic_route_tracking and args.stair_fixed_start
        and hasattr(trainer, "evaluate_fixed_route")
    )
    evaluation_quality_path = log_dir / "best_route_evaluation.json"
    if fixed_route_evaluation:
        best_checkpoint_quality = (-1.0, -1.0)
        if evaluation_quality_path.is_file():
            previous_evaluation = json.loads(evaluation_quality_path.read_text())
            best_checkpoint_quality = (
                float(previous_evaluation["eval/stair_route_success"]),
                float(previous_evaluation["eval/stair_route_progress"]),
            )
    if args.evaluate_gait:
        evaluation_quality_path = log_dir / "best_gait_evaluation.json"
        best_checkpoint_quality = (-1.0, -1.0)
        if evaluation_quality_path.is_file():
            previous_evaluation = json.loads(evaluation_quality_path.read_text())
            if previous_evaluation.get('eval/gait_seed_count') != len(args.gait_evaluation_seeds):
                from sensor_training.gait_evaluation import evaluate_gait_suite
                previous_evaluation.update(evaluate_gait_suite(
                    log_dir / 'model_best.onnx', seconds=args.gait_evaluation_seconds,
                    seeds=args.gait_evaluation_seeds))
                evaluation_quality_path.write_text(json.dumps(previous_evaluation, indent=2)+'\n')
            from sensor_training.gait_evaluation import gait_checkpoint_quality
            best_checkpoint_quality = gait_checkpoint_quality(previous_evaluation, fidelity=args.gait_fidelity_selection)
    try:
        for iteration in range(start_iteration, args.max_iterations):
            current_course_random_start_prob = (
                terrain_course_random_start_prob_at_iteration(
                    iteration,
                    args.terrain_course_random_start_prob,
                    args.terrain_course_random_start_warmup_iterations,
                    args.terrain_course_random_start_ramp_iterations,
                )
            )
            env.set_terrain_course_random_start_prob(
                current_course_random_start_prob
            )
            if args.domain_randomization and args.dr_curriculum_mode == "linear":
                current_dr_scale = domain_randomization_scale_at_iteration(
                    iteration,
                    args.dr_curriculum_iterations,
                )
                env.set_domain_randomization_scale(current_dr_scale)
            if args.domain_randomization and args.warp_global_domain_randomization:
                resample_interval = max(int(args.global_dr_resample_interval), 0)
                if (
                    resample_interval > 0
                    and iteration > start_iteration
                    and iteration % resample_interval == 0
                ):
                    trainer.reset_training_env()
                    run_logger.emit(
                        f"[DR] iter={iteration:05d} synchronized global resample "
                        f"scale={current_dr_scale:.4f}",
                        iteration=iteration,
                    )
            rollout_dr_scale = current_dr_scale
            metrics = trainer.train_iteration()
            last_iteration = iteration
            last_global_step = int(metrics["global_step"])
            if args.memory_cleanup_interval > 0 and (iteration + 1) % args.memory_cleanup_interval == 0:
                release_host_memory()
            metrics["iteration"] = float(iteration)
            metrics["terrain_course_random_start_probability"] = float(
                current_course_random_start_prob
            )
            metrics["terrain_course_random_start_target_probability"] = float(
                args.terrain_course_random_start_prob
            )
            metrics["domain_randomization_scale"] = float(rollout_dr_scale)
            dr_decision = 0
            if args.domain_randomization and args.dr_curriculum_mode == "adaptive":
                dr_window_completed += metrics.get("completed_episode_count", 0.0)
                dr_window_successful += metrics.get("successful_episode_count", 0.0)
                dr_window_bottom_gate += metrics.get("task_bottom_gate", 0.0)
                dr_window_bottom_success += metrics.get("task_bottom_depth_success", 0.0)
                dr_window_termination_sum += metrics.get("termination_fraction", 0.0)
                dr_window_count += 1
                update_interval = max(int(args.dr_curriculum_update_interval), 1)
                if dr_window_count >= update_interval:
                    last_dr_window_success = (
                        dr_window_successful / dr_window_completed
                        if dr_window_completed > 0.0
                        else 0.0
                    )
                    last_dr_window_depth_success = (
                        dr_window_bottom_success / dr_window_bottom_gate
                        if dr_window_bottom_gate > 1.0e-8
                        else 0.0
                    )
                    last_dr_window_termination = dr_window_termination_sum / dr_window_count
                    next_dr_scale, dr_decision = adaptive_domain_randomization_step(
                        current_dr_scale,
                        success_rate=last_dr_window_success,
                        depth_success_rate=last_dr_window_depth_success,
                        termination_fraction=last_dr_window_termination,
                        step=args.dr_curriculum_step,
                        promote_success_rate=args.dr_promote_success_rate,
                        promote_depth_success_rate=args.dr_promote_depth_success_rate,
                        promote_max_termination=args.dr_promote_max_termination,
                        backoff_success_rate=args.dr_backoff_success_rate,
                        backoff_depth_success_rate=args.dr_backoff_depth_success_rate,
                        backoff_termination=args.dr_backoff_termination,
                    )
                    current_dr_scale = next_dr_scale
                    env.set_domain_randomization_scale(current_dr_scale)
                    decision_name = {-1: "backoff", 0: "hold", 1: "promote"}[dr_decision]
                    run_logger.emit(
                        "[DR_CURRICULUM] "
                        f"iter={iteration:05d} decision={decision_name} "
                        f"scale={rollout_dr_scale:.4f}->{current_dr_scale:.4f} "
                        f"success={last_dr_window_success:.3f} "
                        f"depth={last_dr_window_depth_success:.3f} "
                        f"termination={last_dr_window_termination:.5f}",
                        iteration=iteration,
                        force=True,
                    )
                    dr_window_completed = 0.0
                    dr_window_successful = 0.0
                    dr_window_bottom_gate = 0.0
                    dr_window_bottom_success = 0.0
                    dr_window_termination_sum = 0.0
                    dr_window_count = 0
            metrics["domain_randomization_scale_next"] = float(current_dr_scale)
            metrics["dr_window_success_rate"] = float(last_dr_window_success)
            metrics["dr_window_depth_success_rate"] = float(last_dr_window_depth_success)
            metrics["dr_window_termination_fraction"] = float(last_dr_window_termination)
            metrics["dr_curriculum_decision"] = float(dr_decision)
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
                    f"succ={metrics['episode_success_rate']:.3f} "
                    f"depth={metrics.get('bottom_depth_success_rate', 0.0):.3f} "
                    f"term={metrics['termination_fraction']:.3f} "
                    f"bad={metrics.get('task_invalid_state', 0.0):.6f} "
                    f"stall={metrics.get('task_insufficient_progress', 0.0):.6f} "
                    f"miss={metrics.get('task_incomplete_motion_end', 0.0):.6f} "
                    f"stage={metrics.get('task_course_stage', 0.0):.3f} "
                    f"start={metrics.get('task_course_start_stage', 0.0):.3f} "
                    f"advance={metrics.get('task_course_stage_completion', 0.0):.6f} "
                    f"suffix={metrics.get('task_course_suffix_complete', 0.0):.6f} "
                    f"startp={metrics['terrain_course_random_start_probability']:.3f} "
                    f"dr={metrics['domain_randomization_scale']:.3f} "
                    f"pi={metrics['policy_loss']:.4f} "
                    f"v={metrics['value_loss']:.4f} "
                    f"mv={metrics['median_value_loss']:.4f} "
                    f"br={metrics['bpo_ratio_error']:.4f} "
                    f"kl={metrics['approx_kl']:.4f} "
                    f"lr={metrics['learning_rate']:.2e} "
                    f"d={metrics['amp_loss']:.4f} "
                    f"da={metrics['amp_agent_prob']:.3f} "
                    f"dref={metrics['amp_ref_prob']:.3f} "
                    f"fps={metrics['throughput_env_steps_s']:.0f} "
                    f"time={metrics['timing_iteration_s']:.2f}s",
                    iteration=iteration,
                )
            if 'state/joint_position_mae' in metrics:
                names = ('joint_position_mae', 'joint_velocity_mae', 'root_velocity_mae',
                         'contact_accuracy', 'contact_mae', 'foot_position_mae')
                run_logger.emit('[STATE] ' + ' '.join(f'state/{name}={metrics["state/"+name]:.5f}'
                                for name in names if 'state/'+name in metrics), iteration=iteration)
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
                wandb_metrics = prepare_wandb_metrics(metrics)
                # Keep logging global_step explicitly because all charts use
                # step_metric="global_step".
                if should_filter_resume_wandb_metrics(
                    wandb_metrics,
                    iteration=iteration,
                    warmup_until_iteration=wandb_warmup_until_iteration,
                    min_episode_reward=args.resume_wandb_min_episode_reward,
                    min_episode_length=args.resume_wandb_min_episode_length,
                ):
                    run.log(
                        filter_resume_wandb_warmup_metrics(wandb_metrics),
                        step=int(wandb_metrics["global_step"]),
                    )
                else:
                    wandb_metrics["resume_wandb_warmup"] = 0.0
                    run.log(wandb_metrics, step=int(wandb_metrics["global_step"]))
            if (iteration + 1) % max(args.save_interval, 1) == 0:
                ckpt_step = iteration + 1
                ckpt_path = log_dir / f"model_{ckpt_step}.pt"
                saved_pt, saved_onnx = trainer.save(ckpt_path, iteration=iteration)
                run_logger.emit(f"[INFO] saved latest models: {saved_pt}, {saved_onnx}", force=True)
                stair_route = bool(args.deepmimic_ppo and args.stair_fixed_start)
                primary_key = (
                    "rollout_episode_success_rate"
                    if stair_route or not args.deepmimic_ppo
                    else "rollout_mean_task_reward"
                )
                secondary_key = (
                    "rollout_mean_task_reward" if stair_route
                    else "mean_episode_length"
                )
                checkpoint_quality = (
                    float(metrics.get(primary_key, -1.0)),
                    float(metrics.get(secondary_key, -1.0)),
                )
                evaluation = None
                if fixed_route_evaluation:
                    evaluation = trainer.evaluate_fixed_route()
                    checkpoint_quality = (
                        evaluation["eval/stair_route_success"],
                        evaluation["eval/stair_route_progress"],
                    )
                    evaluation_record = dict(evaluation, iteration=iteration, checkpoint=str(saved_pt))
                    with (log_dir / "route_evaluations.jsonl").open("a") as evaluation_file:
                        evaluation_file.write(json.dumps(evaluation_record) + "\n")
                    run_logger.emit(f"[ROUTE_EVAL] {json.dumps(evaluation_record)}", force=True)
                    if run is not None:
                        run.log(evaluation, step=int(trainer.global_step) + 1)
                if args.evaluate_gait:
                    from sensor_training.gait_evaluation import evaluate_gait_suite
                    evaluation = evaluate_gait_suite(
                        saved_onnx, seconds=args.gait_evaluation_seconds,
                        seeds=args.gait_evaluation_seeds,
                        trace_path=log_dir / 'gait_evaluation_latest.csv')
                    from sensor_training.gait_evaluation import gait_checkpoint_quality
                    checkpoint_quality = gait_checkpoint_quality(evaluation, fidelity=args.gait_fidelity_selection)
                    evaluation_record = dict(evaluation, iteration=iteration, checkpoint=str(saved_pt))
                    with (log_dir / 'gait_evaluations.jsonl').open('a') as evaluation_file:
                        evaluation_file.write(json.dumps(evaluation_record)+'\n')
                    run_logger.emit(f'[GAIT_EVAL] {json.dumps(evaluation_record)}', force=True)
                    if run is not None:
                        run.log(evaluation, step=int(trainer.global_step)+1)
                completed_episodes = float(metrics.get("completed_episode_count", 0.0))
                knee_flexion_track = float(
                    metrics.get("task_knee_flexion_track", float("nan"))
                )
                natural_gait_ok = (
                    args.evaluate_gait or args.sensor_knee_gait_reward_coef <= 0.0
                    or (
                        np.isfinite(knee_flexion_track)
                        and knee_flexion_track >= 0.90
                    )
                )
                if (
                    args.save_best_checkpoint
                    and (args.evaluate_gait or fixed_route_evaluation or completed_episodes >= max(int(args.best_checkpoint_min_completed), 1))
                    and all(np.isfinite(value) for value in checkpoint_quality)
                    and natural_gait_ok
                    and checkpoint_quality > best_checkpoint_quality
                ):
                    best_path = log_dir / "model_best.pt"
                    shutil.copy2(saved_pt, best_path)
                    shutil.copy2(saved_onnx, best_path.with_suffix(".onnx"))
                    best_checkpoint_quality = checkpoint_quality
                    if evaluation is not None:
                        evaluation_quality_path.write_text(json.dumps(evaluation_record, indent=2) + "\n")
                        if args.evaluate_gait:
                            shutil.copy2(log_dir / 'gait_evaluation_latest.csv', log_dir / 'gait_evaluation_best.csv')
                    run_logger.emit(
                        "[BEST] "
                        f"iter={iteration:05d} "
                        f"{'gait_accepted' if args.evaluate_gait else 'route_success' if stair_route else 'imitation_reward' if args.deepmimic_ppo else 'rollout_success'}="
                        f"{checkpoint_quality[0]:.3f} "
                        f"{'reference_fidelity_score' if args.evaluate_gait and args.gait_fidelity_selection else 'gait_score' if args.evaluate_gait else 'evaluated_progress' if fixed_route_evaluation else 'imitation_reward' if stair_route else 'episode_length'}={checkpoint_quality[-1]:.3f} "
                        f"knee_flexion_track={knee_flexion_track:.3f} "
                        f"saved {best_path}",
                        iteration=iteration,
                        force=True,
                    )
                if args.stop_on_gait_acceptance and evaluation is not None and evaluation.get('eval/reference_fidelity_accepted' if args.gait_fidelity_selection else 'eval/gait_accepted', 0) > 0:
                    run_logger.emit('[GAIT_ACCEPTED] Continuous walking and reference tracking passed; completing this phase.', force=True)
                    break
            if args.iteration_sleep_ms > 0.0:
                torch.cuda.synchronize()
                time.sleep(args.iteration_sleep_ms / 1000.0)
    finally:
        trainer.global_step = last_global_step
        final_path = log_dir / "model_final.pt"
        try:
            saved_pt, saved_onnx = trainer.save(final_path, iteration=last_iteration)
        except ValueError as exc:
            run_logger.emit(
                f"[CHECKPOINT_GUARD] skipped invalid final checkpoint and preserved existing models: {exc}",
                force=True,
            )
        else:
            run_logger.emit(f"[INFO] saved latest models: {saved_pt}, {saved_onnx}", force=True)
        run_logger.close()
        env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
