"""Student training defaults; rewards are inherited from the Teacher checkpoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from .training_defaults import distillation_cli_defaults


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DISTILLATION_LOG_ROOT = REPO_ROOT / "logs" / "distillation"
STAGE1_GAIT_REFERENCE = (
    REPO_ROOT / "robot" / "reference_data" / "walk1_mjcf.npz"
)
STAGE2_GAIT_REFERENCE = (
    REPO_ROOT / "robot" / "reference_data" / "walk1.npz"
)


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


def _take_option(name: str) -> str | None:
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == name:
            if index + 1 >= len(sys.argv):
                raise SystemExit(f"{name} requires a checkpoint path.")
            value = sys.argv[index + 1]
            del sys.argv[index : index + 2]
            return value
        prefix = f"{name}="
        if arg.startswith(prefix):
            value = arg[len(prefix) :]
            del sys.argv[index]
            return value
    return None


def _apply_defaults(defaults: dict[str, str]) -> None:
    for name, value in defaults.items():
        if not _has_arg(name):
            sys.argv.extend((name, value))


def run_stage1() -> None:
    """Use the 10-06 Student recipe with standing starts and continuous walking."""

    if not _has_arg("--teacher-checkpoint"):
        raise SystemExit(
            "Stage1 distillation requires --teacher-checkpoint PATH."
        )
    _apply_defaults({"--num-steps": "96"})
    decimation, num_steps, num_envs = distillation_cli_defaults(sys.argv[1:])
    _apply_defaults(
        {
            "--stage": "stage1",
            "--device": "cuda",
            "--sim-backend": "warp",
            "--num-envs": str(num_envs),
            "--num-steps": str(num_steps),
            "--decimation": str(decimation),
            "--motion-file": str(STAGE1_GAIT_REFERENCE),
            "--termination-mode": "fall",
            "--motion-loop": "true",
            "--episode-steps": "0",
            "--episode-timeout-jitter-steps": "0",
            # Teacher supervision adds another rollout graph.  Eight recurrent
            # steps per backward graph benchmarks faster than the generic
            # 16-step sensor default for this mixed Student/Teacher workload.
            "--compile-policy-chunk-steps": "8",
            "--max-iterations": "30000",
            "--reference-ctrl-blend": "1.0",
            "--teacher-action-loss-coef": "1.0",
            "--teacher-action-loss-final-coef": "0.30",
            "--teacher-action-loss-decay-iterations": "12000",
            "--teacher-action-loss-scale": "0.10",
            "--teacher-action-probability": "0.80",
            "--teacher-action-probability-final": "0.0",
            "--teacher-action-probability-decay-iterations": "4000",
            "--stand-start-prob": "1.0",
            "--motion-reset-prob": "0.0",
            "--stand-start-motion-phase": "0",
            "--stand-start-phase-delay-steps": "25",
            "--stand-start-transition-min-steps": "80",
            "--stand-start-transition-max-steps": "160",
            "--max-action-std": "0.25",
            "--min-action-std": "0.03",
            "--learning-rate": "1.0e-4",
            "--max-learning-rate": "1.0e-4",
            "--target-kl": "0.010",
            "--target-kl-stop-multiplier": "1.5",
            "--save-interval": "500",
            "--log-root": str(
                REPO_ROOT / "logs" / "command_walking" if "--command-walking" in sys.argv
                else REPO_ROOT / "logs" / "sensor_policy" / "distillation"
            ),
            "--log-role": "student",
            "--wandb-tags": "lytrobot,sensor-only,distillation,stage1-gait,privileged-teacher",
        }
    )
    if not (_has_arg("--evaluate-gait") or _has_arg("--no-evaluate-gait")):
        sys.argv.append("--evaluate-gait")
    if not (_has_arg('--sensor-root-imu-only') or _has_arg('--no-sensor-root-imu-only')):
        sys.argv.append('--no-sensor-root-imu-only')
    if not (_has_arg("--domain-randomization") or _has_arg("--no-domain-randomization")):
        sys.argv.append("--no-domain-randomization")
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    runpy.run_path(str(HERE / "train_distill.py"), run_name="__main__")


def run_stage2() -> None:
    """Continue a distilled student under a privileged robust stage2 teacher."""

    resume_checkpoint = _take_option("--resume-distillation-checkpoint")
    student_checkpoint = _take_option("--student-checkpoint")
    if resume_checkpoint is not None and student_checkpoint is not None:
        raise SystemExit(
            "Use only one of --student-checkpoint or --resume-distillation-checkpoint."
        )
    if resume_checkpoint is None and student_checkpoint is None:
        raise SystemExit(
            "Stage2 distillation requires --student-checkpoint PATH from "
            "distillation/stage1_gait, or --resume-distillation-checkpoint PATH."
        )
    if _has_arg("--resume-checkpoint"):
        raise SystemExit(
            "Use --student-checkpoint or --resume-distillation-checkpoint instead of --resume-checkpoint."
        )
    if not _has_arg("--teacher-checkpoint"):
        raise SystemExit(
            "Stage2 distillation requires --teacher-checkpoint PATH."
        )
    if resume_checkpoint is not None:
        sys.argv.extend(("--resume-checkpoint", resume_checkpoint))
    else:
        assert student_checkpoint is not None
        sys.argv.extend(("--resume-checkpoint", student_checkpoint, "--resume-reset-progress"))
    decimation, num_steps, num_envs = distillation_cli_defaults(sys.argv[1:])
    _apply_defaults(
        {
            "--stage": "stage2",
            "--device": "cuda",
            "--sim-backend": "warp",
            "--num-envs": str(num_envs),
            "--num-steps": str(num_steps),
            "--decimation": str(decimation),
            "--motion-file": str(STAGE2_GAIT_REFERENCE),
            "--termination-mode": "fall",
            "--motion-loop": "true",
            "--episode-steps": "0",
            "--episode-timeout-jitter-steps": "0",
            "--compile-policy-chunk-steps": "8",
            "--max-iterations": "50000",
            "--reference-ctrl-blend": "1.0",
            "--teacher-action-loss-coef": "0.50",
            "--teacher-action-loss-final-coef": "0.30",
            "--teacher-action-loss-scale": "0.10",
            "--teacher-action-probability": "0.30",
            "--teacher-action-probability-final": "0.0",
            "--teacher-action-probability-decay-iterations": "2000",
            "--teacher-action-loss-decay-iterations": "24000",
            "--stand-start-prob": "1.0",
            "--motion-reset-prob": "0.0",
            "--stand-start-motion-phase": "31",
            "--stand-start-phase-delay-steps": "25",
            "--stand-start-transition-min-steps": "80",
            "--stand-start-transition-max-steps": "160",
            "--max-action-std": "0.25",
            "--min-action-std": "0.03",
            "--learning-rate": "1.0e-4",
            "--max-learning-rate": "1.0e-4",
            "--target-kl": "0.010",
            "--target-kl-stop-multiplier": "1.5",
            "--save-interval": "500",
            "--log-root": str(DISTILLATION_LOG_ROOT),
            "--wandb-tags": "lytrobot,sensor-only,distillation,stage2-robust,privileged-teacher",
        }
    )
    if not (_has_arg("--evaluate-gait") or _has_arg("--no-evaluate-gait")):
        sys.argv.append("--evaluate-gait")
    if not (_has_arg('--sensor-root-imu-only') or _has_arg('--no-sensor-root-imu-only')):
        sys.argv.append('--no-sensor-root-imu-only')
    if resume_checkpoint is None and not _has_arg("--distillation-reset-schedule"):
        sys.argv.append("--distillation-reset-schedule")
    if not (_has_arg("--domain-randomization") or _has_arg("--no-domain-randomization")):
        sys.argv.append("--domain-randomization")
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    runpy.run_path(str(HERE / "train_distill.py"), run_name="__main__")
