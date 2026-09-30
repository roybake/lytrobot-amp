#!/usr/bin/env python3
"""Run quality-gated 100 Hz privileged-teacher → sensor-student training.

One invocation trains natural gait; robustness is an optional later extension.
Every phase uses the same reference tendon feedforward at 100 Hz, so teacher
and student actions share the same control baseline. The student Actor observes
only its IMU/tendon history and clock; deployment must provide the reference
feedforward separately.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensor_training.training_defaults import (
    distillation_rollout_shape,
    validate_sensor_decimation,
)


def _run_name(value: str) -> str:
    if Path(value).name != value or value in {"", ".", ".."}:
        raise argparse.ArgumentTypeError("--run-name must be one directory name.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a natural-gait teacher and sensor student, optionally followed by robustness training."
    )
    parser.add_argument(
        "--run-name",
        type=_run_name,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Shared name used in every phase directory and the pipeline manifest.",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument(
        "--student-num-envs",
        type=int,
        default=None,
        help=(
            "Parallel environments for sensor-student phases. By default this is "
            "derived to preserve the teacher's 24-step rollout sample count."
        ),
    )
    parser.add_argument(
        "--student-num-steps",
        type=int,
        default=None,
        help=(
            "Recurrent rollout/BPTT steps for sensor-student phases. "
            "Default: 48 at the required 100 Hz control rate."
        ),
    )
    parser.add_argument("--stage1-iterations", type=int, default=30000)
    parser.add_argument("--stage2-iterations", type=int, default=30000)
    parser.add_argument("--teacher-stage1-iterations", type=int, default=None)
    parser.add_argument("--teacher-stage2-iterations", type=int, default=None)
    parser.add_argument("--teacher-stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--student-stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-quality-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-robust-stage", action="store_true")
    parser.add_argument("--student-root-imu-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--sim-backend", default="warp", choices=("warp",))
    parser.add_argument("--decimation", type=int, default=2)
    parser.add_argument("--wandb-log-interval", type=int, default=10)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected phase commands without launching training.",
    )
    return parser.parse_args()


def _phase_paths(run_name: str) -> dict[str, Path]:
    return {
        "teacher_stage1": REPO_ROOT / "logs" / "teachers" / "stage1_gait" / run_name,
        "teacher_stage2": REPO_ROOT / "logs" / "teachers" / "stage2_robust" / run_name,
        "student_stage1": (
            REPO_ROOT / "logs" / "sensor_policy" / "distillation" / "stage1_gait" / run_name
        ),
        "student_stage2": (
            REPO_ROOT / "logs" / "sensor_policy" / "distillation" / "stage2_robust" / run_name
        ),
    }


def _handoff_checkpoint(directory: Path) -> Path:
    """Prefer the measured-stable policy over a potentially degraded final one."""

    best = directory / "model_best.pt"
    return best if best.is_file() else directory / "model_final.pt"


def _load_manifest(path: Path, *, run_name: str, phases: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    if path.is_file():
        with path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("run_name") != run_name:
            raise ValueError(f"Manifest {path} belongs to another run.")
        return manifest
    return {
        "format_version": 1,
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "control_interface": {
            "reference_ctrl_blend": 1.0,
            "decimation": args.decimation,
            "control_hz": 1.0 / (0.005 * args.decimation),
            "student_num_steps": _student_num_steps(args),
            "student_num_envs": _student_num_envs(args),
        },
        "phases": {
            name: {"log_dir": str(directory), "status": "pending"}
            for name, directory in phases.items()
        },
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary.replace(path)


def _student_num_steps(args: argparse.Namespace) -> int:
    validate_sensor_decimation(int(args.decimation))
    explicit = getattr(args, "student_num_steps", None)
    if explicit is not None:
        return int(explicit)
    return distillation_rollout_shape(int(args.decimation))[0]


def _student_num_envs(args: argparse.Namespace) -> int:
    explicit = getattr(args, "student_num_envs", None)
    if explicit is not None:
        return int(explicit)
    student_steps = _student_num_steps(args)
    return max(1, round(int(args.num_envs) * 24 / student_steps))


def _common_train_args(
    args: argparse.Namespace,
    log_dir: Path,
    iterations: int,
    *,
    sensor_student: bool = False,
) -> list[str]:
    wandb_flag = "--no-wandb" if args.no_wandb else "--wandb"
    command = [
        "--device", args.device,
        "--sim-backend", args.sim_backend,
        "--num-envs", str(_student_num_envs(args) if sensor_student else args.num_envs),
        "--decimation", str(args.decimation),
        "--reference-ctrl-blend", "1.0",
        "--max-iterations", str(iterations),
        "--save-interval", "500",
        "--wandb-log-interval", str(getattr(args, 'wandb_log_interval', 10)),
        "--motion-loop", "true",
        "--episode-steps", "2000",
        "--episode-timeout-jitter-steps", "0",
        "--stand-start-prob", "0.70" if sensor_student else "0.50",
        "--motion-reset-prob", "1.0",
        "--stand-start-motion-phase", "0",
        "--stand-start-phase-delay-steps", "25",
        "--stand-start-transition-min-steps", "80",
        "--stand-start-transition-max-steps", "160",
        "--reference-gait-style-reward-coef", "4.0",
        "--sensor-natural-gait-reward-coef", "2.0",
        "--sensor-hop-penalty-coef", "1.0",
        "--sensor-tiptoe-penalty-coef", "1.0",
        "--sensor-stance-slip-penalty-coef", "0.5",
        "--sensor-sync-forward-penalty-coef", "1.0",
        "--forward-velocity-reward-coef", "2.0",
        "--forward-velocity-deficit-penalty-coef", "2.5",
        "--fall-penalty", "10.0",
        "--learning-rate", "1e-4",
        "--max-learning-rate", "1e-4",
        "--target-kl", "0.015",
        "--evaluate-gait",
        "--log-dir", str(log_dir),
        wandb_flag,
    ]
    if sensor_student:
        command.extend(("--num-steps", str(_student_num_steps(args))))
        command.append('--sensor-root-imu-only' if getattr(args, 'student_root_imu_only', False)
                       else '--no-sensor-root-imu-only')
    return command


def _phase_command(
    name: str,
    args: argparse.Namespace,
    paths: dict[str, Path],
    *,
    resume_checkpoint: Path | None,
) -> list[str]:
    teacher_stage1 = _handoff_checkpoint(paths["teacher_stage1"])
    teacher_stage2 = _handoff_checkpoint(paths["teacher_stage2"])
    student_stage1 = _handoff_checkpoint(paths["student_stage1"])
    script: Path
    command: list[str]

    if name == "teacher_stage1":
        script = REPO_ROOT / "scripts" / "train_stage1.py"
        command = [sys.executable, str(script)] + _common_train_args(
            args, paths[name], getattr(args, 'teacher_stage1_iterations', None) or args.stage1_iterations
        )
        command.append('--stop-on-gait-acceptance')
        if resume_checkpoint is not None:
            command.extend(("--resume-checkpoint", str(resume_checkpoint)))
        elif getattr(args, 'teacher_stage1_checkpoint', None) is not None:
            command.extend(('--resume-checkpoint', str(args.teacher_stage1_checkpoint),
                            '--resume-reset-progress', '--resume-reset-optimizer'))
        return command

    if name == "teacher_stage2":
        script = REPO_ROOT / "scripts" / "train_stage2.py"
        command = [sys.executable, str(script)] + _common_train_args(
            args, paths[name], getattr(args, 'teacher_stage2_iterations', None) or args.stage2_iterations
        )
        command.extend(('--dr-curriculum-mode', 'adaptive', '--dr-curriculum-initial-scale', '0.1'))
        if resume_checkpoint is not None:
            command.extend(("--resume-checkpoint", str(resume_checkpoint)))
        else:
            command.extend(("--stage1-checkpoint", str(teacher_stage1)))
        return command

    if name == "student_stage1":
        script = REPO_ROOT / "sensor_training" / "train_distill_stage1.py"
        command = [sys.executable, str(script)] + _common_train_args(
            args, paths[name], args.stage1_iterations, sensor_student=True
        )
        command.extend(("--teacher-checkpoint", str(teacher_stage1)))
        command.append('--stop-on-gait-acceptance')
        if resume_checkpoint is not None:
            command.extend(("--resume-checkpoint", str(resume_checkpoint)))
        elif getattr(args, 'student_stage1_checkpoint', None) is not None:
            command.extend(('--resume-checkpoint', str(args.student_stage1_checkpoint),
                            '--resume-reset-progress', '--resume-reset-optimizer',
                            '--distillation-reset-schedule'))
        return command

    if name == "student_stage2":
        script = REPO_ROOT / "sensor_training" / "train_distill_stage2.py"
        command = [sys.executable, str(script)] + _common_train_args(
            args, paths[name], args.stage2_iterations, sensor_student=True
        )
        command.extend(("--teacher-checkpoint", str(teacher_stage2)))
        command.extend(('--dr-curriculum-mode', 'adaptive', '--dr-curriculum-initial-scale', '0.1'))
        if resume_checkpoint is not None:
            command.extend(("--resume-distillation-checkpoint", str(resume_checkpoint)))
        else:
            command.extend(("--student-checkpoint", str(student_stage1)))
        return command

    raise ValueError(f"Unknown serial-distillation phase: {name}")


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.student_num_envs is not None and args.student_num_envs <= 0:
        raise ValueError("--student-num-envs must be positive.")
    if args.student_num_steps is not None and args.student_num_steps <= 0:
        raise ValueError("--student-num-steps must be positive.")
    if args.stage1_iterations <= 0 or args.stage2_iterations <= 0:
        raise ValueError("Stage iteration counts must be positive.")
    for count in (args.teacher_stage1_iterations, args.teacher_stage2_iterations):
        if count is not None and count <= 0:
            raise ValueError('Teacher iteration counts must be positive.')
    if args.student_stage1_checkpoint is not None and args.student_root_imu_only:
        import torch
        initial_student = torch.load(args.student_stage1_checkpoint, map_location='cpu', weights_only=False)
        if not initial_student.get('env_config', {}).get('sensor_root_imu_only', False):
            print('[PIPELINE] Previous Student requires multiple IMUs; the root-IMU Student starts from scratch.')
            args.student_stage1_checkpoint = None
    validate_sensor_decimation(args.decimation)

    paths = _phase_paths(args.run_name)
    manifest_path = REPO_ROOT / "logs" / "serial_distillation" / args.run_name / "pipeline.json"
    manifest = _load_manifest(manifest_path, run_name=args.run_name, phases=paths, args=args)
    phase_order = ('teacher_stage1', 'student_stage1')
    if args.include_robust_stage:
        phase_order += ('teacher_stage2', 'student_stage2')
    manifest['enabled_phases'] = list(phase_order)
    if not args.dry_run:
        for name in ('teacher_stage2', 'student_stage2'):
            if name not in phase_order:
                manifest['phases'][name]['status'] = 'not_requested'
        _write_manifest(manifest_path, manifest)

    for name in phase_order:
        log_dir = paths[name]
        checkpoint = log_dir / "model_final.pt"
        phase = manifest.setdefault("phases", {}).setdefault(
            name,
            {"log_dir": str(log_dir), "status": "pending"},
        )
        if phase.get("status") == "completed" and checkpoint.is_file():
            print(f"[PIPELINE] {name}: completed, skip {checkpoint}")
            continue

        resume_checkpoint = checkpoint if checkpoint.is_file() else None
        command = _phase_command(name, args, paths, resume_checkpoint=resume_checkpoint)
        print(f"[PIPELINE] {name}: {shlex.join(command)}")
        if args.dry_run:
            continue

        log_dir.mkdir(parents=True, exist_ok=True)
        phase.update(
            {
                "status": "running",
                "log_dir": str(log_dir),
                "command": command,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        _write_manifest(manifest_path, manifest)
        try:
            if name.startswith('student') and args.teacher_quality_gate:
                from sensor_training.gait_evaluation import evaluate_gait_suite
                teacher_name = 'teacher_stage1' if name == 'student_stage1' else 'teacher_stage2'
                quality = evaluate_gait_suite(_handoff_checkpoint(paths[teacher_name]))
                phase['teacher_quality'] = quality
                _write_manifest(manifest_path, manifest)
                if quality['eval/gait_accepted'] <= 0:
                    raise RuntimeError(f'{teacher_name} failed continuous human-gait evaluation; refusing to distill the failed gait: {quality}')
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            if not checkpoint.is_file():
                raise RuntimeError(f"{name} completed without producing {checkpoint}.")
            if name.startswith('student') and args.teacher_quality_gate:
                from sensor_training.gait_evaluation import evaluate_gait_suite
                quality = evaluate_gait_suite(_handoff_checkpoint(paths[name]))
                phase['student_quality'] = quality
                if quality['eval/gait_accepted'] <= 0:
                    raise RuntimeError(f'{name} needs further natural-gait training; evaluation failed: {quality}')
        except BaseException as exc:
            phase.update(
                {
                    "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "error": str(exc),
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            _write_manifest(manifest_path, manifest)
            raise
        phase.update(
            {
                "status": "completed",
                "checkpoint": str(checkpoint),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        _write_manifest(manifest_path, manifest)
        print(f"[PIPELINE] {name}: completed -> {checkpoint}")

    if args.dry_run:
        print("[PIPELINE] dry run only; no directory, checkpoint, or manifest was written.")
    else:
        print(f"[PIPELINE] all phases completed; manifest={manifest_path}")


if __name__ == "__main__":
    main()
