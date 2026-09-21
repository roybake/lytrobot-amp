#!/usr/bin/env python3
"""Real CUDA+MuJoCo-Warp end-to-end smoke for both sensor routes."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> None:
    started = time.perf_counter()
    print(f"[gpu-smoke] {label}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"[gpu-smoke] {label}: wall_seconds={time.perf_counter()-started:.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='logs/distillation/gpu_smoke')
    parser.add_argument('--num-envs', type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; GPU smoke refuses to fall back to CPU')
    out = (ROOT / args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    python = [sys.executable]
    standalone = ['--device','cuda','--sim-backend','warp','--num-envs',str(args.num_envs),
                  '--num-steps','16','--max-iterations','2','--no-wandb']
    run('standalone phase_no_force', python + ['foot_sensor_training/train_phase_no_force_stage1.py', *standalone])
    run('standalone phase_force', python + ['foot_sensor_training/train_phase_force_stage1.py', *standalone])

    teacher_no_force = out / 'teacher_phase_no_force.pt'
    teacher_force = out / 'teacher_phase_force.pt'
    teacher_common = ['--device','cuda','--sim-backend','warp','--num-envs',str(args.num_envs),
                      '--num-steps','16','--max-iterations','2','--no-wandb']
    run('teacher phase_no_force', python + ['distillation_training/train_teacher.py','--sensor-variant','phase_no_force',
        '--output',str(teacher_no_force),*teacher_common])
    run('teacher phase_force', python + ['distillation_training/train_teacher.py','--sensor-variant','phase_force',
        '--output',str(teacher_force),*teacher_common])

    # One route traverses the complete recurrent pipeline with a genuine
    # 96-step sequence update. Short BC explicitly remains unresolved.
    recurrent = ['--device','cuda','--sim-backend','warp','--num-envs',str(args.num_envs),
                 '--num-steps','96','--max-iterations','1','--no-wandb']
    bc = out / 'student_no_force_bc.pt'
    dagger = out / 'student_no_force_dagger.pt'
    student = out / 'student_no_force_rl.pt'
    run('BC phase_no_force', python + ['distillation_training/train_student_bc.py','--teacher-checkpoint',
        str(teacher_no_force),'--output',str(bc),'--allow-unvalidated-output',*recurrent])
    run('DAgger phase_no_force', python + ['distillation_training/train_student_dagger.py','--teacher-checkpoint',
        str(teacher_no_force),'--student-checkpoint',str(bc),'--output',str(dagger),*recurrent])
    run('Student AMP-RL phase_no_force', python + ['distillation_training/train_student_rl.py','--teacher-checkpoint',
        str(teacher_no_force),'--student-checkpoint',str(dagger),'--output',str(student),'--amp',*recurrent])
    playback = ['--device','cuda','--sim-backend','warp','--num-envs','1','--steps','32']
    run('Oracle phase_no_force', python + ['distillation_training/play.py','--checkpoint',str(student),
        '--teacher-checkpoint',str(teacher_no_force),'--oracle-privileged',*playback])
    run('ONNX phase_no_force', python + ['distillation_training/play.py','--checkpoint',str(student),
        '--export-onnx',str(out/'student_phase_no_force.onnx'),*playback])
    print(f'[gpu-smoke] complete: {out}')


if __name__ == '__main__':
    main()
