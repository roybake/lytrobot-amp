#!/usr/bin/env python3
"""Run the four distillation stages in order with one reproducible manifest.

The teacher trainer is intentionally a separate long-running RL job.  Pass an
already evaluated teacher checkpoint to skip it; this avoids accidentally
starting student BC from a partially trained teacher.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from distillation_training.runtime import validate_checkpoint

def run(command: list[str]) -> None:
    print("[PIPELINE]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

def main() -> None:
    p=argparse.ArgumentParser(description="Teacher → BC → DAgger → RL automation")
    p.add_argument('--teacher-checkpoint', required=True, help='evaluated shared-teacher .pt')
    p.add_argument('--run-dir', default='logs/distillation/pipeline')
    p.add_argument('--skip-dagger', action='store_true')
    p.add_argument('--skip-rl', action='store_true')
    p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--max-iterations',type=int,default=1000)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--sim-backend',choices=('cpu','warp'),default='warp')
    a=p.parse_args(); teacher=Path(a.teacher_checkpoint).resolve()
    if not teacher.is_file(): p.error(f'teacher checkpoint does not exist: {teacher}')
    out=(ROOT/a.run_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    bc=out/'student_bc.pt'; dagger=out/'student_dagger.pt'; final=out/'student_rl.pt'
    validate_checkpoint(teacher,'teacher');common=['--num-envs',str(a.num_envs),'--num-steps','96','--max-iterations',str(a.max_iterations),'--device',a.device,'--sim-backend',a.sim_backend,'--no-wandb']
    run(['uv','run','python','distillation_training/train_student_bc.py','--teacher-checkpoint',str(teacher),'--output',str(bc),*common]);validate_checkpoint(bc,'student')
    if not torch.load(bc,map_location='cpu',weights_only=False)['validation']['passed']: raise RuntimeError('BC validation gate failed')
    current=bc
    if not a.skip_dagger:
        run(['uv','run','python','distillation_training/train_student_dagger.py','--teacher-checkpoint',str(teacher),'--student-checkpoint',str(current),'--output',str(dagger),*common]);validate_checkpoint(dagger,'student')
        current=dagger
    if not a.skip_rl:
        run(['uv','run','python','distillation_training/train_student_rl.py','--teacher-checkpoint',str(teacher),'--student-checkpoint',str(current),'--output',str(final),*common]);validate_checkpoint(final,'student')
        current=final
    (out/'pipeline.json').write_text(json.dumps({'teacher_checkpoint':str(teacher),'bc_checkpoint':str(bc),'dagger_checkpoint':str(dagger),'final_checkpoint':str(final),'skip_dagger':a.skip_dagger,'skip_rl':a.skip_rl},indent=2)+'\n')
    print(f'[PIPELINE] manifest: {out / "pipeline.json"}')
if __name__=='__main__': main()
