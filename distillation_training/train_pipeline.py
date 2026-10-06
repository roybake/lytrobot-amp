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
from distillation_training.runtime import checkpoint_variant,validate_checkpoint
from training_paths import new_run_dir, validate_run_name

def run(command: list[str]) -> None:
    print("[PIPELINE]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

def main() -> None:
    p=argparse.ArgumentParser(description="Teacher → BC → DAgger → RL automation")
    p.add_argument('--teacher-checkpoint', required=True, help='evaluated shared-teacher .pt')
    p.add_argument('--run-dir', default=None, help='Existing timestamp directory to reuse; defaults to logs/sensor_policy/distillation/MM-DD-HH:MM:SS')
    p.add_argument('--skip-dagger', action='store_true')
    p.add_argument('--skip-rl', action='store_true')
    p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--max-iterations',type=int,default=1000)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--sim-backend',choices=('cpu','warp'),default='warp')
    p.add_argument('--max-validation-action-mse',type=float)
    p.add_argument('--max-validation-normalized-action-mse',type=float)
    p.add_argument('--max-validation-kl',type=float)
    a=p.parse_args(); teacher=Path(a.teacher_checkpoint).resolve()
    if not teacher.is_file(): p.error(f'teacher checkpoint does not exist: {teacher}')
    out=(ROOT/a.run_dir).resolve() if a.run_dir else new_run_dir(ROOT/'logs/sensor_policy/distillation')
    validate_run_name(out.name)
    out.mkdir(parents=True,exist_ok=True)
    student = out / 'student'; student.mkdir(exist_ok=True)
    from training_paths import snapshot_teacher
    snapshot_teacher(teacher, out / 'teacher')
    bc=student/'student_bc.pt'; dagger=student/'student_dagger.pt'; final=student/'student_rl.pt'
    teacher_payload=validate_checkpoint(teacher,'teacher');variant=checkpoint_variant(teacher_payload,'teacher');common=['--num-envs',str(a.num_envs),'--num-steps','96','--max-iterations',str(a.max_iterations),'--device',a.device,'--sim-backend',a.sim_backend,'--task-reward-mode',teacher_payload['task_reward_mode'],'--motion-file',teacher_payload['motion_file'],'--target-velocity',str(teacher_payload['target_velocity']),'--no-wandb']
    manual=a.max_validation_action_mse is not None and a.max_validation_kl is not None
    relative=a.max_validation_normalized_action_mse is not None
    if not (manual or relative):p.error('automatic pipeline requires explicit BC manual thresholds or --max-validation-normalized-action-mse')
    gate=[]
    if a.max_validation_action_mse is not None:gate+=['--max-validation-action-mse',str(a.max_validation_action_mse)]
    if a.max_validation_normalized_action_mse is not None:gate+=['--max-validation-normalized-action-mse',str(a.max_validation_normalized_action_mse)]
    if a.max_validation_kl is not None:gate+=['--max-validation-kl',str(a.max_validation_kl)]
    run(['uv','run','python','distillation_training/train_student_bc.py','--teacher-checkpoint',str(teacher),'--output',str(bc),*gate,*common]);validate_checkpoint(bc,'student')
    if not torch.load(bc,map_location='cpu',weights_only=False)['validation']['passed']: raise RuntimeError('BC validation gate failed')
    current=bc
    if not a.skip_dagger:
        run(['uv','run','python','distillation_training/train_student_dagger.py','--teacher-checkpoint',str(teacher),'--student-checkpoint',str(current),'--output',str(dagger),*common]);validate_checkpoint(dagger,'student')
        current=dagger
    if not a.skip_rl:
        run(['uv','run','python','distillation_training/train_student_rl.py','--teacher-checkpoint',str(teacher),'--student-checkpoint',str(current),'--output',str(final),'--amp',*common]);validate_checkpoint(final,'student')
        current=final
    (out/'pipeline.json').write_text(json.dumps({'sensor_variant':variant.name,'sensor_dim':variant.sensor_dim,'estimator_input_dim':variant.estimator_input_dim,'teacher_checkpoint':str(teacher),'bc_checkpoint':str(bc),'dagger_checkpoint':str(dagger),'final_checkpoint':str(final),'skip_dagger':a.skip_dagger,'skip_rl':a.skip_rl},indent=2)+'\n')
    print(f'[PIPELINE] manifest: {out / "pipeline.json"}')
if __name__=='__main__': main()
