#!/usr/bin/env python3
import argparse,sys,torch
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
from distillation_training.networks import SharedActor,HistoryEstimator
from distillation_training.runtime import (add_runtime_args,dimension_metadata,inherit_checkpoint_variant,
 make_env,tensor,env_step,validate_checkpoint,sensor_metadata,task_metadata)
from distillation_training.trainer_student import BoundedReplay,sequence_losses
from distillation_training.config import PRIVILEGED_FIELDS
def rollout(env,actor,estimator,steps,device,student_control):
 obs=tensor(env.reset(),device);n=obs.shape[0];pa=torch.zeros(n,14,device=device);pt=obs[:,42:56].clone();start=torch.ones(n,device=device);hidden=torch.zeros(n,128,device=device);rows=[];force={key:[] for key in ('nonzero_fraction','saturation_fraction')}
 for _ in range(steps):
  truth=tensor(env.privileged_vector(),device)
  with torch.no_grad():
   teacher=actor.mean(obs,truth);pred,hidden=estimator(obs,pa,pt,hidden,start);student=actor.mean(obs,pred);action=student if student_control else teacher
  rows.append({'sensor':obs,'previous_action':pa,'previous_tendon':pt,'privileged':truth,'teacher_mean':teacher,'student_mean':student,'episode_start':start})
  nxt,_,done,info=env_step(env,action)
  for key in force:
   if f'foot_force/{key}' in info:force[key].append(tensor(info[f'foot_force/{key}'],device))
  pt=torch.where(done[:,None]>0,nxt[:,42:56],obs[:,42:56]);pa=torch.where(done[:,None]>0,torch.zeros_like(action),action);start=done;obs=nxt
 zero=torch.zeros((),device=device);health={key:(torch.stack(values).mean() if values else zero) for key,values in force.items()}
 return {k:torch.stack([r[k] for r in rows],1) for k in rows[0]},health # [N,T,...]
from distillation_training.runtime import resolve_training_output

def main():
 p=argparse.ArgumentParser();p.add_argument('--teacher-checkpoint',required=True);p.add_argument('--student-checkpoint',required=True);p.add_argument('--max-buffer',type=int,default=4096);add_runtime_args(p);p.set_defaults(output='student_dagger.pt');a=p.parse_args();resolve_training_output(a, "student_dagger");device=torch.device(a.device);torch.manual_seed(a.seed)
 teacher=validate_checkpoint(a.teacher_checkpoint,'teacher',allow_legacy_reward_checkpoint=a.allow_legacy_reward_checkpoint);student=validate_checkpoint(a.student_checkpoint,'student',allow_legacy_reward_checkpoint=a.allow_legacy_reward_checkpoint);spec=inherit_checkpoint_variant(a,('teacher',teacher),('student',student));actor=SharedActor(spec.sensor_dim).to(device);actor.load_state_dict(teacher['shared_actor']);actor.eval()
 for x in actor.parameters():x.requires_grad=False
 estimator=HistoryEstimator(spec.sensor_dim).to(device);estimator.load_state_dict(student['history_estimator']);optimizer=torch.optim.Adam(estimator.parameters(),lr=a.learning_rate);env,cfg=make_env(a);replay=BoundedReplay(a.max_buffer,storage_device=device);global_step=0
 for iteration in range(1,a.max_iterations+1):
  teacher_rollout,teacher_force=rollout(env,actor,estimator,a.num_steps,device,False);student_rollout,student_force=rollout(env,actor,estimator,a.num_steps,device,True)
  if spec.use_foot_force and a.sim_backend=='warp' and (min(float(teacher_force['nonzero_fraction']),float(student_force['nonzero_fraction']))<=0.0 or max(float(teacher_force['saturation_fraction']),float(student_force['saturation_fraction']))>.95):
   emergency=Path(a.output).with_suffix('.emergency.pt');emergency.parent.mkdir(parents=True,exist_ok=True);torch.save({'stage':'dagger','iteration':iteration,'history_estimator':estimator.state_dict(),'teacher_force':{k:v.detach().cpu() for k,v in teacher_force.items()},'student_force':{k:v.detach().cpu() for k,v in student_force.items()}},emergency);raise RuntimeError(f'Warp foot force health check failed at iteration {iteration}; saved {emergency}')
  replay.add('teacher',**teacher_rollout);replay.add('student',**student_rollout);fraction=.5-.3*((iteration-1)/max(a.max_iterations-1,1));sample=replay.sample(min(a.num_envs,len(replay.items)),fraction);batch={k:v.transpose(0,1) for k,v in sample.items() if k!='student_mean'};losses=sequence_losses(actor,estimator,batch,device)
  if not bool(torch.stack([v for v in losses.values() if torch.is_tensor(v)]).isfinite().all()):
   emergency=Path(a.output).with_suffix('.emergency.pt');emergency.parent.mkdir(parents=True,exist_ok=True);torch.save({'stage':'dagger','iteration':iteration,'history_estimator':estimator.state_dict(),'optimizer':optimizer.state_dict(),'metrics':{k:v.detach().cpu() for k,v in losses.items()}},emergency);raise RuntimeError(f'DAgger loss became NaN/Inf at iteration {iteration}; saved {emergency}')
  optimizer.zero_grad(set_to_none=True);losses['loss'].backward();grad_tensor=torch.nn.utils.clip_grad_norm_(estimator.parameters(),1)
  if not bool(torch.isfinite(grad_tensor)):
   emergency=Path(a.output).with_suffix('.emergency.pt');emergency.parent.mkdir(parents=True,exist_ok=True);torch.save({'stage':'dagger','iteration':iteration,'history_estimator':estimator.state_dict(),'optimizer':optimizer.state_dict(),'gradient_norm':grad_tensor.detach().cpu()},emergency);raise RuntimeError(f'DAgger gradient became NaN/Inf at iteration {iteration}; saved {emergency}')
  grad=float(grad_tensor);optimizer.step();global_step+=2*a.num_envs*a.num_steps;counts=replay.last_sample_counts;print(f"[dagger] iter={iteration} replay={len(replay.items)} requested_teacher_fraction={fraction:.3f} actual_teacher_samples={counts['teacher']} actual_student_samples={counts['student']} loss={losses['loss'].item():.6f}")
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save({'shared_actor':actor.state_dict(),'history_estimator':estimator.state_dict(),'log_std':actor.log_std.detach().cpu(),'optimizer':optimizer.state_dict(),'teacher_checkpoint':str(a.teacher_checkpoint),'sensor_metadata':sensor_metadata(env,spec),'phase_clock':env.observation_metadata['phase_clock'],'privileged_schema':{'fields':list(PRIVILEGED_FIELDS),'dim':20},'iteration':a.max_iterations,'global_step':global_step,'bptt_steps':a.num_steps,'replay_format':'bounded device-resident ReplaySequence[N,96,...]','replay_storage_device':str(device),'replay_transfer_strategy':'device_resident_no_per_batch_transfer','dagger_sequence_starts_from_env_reset':True,'gradient_norm':grad,**task_metadata(env),**dimension_metadata(spec)},out);env.close();print(out.resolve());return out
if __name__=='__main__':main()
