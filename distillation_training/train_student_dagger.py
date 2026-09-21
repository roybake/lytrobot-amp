#!/usr/bin/env python3
import argparse,sys,torch
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
from distillation_training.networks import SharedActor,HistoryEstimator
from distillation_training.runtime import add_runtime_args,make_env,tensor,env_step,validate_checkpoint,sensor_metadata
from distillation_training.trainer_student import BoundedReplay,sequence_losses
from distillation_training.config import PRIVILEGED_FIELDS
def rollout(env,actor,estimator,steps,device,student_control):
 obs=tensor(env.reset(),device);n=obs.shape[0];pa=torch.zeros(n,14,device=device);pt=obs[:,42:56].clone();start=torch.ones(n,device=device);hidden=torch.zeros(n,128,device=device);rows=[]
 for _ in range(steps):
  truth=tensor(env.privileged_vector(),device)
  with torch.no_grad():
   teacher=actor.mean(obs,truth);pred,hidden=estimator(obs,pa,pt,hidden,start);student=actor.mean(obs,pred);action=student if student_control else teacher
  rows.append({'sensor':obs,'previous_action':pa,'previous_tendon':pt,'privileged':truth,'teacher_mean':teacher,'student_mean':student,'episode_start':start})
  nxt,_,done,_=env_step(env,action);pt=torch.where(done[:,None]>0,nxt[:,42:56],obs[:,42:56]);pa=torch.where(done[:,None]>0,torch.zeros_like(action),action);start=done;obs=nxt
 return {k:torch.stack([r[k] for r in rows],1) for k in rows[0]} # [N,T,...]
def main():
 p=argparse.ArgumentParser();p.add_argument('--teacher-checkpoint',required=True);p.add_argument('--student-checkpoint',required=True);p.add_argument('--max-buffer',type=int,default=4096);add_runtime_args(p);p.set_defaults(output='student_dagger.pt');a=p.parse_args();device=torch.device(a.device);torch.manual_seed(a.seed)
 teacher=validate_checkpoint(a.teacher_checkpoint,'teacher');student=validate_checkpoint(a.student_checkpoint,'student');actor=SharedActor().to(device);actor.load_state_dict(teacher['shared_actor']);actor.eval()
 for x in actor.parameters():x.requires_grad=False
 estimator=HistoryEstimator().to(device);estimator.load_state_dict(student['history_estimator']);optimizer=torch.optim.Adam(estimator.parameters(),lr=a.learning_rate);env,cfg=make_env(a);replay=BoundedReplay(a.max_buffer);global_step=0
 for iteration in range(1,a.max_iterations+1):
  replay.add('teacher',**rollout(env,actor,estimator,a.num_steps,device,False));replay.add('student',**rollout(env,actor,estimator,a.num_steps,device,True));fraction=.5-.3*((iteration-1)/max(a.max_iterations-1,1));sample=replay.sample(min(a.num_envs,len(replay.items)),fraction);batch={k:v.transpose(0,1) for k,v in sample.items() if k!='student_mean'};losses=sequence_losses(actor,estimator,batch,device);optimizer.zero_grad();losses['loss'].backward();grad=float(torch.nn.utils.clip_grad_norm_(estimator.parameters(),1));optimizer.step();global_step+=2*a.num_envs*a.num_steps;print(f"[dagger] iter={iteration} replay={len(replay.items)} teacher_fraction={fraction:.3f} loss={losses['loss'].item():.6f}")
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save({'shared_actor':actor.state_dict(),'history_estimator':estimator.state_dict(),'log_std':actor.log_std.detach().cpu(),'optimizer':optimizer.state_dict(),'teacher_checkpoint':str(a.teacher_checkpoint),'sensor_metadata':sensor_metadata(env),'privileged_schema':{'fields':list(PRIVILEGED_FIELDS),'dim':20},'iteration':a.max_iterations,'global_step':global_step,'bptt_steps':a.num_steps,'replay_format':'bounded ReplaySequence[N,96,...]','gradient_norm':grad},out);env.close();print(out.resolve());return out
if __name__=='__main__':main()
