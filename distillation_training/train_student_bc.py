#!/usr/bin/env python3
import argparse,torch,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
from distillation_training.networks import SharedActor,HistoryEstimator
from distillation_training.runtime import add_runtime_args,make_env,tensor,env_step,validate_checkpoint,sensor_metadata
from distillation_training.trainer_student import sequence_losses
from distillation_training.config import PRIVILEGED_FIELDS
def collect(env,actor,steps,device):
 obs=tensor(env.reset(),device);n=obs.shape[0];pa=torch.zeros(n,14,device=device);pt=obs[:,42:56].clone();start=torch.ones(n,device=device);rows=[]
 for _ in range(steps):
  privileged=tensor(env.privileged_vector(),device)
  with torch.no_grad():mean=actor.mean(obs,privileged);action=actor.distribution(obs,privileged).sample()
  rows.append({'sensor':obs,'previous_action':pa,'previous_tendon':pt,'privileged':privileged,'teacher_mean':mean,'episode_start':start})
  nxt,_,done,_=env_step(env,action);pt=torch.where(done[:,None]>0,nxt[:,42:56],obs[:,42:56]);pa=torch.where(done[:,None]>0,torch.zeros_like(action),action);start=done;obs=nxt
 return {k:torch.stack([r[k] for r in rows]) for k in rows[0]}
def split(batch,fraction=.8):
 n=batch['sensor'].shape[1];cut=max(1,min(n-1,round(n*fraction))) if n>1 else 1
 return ({k:v[:,:cut] for k,v in batch.items()},{k:v[:,cut:] if cut<n else v[:,:1] for k,v in batch.items()})
def main():
 p=argparse.ArgumentParser();p.add_argument('--teacher-checkpoint',required=True);add_runtime_args(p);p.set_defaults(output='student_bc.pt');p.add_argument('--max-validation-action-mse',type=float,default=float('inf'));p.add_argument('--max-validation-kl',type=float,default=float('inf'));a=p.parse_args();device=torch.device(a.device);torch.manual_seed(a.seed)
 ck=validate_checkpoint(a.teacher_checkpoint,'teacher');actor=SharedActor().to(device);actor.load_state_dict(ck['shared_actor']);actor.eval()
 for parameter in actor.parameters():parameter.requires_grad=False
 estimator=HistoryEstimator().to(device);optimizer=torch.optim.Adam(estimator.parameters(),lr=a.learning_rate);env,cfg=make_env(a);global_step=0
 for iteration in range(1,a.max_iterations+1):
  train,val=split(collect(env,actor,a.num_steps,device));losses=sequence_losses(actor,estimator,train,device);optimizer.zero_grad();losses['loss'].backward();grad=float(torch.nn.utils.clip_grad_norm_(estimator.parameters(),1));optimizer.step();global_step+=a.num_envs*a.num_steps
  with torch.no_grad():metrics=sequence_losses(actor,estimator,val,device)
  print(f"[bc] iter={iteration} loss={losses['loss'].item():.6f} val_action_mse={metrics['action_mse'].item():.6f} val_kl={metrics['kl'].item():.6f}")
 passed=metrics['action_mse'].item()<=a.max_validation_action_mse and metrics['kl'].item()<=a.max_validation_kl
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);validation={f'val_{k}':float(metrics[k]) for k in ('privileged_mae','root_velocity_mae','gravity_error','tendon_velocity_mae','action_mse','kl')};validation.update({'passed':passed,'gradient_norm':grad})
 torch.save({'shared_actor':actor.state_dict(),'history_estimator':estimator.state_dict(),'log_std':actor.log_std.detach().cpu(),'optimizer':optimizer.state_dict(),'teacher_checkpoint':str(a.teacher_checkpoint),'sensor_metadata':sensor_metadata(env),'privileged_schema':{'fields':list(PRIVILEGED_FIELDS),'dim':20},'validation':validation,'iteration':a.max_iterations,'global_step':global_step,'bptt_steps':a.num_steps},out);env.close();print(out.resolve())
 if not passed:raise SystemExit(2)
 return out
if __name__=='__main__':main()
