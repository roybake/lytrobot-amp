#!/usr/bin/env python3
import argparse,sys,torch
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
from distillation_training.networks import SharedActor,HistoryEstimator,TeacherCritic
from distillation_training.runtime import add_runtime_args,make_env,tensor,env_step,validate_checkpoint,sensor_metadata
from distillation_training.trainer_student import distillation_losses
from distillation_training.config import PRIVILEGED_FIELDS
def main():
 p=argparse.ArgumentParser();p.add_argument('--teacher-checkpoint',required=True);p.add_argument('--student-checkpoint',required=True);add_runtime_args(p);p.set_defaults(output='student_rl.pt');p.add_argument('--lambda-p',type=float,default=.5);p.add_argument('--lambda-kl',type=float,default=.5);a=p.parse_args();device=torch.device(a.device);torch.manual_seed(a.seed)
 teacher=validate_checkpoint(a.teacher_checkpoint,'teacher');student=validate_checkpoint(a.student_checkpoint,'student');actor=SharedActor().to(device);actor.load_state_dict(student['shared_actor'])
 for name,parameter in actor.named_parameters():parameter.requires_grad=(name=='log_std')
 teacher_actor=SharedActor().to(device);teacher_actor.load_state_dict(teacher['shared_actor']);teacher_actor.eval()
 for parameter in teacher_actor.parameters():parameter.requires_grad=False
 estimator=HistoryEstimator().to(device);estimator.load_state_dict(student['history_estimator']);critic=TeacherCritic().to(device);optimizer=torch.optim.Adam([actor.log_std,*estimator.parameters(),*critic.parameters()],lr=a.learning_rate);env,cfg=make_env(a);obs=tensor(env.reset(),device);n=a.num_envs;pa=torch.zeros(n,14,device=device);pt=obs[:,42:56].clone();start=torch.ones(n,device=device);hidden=torch.zeros(n,128,device=device);global_step=0
 for iteration in range(1,a.max_iterations+1):
  rows=[];initial_hidden=hidden.detach().clone()
  for _ in range(a.num_steps):
   truth=tensor(env.privileged_vector(),device)
   with torch.no_grad():
    predicted,hidden=estimator(obs,pa,pt,hidden,start);dist=actor.distribution(obs,predicted);action=dist.sample();logp=dist.log_prob(action).sum(-1);value,_=critic(obs,truth);teacher_mean=teacher_actor.mean(obs,truth)
   rows.append((obs,pa,pt,start,truth,teacher_mean,action,logp,value));nxt,reward,done,_=env_step(env,action);rows[-1]=(*rows[-1],reward,done);pt=torch.where(done[:,None]>0,nxt[:,42:56],obs[:,42:56]);pa=torch.where(done[:,None]>0,torch.zeros_like(action),action);start=done;obs=nxt;global_step+=n
  with torch.no_grad():next_value,_=critic(obs,tensor(env.privileged_vector(),device))
  adv=[];gae=torch.zeros(n,device=device)
  for r in reversed(rows):
   delta=r[9]+.99*(1-r[10])*next_value-r[8];gae=delta+.99*.95*(1-r[10])*gae;adv.append(gae);next_value=r[8]
  adv=torch.stack(list(reversed(adv)));ret=adv+torch.stack([r[8] for r in rows]);adv=(adv-adv.mean())/(adv.std()+1e-8);h=initial_hidden;logps=[];values=[];aux=[]
  anneal=max(0.,1.-iteration/max(a.max_iterations,1))
  for t,r in enumerate(rows):
   losses=distillation_losses(actor,estimator,r[0],r[1],r[2],h,r[4],r[5],r[3],a.lambda_p*anneal,a.lambda_kl*anneal);h=losses['hidden'];logps.append(actor.distribution(r[0],losses['predicted']).log_prob(r[6]).sum(-1));values.append(critic(r[0],r[4])[0]);aux.append(losses['loss'])
  new=torch.stack(logps);old=torch.stack([r[7] for r in rows]);ratio=(new-old).exp();policy=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean();value_loss=.5*(torch.stack(values)-ret).square().mean();loss=policy+value_loss+torch.stack(aux).mean();optimizer.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_([*actor.parameters(),*estimator.parameters(),*critic.parameters()],1));optimizer.step();hidden=hidden.detach();print(f'[student-rl] iter={iteration} loss={loss.item():.6f} auxiliary_scale={anneal:.4f}')
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save({'shared_actor':actor.state_dict(),'history_estimator':estimator.state_dict(),'critic':critic.state_dict(),'log_std':actor.log_std.detach().cpu(),'optimizer':optimizer.state_dict(),'teacher_checkpoint':str(a.teacher_checkpoint),'sensor_metadata':sensor_metadata(env),'privileged_schema':{'fields':list(PRIVILEGED_FIELDS),'dim':20},'iteration':a.max_iterations,'global_step':global_step,'bptt_steps':a.num_steps,'recurrent_sequence_update':True,'rollout_initial_hidden_saved':True,'rl_finetune_stage':'A','frozen_modules':['sensor_encoder','privileged_encoder','actor_head'],'trainable_modules':['history_estimator','critic','log_std'],'gradient_norm':grad},out);env.close();print(out.resolve());return out
if __name__=='__main__':main()
