#!/usr/bin/env python3
"""Real 64D deployable-sensor + 20D asymmetric teacher PPO trainer."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
import argparse,torch
import numpy as np
from torch.nn import functional as F
from dataclasses import asdict
from distillation_training.networks import SharedActor,TeacherCritic
from lytrobot_amp.networks import AmpDiscriminator
from distillation_training.runtime import add_runtime_args,make_env,tensor,env_step,sensor_metadata
from distillation_training.config import PRIVILEGED_FIELDS
def main():
 p=argparse.ArgumentParser();add_runtime_args(p,steps=24);p.add_argument('--epochs',type=int,default=4);p.add_argument('--clip-coef',type=float,default=.2);p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True);a=p.parse_args();torch.manual_seed(a.seed)
 device=torch.device(a.device);env,cfg=make_env(a);actor=SharedActor().to(device);critic=TeacherCritic().to(device);discriminator=AmpDiscriminator(env.amp_obs_dim).to(device);optimizer=torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=a.learning_rate);disc_optimizer=torch.optim.Adam(discriminator.parameters(),lr=3e-5);obs=tensor(env.reset(),device);global_step=0;rng=np.random.default_rng(a.seed+1000)
 for iteration in range(1,a.max_iterations+1):
  rows=[]
  for _ in range(a.num_steps):
   privileged=tensor(env.privileged_vector(),device)
   with torch.no_grad():
    dist=actor.distribution(obs,privileged);action=dist.sample();logp=dist.log_prob(action).sum(-1);value,median=critic(obs,privileged)
   amp_feature=tensor(env.amp_observe(),device)
   next_obs,reward,done,_=env_step(env,action)
   if a.amp:reward=.5*reward+.6*discriminator.reward(amp_feature)
   rows.append((obs,privileged,action,logp,value,median,reward,done,amp_feature));obs=next_obs;global_step+=a.num_envs
  with torch.no_grad():next_value,_=critic(obs,tensor(env.privileged_vector(),device))
  advantages=[];gae=torch.zeros(a.num_envs,device=device)
  for row in reversed(rows):
   delta=row[6]+.99*(1-row[7])*next_value-row[4];gae=delta+.99*.95*(1-row[7])*gae;advantages.append(gae);next_value=row[4]
  advantages=torch.stack(list(reversed(advantages)));returns=advantages+torch.stack([r[4] for r in rows]);S=torch.cat([r[0] for r in rows]);P=torch.cat([r[1] for r in rows]);A=torch.cat([r[2] for r in rows]);OLD=torch.cat([r[3] for r in rows]);ADV=advantages.flatten();RET=returns.flatten();ADV=(ADV-ADV.mean())/(ADV.std()+1e-8)
  for _ in range(a.epochs):
   dist=actor.distribution(S,P);new=dist.log_prob(A).sum(-1);ratio=(new-OLD).exp();policy=-torch.minimum(ratio*ADV,ratio.clamp(1-a.clip_coef,1+a.clip_coef)*ADV).mean();value,median=critic(S,P);value_loss=.5*((value-RET).square().mean()+(median-RET.median()).square().mean());loss=policy+value_loss-.002*dist.entropy().sum(-1).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([*actor.parameters(),*critic.parameters()],1);optimizer.step()
  if a.amp:
   agent=torch.cat([r[8] for r in rows]);reference=tensor(env.motion.sample_features(agent.shape[0],rng),device);disc_loss=F.binary_cross_entropy_with_logits(discriminator(agent.detach()),torch.zeros(agent.shape[0],device=device))+F.binary_cross_entropy_with_logits(discriminator(reference),torch.ones(agent.shape[0],device=device));disc_optimizer.zero_grad();disc_loss.backward();disc_optimizer.step()
  print(f'[teacher] iter={iteration} step={global_step} reward={torch.stack([r[6] for r in rows]).mean().item():.4f}')
 out=Path(a.output or 'teacher.pt');out.parent.mkdir(parents=True,exist_ok=True);meta=sensor_metadata(env);payload={'shared_actor':actor.state_dict(),'critic':critic.value_net.state_dict(),'median_critic':critic.median_net.state_dict(),'log_std':actor.log_std.detach().cpu(),'optimizer':optimizer.state_dict(),'discriminator':discriminator.state_dict(),'discriminator_optimizer':disc_optimizer.state_dict(),'ppo_config':vars(a),'env_config':asdict(cfg),'sensor_metadata':meta,'foot_force_metadata':meta['foot_force'],'privileged_schema':{'fields':list(PRIVILEGED_FIELDS),'dim':20},'iteration':a.max_iterations,'global_step':global_step,'evaluation':{'mean_rollout_reward':float(torch.stack([r[6] for r in rows]).mean())}};torch.save(payload,out);print(out.resolve());env.close();return out
if __name__=='__main__':main()
