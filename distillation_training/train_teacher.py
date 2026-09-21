#!/usr/bin/env python3
"""Real 64D deployable-sensor + 20D asymmetric Teacher PPO+AMP."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))

import argparse
from dataclasses import asdict
import numpy as np
import torch

from distillation_training.amp import AMPTrainingModule,add_amp_args,config_from_args
from distillation_training.config import PRIVILEGED_FIELDS
from distillation_training.networks import SharedActor,TeacherCritic
from distillation_training.runtime import add_runtime_args,make_env,tensor,env_step,sensor_metadata


def _emergency(path, iteration, actor, critic, optimizer, amp, diagnostics):
    target=Path(path or 'teacher.pt').with_suffix('.emergency.pt')
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'stage':'teacher','iteration':iteration,'shared_actor':actor.state_dict(),
      'critic':critic.state_dict(),'optimizer':optimizer.state_dict(),
      'discriminator':amp.discriminator.state_dict(),
      'discriminator_optimizer':amp.optimizer.state_dict(),'diagnostics':diagnostics},target)
    return target


def main():
    p=argparse.ArgumentParser();add_runtime_args(p,steps=24);p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--clip-coef',type=float,default=.2);add_amp_args(p);a=p.parse_args();torch.manual_seed(a.seed)
    device=torch.device(a.device);env,cfg=make_env(a);actor=SharedActor().to(device);critic=TeacherCritic().to(device)
    optimizer=torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=a.learning_rate)
    amp=AMPTrainingModule(env.amp_obs_dim,device,config_from_args(a));obs=tensor(env.reset(),device)
    global_step=0;rng=np.random.default_rng(a.seed+1000);last_metrics={}
    for iteration in range(1,a.max_iterations+1):
        rows=[];force_nonzero=[];force_saturation=[]
        for _ in range(a.num_steps):
            privileged=tensor(env.privileged_vector(),device)
            with torch.no_grad():
                dist=actor.distribution(obs,privileged);action=dist.sample();logp=dist.log_prob(action).sum(-1)
                value,median=critic(obs,privileged);amp_feature=tensor(env.amp_observe(),device)
                amp_reward=amp.policy_reward(amp_feature)
            next_obs,task_reward,done,info=env_step(env,action)
            reward=amp.combine_reward(task_reward,amp_reward)
            rows.append((obs,privileged,action,logp,value,median,reward,done,amp_feature,task_reward,amp_reward))
            if 'foot_force/nonzero_fraction' in info:force_nonzero.append(tensor(info['foot_force/nonzero_fraction'],device))
            if 'foot_force/saturation_fraction' in info:force_saturation.append(tensor(info['foot_force/saturation_fraction'],device))
            obs=next_obs;global_step+=a.num_envs
        with torch.no_grad():next_value,_=critic(obs,tensor(env.privileged_vector(),device))
        advantages=[];gae=torch.zeros(a.num_envs,device=device)
        for row in reversed(rows):
            delta=row[6]+.99*(1-row[7])*next_value-row[4];gae=delta+.99*.95*(1-row[7])*gae
            advantages.append(gae);next_value=row[4]
        advantages=torch.stack(list(reversed(advantages)));returns=advantages+torch.stack([r[4] for r in rows])
        S=torch.cat([r[0] for r in rows]);P=torch.cat([r[1] for r in rows]);A=torch.cat([r[2] for r in rows])
        OLD=torch.cat([r[3] for r in rows]);ADV=advantages.flatten();RET=returns.flatten();ADV=(ADV-ADV.mean())/(ADV.std()+1e-8)
        for _ in range(a.epochs):
            dist=actor.distribution(S,P);new=dist.log_prob(A).sum(-1);ratio=(new-OLD).exp()
            policy=-torch.minimum(ratio*ADV,ratio.clamp(1-a.clip_coef,1+a.clip_coef)*ADV).mean()
            value,median=critic(S,P);value_loss=.5*((value-RET).square().mean()+(median-RET.median()).square().mean())
            loss=policy+value_loss-.002*dist.entropy().sum(-1).mean()
            if not bool(torch.stack((loss,policy,value_loss)).isfinite().all()):
                emergency=_emergency(a.output,iteration,actor,critic,optimizer,amp,{'loss':loss.detach().cpu(),'action_std':actor.log_std.exp().detach().cpu()})
                raise RuntimeError(f'Teacher loss became NaN/Inf at iteration {iteration}; saved {emergency}')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_([*actor.parameters(),*critic.parameters()],1)
            if not bool(torch.isfinite(grad)):
                emergency=_emergency(a.output,iteration,actor,critic,optimizer,amp,{'gradient_norm':grad.detach().cpu(),'action_std':actor.log_std.exp().detach().cpu()})
                raise RuntimeError(f'Teacher gradient became NaN/Inf at iteration {iteration}; saved {emergency}')
            optimizer.step()
        agent=torch.stack([r[8] for r in rows])
        try:amp_metrics=amp.update(agent,env.motion,rng,freeze=a.freeze_amp_discriminator)
        except RuntimeError as exc:
            emergency=_emergency(a.output,iteration,actor,critic,optimizer,amp,{'error':str(exc),'action_std':actor.log_std.exp().detach().cpu()})
            raise RuntimeError(f'{exc}; emergency checkpoint: {emergency}') from exc
        task_mean=torch.stack([r[9] for r in rows]).mean();amp_mean=torch.stack([r[10] for r in rows]).mean()
        combined_mean=torch.stack([r[6] for r in rows]).mean()
        nonzero=torch.stack(force_nonzero).mean() if force_nonzero else torch.zeros((),device=device)
        saturation=torch.stack(force_saturation).mean() if force_saturation else torch.zeros((),device=device)
        if a.sim_backend=='warp' and (float(nonzero)<=0.0 or float(saturation)>.95):
            emergency=_emergency(a.output,iteration,actor,critic,optimizer,amp,
              {'force_nonzero_fraction':nonzero.detach().cpu(),'force_saturation_fraction':saturation.detach().cpu(),
               'action_std':actor.log_std.exp().detach().cpu()})
            raise RuntimeError(f'Warp foot force health check failed at iteration {iteration}; saved {emergency}')
        last_metrics={'teacher/task_reward':task_mean,'teacher/amp_reward':amp_mean,'teacher/combined_reward':combined_mean,
          'teacher/policy_loss':policy.detach(),'teacher/value_loss':value_loss.detach(),'teacher/entropy':dist.entropy().sum(-1).mean().detach(),
          'amp/discriminator_loss':amp_metrics['discriminator_loss'],'amp/agent_logit':amp_metrics['agent_logit'],
          'amp/expert_logit':amp_metrics['expert_logit'],'amp/gradient_penalty':amp_metrics['gradient_penalty'],
          'foot_force/nonzero_fraction':nonzero,'foot_force/saturation_fraction':saturation}
        rendered=' '.join(f'{key}={float(value):.5f}' for key,value in last_metrics.items())
        print(f'[teacher] iter={iteration} step={global_step} {rendered}')
    out=Path(a.output or 'teacher.pt');out.parent.mkdir(parents=True,exist_ok=True);meta=sensor_metadata(env)
    payload={'shared_actor':actor.state_dict(),'critic':critic.value_net.state_dict(),'median_critic':critic.median_net.state_dict(),
      'log_std':actor.log_std.detach().cpu(),'optimizer':optimizer.state_dict(),'discriminator':amp.discriminator.state_dict(),
      'discriminator_optimizer':amp.optimizer.state_dict(),'ppo_config':vars(a),'env_config':asdict(cfg),'sensor_metadata':meta,
      'foot_force_metadata':meta['foot_force'],'phase_clock':meta['phase_clock'],'privileged_schema':{'fields':list(PRIVILEGED_FIELDS),'dim':20},
      'amp_config':amp.metadata,'amp_feature_dim':env.amp_obs_dim,'amp_actor_input':False,'amp_training_only':True,
      'iteration':a.max_iterations,'global_step':global_step,'evaluation':{key:float(value) for key,value in last_metrics.items()}}
    torch.save(payload,out);print(out.resolve());env.close();return out


if __name__=='__main__':main()
