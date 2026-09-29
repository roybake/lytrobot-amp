#!/usr/bin/env python3
"""Variant-aware deployable-sensor + 20D asymmetric Teacher PPO+AMP."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))

import argparse
from dataclasses import asdict
import numpy as np
import torch

from distillation_training.amp import AMPTrainingModule,add_amp_args,config_from_args
from distillation_training.config import PHASE_FORCE,PRIVILEGED_FIELDS
from distillation_training.networks import SharedActor,TeacherCritic
from distillation_training.returns import compute_gae_with_time_limit_bootstrap
from distillation_training.runtime import (add_runtime_args,dimension_metadata,make_env,
  selected_variant,tensor,env_step,sensor_metadata,task_metadata,
  validate_checkpoint,inherit_checkpoint_variant)


REWARD_COMPONENTS = {
    'pose_track':'pose_track','vel_track':'vel_track','height':'height',
    'root_orientation':'root_orientation','root_vel':'root_vel','foot_pos':'foot_pos',
    'foot_vel':'foot_vel','stance_foot':'stance_foot','contact':'contact',
    'missed_contact':'missed_contact','symmetry':'symmetry','ctrl_track':'ctrl_track',
    'ctrl_penalty':'ctrl_penalty','action_rate':'action_rate',
}


def _init_wandb(args, spec):
    if args.no_wandb:
        print('[INFO] wandb disabled for this run.')
        return None
    try:
        import wandb
    except ImportError:
        print('[WARN] wandb is not installed; continuing without online logging.')
        return None
    run_name=args.wandb_run_name or f'teacher-{spec.name}'
    return wandb.init(project=args.wandb_project,entity=args.wandb_entity,name=run_name,
      tags=[tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()],
      config={**vars(args),'sensor_dim':spec.sensor_dim,'estimator_input_dim':spec.estimator_input_dim},
      settings=wandb.Settings(console='off'))


def _emergency(path, iteration, actor, critic, optimizer, amp, diagnostics):
    target=Path(path or 'teacher.pt').with_suffix('.emergency.pt')
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'stage':'teacher','iteration':iteration,'shared_actor':actor.state_dict(),
      'critic':critic.state_dict(),'optimizer':optimizer.state_dict(),
      'discriminator':amp.discriminator.state_dict(),
      'discriminator_optimizer':amp.optimizer.state_dict(),'diagnostics':diagnostics},target)
    return target


def main():
    p=argparse.ArgumentParser();add_runtime_args(p,steps=96,sensor_variant_default=PHASE_FORCE.name);p.add_argument('--epochs',type=int,default=4)
    p.set_defaults(learning_rate=1e-4)
    p.add_argument('--clip-coef',type=float,default=.2)
    p.add_argument('--target-kl',type=float,default=.025)
    p.add_argument('--resume',default=None,help='Resume an optimized Teacher checkpoint.')
    add_amp_args(p);a=p.parse_args();torch.manual_seed(a.seed)
    device=torch.device(a.device);env,cfg=make_env(a);spec=selected_variant(a)
    wandb_run=_init_wandb(a,spec)
    actor=SharedActor(spec.sensor_dim).to(device);critic=TeacherCritic(spec.sensor_dim).to(device)
    optimizer=torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=a.learning_rate)
    amp=AMPTrainingModule(env.amp_obs_dim,device,config_from_args(a),cfg.motion_file)
    start_iteration=0;global_step=0
    if a.resume:
        previous=validate_checkpoint(a.resume,'teacher')
        inherit_checkpoint_variant(a,('teacher',previous))
        old_amp=previous.get('amp_config',{})
        amp_mismatch={key:(old_amp.get(key),amp.metadata[key]) for key in (
            'enabled','task_reward_weight','amp_reward_weight'
        ) if old_amp.get(key)!=amp.metadata[key]}
        if amp_mismatch:
            raise ValueError(f'resume AMP reward configuration mismatch: {amp_mismatch}')
        actor.load_state_dict(previous['shared_actor'])
        critic.value_net.load_state_dict(previous['critic'])
        critic.median_net.load_state_dict(previous['median_critic'])
        optimizer.load_state_dict(previous['optimizer'])
        amp.load(previous,load_optimizer=True)
        start_iteration=int(previous['iteration'])
        global_step=int(previous['global_step'])
        if start_iteration>=a.max_iterations:
            raise ValueError(f'checkpoint iteration {start_iteration} already reached --max-iterations {a.max_iterations}')
        print(f'[INFO] resumed Teacher at iteration {start_iteration} step {global_step}')
    obs=tensor(env.reset(),device)
    rng=np.random.default_rng(a.seed+1000+start_iteration);last_metrics={}
    episode_returns=torch.zeros(a.num_envs,device=device)
    for iteration in range(start_iteration+1,a.max_iterations+1):
        rows=[];force_nonzero=[];force_saturation=[]
        reward_components={name:[] for name in REWARD_COMPONENTS}
        completed_returns=[];completed_lengths=[];terminations=[];forward_velocities=[]
        for _ in range(a.num_steps):
            privileged=tensor(env.privileged_vector(),device)
            with torch.no_grad():
                dist=actor.distribution(obs,privileged);action=dist.sample();logp=dist.log_prob(action).sum(-1)
                value,median=critic(obs,privileged);amp_feature=tensor(env.amp_observe(),device)
                amp_reward=amp.policy_reward(amp_feature)
            next_obs,task_reward,done,info=env_step(env,action)
            reward=amp.combine_reward(task_reward,amp_reward)
            episode_returns+=reward
            done_mask=done>0
            if bool(done_mask.any()):
                completed_returns.append(episode_returns[done_mask].detach())
                completed_lengths.append(tensor(info['episode_step'],device)[done_mask])
                episode_returns[done_mask]=0
            term=tensor(info.get('termination',torch.zeros_like(done)),device)
            terminations.append(term)
            if 'forward_velocity' in info:forward_velocities.append(tensor(info['forward_velocity'],device))
            for log_name,info_name in REWARD_COMPONENTS.items():
                if info_name in info:reward_components[log_name].append(tensor(info[info_name],device))
            time_out=tensor(info.get('time_outs',torch.zeros_like(done)),device)
            timeout_bootstrap=torch.zeros(a.num_envs,device=device)
            if 'terminal_observation' in info:
                terminal_ids=torch.as_tensor(info['terminal_env_ids'],device=device,dtype=torch.long)
                terminal_obs=tensor(info['terminal_observation'],device)
                terminal_privileged=tensor(info['terminal_privileged_vector'],device)
                with torch.no_grad():terminal_value,_=critic(terminal_obs,terminal_privileged)
                timeout_bootstrap[terminal_ids]=terminal_value
            rows.append((obs,privileged,action,logp,value,median,reward,done,time_out,
              timeout_bootstrap,amp_feature,task_reward,amp_reward))
            if 'foot_force/nonzero_fraction' in info:force_nonzero.append(tensor(info['foot_force/nonzero_fraction'],device))
            if 'foot_force/saturation_fraction' in info:force_saturation.append(tensor(info['foot_force/saturation_fraction'],device))
            obs=next_obs;global_step+=a.num_envs
        with torch.no_grad():next_value,_=critic(obs,tensor(env.privileged_vector(),device))
        rollout_rewards=torch.stack([r[6] for r in rows])
        if not bool(rollout_rewards.isfinite().all() & actor.log_std.exp().isfinite().all()):
            emergency=_emergency(a.output,iteration,actor,critic,optimizer,amp,
              {'sensor_variant':spec.name,'reward_min':rollout_rewards.nan_to_num().min().detach().cpu(),
               'reward_max':rollout_rewards.nan_to_num().max().detach().cpu(),
               'action_std':actor.log_std.exp().detach().cpu()})
            raise RuntimeError(f'Teacher rollout became NaN/Inf at iteration {iteration}; saved {emergency}')
        advantages,returns=compute_gae_with_time_limit_bootstrap(
          rollout_rewards,torch.stack([r[4] for r in rows]),
          torch.stack([r[7] for r in rows]),torch.stack([r[8] for r in rows]),
          torch.stack([r[9] for r in rows]),next_value,.99,.95)
        S=torch.cat([r[0] for r in rows]);P=torch.cat([r[1] for r in rows]);A=torch.cat([r[2] for r in rows])
        OLD=torch.cat([r[3] for r in rows]);ADV=advantages.flatten();RET=returns.flatten()
        advantage_mean=ADV.mean().detach();advantage_std=ADV.std().detach()
        ADV=(ADV-advantage_mean)/(advantage_std+1e-8)
        epochs_completed=0
        for _ in range(a.epochs):
            dist=actor.distribution(S,P);new=dist.log_prob(A).sum(-1);ratio=(new-OLD).exp()
            with torch.no_grad():
                epoch_kl=((ratio-1.0)-(new-OLD)).mean()
            if epochs_completed and float(epoch_kl)>2.0*a.target_kl:
                break
            policy=-torch.minimum(ratio*ADV,ratio.clamp(1-a.clip_coef,1+a.clip_coef)*ADV).mean()
            value,median=critic(S,P);value_loss=.5*((value-RET).square().mean()+(median-RET.median()).square().mean())
            loss=policy+value_loss-.002*dist.entropy().sum(-1).mean()
            if not bool(torch.stack((loss,policy,value_loss,dist.entropy().mean())).isfinite().all()):
                emergency=_emergency(a.output,iteration,actor,critic,optimizer,amp,{'loss':loss.detach().cpu(),'action_std':actor.log_std.exp().detach().cpu()})
                raise RuntimeError(f'Teacher loss became NaN/Inf at iteration {iteration}; saved {emergency}')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_([*actor.parameters(),*critic.parameters()],1)
            if not bool(torch.isfinite(grad)):
                emergency=_emergency(a.output,iteration,actor,critic,optimizer,amp,{'gradient_norm':grad.detach().cpu(),'action_std':actor.log_std.exp().detach().cpu()})
                raise RuntimeError(f'Teacher gradient became NaN/Inf at iteration {iteration}; saved {emergency}')
            optimizer.step()
            actor.clamp_log_std_()
            epochs_completed+=1
        with torch.no_grad():
            final_logp=actor.distribution(S,P).log_prob(A).sum(-1)
            log_ratio=final_logp-OLD
            final_ratio=log_ratio.clamp(-20.0,20.0).exp()
            approx_kl=((final_ratio-1.0)-log_ratio).mean()
            clip_fraction=((final_ratio-1.0).abs()>a.clip_coef).float().mean()
        agent=torch.stack([r[10] for r in rows])
        try:amp_metrics=amp.update(agent,env.motion,rng,freeze=a.freeze_amp_discriminator)
        except RuntimeError as exc:
            emergency=_emergency(a.output,iteration,actor,critic,optimizer,amp,{'error':str(exc),'action_std':actor.log_std.exp().detach().cpu()})
            raise RuntimeError(f'{exc}; emergency checkpoint: {emergency}') from exc
        task_mean=torch.stack([r[11] for r in rows]).mean();amp_mean=torch.stack([r[12] for r in rows]).mean()
        combined_mean=torch.stack([r[6] for r in rows]).mean()
        nonzero=torch.stack(force_nonzero).mean() if force_nonzero else torch.zeros((),device=device)
        saturation=torch.stack(force_saturation).mean() if force_saturation else torch.zeros((),device=device)
        if spec.use_foot_force and a.sim_backend=='warp' and (float(nonzero)<=0.0 or float(saturation)>.95):
            emergency=_emergency(a.output,iteration,actor,critic,optimizer,amp,
              {'force_nonzero_fraction':nonzero.detach().cpu(),'force_saturation_fraction':saturation.detach().cpu(),
               'action_std':actor.log_std.exp().detach().cpu()})
            raise RuntimeError(f'Warp foot force health check failed at iteration {iteration}; saved {emergency}')
        zero=torch.zeros((),device=device)
        component_means={f'teacher/reward_{name}':(
          torch.stack(values).mean() if values else zero) for name,values in reward_components.items()}
        episode_reward=(torch.cat(completed_returns).mean() if completed_returns else episode_returns.mean())
        episode_length=(torch.cat(completed_lengths).mean() if completed_lengths else zero)
        fall_rate=torch.stack(terminations).mean() if terminations else zero
        last_metrics={'teacher/task_reward':task_mean,'teacher/amp_reward':amp_mean,'amp/reward':amp_mean,'teacher/combined_reward':combined_mean,
          'teacher/policy_loss':policy.detach(),'teacher/value_loss':value_loss.detach(),'teacher/entropy':dist.entropy().sum(-1).mean().detach(),
          'teacher/approx_kl':approx_kl,'teacher/clip_fraction':clip_fraction,
          'teacher/update_epochs':torch.as_tensor(float(epochs_completed),device=device),
          'teacher/action_std':actor.log_std.exp().mean().detach(),
          'teacher/advantage_mean':advantage_mean,'teacher/advantage_std':advantage_std,
          'teacher/forward_velocity':torch.stack(forward_velocities).mean() if forward_velocities else zero,
          'teacher/episode_length':episode_length,'teacher/episode_reward':episode_reward,
          'teacher/fall_rate':fall_rate,
          'amp/discriminator_loss':amp_metrics['discriminator_loss'],'amp/agent_logit':amp_metrics['agent_logit'],
          'amp/expert_logit':amp_metrics['expert_logit'],'amp/gradient_penalty':amp_metrics['gradient_penalty'],
          'foot_force/nonzero_fraction':nonzero,'foot_force/saturation_fraction':saturation,
          **component_means}
        if wandb_run is not None and iteration % max(a.wandb_log_interval,1)==0:
            wandb_run.log({**{key:float(value) for key,value in last_metrics.items()},
              'iteration':iteration,'global_step':global_step},step=global_step)
        rendered=' '.join(f'{key}={float(value):.5f}' for key,value in last_metrics.items())
        print(f'[teacher] iter={iteration} step={global_step} {rendered}')
    out=Path(a.output or 'teacher.pt');out.parent.mkdir(parents=True,exist_ok=True);meta=sensor_metadata(env,spec)
    payload={'shared_actor':actor.state_dict(),'critic':critic.value_net.state_dict(),'median_critic':critic.median_net.state_dict(),
      'log_std':actor.log_std.detach().cpu(),'optimizer':optimizer.state_dict(),'discriminator':amp.discriminator.state_dict(),
      'discriminator_optimizer':amp.optimizer.state_dict(),'ppo_config':vars(a),'env_config':asdict(cfg),'sensor_metadata':meta,
      'foot_force_metadata':meta.get('foot_force',{}),'phase_clock':meta['phase_clock'],'privileged_schema':{'fields':list(PRIVILEGED_FIELDS),'dim':20},
      'amp_config':amp.metadata,'amp_feature_dim':env.amp_obs_dim,'amp_actor_input':False,'amp_training_only':True,
      'iteration':a.max_iterations,'global_step':global_step,'evaluation':{key:float(value) for key,value in last_metrics.items()},
      **task_metadata(env),**dimension_metadata(spec)}
    torch.save(payload,out);print(out.resolve());env.close()
    if wandb_run is not None:wandb_run.finish()
    return out


if __name__=='__main__':main()
