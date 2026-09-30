#!/usr/bin/env python3
"""96-step recurrent Student PPO fine-tuning with persistent AMP style reward."""
import argparse,sys,torch
import numpy as np
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))

from distillation_training.amp import AMPTrainingModule,add_amp_args,config_from_args
from distillation_training.networks import SharedActor,HistoryEstimator,TeacherCritic
from distillation_training.returns import compute_gae_with_time_limit_bootstrap
from distillation_training.runtime import (add_runtime_args,dimension_metadata,inherit_checkpoint_variant,
  make_env,selected_variant,tensor,env_step,validate_checkpoint,sensor_metadata,task_metadata)
from distillation_training.trainer_student import distillation_losses
from distillation_training.config import PRIVILEGED_FIELDS


def _checkpoint_payload(a,actor,estimator,critic,optimizer,amp,env,iteration,global_step,grad):
    spec=selected_variant(a);meta=sensor_metadata(env,spec)
    return {'shared_actor':actor.state_dict(),'history_estimator':estimator.state_dict(),
      'critic':critic.state_dict(),'log_std':actor.log_std.detach().cpu(),'optimizer':optimizer.state_dict(),
      'discriminator':amp.discriminator.state_dict(),'discriminator_optimizer':amp.optimizer.state_dict(),
      'teacher_checkpoint':str(a.teacher_checkpoint),'sensor_metadata':meta,'phase_clock':meta['phase_clock'],
      'privileged_schema':{'fields':list(PRIVILEGED_FIELDS),'dim':20},'amp_config':amp.metadata,
      'amp_feature_dim':env.amp_obs_dim,'amp_actor_input':False,'amp_training_only':True,
      'amp_discriminator_source':getattr(a,'amp_discriminator_source','teacher'),
      'iteration':iteration,'global_step':global_step,'bptt_steps':a.num_steps,
      'recurrent_sequence_update':True,'rollout_initial_hidden_saved':True,'rl_finetune_stage':'A',
      'frozen_modules':['sensor_encoder','privileged_encoder','actor_head'],
      'trainable_modules':['history_estimator','critic','log_std'],
      'amp_discriminator_frozen':bool(a.freeze_amp_discriminator),'gradient_norm':grad,
      **task_metadata(env),**dimension_metadata(spec)}


def _emergency(a,iteration,actor,estimator,critic,optimizer,amp,env,global_step,diagnostics):
    target=Path(a.output).with_suffix('.emergency.pt');target.parent.mkdir(parents=True,exist_ok=True)
    payload=_checkpoint_payload(a,actor,estimator,critic,optimizer,amp,env,iteration,global_step,float('nan'))
    payload.update({'emergency':True,'diagnostics':diagnostics});torch.save(payload,target);return target


from distillation_training.runtime import resolve_training_output

def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher-checkpoint',required=True)
    p.add_argument('--student-checkpoint',required=True);add_runtime_args(p);p.set_defaults(output='student_rl.pt')
    p.add_argument('--lambda-p',type=float,default=.5);p.add_argument('--lambda-kl',type=float,default=.5)
    add_amp_args(p);a=p.parse_args();resolve_training_output(a, "student_rl");device=torch.device(a.device);torch.manual_seed(a.seed)
    teacher=validate_checkpoint(a.teacher_checkpoint,'teacher',allow_legacy_reward_checkpoint=a.allow_legacy_reward_checkpoint);student=validate_checkpoint(a.student_checkpoint,'student',allow_legacy_reward_checkpoint=a.allow_legacy_reward_checkpoint)
    spec=inherit_checkpoint_variant(a,('teacher',teacher),('student',student))
    actor=SharedActor(spec.sensor_dim).to(device);actor.load_state_dict(student['shared_actor'])
    for name,parameter in actor.named_parameters():parameter.requires_grad=(name=='log_std')
    teacher_actor=SharedActor(spec.sensor_dim).to(device);teacher_actor.load_state_dict(teacher['shared_actor']);teacher_actor.eval()
    for parameter in teacher_actor.parameters():parameter.requires_grad=False
    estimator=HistoryEstimator(spec.sensor_dim).to(device);estimator.load_state_dict(student['history_estimator'])
    critic=TeacherCritic(spec.sensor_dim).to(device)
    if student.get('rl_finetune_stage') and student.get('critic') is not None:critic.load_state_dict(student['critic'])
    optimizer=torch.optim.Adam([actor.log_std,*estimator.parameters(),*critic.parameters()],lr=a.learning_rate)
    if student.get('rl_finetune_stage') and student.get('optimizer') is not None:optimizer.load_state_dict(student['optimizer'])
    env,cfg=make_env(a);amp=AMPTrainingModule(env.amp_obs_dim,device,config_from_args(a),cfg.motion_file)
    amp_source=student if student.get('discriminator') is not None else teacher
    a.amp_discriminator_source='student_resume' if amp_source is student else 'teacher'
    amp.load(amp_source,load_optimizer=amp_source is student and student.get('discriminator_optimizer') is not None)
    obs=tensor(env.reset(),device);n=a.num_envs;pa=torch.zeros(n,14,device=device)
    pt=obs[:,42:56].clone();start=torch.ones(n,device=device);hidden=torch.zeros(n,128,device=device)
    global_step=int(student.get('global_step',0)) if student.get('rl_finetune_stage') else 0
    start_iteration=int(student.get('iteration',0)) if student.get('rl_finetune_stage') else 0
    rng=np.random.default_rng(a.seed+1000);last_metrics={};grad=torch.zeros((),device=device)
    if start_iteration>=a.max_iterations:
        raise ValueError(f'checkpoint iteration {start_iteration} already reached --max-iterations {a.max_iterations}')
    for iteration in range(start_iteration+1,a.max_iterations+1):
        rows=[];initial_hidden=hidden.detach().clone();force_metrics={key:[] for key in ('left_total','right_total','nonzero_fraction','saturation_fraction')}
        for _ in range(a.num_steps):
            truth=tensor(env.privileged_vector(),device)
            with torch.no_grad():
                predicted,hidden=estimator(obs,pa,pt,hidden,start);dist=actor.distribution(obs,predicted)
                action=dist.sample();logp=dist.log_prob(action).sum(-1);value,_=critic(obs,truth)
                teacher_mean=teacher_actor.mean(obs,truth);amp_feature=tensor(env.amp_observe(),device)
                amp_reward=amp.policy_reward(amp_feature)
            nxt,task_reward,done,info=env_step(env,action);reward=amp.combine_reward(task_reward,amp_reward)
            time_out=tensor(info.get('time_outs',torch.zeros_like(done)),device)
            timeout_bootstrap=torch.zeros(n,device=device)
            if 'terminal_observation' in info:
                terminal_ids=torch.as_tensor(info['terminal_env_ids'],device=device,dtype=torch.long)
                terminal_obs=tensor(info['terminal_observation'],device)
                terminal_privileged=tensor(info['terminal_privileged_vector'],device)
                with torch.no_grad():terminal_value,_=critic(terminal_obs,terminal_privileged)
                timeout_bootstrap[terminal_ids]=terminal_value
            rows.append({'obs':obs,'previous_action':pa,'previous_tendon':pt,'episode_start':start,
              'truth':truth,'teacher_mean':teacher_mean,'action':action,'old_logp':logp,'value':value,
              'reward':reward,'done':done,'time_out':time_out,'timeout_bootstrap':timeout_bootstrap,
              'task_reward':task_reward,'amp_reward':amp_reward,'amp_feature':amp_feature})
            for key in force_metrics:
                info_key=f'foot_force/{key}'
                if info_key in info:force_metrics[key].append(tensor(info[info_key],device))
            pt=torch.where(done[:,None]>0,nxt[:,42:56],obs[:,42:56]);pa=torch.where(done[:,None]>0,torch.zeros_like(action),action)
            start=done;obs=nxt;global_step+=n
        with torch.no_grad():next_value,_=critic(obs,tensor(env.privileged_vector(),device))
        adv,ret=compute_gae_with_time_limit_bootstrap(
          torch.stack([row['reward'] for row in rows]),torch.stack([row['value'] for row in rows]),
          torch.stack([row['done'] for row in rows]),torch.stack([row['time_out'] for row in rows]),
          torch.stack([row['timeout_bootstrap'] for row in rows]),next_value,.99,.95)
        adv=(adv-adv.mean())/(adv.std()+1e-8);h=initial_hidden;logps=[];values=[];entropies=[];aux=[];step_losses=[]
        anneal=max(0.,1.-iteration/max(a.max_iterations,1))
        for row in rows:
            losses=distillation_losses(actor,estimator,row['obs'],row['previous_action'],row['previous_tendon'],h,
              row['truth'],row['teacher_mean'],row['episode_start'],a.lambda_p*anneal,a.lambda_kl*anneal)
            h=losses['hidden'];distribution=actor.distribution(row['obs'],losses['predicted'])
            logps.append(distribution.log_prob(row['action']).sum(-1));entropies.append(distribution.entropy().sum(-1))
            values.append(critic(row['obs'],row['truth'])[0]);aux.append(losses['loss']);step_losses.append(losses)
        new=torch.stack(logps);old=torch.stack([row['old_logp'] for row in rows]);ratio=(new-old).exp()
        policy=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
        value_loss=.5*(torch.stack(values)-ret).square().mean();privileged_loss=torch.stack([x['privileged'] for x in step_losses]).mean()
        action_kl=torch.stack([x['kl'] for x in step_losses]).mean();teacher_mse=torch.stack([x['action_mse'] for x in step_losses]).mean()
        estimator_loss=torch.stack(aux).mean();entropy=torch.stack(entropies).mean();loss=policy+value_loss+estimator_loss-.002*entropy
        rollout_rewards=torch.stack([row['reward'] for row in rows])
        finite_values=torch.stack((loss,policy,value_loss,entropy,privileged_loss,action_kl,teacher_mse,estimator_loss))
        finite=(finite_values.isfinite().all() & rollout_rewards.isfinite().all()
          & actor.log_std.exp().isfinite().all() & h.isfinite().all())
        if not bool(finite):
            emergency=_emergency(a,iteration,actor,estimator,critic,optimizer,amp,env,global_step,
              {'sensor_variant':spec.name,'losses':finite_values.detach().cpu(),
               'reward_min':rollout_rewards.nan_to_num().min().detach().cpu(),
               'reward_max':rollout_rewards.nan_to_num().max().detach().cpu(),
               'hidden_norm':h.nan_to_num().norm(dim=1).mean().detach().cpu(),
               'action_std':actor.log_std.exp().detach().cpu()})
            raise RuntimeError(f'Student RL loss became NaN/Inf at iteration {iteration}; saved {emergency}')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_([*actor.parameters(),*estimator.parameters(),*critic.parameters()],1)
        if not bool(torch.isfinite(grad)):
            emergency=_emergency(a,iteration,actor,estimator,critic,optimizer,amp,env,global_step,
              {'gradient_norm':grad.detach().cpu(),'action_std':actor.log_std.exp().detach().cpu()})
            raise RuntimeError(f'Student RL gradient became NaN/Inf at iteration {iteration}; saved {emergency}')
        optimizer.step();actor.clamp_log_std_();hidden=hidden.detach()
        try:amp_metrics=amp.update(torch.stack([row['amp_feature'] for row in rows]),env.motion,rng,freeze=a.freeze_amp_discriminator)
        except RuntimeError as exc:
            emergency=_emergency(a,iteration,actor,estimator,critic,optimizer,amp,env,global_step,{'error':str(exc)})
            raise RuntimeError(f'{exc}; emergency checkpoint: {emergency}') from exc
        zero=torch.zeros((),device=device)
        force_means={key:(torch.stack(values).mean() if values else zero) for key,values in force_metrics.items()}
        if spec.use_foot_force and a.sim_backend=='warp' and float(force_means['nonzero_fraction'])<=0:
            emergency=_emergency(a,iteration,actor,estimator,critic,optimizer,amp,env,global_step,{'force_metrics':{k:v.detach().cpu() for k,v in force_means.items()}})
            raise RuntimeError(f'Warp foot force stayed zero for the full rollout; saved {emergency}')
        if spec.use_foot_force and a.sim_backend=='warp' and float(force_means['saturation_fraction'])>.95:
            emergency=_emergency(a,iteration,actor,estimator,critic,optimizer,amp,env,global_step,{'force_metrics':{k:v.detach().cpu() for k,v in force_means.items()}})
            raise RuntimeError(f'Warp foot force saturation exceeded 0.95; saved {emergency}')
        last_metrics={'student_rl/task_reward':torch.stack([row['task_reward'] for row in rows]).mean(),
          'student_rl/amp_reward':torch.stack([row['amp_reward'] for row in rows]).mean(),
          'student_rl/combined_reward':torch.stack([row['reward'] for row in rows]).mean(),
          'student_rl/policy_loss':policy.detach(),'student_rl/value_loss':value_loss.detach(),
          'student_rl/entropy':entropy.detach(),'student_rl/privileged_loss':privileged_loss.detach(),
          'student_rl/action_kl':action_kl.detach(),'student_rl/teacher_action_mse':teacher_mse.detach(),
          'amp/discriminator_loss':amp_metrics['discriminator_loss'],'amp/agent_logit':amp_metrics['agent_logit'],
          'amp/expert_logit':amp_metrics['expert_logit'],'amp/gradient_penalty':amp_metrics['gradient_penalty'],
          'estimator/root_velocity_mae':torch.stack([x['root_velocity_mae'] for x in step_losses]).mean().detach(),
          'estimator/gravity_error':torch.stack([x['gravity_error'] for x in step_losses]).mean().detach(),
          'estimator/tendon_velocity_mae':torch.stack([x['tendon_velocity_mae'] for x in step_losses]).mean().detach(),
          'recurrent/hidden_norm':h.norm(dim=1).mean().detach(),'foot_force/left_total':force_means['left_total'],
          'foot_force/right_total':force_means['right_total'],'foot_force/nonzero_fraction':force_means['nonzero_fraction'],
          'foot_force/saturation_fraction':force_means['saturation_fraction']}
        rendered=' '.join(f'{key}={float(value):.5f}' for key,value in last_metrics.items())
        print(f'[student-rl] iter={iteration} step={global_step} auxiliary_scale={anneal:.4f} {rendered}')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    payload=_checkpoint_payload(a,actor,estimator,critic,optimizer,amp,env,a.max_iterations,global_step,float(grad))
    payload['training_metrics']={key:float(value) for key,value in last_metrics.items()};torch.save(payload,out)
    env.close();print(out.resolve());return out


if __name__=='__main__':main()
