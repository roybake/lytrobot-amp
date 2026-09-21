#!/usr/bin/env python3
import argparse,torch,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
from distillation_training.networks import SharedActor,HistoryEstimator
from distillation_training.runtime import (add_runtime_args,dimension_metadata,inherit_checkpoint_variant,
  make_env,tensor,env_step,validate_checkpoint,sensor_metadata)
from distillation_training.trainer_student import BoundedReplay,sequence_losses
from distillation_training.config import PRIVILEGED_FIELDS


def collect(env,actor,steps,device):
    obs=tensor(env.reset(),device);n=obs.shape[0];pa=torch.zeros(n,14,device=device)
    pt=obs[:,42:56].clone();start=torch.ones(n,device=device);rows=[];force={key:[] for key in ('nonzero_fraction','saturation_fraction')}
    for _ in range(steps):
        privileged=tensor(env.privileged_vector(),device)
        with torch.no_grad():mean=actor.mean(obs,privileged);action=actor.distribution(obs,privileged).sample()
        rows.append({'sensor':obs,'previous_action':pa,'previous_tendon':pt,'privileged':privileged,
          'teacher_mean':mean,'episode_start':start})
        nxt,_,done,info=env_step(env,action)
        for key in force:
            if f'foot_force/{key}' in info:force[key].append(tensor(info[f'foot_force/{key}'],device))
        pt=torch.where(done[:,None]>0,nxt[:,42:56],obs[:,42:56])
        pa=torch.where(done[:,None]>0,torch.zeros_like(action),action);start=done;obs=nxt
    zero=torch.zeros((),device=device)
    health={key:(torch.stack(values).mean() if values else zero) for key,values in force.items()}
    return {k:torch.stack([r[k] for r in rows]) for k in rows[0]},health


def split(batch,fraction=.8):
    """Fixed disjoint train/validation environments within each collection."""
    n=batch['sensor'].shape[1];cut=max(1,min(n-1,round(n*fraction))) if n>1 else 1
    return ({k:v[:,:cut] for k,v in batch.items()},
      {k:v[:,cut:] if cut<n else v[:,:1] for k,v in batch.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher-checkpoint',required=True);add_runtime_args(p)
    p.set_defaults(output='student_bc.pt')
    p.add_argument('--max-validation-action-mse',type=float)
    p.add_argument('--max-validation-normalized-action-mse',type=float)
    p.add_argument('--max-validation-kl',type=float)
    p.add_argument('--require-validation-thresholds',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--allow-unvalidated-output',action='store_true')
    p.add_argument('--max-buffer',type=int,default=4096);p.add_argument('--updates-per-iteration',type=int,default=2)
    a=p.parse_args();device=torch.device(a.device);torch.manual_seed(a.seed)
    ck=validate_checkpoint(a.teacher_checkpoint,'teacher');spec=inherit_checkpoint_variant(a,('teacher',ck))
    actor=SharedActor(spec.sensor_dim).to(device);actor.load_state_dict(ck['shared_actor']);actor.eval()
    for parameter in actor.parameters():parameter.requires_grad=False
    estimator=HistoryEstimator(spec.sensor_dim).to(device);optimizer=torch.optim.Adam(estimator.parameters(),lr=a.learning_rate)
    env,cfg=make_env(a);replay=BoundedReplay(a.max_buffer,storage_device=device);global_step=0
    for iteration in range(1,a.max_iterations+1):
        collected,force_health=collect(env,actor,a.num_steps,device);train,val=split(collected)
        if spec.use_foot_force and a.sim_backend=='warp' and (float(force_health['nonzero_fraction'])<=0.0 or float(force_health['saturation_fraction'])>.95):
            emergency=Path(a.output).with_suffix('.emergency.pt');emergency.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'stage':'bc','iteration':iteration,'history_estimator':estimator.state_dict(),
              'force_metrics':{k:v.detach().cpu() for k,v in force_health.items()}},emergency)
            raise RuntimeError(f'Warp foot force health check failed at iteration {iteration}; saved {emergency}')
        replay.add('teacher',**{k:v.transpose(0,1) for k,v in train.items()})
        for _ in range(a.updates_per_iteration):
            sampled=replay.sample(min(a.num_envs,len(replay.items)),1.0)
            batch={k:v.transpose(0,1) for k,v in sampled.items()};losses=sequence_losses(actor,estimator,batch,device)
            finite=torch.stack([value for value in losses.values() if torch.is_tensor(value)]).isfinite().all()
            if not bool(finite):
                emergency=Path(a.output).with_suffix('.emergency.pt');emergency.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'stage':'bc','iteration':iteration,'history_estimator':estimator.state_dict(),
                  'optimizer':optimizer.state_dict(),'metrics':{k:v.detach().cpu() for k,v in losses.items()}},emergency)
                raise RuntimeError(f'BC loss became NaN/Inf at iteration {iteration}; saved {emergency}')
            optimizer.zero_grad(set_to_none=True);losses['loss'].backward()
            grad=torch.nn.utils.clip_grad_norm_(estimator.parameters(),1)
            if not bool(torch.isfinite(grad)):
                emergency=Path(a.output).with_suffix('.emergency.pt');emergency.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'stage':'bc','iteration':iteration,'history_estimator':estimator.state_dict(),
                  'optimizer':optimizer.state_dict(),'gradient_norm':grad.detach().cpu()},emergency)
                raise RuntimeError(f'BC gradient became NaN/Inf at iteration {iteration}; saved {emergency}')
            optimizer.step()
        global_step+=a.num_envs*a.num_steps
        with torch.no_grad():metrics=sequence_losses(actor,estimator,val,device)
        print(f"[bc] iter={iteration} loss={float(losses['loss']):.6f} val_action_mse={float(metrics['action_mse']):.6f} val_normalized_action_mse={float(metrics['normalized_action_mse']):.6f} val_kl={float(metrics['kl']):.6f}")
    manual_resolved=a.max_validation_action_mse is not None and a.max_validation_kl is not None
    relative_resolved=a.max_validation_normalized_action_mse is not None
    resolved=manual_resolved or relative_resolved
    passed=bool(resolved)
    if manual_resolved:
        passed=passed and float(metrics['action_mse'])<=a.max_validation_action_mse and float(metrics['kl'])<=a.max_validation_kl
    if relative_resolved:
        passed=passed and float(metrics['normalized_action_mse'])<=a.max_validation_normalized_action_mse
    status='passed' if passed else ('failed' if resolved else 'unresolved')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    metric_names=('privileged_mae','root_velocity_mae','gravity_error','tendon_velocity_mae','action_mse',
      'normalized_action_mse','teacher_action_variance','kl')
    validation={f'val_{key}':float(metrics[key]) for key in metric_names}
    validation.update({'passed':passed,'resolved':resolved,'status':status,'gradient_norm':float(grad),
      'thresholds':{'max_action_mse':a.max_validation_action_mse,
        'max_normalized_action_mse':a.max_validation_normalized_action_mse,'max_kl':a.max_validation_kl}})
    torch.save({'shared_actor':actor.state_dict(),'history_estimator':estimator.state_dict(),
      'log_std':actor.log_std.detach().cpu(),'optimizer':optimizer.state_dict(),
      'teacher_checkpoint':str(a.teacher_checkpoint),'sensor_metadata':sensor_metadata(env,spec),
      'phase_clock':env.observation_metadata['phase_clock'],
      'privileged_schema':{'fields':list(PRIVILEGED_FIELDS),'dim':20},'validation':validation,
      'iteration':a.max_iterations,'global_step':global_step,'bptt_steps':a.num_steps,
      'replay_format':'bounded device-resident ReplaySequence[N,96,...]',
      'replay_storage_device':str(device),'replay_transfer_strategy':'device_resident_no_per_batch_transfer',
      'bc_sequence_starts_from_env_reset':True,
      'bc_updates_per_iteration':a.updates_per_iteration,**dimension_metadata(spec)},out)
    env.close();print(out.resolve())
    if a.require_validation_thresholds and not passed and not a.allow_unvalidated_output:raise SystemExit(2)
    return out


if __name__=='__main__':main()
