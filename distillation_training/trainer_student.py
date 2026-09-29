"""Reusable BC/DAgger losses and bounded, mixed-source replay."""
from __future__ import annotations
from collections import deque
import random
import torch
from torch.nn import functional as F

class BoundedReplay:
    """Sequence replay whose storage stays on the selected training device."""
    def __init__(self,max_items=200000,storage_device=None):
        self.items=deque(maxlen=max_items);self.storage_device=storage_device
        self.last_sample_counts={'teacher':0,'student':0}
    def add(self,source,**batch):
        n=batch['sensor'].shape[0]
        for i in range(n):
            row={k:v[i].detach().to(self.storage_device) if self.storage_device is not None else v[i].detach() for k,v in batch.items()}
            self.items.append((source,row))
    def sample(self,n,teacher_fraction=.5):
        teacher=[x for s,x in self.items if s=='teacher']; student=[x for s,x in self.items if s=='student']
        take_t=min(round(n*teacher_fraction),len(teacher)); take_s=min(n-take_t,len(student))
        rows=(random.sample(teacher,take_t)+random.sample(student,take_s))
        if not rows: raise RuntimeError('replay is empty')
        self.last_sample_counts={'teacher':take_t,'student':take_s}
        return {k:torch.stack([r[k] for r in rows]) for k in rows[0]}

def distillation_losses(actor, estimator, sensor, previous_action, previous_tendon, hidden, privileged, teacher_mean,episode_start=None,lambda_p=1.,lambda_kl=1.):
    predicted,hidden=estimator(sensor,previous_action,previous_tendon,hidden,episode_start)
    student_mean=actor.mean(sensor,predicted)
    reconstruction=F.smooth_l1_loss(predicted,privileged)
    action_mse=F.mse_loss(student_mean,teacher_mean)
    # Equal fixed diagonal std means Gaussian KL is 0.5 * squared normalized mean error.
    std=actor.log_std.exp().clamp_min(1e-4)
    kl=.5*(((teacher_mean-student_mean)/std)**2).mean()
    return {'loss':lambda_p*reconstruction+lambda_kl*kl,'privileged':reconstruction,
      'privileged_mae':F.l1_loss(predicted,privileged),'root_velocity_mae':F.l1_loss(predicted[:,:3],privileged[:,:3]),
      'gravity_error':torch.linalg.vector_norm(predicted[:,3:6]-privileged[:,3:6],dim=1).mean(),
      'tendon_velocity_mae':F.l1_loss(predicted[:,6:],privileged[:,6:]),
      'action_mse':action_mse,'kl':kl,'hidden':hidden,'predicted':predicted,'student_mean':student_mean}

def sequence_losses(actor,estimator,batch,device,lambda_p=1.,lambda_kl=1.):
    """Unroll a complete [T,N,...] batch; never flatten recurrent time."""
    sensor=batch['sensor'].to(device,non_blocking=True);n=sensor.shape[1];hidden=torch.zeros(n,128,device=device);totals={}
    for t in range(sensor.shape[0]):
        losses=distillation_losses(actor,estimator,sensor[t],batch['previous_action'][t].to(device,non_blocking=True),
          batch['previous_tendon'][t].to(device,non_blocking=True),hidden,batch['privileged'][t].to(device,non_blocking=True),
          batch['teacher_mean'][t].to(device,non_blocking=True),batch['episode_start'][t].to(device,non_blocking=True),lambda_p,lambda_kl)
        hidden=losses.pop('hidden')
        for key,value in losses.items():
            if key not in ('predicted','student_mean'): totals[key]=totals.get(key,0)+value
    result={key:value/sensor.shape[0] for key,value in totals.items()}
    teacher=batch['teacher_mean'].to(device,non_blocking=True)
    result['teacher_action_variance']=teacher.var(unbiased=False)
    result['normalized_action_mse']=result['action_mse']/(result['teacher_action_variance']+1e-8)
    return result
