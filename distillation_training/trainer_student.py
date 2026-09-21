"""Reusable BC/DAgger losses and bounded, mixed-source replay."""
from __future__ import annotations
from collections import deque
import random
import torch
from torch.nn import functional as F

class BoundedReplay:
    def __init__(self,max_items=200000): self.items=deque(maxlen=max_items)
    def add(self,source,**batch):
        n=batch['sensor'].shape[0]
        for i in range(n): self.items.append((source,{k:v[i].detach().cpu() for k,v in batch.items()}))
    def sample(self,n,teacher_fraction=.5):
        teacher=[x for s,x in self.items if s=='teacher']; student=[x for s,x in self.items if s=='student']
        take_t=min(round(n*teacher_fraction),len(teacher)); take_s=min(n-take_t,len(student))
        rows=(random.sample(teacher,take_t)+random.sample(student,take_s))
        if not rows: raise RuntimeError('replay is empty')
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
    sensor=batch['sensor'].to(device);n=sensor.shape[1];hidden=torch.zeros(n,128,device=device);totals={}
    for t in range(sensor.shape[0]):
        losses=distillation_losses(actor,estimator,sensor[t],batch['previous_action'][t].to(device),
          batch['previous_tendon'][t].to(device),hidden,batch['privileged'][t].to(device),
          batch['teacher_mean'][t].to(device),batch['episode_start'][t].to(device),lambda_p,lambda_kl)
        hidden=losses.pop('hidden')
        for key,value in losses.items():
            if key not in ('predicted','student_mean'): totals[key]=totals.get(key,0)+value
    return {key:value/sensor.shape[0] for key,value in totals.items()}
