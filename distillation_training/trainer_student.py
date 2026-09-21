"""Reusable BC/DAgger losses and bounded, mixed-source replay."""
from __future__ import annotations
from collections import deque
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
        rows=(teacher[:take_t]+student[:take_s])
        if not rows: raise RuntimeError('replay is empty')
        return {k:torch.stack([r[k] for r in rows]) for k in rows[0]}

def distillation_losses(actor, estimator, sensor, previous_action, previous_tendon, hidden, privileged, teacher_mean):
    predicted,hidden=estimator(sensor,previous_action,previous_tendon,hidden)
    student_mean=actor.mean(sensor,predicted)
    reconstruction=F.smooth_l1_loss(predicted,privileged)
    action_mse=F.mse_loss(student_mean,teacher_mean)
    # Equal fixed diagonal std means Gaussian KL is 0.5 * squared normalized mean error.
    std=actor.log_std.exp().clamp_min(1e-4)
    kl=.5*(((teacher_mean-student_mean)/std)**2).mean()
    return {'loss':reconstruction+kl,'privileged':reconstruction,'action_mse':action_mse,'kl':kl,'hidden':hidden}
