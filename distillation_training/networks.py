"""Shared-head teacher/student policy.  The deployment graph has no truth input."""
import torch
from torch import nn
from sensor_training.networks import build_mlp
from .config import SENSOR_DIM,ACTION_DIM,PRIVILEGED_DIM,ESTIMATOR_INPUT_DIM,HIDDEN_DIM

class SharedActor(nn.Module):
    def __init__(self):
        super().__init__(); self.sensor_encoder=build_mlp(SENSOR_DIM,(128,),128); self.privileged_encoder=build_mlp(PRIVILEGED_DIM,(64,),64); self.actor_head=build_mlp(192,(256,128),ACTION_DIM); self.log_std=nn.Parameter(torch.zeros(ACTION_DIM))
    def mean(self,s,p): return torch.tanh(self.actor_head(torch.cat((self.sensor_encoder(s),self.privileged_encoder(p)),1)))
    def distribution(self,s,p):
        return torch.distributions.Normal(self.mean(s,p),self.log_std.exp().expand(s.shape[0],-1))
    def forward(self,s,p,deterministic=False):
        distribution=self.distribution(s,p)
        action=distribution.mean if deterministic else distribution.rsample()
        return action,distribution.log_prob(action).sum(-1),distribution.entropy().sum(-1)

class TeacherCritic(nn.Module):
    def __init__(self):
        super().__init__(); self.value_net=build_mlp(SENSOR_DIM+PRIVILEGED_DIM,(256,128),1); self.median_net=build_mlp(SENSOR_DIM+PRIVILEGED_DIM,(256,128),1)
    def forward(self,s,p):
        x=torch.cat((s,p),1); return self.value_net(x).squeeze(-1),self.median_net(x).squeeze(-1)
class HistoryEstimator(nn.Module):
    def __init__(self,policy_dt=.01,tendon_velocity_scale=10.0):
        super().__init__(); self.gru=nn.GRUCell(ESTIMATOR_INPUT_DIM,HIDDEN_DIM); self.out=build_mlp(HIDDEN_DIM,(128,),PRIVILEGED_DIM)
        self.policy_dt=float(policy_dt); self.tendon_velocity_scale=float(tendon_velocity_scale)
    def forward(self,s,previous_action,previous_tendon,h,episode_start=None):
        if episode_start is not None: h=h*(1-episode_start.reshape(-1,1).to(h.dtype))
        tendon_velocity=(s[:,42:56]-previous_tendon)/(self.policy_dt*self.tendon_velocity_scale)
        h=self.gru(torch.cat((s,previous_action,tendon_velocity),1),h); return self.out(h),h
class StudentExport(nn.Module):
    def __init__(self,actor,estimator): super().__init__();self.actor=actor;self.estimator=estimator
    def forward(self,observation,previous_action,recurrent_state,episode_start):
        # recurrent state = estimator hidden(128) + previous tendon length(14)
        p,h=self.estimator(observation,previous_action,recurrent_state[:,128:],recurrent_state[:,:128],episode_start)
        action=self.actor.mean(observation,p); return action,torch.cat((h,observation[:,42:56]),1)
