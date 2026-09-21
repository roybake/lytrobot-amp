"""Shared simulator construction, tensor conversion and checkpoint validation."""
from __future__ import annotations
from pathlib import Path
import torch
from lytrobot_amp.env import LytRobotEnvCfg
from .envs import DistillationCpuEnv,DistillationWarpEnv
from .config import PRIVILEGED_FIELDS

ROOT=Path(__file__).resolve().parents[1]
def add_runtime_args(p,*,steps=96):
    p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--num-steps',type=int,default=steps)
    p.add_argument('--max-iterations',type=int,default=1000);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--sim-backend',choices=('cpu','warp'),default='warp');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--no-wandb',action='store_true');p.add_argument('--output',default=None)
def make_env(a):
    if a.sim_backend=='warp' and a.device!='cuda': raise ValueError('Warp requires --device cuda')
    cfg=LytRobotEnvCfg(model_xml=str(ROOT/'robot/MJCFmodelposition.xml'),motion_file=str(ROOT/'robot/reference_data/walk1_mjcf.npz'),num_envs=a.num_envs,seed=a.seed,reference_ctrl_blend=0.0,domain_randomization=False)
    return (DistillationWarpEnv(cfg,a.device) if a.sim_backend=='warp' else DistillationCpuEnv(cfg)),cfg
def tensor(value,device):
    if torch.is_tensor(value): return value.to(device=device,dtype=torch.float32)
    return torch.as_tensor(value,device=device,dtype=torch.float32)
def env_step(env,action):
    sent=action if env.is_torch_env else action.detach().cpu().numpy();obs,reward,done,info=env.step(sent)
    return tensor(obs,action.device),tensor(reward,action.device),tensor(done,action.device),info
def sensor_metadata(env):
    data=dict(env.observation_metadata);data['privileged_schema']={'fields':list(PRIVILEGED_FIELDS),'dim':20,'actor_input':False};return data
TEACHER_KEYS={'shared_actor','critic','median_critic','log_std','optimizer','ppo_config','env_config','sensor_metadata','foot_force_metadata','privileged_schema','iteration','global_step'}
STUDENT_KEYS={'shared_actor','history_estimator','log_std','sensor_metadata','privileged_schema','iteration','global_step'}
def validate_checkpoint(path,kind):
    ck=torch.load(path,map_location='cpu',weights_only=False);required=TEACHER_KEYS if kind=='teacher' else STUDENT_KEYS;missing=required-set(ck)
    if missing: raise ValueError(f'{kind} checkpoint missing keys: {sorted(missing)}')
    return ck
