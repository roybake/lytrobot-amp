"""Shared simulator construction, tensor conversion and checkpoint validation."""
from __future__ import annotations
from pathlib import Path
import torch
from lytrobot_amp.env import LytRobotEnvCfg
from .envs import DistillationCpuEnv,DistillationWarpEnv
from .config import (PRIVILEGED_FIELDS,SENSOR_DIM,ESTIMATOR_INPUT_DIM,HIDDEN_DIM,
                     PRIVILEGED_DIM,RECURRENT_STATE_DIM)

ROOT=Path(__file__).resolve().parents[1]
def add_runtime_args(p,*,steps=96):
    p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--num-steps',type=int,default=steps)
    p.add_argument('--max-iterations',type=int,default=1000);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--sim-backend',choices=('cpu','warp'),default='warp');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--no-wandb',action='store_true');p.add_argument('--output',default=None)
def make_env(a):
    if a.sim_backend=='warp' and a.device!='cuda': raise ValueError('Warp requires --device cuda')
    cfg=LytRobotEnvCfg(model_xml=str(ROOT/'robot/MJCFmodelposition.xml'),motion_file=str(ROOT/'robot/reference_data/walk1_mjcf.npz'),num_envs=a.num_envs,seed=a.seed,reference_ctrl_blend=0.0,domain_randomization=False,sensor_stability_reward=True,stand_start_prob=1.0,motion_reset_prob=0.0,stand_start_transition_min_steps=120,stand_start_transition_max_steps=120)
    env=DistillationWarpEnv(cfg,a.device) if a.sim_backend=='warp' else DistillationCpuEnv(cfg)
    validate_runtime_schema(env,getattr(a,'num_steps',getattr(a,'steps',0)))
    return env,cfg
def tensor(value,device):
    if torch.is_tensor(value): return value.to(device=device,dtype=torch.float32)
    return torch.as_tensor(value,device=device,dtype=torch.float32)
def env_step(env,action):
    sent=action if env.is_torch_env else action.detach().cpu().numpy();obs,reward,done,info=env.step(sent)
    return tensor(obs,action.device),tensor(reward,action.device),tensor(done,action.device),info
def sensor_metadata(env):
    data=dict(env.observation_metadata);data['privileged_schema']={'fields':list(PRIVILEGED_FIELDS),'dim':20,'actor_input':False};return data
def validate_runtime_schema(env,bptt_steps):
    metadata=env.observation_metadata;fields=tuple(metadata.get('actor_input_fields',()))
    values={"Actor observation dim":env.obs_dim,"IMU":env.imu_layout.dim,
      "Tendon":env.base_env.position_action_dim,
      "Foot force":len(metadata.get('foot_force',{}).get('channel_order',())),
      "Phase":metadata.get('phase_dim',0),"Estimator input":ESTIMATOR_INPUT_DIM,"GRU hidden":HIDDEN_DIM,
      "Privileged target":PRIVILEGED_DIM,"Recurrent state":RECURRENT_STATE_DIM,"BPTT":bptt_steps,
      "Control rate":round(1.0/env.base_env.dt),"reference_ctrl_blend":env.base_env.cfg.reference_ctrl_blend,
      "reference tendon actor input":any('reference' in str(field) for field in fields),
      "privileged actor input":bool(metadata.get('privileged_actor_input',True))}
    expected={"Actor observation dim":SENSOR_DIM,"IMU":42,"Tendon":14,"Foot force":6,
      "Phase":2,"Estimator input":92,"GRU hidden":128,"Privileged target":20,
      "Recurrent state":142,"Control rate":100,"reference_ctrl_blend":0.0,
      "reference tendon actor input":False,"privileged actor input":False}
    for key,value in expected.items():
        if values[key]!=value:raise ValueError(f"schema mismatch: {key}={values[key]!r}, expected {value!r}")
    print("\n".join(f"{key}: {value}" for key,value in values.items()))
TEACHER_KEYS={'shared_actor','critic','median_critic','log_std','optimizer','ppo_config','env_config','sensor_metadata','foot_force_metadata','privileged_schema','iteration','global_step'}
STUDENT_KEYS={'shared_actor','history_estimator','log_std','sensor_metadata','privileged_schema','iteration','global_step'}
def validate_checkpoint(path,kind):
    ck=torch.load(path,map_location='cpu',weights_only=False);required=TEACHER_KEYS if kind=='teacher' else STUDENT_KEYS;missing=required-set(ck)
    if missing: raise ValueError(f'{kind} checkpoint missing keys: {sorted(missing)}')
    metadata=ck.get('sensor_metadata',{})
    if metadata.get('actor_frame_dim')!=SENSOR_DIM or metadata.get('phase_dim')!=2:
        raise ValueError(f'{kind} checkpoint is not the required 64D phase-enabled schema')
    return ck
