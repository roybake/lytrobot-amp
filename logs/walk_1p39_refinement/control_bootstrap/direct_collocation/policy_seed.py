"""Use measured settled walking only as a numerical trajectory optimization seed."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import mujoco,numpy as np
from scripts.play import OnnxPolicy,load_onnx_metadata,LytRobotEnvCfg,LytRobotAmpEnv
from lytrobot_amp.motion import MotionReference
source=ROOT/'logs/left_leg_diagnosis/teacher_best.onnx'
metadata=load_onnx_metadata(source);cfg=LytRobotEnvCfg(**metadata['env_config']);cfg.num_envs=1;cfg.randomize_episode_seed=False;cfg.reset_noise_scale=0.
env=LytRobotAmpEnv(cfg);policy=OnnxPolicy(source);obs=env.reset();rows=[]
for step in range(2000):
 obs,_,done,_=env.step(policy.act_inference(obs))
 if done[0]:raise RuntimeError('Seed trajectory reset')
 if step>=500:
  d=env.datas[0];rows.append((float(env.motion_phase[0]),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),-d.actuator_force.copy()))
m=env.model;ref=env.motion;N=80;nv=m.nv;nu=m.nu;freq=100.;variables=np.zeros((N,2*nv+nu));qs=[];vs=[];cs=[];gap=[]
for i in range(N):
 phase=i/N*ref.frame_count;j=int(np.floor(phase));k=(j+1)%ref.frame_count;alpha=phase-j
 base=((1-alpha)*ref.reference_qpos[j]+alpha*ref.reference_qpos[k]).astype(float)
 mujoco.mj_normalizeQuat(m,base)
 row=min(rows,key=lambda r:abs((r[0]-phase+45)%90-45));q=row[1].copy();q[0]=base[0]
 offset=np.empty(nv);mujoco.mj_differentiatePos(m,offset,1.,base,q)
 scale=np.full(nv,np.deg2rad(10));scale[:3]=.03
 for name in ('patella_r_j1','patella_l_j1'):scale[int(m.joint(name).dofadr[0])]=.01
 offset=np.clip(offset,-scale*.995,scale*.995)
 length=np.linalg.norm(offset[3:6]);limit=np.deg2rad(9.9)
 if length>limit:offset[3:6]*=limit/length
 q=base.copy();mujoco.mj_integratePos(m,q,offset,1.)
 v=row[2].copy();v[0]+=1.39-row[2][0]
 velocityscale=np.ones(nv);velocityscale[3:]=5.
 for name in ('patella_r_j1','patella_l_j1'):velocityscale[int(m.joint(name).dofadr[0])]=.1
 variables[i,:nv]=offset/scale;variables[i,nv:2*nv]=v/velocityscale
 d=mujoco.MjData(m);d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
 caps=-m.actuator_forcerange[:,0];gain=m.actuator_gainprm[:,0];lo=np.clip(gain*(d.actuator_length-m.actuator_ctrlrange[:,1]),0,caps);hi=np.clip(gain*(d.actuator_length-m.actuator_ctrlrange[:,0]),0,caps)
 forces=np.minimum(np.maximum(row[4],lo),hi)
 fraction=np.divide(forces-lo,hi-lo,out=np.full(nu,.2),where=hi-lo>1e-8)
 variables[i,2*nv:]=np.clip(fraction*2-1,-.99,.99)
 qs.append(q);vs.append(v);cs.append(row[3])
out=Path(__file__).resolve().parent/'policy_seed.npz';np.savez_compressed(out,variables=variables,qpos=np.array(qs),qvel=np.array(vs),ctrl=np.array(cs),frequency=100.)
print(out);env.close()
