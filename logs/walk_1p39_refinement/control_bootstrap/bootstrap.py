"""Offline tendon force allocation followed by unassisted native MuJoCo simulation."""
from pathlib import Path
import sys, json, time, argparse
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src')]
import mujoco
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import lsq_linear
from lytrobot_amp.motion import MotionReference

p=argparse.ArgumentParser()
p.add_argument('--motion', default='logs/walk_1p39_refinement/fft3_contact_candidate.npz')
p.add_argument('--seconds',type=float,default=3.2)
p.add_argument('--joint-kp',type=float,default=150.)
p.add_argument('--root-weight',type=float,default=8.)
p.add_argument('--contact-weight',type=float,default=2.)
p.add_argument('--phase',type=int,default=0)
p.add_argument('--timestep',type=float,default=.005)
p.add_argument('--name',default='trial')
a=p.parse_args()
OUT=Path(__file__).resolve().parent/a.name; OUT.mkdir(exist_ok=True)
m=mujoco.MjModel.from_xml_path(str(ROOT/'robot/MJCFmodelposition.xml'))
m.opt.timestep=a.timestep; m.opt.iterations=10; m.opt.ls_iterations=20
d=mujoco.MjData(m)
r=MotionReference(ROOT/a.motion,m)
base=MotionReference(ROOT/'robot/reference_data/walk1_lyt_contact_1p39.npz',m)
dt=m.opt.timestep
period=r.frame_count/r.frequency
disp=np.array([1.112,0.,0.])
caps=-m.actuator_forcerange[:,0]
footids=[m.geom(n).id for n in ['foot_r','toe_r','foot_l','toe_l']]
floor=m.geom('floor').id
excluded=[int(m.joint(n).dofadr[0]) for n in ['patella_r_j1','patella_l_j1','mtp_angle_r','mtp_angle_l']]
M=np.empty((m.nv,m.nv))
def sample(ref,t):
    phase=t*ref.frequency+a.phase
    i=int(np.floor(phase)); f=phase-i
    j=i%ref.frame_count; k=(i+1)%ref.frame_count
    q=ref.reference_qpos[j].astype(float).copy()
    hi=ref.reference_qpos[k].astype(float).copy()
    if k==0: hi[:3]+=disp
    q=(1-f)*q+f*hi; q[:3]+=(i//ref.frame_count)*disp
    mujoco.mj_normalizeQuat(m,q)
    v=(1-f)*ref.reference_qvel[j]+f*ref.reference_qvel[k]
    acc=(ref.reference_qvel[k]-ref.reference_qvel[j])*ref.frequency
    return q,v,acc

d.qpos[:],d.qvel[:],_=sample(r,0.)
mujoco.mj_forward(m,d)
q_initial=d.qpos.copy()
rows={k:[] for k in ['time','qpos','qvel','ctrl','tension','planned_tension','contact','normal_force','qacc_prediction','qacc_actual','root_deviation','joint_deviation','orientation_deviation']}
statuses=[]
started=time.perf_counter(); fell=False
for step in range(int(a.seconds/dt)):
    t=step*dt
    mujoco.mj_forward(m,d)
    qt,vt,at=sample(r,t)
    dq=np.empty(m.nv); mujoco.mj_differentiatePos(m,dq,1.,d.qpos,qt)
    desired=at+a.joint_kp*dq+2*np.sqrt(a.joint_kp)*(vt-d.qvel)
    desired[0]=15*(1.39-d.qvel[0])
    desired[1]=150*dq[1]+25*(vt[1]-d.qvel[1])
    desired[2]=at[2]+400*dq[2]+40*(vt[2]-d.qvel[2])
    desired[3:6]=at[3:6]+250*dq[3:6]+30*(vt[3:6]-d.qvel[3:6])
    # Passive toe/patella accelerations are not target tasks or virtual actuators.
    desired[excluded]=0
    act=np.zeros((m.nv,m.nu))
    for u in range(m.nu):
        start,count=int(d.moment_rowadr[u]),int(d.moment_rownnz[u])
        act[d.moment_colind[start:start+count],u]=-d.actuator_moment[start:start+count]
    J=d.efc_J.reshape(d.nefc,m.nv)
    types=np.asarray(d.efc_type)
    used=np.flatnonzero((types==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))|(types==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))) .tolist()
    ground=[]
    for c in d.contact:
        if (c.geom1==floor and c.geom2 in footids) or (c.geom2==floor and c.geom1 in footids):
            if c.efc_address>=0:
                n=1 if c.dim==1 else 2*(c.dim-1)
                ground.extend(range(c.efc_address,c.efc_address+n))
    used=used+ground
    Ju=J[used]
    B=np.column_stack((act,Ju.T))
    mujoco.mj_fullM(m,d,M)
    factor=cho_factor(M)
    H=cho_solve(factor,B)
    biasacc=cho_solve(factor,d.qfrc_passive-d.qfrc_bias)
    w=np.ones(m.nv); w[:3]=a.root_weight; w[3:6]=a.root_weight*.3; w[excluded]=0
    C=[w[:,None]*H]; target=[w*(desired-biasacc)]
    if used:
        # Match native constraint reference acceleration and impedance. Only
        # the subsequent mj_step determines actual contact reaction forces.
        soft=Ju@H
        soft[:,m.nu:]+=np.diag(d.efc_R[used])
        cw=np.full(len(used),a.contact_weight)
        cw[:len(used)-len(ground)]=10
        C.append(cw[:,None]*soft)
        target.append(cw*(d.efc_aref[used]-Ju@biasacc))
    reg=np.zeros((m.nu,B.shape[1])); reg[:,:m.nu]=np.eye(m.nu)/caps[:,None]*.03
    C.append(reg);target.append(np.zeros(m.nu))
    gain=m.actuator_gainprm[:,0]
    bias=(m.actuator_biasprm[:,0]+m.actuator_biasprm[:,1]*d.actuator_length+m.actuator_biasprm[:,2]*d.actuator_velocity)
    lo=np.clip(-(gain*m.actuator_ctrlrange[:,1]+bias),0,caps)
    hi=np.clip(-(gain*m.actuator_ctrlrange[:,0]+bias),0,caps)
    fixed=hi-lo<1e-8
    # lsq_linear needs strict bounds; fixed variables are eliminated exactly.
    lower=np.concatenate((lo,np.where(types[used]==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY),-np.inf,0.)))
    upper=np.concatenate((hi,np.full(len(used),np.inf)))
    mat=np.concatenate(C); tar=np.concatenate(target)
    free=np.ones(B.shape[1],bool);free[:m.nu]=~fixed
    x=np.zeros(B.shape[1]);x[:m.nu][fixed]=lo[fixed]
    ans=lsq_linear(mat[:,free],tar-mat[:,~free]@x[~free],bounds=(lower[free],upper[free]),method='bvls',tol=1e-5,max_iter=80)
    x[free]=ans.x
    predicted=H@x+biasacc
    control=(-x[:m.nu]-bias)/gain
    d.ctrl[:]=np.clip(control,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    # There is no qfrc_applied/xfrc_applied assignment and no root actuator.
    mujoco.mj_forward(m,d)
    contact=np.zeros(2);nf=np.zeros(2)
    force=np.zeros(6)
    for ci,c in enumerate(d.contact):
        if (c.geom1==floor and c.geom2 in footids) or (c.geom2==floor and c.geom1 in footids):
            gi=int(c.geom2 if c.geom1==floor else c.geom1)
            side=footids.index(gi)//2
            mujoco.mj_contactForce(m,d,ci,force)
            if force[0]>.1: contact[side]=1
            nf[side]+=force[0]
    qb,_,_=sample(base,t)
    bd=np.empty(m.nv);mujoco.mj_differentiatePos(m,bd,1.,qb,d.qpos)
    values=dict(time=t,qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),tension=-d.actuator_force.copy(),planned_tension=x[:m.nu],contact=contact,normal_force=nf,qacc_prediction=predicted,qacc_actual=d.qacc.copy(),root_deviation=bd[:3],joint_deviation=bd[6:],orientation_deviation=bd[3:6])
    for k,v in values.items(): rows[k].append(v)
    statuses.append(int(ans.status))
    mujoco.mj_step(m,d)
    if d.qpos[2]<.55 or np.linalg.norm(d.qvel)>200 or not np.isfinite(d.qpos).all():
        fell=True; break
    if step%100==0: print(json.dumps(dict(step=step,time_s=round(t,3),vx=round(float(d.qvel[0]),3),z=round(float(d.qpos[2]),3),contacts=contact.tolist())),flush=True)
arr={k:np.asarray(v) for k,v in rows.items()}
np.savez_compressed(OUT/'trajectory.npz',**arr)
summary=dict(motion=a.motion,parameters=vars(a),duration_s=float(d.time),fell=fell,mean_vx_mps=float(arr['qvel'][:,0].mean()),distance_speed_mps=float((d.qpos[0]-q_initial[0])/d.time),final_vx_mps=float(d.qvel[0]),minimum_root_height_m=float(arr['qpos'][:,2].min()),maximum_root_axis_deviation_mm=(1000*np.abs(arr['root_deviation']).max(axis=0)).tolist(),maximum_joint_deviation_deg=float(np.rad2deg(np.abs(arr['joint_deviation']).max())),maximum_orientation_deviation_deg=float(np.rad2deg(np.linalg.norm(arr['orientation_deviation'],axis=1).max())),peak_tension_N=arr['tension'].max(axis=0).tolist(),stance_fractions=arr['contact'].mean(axis=0).tolist(),solver_unsuccessful_steps=sum(s<0 for s in statuses),elapsed_s=time.perf_counter()-started,method='Bounded nonnegative tendon tension/native EFC allocation; actual MuJoCo forward simulation with no external forces and passive MTP.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
