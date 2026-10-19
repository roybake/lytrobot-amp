"""Bounded native MuJoCo multiple-shooting reference refinement (offline)."""
from pathlib import Path
import sys, argparse, json, time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from lytrobot_amp.motion import MotionReference

p=argparse.ArgumentParser()
p.add_argument('--nodes',type=int,default=40)
p.add_argument('--max-nfev',type=int,default=150)
p.add_argument('--name',default='nodes40')
p.add_argument('--seed',default='logs/walk_1p39_refinement/bounded_reachable_reference.npz')
p.add_argument('--initial-solution')
p.add_argument('--control-parameter',choices=('position','force'),default='force')
args=p.parse_args()
OUT=Path(__file__).resolve().parent/args.name;OUT.mkdir(exist_ok=True)
m=mujoco.MjModel.from_xml_path(str(ROOT/'robot/MJCFmodelposition.xml'))
m.opt.timestep=.005;m.opt.iterations=80;m.opt.ls_iterations=50
m.opt.tolerance=1e-10
m.opt.ccd_tolerance=1e-10
m.opt.ccd_iterations=100
d=mujoco.MjData(m)
nv,nu=m.nv,m.nu;nx=2*nv;nvar=nx+nu;N=args.nodes
h=.8/N; substeps=round(h/.005)
if abs(h-substeps*.005)>1e-10: raise ValueError('Nodes must divide 160 physics steps.')
disp=np.array([1.112,0.,0.])
base=MotionReference(ROOT/'robot/reference_data/walk1_lyt_contact_1p39.npz',m)
seed=MotionReference(ROOT/args.seed,m)
def sample(ref,t):
    ph=t*ref.frequency;i=int(np.floor(ph));f=ph-i
    j=i%ref.frame_count;k=(i+1)%ref.frame_count
    lo=ref.reference_qpos[j].astype(float).copy();hi=ref.reference_qpos[k].astype(float).copy()
    if k==0:hi[:3]+=disp
    q=lo*(1-f)+hi*f;q[:3]+=(i//ref.frame_count)*disp
    mujoco.mj_normalizeQuat(m,q)
    v=ref.reference_qvel[j]*(1-f)+ref.reference_qvel[k]*f
    return q,v
qbase=np.array([sample(base,i*h)[0] for i in range(N)])
vbase=np.array([sample(base,i*h)[1] for i in range(N)])
qseed=np.array([sample(seed,i*h)[0] for i in range(N)])
vseed=np.array([sample(seed,i*h)[1] for i in range(N)])
# Pose offsets parameterize each node relative to the same phase in the original.
pscale=np.full(nv,np.deg2rad(10.));pscale[:3]=.03
# A conservative box on the root rotation vector guarantees geodesic <= 10deg.
pscale[3:6]=np.deg2rad(10.)
for name in ['patella_r_j1','patella_l_j1']:
    pscale[int(m.joint(name).dofadr[0])]=.01
vscale=np.ones(nv);vscale[3:]=5.
for name in ['patella_r_j1','patella_l_j1']:
    vscale[int(m.joint(name).dofadr[0])]=.1
mid=m.actuator_ctrlrange.mean(axis=1);half=np.diff(m.actuator_ctrlrange,axis=1)[:,0]/2
lower=np.full((N,nvar),-1.);upper=-lower
lower[:,nv:2*nv]=-10/vscale;upper[:,nv:2*nv]=10/vscale
for joint in range(1,m.njnt):
    qa=int(m.jnt_qposadr[joint]);va=int(m.jnt_dofadr[joint])
    if m.jnt_limited[joint]:
        lower[:,va]=np.maximum(-1,(m.jnt_range[joint,0]-qbase[:,qa])/pscale[va])
        upper[:,va]=np.minimum(1,(m.jnt_range[joint,1]-qbase[:,qa])/pscale[va])
x0=np.zeros((N,nvar))
for i in range(N):
    dv=np.empty(nv);mujoco.mj_differentiatePos(m,dv,1.,qbase[i],qseed[i])
    x0[i,:nv]=dv/pscale;x0[i,nv:2*nv]=vseed[i]/vscale
    d.qpos[:]=qseed[i];d.qvel[:]=vseed[i];mujoco.mj_forward(m,d)
    x0[i,2*nv:]=-0.6
if args.initial_solution:
    initial=np.load(args.initial_solution)
    old=initial['variables'];oldN=len(old)
    old=old.copy()
    for c in range(nvar):
        x0[:,c]=np.interp(np.arange(N)/N,np.arange(oldN+1)/oldN,np.append(old[:,c],old[0,c]))
x0=np.clip(x0,lower+1e-8,upper-1e-8)
stateweights=np.ones(nx)
stateweights[:nv]=1/pscale
stateweights[nv:]=1/vscale
stateweights[nv:nv+3]*=5
stateweights[nv+3:]*=2
regular=.00001
cache={};calls=0;started=time.perf_counter()

def qdiff(q1,q2):
    out=np.empty(nv);mujoco.mj_differentiatePos(m,out,1.,q1,q2);return out
capacity=-m.actuator_forcerange[:,0]
gain=m.actuator_gainprm[:,0]
position_data=mujoco.MjData(m)
def force_control(pose,parameters):
    position_data.qpos[:]=pose
    mujoco.mj_fwdPosition(m,position_data)
    length=position_data.actuator_length.copy()
    length_jac=np.zeros((nu,nv))
    for i in range(nu):
        adr=int(position_data.moment_rowadr[i]);count=int(position_data.moment_rownnz[i])
        length_jac[i,position_data.moment_colind[adr:adr+count]]=position_data.actuator_moment[adr:adr+count]
    lo=np.clip(gain*(length-m.actuator_ctrlrange[:,1]),0,capacity)
    hi=np.clip(gain*(length-m.actuator_ctrlrange[:,0]),0,capacity)
    activation=(parameters+1.)/2
    tension=lo+(hi-lo)*activation
    raw=length-tension/gain
    ctrl=np.clip(raw,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    dlo=np.where((gain*(length-m.actuator_ctrlrange[:,1])>0)&(gain*(length-m.actuator_ctrlrange[:,1])<capacity),gain,0.)
    dhi=np.where((gain*(length-m.actuator_ctrlrange[:,0])>0)&(gain*(length-m.actuator_ctrlrange[:,0])<capacity),gain,0.)
    ctrl_dl=1.-((1.-activation)*dlo+activation*dhi)/gain
    ctrl_dl[(raw<=m.actuator_ctrlrange[:,0])|(raw>=m.actuator_ctrlrange[:,1])]=0.
    du_dparam=-(hi-lo)/(2*gain)
    du_dparam[(raw<=m.actuator_ctrlrange[:,0])|(raw>=m.actuator_ctrlrange[:,1])]=0.
    return ctrl,ctrl_dl[:,None]*length_jac,du_dparam

def pose_offset(row):
    offset=row[:nv]*pscale
    angular=offset[3:6]
    length=np.linalg.norm(angular)
    if length>np.deg2rad(10.):offset[3:6]=angular*np.deg2rad(10.)/length
    return offset

def decode(x):
    xr=x.reshape(N,nvar)
    qs=qbase.copy()
    for i in range(N):mujoco.mj_integratePos(m,qs[i],pose_offset(xr[i]),1.)
    vs=xr[:,nv:2*nv]*vscale
    cs=np.empty((N,nu))
    for i in range(N):cs[i]=force_control(qs[i],xr[i,2*nv:])[0]
    return qs,vs,cs
def integration_matrix(qref,row,qcurrent):
    S=np.diag(pscale)
    eps=1e-7
    for j in range(3,6):
        perturbed=row.copy();perturbed[j]+=eps
        qp=qref.copy();mujoco.mj_integratePos(m,qp,pose_offset(perturbed),1.)
        S[:,j]=qdiff(qcurrent,qp)/eps
    return S

def difference_matrices(qnext,qpred):
    ep=qdiff(qnext,qpred)
    JP=np.eye(nv);JN=-np.eye(nv);eps=1e-7
    for j in range(3,6):
        inc=np.zeros(nv);inc[j]=eps
        qp=qpred.copy();mujoco.mj_integratePos(m,qp,inc,1.)
        qn=qnext.copy();mujoco.mj_integratePos(m,qn,inc,1.)
        JP[:,j]=(qdiff(qnext,qp)-ep)/eps
        JN[:,j]=(qdiff(qn,qpred)-ep)/eps
    return ep,JP,JN

def evaluate(x,need_jac):
    global calls
    # scipy calls fun then jac at exactly the same x; cache the computed pair.
    if cache.get('x') is not None and np.array_equal(x,cache['x']):
        if not need_jac or cache.get('jac') is not None:return cache
    xr=x.reshape(N,nvar);qs,vs,cs=decode(x)
    residual=np.empty((N,nx));predqs=np.empty_like(qs);predvs=np.empty_like(vs)
    rr=[];cc=[];vv=[]
    for i in range(N):
        mujoco.mj_resetData(m,d);d.qpos[:]=qs[i];d.qvel[:]=vs[i];d.ctrl[:]=cs[i]
        d.qacc_warmstart[:]=0
        Aall=np.eye(nx);Ball=np.zeros((nx,nu))
        for _ in range(substeps):
            if need_jac:
                A=np.empty((nx,nx));B=np.empty((nx,nu))
                mujoco.mjd_transitionFD(m,d,1e-6,True,A,B,None,None)
                Aall=A@Aall;Ball=A@Ball+B
            mujoco.mj_step(m,d)
        qp=d.qpos.copy();vp=d.qvel.copy();predqs[i]=qp;predvs[i]=vp
        j=(i+1)%N;qn=qs[j].copy()
        if j==0:qn[:3]+=disp
        if need_jac:ep,JP,JN=difference_matrices(qn,qp)
        else:ep=qdiff(qn,qp)
        residual[i]=np.concatenate((ep,vp-vs[j]))*stateweights
        if need_jac:
            Si=np.zeros((nx,nx));Si[:nv,:nv]=integration_matrix(qbase[i],xr[i],qs[i]);Si[nv:,nv:]=np.diag(vscale)
            Sj=integration_matrix(qbase[j],xr[j],qs[j])
            output=np.eye(nx);output[:nv,:nv]=JP
            _,control_pose_jac,control_parameter_jac=force_control(qs[i],xr[i,2*nv:])
            control_state_jac=np.zeros((nu,nx));control_state_jac[:,:nv]=control_pose_jac
            own=np.column_stack((output@(Aall+Ball@control_state_jac)@Si,
                                 output@Ball*control_parameter_jac[None,:]))*stateweights[:,None]
            following=np.zeros((nx,nvar));following[:nv,:nv]=JN@Sj;following[nv:,nv:2*nv]=-np.diag(vscale)
            following*=stateweights[:,None]
            for col,node in [(own,i),(following,j)]:
                rloc,cloc=np.nonzero(col)
                rr.extend((i*nx+rloc).tolist());cc.extend((node*nvar+cloc).tolist());vv.extend(col[rloc,cloc].tolist())
    fun=np.concatenate((residual.ravel(),regular*x))
    jac=None
    if need_jac:
        idx=np.arange(N*nvar);rr.extend((N*nx+idx).tolist());cc.extend(idx.tolist());vv.extend(np.full(N*nvar,regular).tolist())
        jac=coo_matrix((vv,(rr,cc)),shape=(len(fun),N*nvar)).tocsr()
    cache.clear();cache.update(x=x.copy(),fun=fun,jac=jac,residual=residual,qs=qs,vs=vs,cs=cs,predqs=predqs,predvs=predvs)
    if not need_jac:
        calls+=1
        if calls%5==1:
            print(json.dumps(dict(evaluation=calls,elapsed_s=round(time.perf_counter()-started,2),defect_rms=float(np.sqrt(np.mean(residual**2))),max_pose_scaled=float(np.abs(residual[:,:nv]).max()),max_velocity_scaled=float(np.abs(residual[:,nv:]).max()))),flush=True)
    return cache

initial=evaluate(x0.ravel(),False)
initial_rms=float(np.sqrt(np.mean(initial['residual']**2)))
# Numeric analysis of a directional derivative at the initial trajectory.
x=x0.ravel();direction=np.random.default_rng(42).normal(size=x.size);direction/=np.linalg.norm(direction)
jac=evaluate(x,True)['jac'].copy();eps=1e-7
plus=evaluate(x+eps*direction,False)['fun'].copy();minus=evaluate(x-eps*direction,False)['fun'].copy()
fd=(plus-minus)/(2*eps);analytic=jac@direction
print(json.dumps({'directional_derivative_relative_error':float(np.linalg.norm(fd-analytic)/np.linalg.norm(fd)),'fd_norm':float(np.linalg.norm(fd)),'predicted_norm':float(np.linalg.norm(analytic))}),flush=True)
solution=least_squares(lambda x:evaluate(x,False)['fun'],x0.ravel(),jac=lambda x:evaluate(x,True)['jac'],bounds=(lower.ravel(),upper.ravel()),max_nfev=args.max_nfev,method='trf',tr_solver='lsmr',x_scale='jac',ftol=1e-12,gtol=1e-10,xtol=1e-12,tr_options=dict(atol=1e-8,btol=1e-8,maxiter=3000),verbose=0)
final=evaluate(solution.x,False)
qs,vs,cs=final['qs'],final['vs'],final['cs']
np.savez_compressed(OUT/'solution.npz',variables=solution.x.reshape(N,nvar),qpos=qs,qvel=vs,ctrl=cs,defects=final['residual']/stateweights,cycle_translation=disp,frequency=N/.8)

# Continuous native replay at 100 Hz commands and 200 Hz physics is separate
# from the independent shooting nodes. Commands are held in each coarse block.
m.opt.iterations=10;m.opt.ls_iterations=20;m.opt.tolerance=1e-8;m.opt.ccd_tolerance=1e-6
mujoco.mj_resetData(m,d);d.qpos[:]=qs[0];d.qvel[:]=vs[0]
replay={k:[] for k in ['time','qpos','qvel','ctrl','tension','root_deviation','joint_deviation','orientation_deviation']}
fell=False
for tick in range(3*80):
    t=tick*.01;node=int((t%.8)/h+1e-9)%N
    d.ctrl[:]=cs[node]
    qb,_=sample(base,t)
    dd=qdiff(qb,d.qpos)
    vals=dict(time=t,qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),tension=-d.actuator_force.copy(),root_deviation=dd[:3],joint_deviation=dd[6:],orientation_deviation=dd[3:6])
    for key,value in vals.items():replay[key].append(value)
    for _ in range(2):mujoco.mj_step(m,d)
    if d.qpos[2]<.55 or not np.isfinite(d.qpos).all() or np.linalg.norm(d.qvel)>200:fell=True;break
ra={k:np.asarray(v) for k,v in replay.items()}
np.savez_compressed(OUT/'continuous_replay.npz',**ra)
defects=final['residual']/stateweights
summary=dict(nodes=N,substeps=substeps,physics_timestep_s=.005,control_period_s=h,original_reference=str(ROOT/'robot/reference_data/walk1_lyt_contact_1p39.npz'),seed=args.seed,initial_scaled_defect_rms=initial_rms,final_scaled_defect_rms=float(np.sqrt(np.mean(final['residual']**2))),max_position_defect_root_mm=(1000*abs(defects[:,:3]).max(axis=0)).tolist(),max_position_defect_orientation_deg=float(np.rad2deg(np.linalg.norm(defects[:,3:6],axis=1).max())),max_position_defect_joint_deg=float(np.rad2deg(abs(defects[:,6:nv]).max())),max_velocity_defect=float(abs(defects[:,nv:]).max()),nfev=solution.nfev,status=int(solution.status),message=solution.message,cost=float(solution.cost),optimality=float(solution.optimality),elapsed_s=time.perf_counter()-started,replay_duration_s=float(d.time),replay_fell=fell,replay_distance_speed_mps=float((d.qpos[0]-qs[0,0])/d.time),replay_mean_vx_mps=float(ra['qvel'][:,0].mean()),replay_root_max_deviation_mm=(1000*abs(ra['root_deviation']).max(axis=0)).tolist(),replay_joint_max_deviation_deg=float(np.rad2deg(abs(ra['joint_deviation']).max())),replay_orientation_max_deviation_deg=float(np.rad2deg(np.linalg.norm(ra['orientation_deviation'],axis=1).max())),pose_nodes_within_hard_bounds=bool(np.all(solution.x>=lower.ravel()-1e-9) and np.all(solution.x<=upper.ravel()+1e-9)),actual_walking_success=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
