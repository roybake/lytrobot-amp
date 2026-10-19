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
args=p.parse_args()
OUT=Path(__file__).resolve().parent/args.name;OUT.mkdir(exist_ok=True)
m=mujoco.MjModel.from_xml_path(str(ROOT/'robot/MJCFmodelposition.xml'))
m.opt.timestep=.005;m.opt.iterations=10;m.opt.ls_iterations=20
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
pscale[3:6]=np.deg2rad(10.)/np.sqrt(3)
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
    x0[i,2*nv:]=(np.clip(d.actuator_length-.0015,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])-mid)/half
if args.initial_solution:
    initial=np.load(args.initial_solution)
    old=initial['variables'];oldN=len(old)
    for c in range(nvar):
        x0[:,c]=np.interp(np.arange(N)/N,np.arange(oldN+1)/oldN,np.append(old[:,c],old[0,c]))
x0=np.clip(x0,lower+1e-8,upper-1e-8)
stateweights=np.ones(nx)
stateweights[:nv]=1/pscale
stateweights[nv:]=1/vscale
stateweights[nv:nv+3]*=5
stateweights[nv+3:]*=2
regular=.001
cache={};calls=0;started=time.perf_counter()

def qdiff(q1,q2):
    out=np.empty(nv);mujoco.mj_differentiatePos(m,out,1.,q1,q2);return out
def decode(x):
    xr=x.reshape(N,nvar)
    qs=qbase.copy()
    for i in range(N):mujoco.mj_integratePos(m,qs[i],xr[i,:nv]*pscale,1.)
    return qs,xr[:,nv:2*nv]*vscale,mid+xr[:,2*nv:]*half
def integration_matrix(qref,offset,qcurrent):
    S=np.diag(pscale)
    eps=1e-6
    for j in range(3,6):
        oo=offset.copy();oo[j]+=eps
        qp=qref.copy();mujoco.mj_integratePos(m,qp,oo,1.)
        S[:,j]=qdiff(qcurrent,qp)/eps*pscale[j]
    return S
def difference_matrices(qnext,qpred):
    ep=qdiff(qnext,qpred)
    JP=np.eye(nv);JN=-np.eye(nv);eps=1e-6
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
                mujoco.mjd_transitionFD(m,d,1e-5,False,A,B,None,None)
                Aall=A@Aall;Ball=A@Ball+B
            mujoco.mj_step(m,d)
        qp=d.qpos.copy();vp=d.qvel.copy();predqs[i]=qp;predvs[i]=vp
        j=(i+1)%N;qn=qs[j].copy()
        if j==0:qn[:3]+=disp
        if need_jac:ep,JP,JN=difference_matrices(qn,qp)
        else:ep=qdiff(qn,qp)
        residual[i]=np.concatenate((ep,vp-vs[j]))*stateweights
        if need_jac:
            Si=np.zeros((nx,nx));Si[:nv,:nv]=integration_matrix(qbase[i],xr[i,:nv]*pscale,qs[i]);Si[nv:,nv:]=np.diag(vscale)
            Sj=integration_matrix(qbase[j],xr[j,:nv]*pscale,qs[j])
            output=np.eye(nx);output[:nv,:nv]=JP
            own=np.column_stack((output@Aall@Si,output@Ball*np.asarray(half)[None,:]))*stateweights[:,None]
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
solution=least_squares(lambda x:evaluate(x,False)['fun'],x0.ravel(),jac=lambda x:evaluate(x,True)['jac'],bounds=(lower.ravel(),upper.ravel()),max_nfev=args.max_nfev,method='trf',tr_solver='lsmr',x_scale='jac',ftol=1e-8,gtol=1e-6,xtol=1e-8,tr_options=dict(atol=1e-5,btol=1e-5,maxiter=500),verbose=0)
final=evaluate(solution.x,False)
qs,vs,cs=final['qs'],final['vs'],final['cs']
np.savez_compressed(OUT/'solution.npz',variables=solution.x.reshape(N,nvar),qpos=qs,qvel=vs,ctrl=cs,defects=final['residual']/stateweights,cycle_translation=disp,frequency=N/.8)

# Continuous native replay at 100 Hz commands and 200 Hz physics is separate
# from the independent shooting nodes. Commands are held in each coarse block.
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
