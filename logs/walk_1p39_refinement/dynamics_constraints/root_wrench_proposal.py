"""Propose bounded Fourier pelvis corrections against frozen support points.

This is a centroidal optimization proposal. The support points remain fixed in
world space; the limbs must subsequently be projected to them by bounded IK.
It does not assert full-body dynamics or actual native-contact feasibility.
"""
from pathlib import Path
import argparse, json, sys, time
import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.optimize_walk_contacts import cyclic_velocities, mesh_points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--baseline', type=Path, default=ROOT/'robot/reference_data/walk1_lyt_contact_1p39.npz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--harmonics', type=int, default=3)
    p.add_argument('--iterations', type=int, default=6)
    p.add_argument('--support-source', type=Path)
    p.add_argument('--support-depth-mm', type=float, default=3.)
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(ROOT/'robot/MJCFmodelposition.xml'))
    model.opt.timestep = .005
    with np.load(args.seed, allow_pickle=True) as a:
        fields = {k: a[k] for k in a.files}
    baseline = np.load(args.baseline, allow_pickle=True)['qpos'].astype(float)
    seed = fields['qpos'].astype(float)
    support_poses = (np.load(args.support_source,allow_pickle=True)['qpos'].astype(float)
                     if args.support_source else baseline)
    n = len(seed)
    freq = float(fields['frequency'])
    shift = fields['cycle_translation'].astype(float)
    phase = 2*np.pi*np.arange(n)/n
    basis = np.column_stack([np.ones(n)] + [f(k*phase) for k in range(1,args.harmonics+1) for f in (np.sin,np.cos)])
    nb = basis.shape[1]
    scales = np.r_[np.full(3,.03),np.full(3,np.deg2rad(10))]
    # Six coordinate groups; coefficients are dimensionless.
    root_map = np.kron(np.eye(6), basis) * np.repeat(scales, n)[:,None]
    seed_euler = Rotation.from_quat(seed[:,[4,5,6,3]]).as_euler('xyz')
    base_euler = Rotation.from_quat(baseline[:,[4,5,6,3]]).as_euler('xyz')
    prior = np.column_stack([seed[:,:3]-baseline[:,:3],seed_euler-base_euler]).T.ravel()
    limits = np.repeat(scales,n)
    data = [mujoco.MjData(model) for _ in seed]
    floor = model.geom('floor').id
    ids = [model.geom(s).id for s in ('foot_r','toe_r','foot_l','toe_l')]
    origin = seed[:,:3].copy()
    cones = []
    meshes=[mesh_points(model,g) for g in ids]
    for i,d in enumerate(data):
        d.qpos[:] = support_poses[i]
        mujoco.mj_forward(model,d)
        cols=[]
        clouds=[]
        for geom,mesh in zip(ids,meshes):
            body=model.geom_bodyid[geom]
            rot=d.xmat[body].reshape(3,3)
            clouds.append(d.xpos[body]+mesh@rot.T)
        for side in range(2):
            if fields['foot_contact'][i,side]<.5: continue
            points=np.concatenate(clouds[2*side:2*side+2])
            points=points[points[:,2] <= points[:,2].min()+args.support_depth_mm/1000]
            # Optimistic frozen sole region; each point is vertically projected
            # onto the plane. The native contact margin is not asserted here.
            xy=points[:,:2]
            from scipy.spatial import ConvexHull
            if len(xy)>=3 and np.linalg.matrix_rank(xy-xy.mean(axis=0))==2:
                xy=xy[ConvexHull(xy).vertices]
            for point in xy:
                contact=np.r_[point,0.]
                mu=max(model.geom_friction[ids[2*side],0],model.geom_friction[floor,0])
                edge=np.array([[mu,-mu,0,0],[0,0,mu,-mu],[1,1,1,1]])
                cols.append(np.vstack([edge,np.cross(np.broadcast_to(contact-origin[i],edge.T.shape),edge.T).T]))
        cones.append(np.concatenate(cols,axis=1) if cols else np.zeros((6,0)))
    counts=[c.shape[1] for c in cones]
    force_count=sum(counts)
    rows,cols,vals=[],[],[]
    start=0
    # Normalize moments by a 0.7 m characteristic height and forces by mg.
    wrench_scale=np.r_[np.ones(3),np.full(3,1/.7)]/(model.body_mass.sum()*9.81)
    for i,c in enumerate(cones):
        rr,cc=np.nonzero(c)
        rows.extend(i*6+rr);cols.extend(start+cc);vals.extend((c*wrench_scale[:,None])[rr,cc]);start+=counts[i]
    ground=csr_matrix((vals,(rows,cols)),shape=(n*6,force_count))

    def positions(coeff):
        delta=(root_map@coeff).reshape(6,n).T
        q=seed.copy();q[:,:3]+=delta[:,:3]
        q[:,3:7]=Rotation.from_euler('xyz',seed_euler+delta[:,3:]).as_quat()[:,[3,0,1,2]]
        return q

    def wrench(coeff):
        q=positions(coeff)
        velocity=cyclic_velocities(model,q,freq,shift)
        acceleration=(np.roll(velocity,-1,axis=0)-np.roll(velocity,1,axis=0))*freq/2
        targets=[]
        for i,d in enumerate(data):
            d.qpos[:]=q[i];d.qvel[:]=velocity[i]
            mujoco.mj_forward(model,d)
            inertia=np.empty(model.nv);mujoco.mj_mulM(model,d,inertia,acceleration[i])
            required=inertia+d.qfrc_bias-d.qfrc_passive
            rot=d.xmat[model.body('pelvis').id].reshape(3,3)
            moment=rot@required[3:6]+np.cross(q[i,:3]-origin[i],required[:3])
            targets.append(np.r_[required[:3],moment]*wrench_scale)
        return np.array(targets).ravel()

    def closest(coeff):
        target=wrench(coeff)
        z=eye(n*6,format='csr');zero=csr_matrix((n*6,n*6))
        eq=hstack([ground,z,-z],format='csr')
        objective=np.r_[np.full(force_count,1e-8),np.ones(2*n*6)]
        lp=linprog(objective,A_eq=eq,b_eq=target,bounds=[(0,None)]*len(objective),method='highs')
        return float(lp.fun),target

    coeff=np.zeros(6*nb)
    started=time.monotonic()
    value,target=closest(coeff)
    history=[{'iteration':0,'normalized_L1_wrench_deficit':value}]
    for iteration in range(1,args.iterations+1):
        if time.monotonic()-started>150:break
        h=1e-4
        derivative=np.column_stack([(wrench(coeff+np.eye(len(coeff))[k]*h)-target)/h for k in range(len(coeff))])
        # Variables: coefficient increment, nonnegative contact forces, positive
        # and negative balance deficits, absolute coefficient increment.
        nc,nr=len(coeff),n*6
        z=eye(nr,format='csr')
        eq=hstack([csr_matrix(derivative),-ground,z,-z,csr_matrix((nr,nc))],format='csr')
        root_upper=limits-prior-root_map@coeff
        root_lower=limits+prior+root_map@coeff
        pad=csr_matrix((len(prior),force_count+2*nr+nc))
        # L1 Euler-angle changes <= 10 degrees imply a true rotation distance
        # <= 10 degrees by the triangle inequality for the three axis turns.
        signs=np.array([[a,b,c] for a in (-1.,1.) for b in (-1.,1.) for c in (-1.,1.)])
        angle_rows=np.concatenate([sum(signs[k,j]*root_map[(j+3)*n:(j+4)*n] for j in range(3)) for k in range(8)])
        angle_prior=np.concatenate([sum(signs[k,j]*prior[(j+3)*n:(j+4)*n] for j in range(3)) for k in range(8)])
        angle_upper=np.deg2rad(10)-angle_prior-angle_rows@coeff
        angle_pad=csr_matrix((len(angle_upper),force_count+2*nr+nc))
        ub=vstack([hstack([csr_matrix(root_map),pad]),hstack([-csr_matrix(root_map),pad]),
                   hstack([csr_matrix(angle_rows),angle_pad]),
                   hstack([eye(nc),csr_matrix((nc,force_count+2*nr)),-eye(nc)]),
                   hstack([-eye(nc),csr_matrix((nc,force_count+2*nr)),-eye(nc)])],format='csr')
        rhs=np.r_[root_upper,root_lower,angle_upper,np.zeros(2*nc)]
        objective=np.r_[np.zeros(nc),np.full(force_count,1e-8),np.ones(2*nr),np.full(nc,.02)]
        result=linprog(objective,A_ub=ub,b_ub=rhs,A_eq=eq,b_eq=-target,
                       bounds=[(-.2,.2)]*nc+[(0,None)]*(force_count+2*nr+nc),method='highs')
        if not result.success:
            history.append({'iteration':iteration,'failure':result.message});break
        step=result.x[:nc]
        accepted=False
        for alpha in (1.,.5,.25,.125):
            proposal=coeff+alpha*step
            proposal_value,proposal_target=closest(proposal)
            if proposal_value<value-1e-6:
                coeff,value,target=proposal,proposal_value,proposal_target;accepted=True;break
        history.append({'iteration':iteration,'normalized_L1_wrench_deficit':value,'accepted':accepted})
        print(json.dumps(history[-1]),flush=True)
        if not accepted:break
    qpos=positions(coeff)
    qvel=cyclic_velocities(model,qpos,freq,shift)
    actual=np.column_stack([qpos[:,:3]-baseline[:,:3],Rotation.from_quat(qpos[:,[4,5,6,3]]).as_euler('xyz')-base_euler])
    rotation_distance=np.linalg.norm((Rotation.from_quat(baseline[:,[4,5,6,3]]).inv()*Rotation.from_quat(qpos[:,[4,5,6,3]])).as_rotvec(),axis=1)
    if np.any(abs(actual)>scales+1e-7):
        raise RuntimeError('Proposal exceeds user pose bounds')
    if rotation_distance.max()>np.deg2rad(10)+1e-7:
        raise RuntimeError('Proposal exceeds user overall rotation bound')
    metadata=json.loads(str(fields['metadata']))
    for stale in ('kinematic_audit','loader_validation','smoothness_comparison'):
        metadata.pop(stale,None)
    metadata['centroidal_proposal']={'method':'Bounded pelvis Fourier corrections against frozen sole-polygon contact cones',
         'support_source':str(args.support_source or args.baseline),'full_body_dynamic_feasibility':'Not asserted; support-preserving bounded IK and native contact audit required.'}
    fields['metadata']=json.dumps(metadata)
    fields.update(qpos=qpos.astype(np.float32),qvel=qvel.astype(np.float32))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**fields)
    report={'purpose':'Frozen-support centroidal proposal; requires bounded contact IK and full dynamics audit.',
            'seed':str(args.seed),'support_contact_edge_counts':counts,'history':history,
            'max_root_xyz_change_mm':(np.max(abs(actual[:,:3]),axis=0)*1000).tolist(),
            'max_root_euler_change_deg':np.rad2deg(np.max(abs(actual[:,3:]),axis=0)).tolist(),
            'max_root_rotation_distance_deg':float(np.rad2deg(rotation_distance.max())),
            'elapsed_s':time.monotonic()-started,
            'coefficients':coeff.reshape(6,nb).tolist()}
    args.output.with_suffix('.proposal.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
