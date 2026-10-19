"""Low-frequency full-pose wrench refinement with support reachability.

All original reference/model files remain untouched. Frozen sole cones are an
optimization approximation; the saved pose requires a native contact audit.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np,mujoco
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,eye,hstack,vstack,block_diag
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from logs.walk_1p39_refinement.project_fft3 import Projection,encode,artifact_metadata,summarize
from scripts.optimize_walk_contacts import cyclic_velocities,mesh_points


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--harmonics',type=int,default=3)
    p.add_argument('--support-source',type=Path)
    p.add_argument('--support-point-tolerance-mm',type=float,default=.3)
    args=p.parse_args();started=time.monotonic()
    with np.load(ROOT/'robot/reference_data/walk1_lyt_contact_1p39.npz',allow_pickle=True) as a:
        fields={k:a[k] for k in a.files}
    seed=np.load(args.seed,allow_pickle=True)['qpos'].astype(float)
    support_seed=(np.load(args.support_source,allow_pickle=True)['qpos'].astype(float)
                  if args.support_source else seed)
    m=mujoco.MjModel.from_xml_path(str(ROOT/'robot/MJCFmodelposition.xml'));m.opt.timestep=.005
    fit=Projection(m,fields,seed)
    n,c=fit.count,fit.columns
    phase=2*np.pi*np.arange(n)/n
    basis=np.column_stack([np.ones(n)]+[f(k*phase) for k in range(1,args.harmonics+1) for f in (np.sin,np.cos)])
    nb=basis.shape[1];nc=c*nb
    scales=np.r_[np.full(3,.03),np.full(c-3,np.deg2rad(10))]
    # Frame-major coordinate map makes geometry constraints direct products.
    mapping=np.zeros((n,c,nc))
    for j in range(c):mapping[:,j,j*nb:(j+1)*nb]=basis*scales[j]
    seed_values=encode(fit,seed)
    lower,upper=fit.bounds();lower=lower.reshape(n,c);upper=upper.reshape(n,c)
    pose_map=mapping.reshape(n*c,nc)
    geometry_points=np.empty((n,2,3));minimum_gaps=[]
    fit.state(encode(fit,support_seed).ravel())
    for i,d in enumerate(fit.data):
        minimum_gaps.append(fit.gaps[i].copy())
        for side in range(2):
            body=fit.constraint_bodies[i,side]
            geometry_points[i,side]=d.xpos[body]+d.xmat[body].reshape(3,3)@fit.material[i,side]
    # Allow only tiny support motion from the already contact-projected seed.
    contact_tolerance=args.support_point_tolerance_mm/1000
    floor=m.geom('floor').id
    geoms=[m.geom(name).id for name in ('foot_r','toe_r','foot_l','toe_l')]
    meshes=[mesh_points(m,g) for g in geoms]
    cones=[];origin=support_seed[:,:3].copy()
    for i,d in enumerate(fit.data):
        clouds=[d.xpos[b]+mesh@d.xmat[b].reshape(3,3).T for b,mesh in zip(fit.bodies,meshes)]
        cols=[]
        for side in range(2):
            if fit.contact[i,side]<.5:continue
            points=np.concatenate(clouds[2*side:2*side+2]);points=points[points[:,2]<=points[:,2].min()+.003]
            xy=points[:,:2]
            if len(xy)>=3 and np.linalg.matrix_rank(xy-xy.mean(axis=0))==2:xy=xy[ConvexHull(xy).vertices]
            mu=max(m.geom_friction[geoms[2*side],0],m.geom_friction[floor,0])
            edge=np.array([[mu,-mu,0,0],[0,0,mu,-mu],[1,1,1,1]])
            for xy_point in xy:
                point=np.r_[xy_point,0.]
                cols.append(np.vstack([edge,np.cross(np.broadcast_to(point-origin[i],edge.T.shape),edge.T).T]))
        cones.append(np.concatenate(cols,axis=1))
    mg=m.body_mass.sum()*9.81;wscale=np.r_[np.ones(3),np.full(3,1/.7)]/mg
    ground=block_diag([csr_matrix(x*wscale[:,None]) for x in cones],format='csr')
    nf=ground.shape[1];nr=6*n

    def state(coef):
        values=seed_values+(pose_map@coef).reshape(n,c)
        fit.state(values.ravel())
        return values,fit.poses

    def wrench(coef):
        values,q=state(coef)
        v=cyclic_velocities(m,q,fit.frequency,fit.translation)
        a=(np.roll(v,-1,axis=0)-np.roll(v,1,axis=0))*fit.frequency/2
        target=[]
        for i,d in enumerate(fit.data):
            d.qpos[:]=q[i];d.qvel[:]=v[i];mujoco.mj_forward(m,d)
            inertia=np.empty(m.nv);mujoco.mj_mulM(m,d,inertia,a[i])
            required=inertia+d.qfrc_bias-d.qfrc_passive
            rot=d.xmat[m.body('pelvis').id].reshape(3,3)
            world=np.r_[required[:3],rot@required[3:6]+np.cross(q[i,:3]-origin[i],required[:3])]
            target.extend(world*wscale)
        return np.array(target)

    def geometry(coef,with_jac=False):
        values,q=state(coef);errors=[];jg=[];gaps=[];jgap=[]
        for i,d in enumerate(fit.data):
            local_map=fit.maps[i]
            for side in range(2):
                if fit.contact[i,side]<.5:continue
                body=fit.constraint_bodies[i,side]
                point=d.xpos[body]+d.xmat[body].reshape(3,3)@fit.material[i,side]
                errors.extend(point-geometry_points[i,side])
                if with_jac:
                    mujoco.mj_jac(m,d,fit.jacp,None,point,body)
                    jg.extend(fit.jacp@local_map@mapping[i])
            for k,(body,point,gap) in enumerate(zip(fit.bodies,fit.bottoms[i],fit.gaps[i])):
                gaps.append(gap)
                if with_jac:
                    mujoco.mj_jac(m,d,fit.jacp,None,point,body)
                    jgap.append(fit.jacp[2]@local_map@mapping[i])
        return values,np.array(errors),np.array(gaps),np.array(jg),np.array(jgap)

    def geometry_violation(errors,gaps):
        support=np.array([min(gaps[4*i+2*s:4*i+2*s+2]) for i in range(n) for s in range(2) if fit.contact[i,s]>.5])
        return float(np.maximum(abs(errors)-contact_tolerance,0).sum()+np.maximum(-gaps-.0001,0).sum()+np.maximum(support-.00095,0).sum())

    def merit(coef,wrench_value,target):
        speed_loss=.3*np.sum(abs(seed_velocity+velocity_map@coef))
        friction_loss=np.maximum(abs(target[0::6])+abs(target[1::6])-target[2::6],0).sum()
        return wrench_value+speed_loss+friction_loss

    cone_eq=hstack([ground,eye(nr),-eye(nr)],format='csr')
    cone_cost=np.r_[np.full(nf,1e-8),np.ones(2*nr)]
    def closest(coef):
        target=wrench(coef)
        answer=linprog(cone_cost,A_eq=cone_eq,b_eq=target,bounds=[(0,None)]*len(cone_cost),method='highs')
        return float(answer.fun),target

    coeff=np.zeros(nc);value,target=closest(coeff)
    velocity_map=(np.roll(mapping[:,0],-1,axis=0)-np.roll(mapping[:,0],1,axis=0))*fit.frequency/(2*1.39)
    seed_velocity=cyclic_velocities(m,seed,fit.frequency,fit.translation)[:,0]/1.39-1
    history=[{'iteration':0,'normalized_L1_wrench_deficit':value}]
    for iteration in range(1,args.iterations+1):
        if time.monotonic()-started>150:break
        h=1e-4
        derivative=np.column_stack([(wrench(coeff+np.eye(nc)[k]*h)-target)/h for k in range(nc)])
        values,contact_error,gaps,Jcontact,Jgap=geometry(coeff,True)
        old_geometry_violation=geometry_violation(contact_error,gaps)
        old_merit=merit(coeff,value,target)
        # Additional epigraph variables penalize forward-speed fluctuation and
        # net friction violations without inventing unavailable ground forces.
        auxiliary=nc+2*n
        eq=hstack([csr_matrix(derivative),-ground,eye(nr),-eye(nr),csr_matrix((nr,auxiliary))],format='csr')
        padding=csr_matrix((n*c,nf+2*nr+auxiliary))
        blocks=[hstack([csr_matrix(pose_map),padding]),hstack([-csr_matrix(pose_map),padding])]
        rhs=[(upper-values).ravel(),(values-lower).ravel()]
        cp=csr_matrix((len(contact_error),nf+2*nr+auxiliary))
        blocks.extend([hstack([csr_matrix(Jcontact),cp]),hstack([-csr_matrix(Jcontact),cp])])
        rhs.extend([contact_tolerance-contact_error,contact_tolerance+contact_error])
        # Existing 0.16 mm penetration is preserved; no new deep penetration.
        gp=csr_matrix((len(gaps),nf+2*nr+auxiliary))
        blocks.append(hstack([-csr_matrix(Jgap),gp]));rhs.append(gaps+.0001)
        support_indices=[]
        for i in range(n):
            for side in range(2):
                if fit.contact[i,side]>.5:
                    support_indices.append(4*i+2*side+int(gaps[4*i+2*side+1]<gaps[4*i+2*side]))
        support_indices=np.array(support_indices)
        sp=csr_matrix((len(support_indices),nf+2*nr+auxiliary))
        blocks.append(hstack([csr_matrix(Jgap[support_indices]),sp]));rhs.append(.00095-gaps[support_indices])
        zrows=np.arange(n)*6+2
        zp=csr_matrix((n,nf+2*nr+auxiliary))
        blocks.append(hstack([-csr_matrix(derivative[zrows]),zp]));rhs.append(target[zrows]-30./mg)
        signs=np.array([[a,b,c] for a in (-1.,1.) for b in (-1.,1.) for c in (-1.,1.)])
        angle_rows=np.concatenate([np.einsum('j,njk->nk',sign,mapping[:,3:6]) for sign in signs])
        angle_delta=values[:,3:6]-fit.reference_values[:,3:6]
        angle_rhs=np.concatenate([np.deg2rad(10)-angle_delta@sign for sign in signs])
        ap=csr_matrix((len(angle_rhs),nf+2*nr+auxiliary))
        blocks.append(hstack([csr_matrix(angle_rows),ap]));rhs.append(angle_rhs)
        blocks.extend([hstack([eye(nc),csr_matrix((nc,nf+2*nr)),-eye(nc),csr_matrix((nc,2*n))]),
                       hstack([-eye(nc),csr_matrix((nc,nf+2*nr)),-eye(nc),csr_matrix((nc,2*n))])])
        rhs.append(np.zeros(2*nc))
        velocity_delta=seed_velocity+velocity_map@coeff
        blocks.extend([hstack([csr_matrix(velocity_map),csr_matrix((n,nf+2*nr+nc)),-eye(n),csr_matrix((n,n))]),
                       hstack([-csr_matrix(velocity_map),csr_matrix((n,nf+2*nr+nc)),-eye(n),csr_matrix((n,n))])])
        rhs.extend([-velocity_delta,velocity_delta])
        for sx,sy in ((-1,-1),(-1,1),(1,-1),(1,1)):
            friction_derivative=sx*derivative[0::6]+sy*derivative[1::6]-derivative[2::6]
            friction_now=sx*target[0::6]+sy*target[1::6]-target[2::6]
            blocks.append(hstack([csr_matrix(friction_derivative),csr_matrix((n,nf+2*nr+nc+n)),-eye(n)]))
            rhs.append(-friction_now)
        cost=np.r_[np.zeros(nc),np.full(nf,1e-8),np.ones(2*nr),np.full(nc,.04),np.full(n,.3),np.full(n,1.)]
        result=linprog(cost,A_ub=vstack(blocks,format='csr'),b_ub=np.concatenate(rhs),
            A_eq=eq,b_eq=-target,bounds=[(-.1,.1)]*nc+[(0,None)]*(nf+2*nr+auxiliary),method='highs')
        if not result.success:
            history.append({'iteration':iteration,'failure':result.message});break
        accepted=False
        trials=[]
        for alpha in (1.,.5,.25,.125,.0625):
            proposal=coeff+alpha*result.x[:nc]
            vv,ee,gg,_,_=geometry(proposal)
            candidate_support=[min(gg[4*i+2*s:4*i+2*s+2]) for i in range(n) for s in range(2) if fit.contact[i,s]>.5]
            new_geometry_violation=geometry_violation(ee,gg)
            val,tt=closest(proposal)
            if np.min(tt[2::6])<0:continue
            new_merit=merit(proposal,val,tt)
            trials.append({'alpha':alpha,'geometry_violation_mm':new_geometry_violation*1000,'merit':new_merit,
                           'support_gap_mm':max(candidate_support)*1000,'penetration_mm':-min(gg)*1000})
            restoring=(old_geometry_violation>1e-7 and new_geometry_violation<old_geometry_violation*.98)
            feasible_descent=(new_geometry_violation<=1e-7 and new_merit<old_merit-1e-5)
            if restoring or feasible_descent:
                coeff,value,target=proposal,val,tt;accepted=True;break
        vv,ee,gg,_,_=geometry(coeff)
        history.append({'iteration':iteration,'normalized_L1_wrench_deficit':value,'accepted':accepted,
                        'support_point_max_error_mm':float(abs(ee).max()*1000),'foot_lowest_gap_mm':float(gg.min()*1000)})
        history[-1]['line_search']=trials
        print(json.dumps(history[-1]),flush=True)
        if not accepted:break
    values,qpos=state(coeff)
    if np.any(values<lower-1e-8) or np.any(values>upper+1e-8):raise RuntimeError('Pose bounds exceeded')
    summary=summarize(m,fields,qpos)
    if summary['max_root_quaternion_angle_change_deg']>10.0001:raise RuntimeError('Rotation limit exceeded')
    vv,ee,gg,_,_=geometry(coeff)
    fields['qpos']=qpos.astype(np.float32)
    fields['qvel']=cyclic_velocities(m,qpos,fit.frequency,fit.translation).astype(np.float32)
    fields['metadata']=artifact_metadata(fields,'Joint and pelvis Fourier wrench refinement with bounded support reachability; frozen-cone feasibility not asserted',summary)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**fields)
    report={'seed':str(args.seed),'candidate':str(args.output),'history':history,'summary':summary,
            'support_point_tolerance_mm':args.support_point_tolerance_mm,
            'support_point_max_error_mm':float(abs(ee).max()*1000),'foot_lowest_gap_mm':float(gg.min()*1000),
            'elapsed_s':time.monotonic()-started,'coefficients':coeff.reshape(c,nb).tolist(),
            'geometry_violation_mm':geometry_violation(ee,gg)*1000,
            'status':'Frozen-cone numerical proposal; see measured geometry violation. Actual native contact and muscle audit required.'}
    args.output.with_suffix('.proposal.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
