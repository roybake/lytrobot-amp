"""Restore stance contacts with a prescribed bounded root trajectory."""
from pathlib import Path
import argparse, json, sys, time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from logs.walk_1p39_refinement.project_fft3 import Projection, encode, summarize, artifact_metadata
from scripts.optimize_walk_contacts import cyclic_velocities, mesh_points

OUT=Path(__file__).resolve().parent
REFERENCE=ROOT/'robot/reference_data/walk1_lyt_contact_1p39.npz'
PROPOSAL=OUT/'dynamics_constraints/contact_root_wrench_candidate.npz'
CONTACTS=OUT/'fft3_contact_candidate.npz'


class LockedRootProjection(Projection):
    def __init__(self,model,fields,proposal,contact_targets):
        super().__init__(model,fields,proposal)
        self.locked_root=self.initial[:,:6].copy()
        self.state(encode(self,contact_targets).ravel())
        for i,data in enumerate(self.data):
            for side in range(2):
                body=self.constraint_bodies[i,side]
                rotation=data.xmat[body].reshape(3,3)
                self.targets_p[i,side]=data.xpos[body]+rotation@self.material[i,side]
                self.targets_r[i,side]=rotation
                self.position_weights[i,side]=10000. if self.contact[i,side]>.5 else 150.
                self.rotation_weights[i,side]=150. if self.contact[i,side]>.5 else 30.
        self.initial=encode(self,proposal)
        self.nominal=self.initial.copy()
        self.cached=None

    def bounds(self):
        lower,upper=super().bounds()
        lower=lower.reshape(self.count,self.columns)
        upper=upper.reshape(self.count,self.columns)
        lower[:,:6]=self.locked_root-1e-11
        upper[:,:6]=self.locked_root+1e-11
        return lower.ravel(),upper.ravel()

    def contact_row_scale(self):
        scale=np.ones(self.count*self.rows + 2*self.count + self.count*self.columns)
        # The target clearances are 0.1 mm above the floor and at most 0.8 mm
        # for support, leaving room for float32 output rounding.
        gap_start=12+self.columns+2
        for i in range(self.count):scale[i*self.rows+gap_start:i*self.rows+gap_start+4]=30.
        scale[self.count*self.rows:self.count*self.rows+2*self.count]=10.
        return scale

    def residual(self,flat):
        return super().residual(flat)*self.contact_row_scale()

    def jacobian(self,flat):
        return super().jacobian(flat).multiply(self.contact_row_scale()[:,None]).tocsr()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',type=Path,default=PROPOSAL)
    parser.add_argument('--output-prefix',default='root_contact_candidate')
    args=parser.parse_args()
    with np.load(REFERENCE) as a:fields={k:a[k] for k in a.files}
    with np.load(args.proposal) as a:proposal=a['qpos'].astype(float)
    with np.load(CONTACTS) as a:contact_targets=a['qpos'].astype(float)
    model=mujoco.MjModel.from_xml_path(str(ROOT/'robot/MJCFmodelposition.xml'))
    fit=LockedRootProjection(model,fields,proposal,contact_targets)
    lower,upper=fit.bounds();initial=np.clip(fit.initial.ravel(),lower+1e-12,upper-1e-12)
    started=time.monotonic()
    result=least_squares(fit.residual,initial,jac=fit.jacobian,bounds=(lower,upper),max_nfev=300,
        ftol=1e-7,xtol=1e-7,gtol=1e-7,tr_options={'atol':1e-7,'btol':1e-7})
    positions=fit.decode(result.x.reshape(fit.count,fit.columns))
    summary=summarize(model,fields,positions)
    original=fields['qpos'].astype(float)
    boundaries=[]
    for joint,address in zip(fit.hinges,fit.qadr):
        change=np.rad2deg(positions[:,address]-original[:,address])
        frames=np.flatnonzero(abs(change)>9.99)
        if len(frames):boundaries.append({'joint':model.joint(int(joint)).name,'frames':frames.tolist(),
                                        'signed_change_deg':change[frames].tolist()})
    bodies=[model.body(name).id for name in ('foot_r','toe_r','foot_l','toe_l')]
    vertices=[mesh_points(model,model.geom(name).id) for name in ('foot_r','toe_r','foot_l','toe_l')]
    contact_errors=[];d=mujoco.MjData(model)
    for i,p in enumerate(positions):
        d.qpos[:]=p;mujoco.mj_forward(model,d)
        clearance=np.array([(d.xpos[b]+v@d.xmat[b].reshape(3,3).T)[:,2].min() for b,v in zip(bodies,vertices)])
        for side in range(2):
            if fields['foot_contact'][i,side]>.5:
                gap=min(clearance[2*side:2*side+2])
                if abs(gap)>.002:contact_errors.append({'frame':i,'phase':i/len(positions),
                    'side':['right','left'][side],'gap_m':float(gap),
                    'roll_phase':float(fields['foot_roll_phase'][i,side])})
    report={'proposal':str(args.proposal),'frozen_support_target':str(CONTACTS),
            'reference':str(REFERENCE),'candidate':summary,'joints_at_10deg_limit':boundaries,
            'stance_clearance_outside_2mm':contact_errors,
            'root_max_component_difference_from_proposal':float(abs(positions[:,:7]-proposal[:,:7]).max()),
            'optimizer':{'message':result.message,'nfev':result.nfev,'residual_calls':fit.calls,
                         'cost':result.cost,'elapsed_s':time.monotonic()-started}}
    candidate=fields.copy();candidate['qpos']=positions.astype(np.float32)
    candidate['qvel']=cyclic_velocities(model,positions,float(fields['frequency']),
                                        fields['cycle_translation'].astype(float)).astype(np.float32)
    candidate['metadata']=artifact_metadata(fields,'Fixed-root bounded contact projection of dynamic root proposal',summary)
    np.savez_compressed(OUT/f'{args.output_prefix}.npz',**candidate)
    (OUT/f'{args.output_prefix}_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
