"""Bounded numerical projection of a low-frequency gait onto foot contacts."""
from pathlib import Path
import hashlib, json, sys, time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, vstack
from scipy.spatial.transform import Rotation
from scripts.optimize_walk_contacts import ContactFit, cyclic_velocities
from scripts.repair_walk_reference import enforce_passive

OUT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'robot/reference_data/walk1_lyt_contact_1p39.npz'


def artifact_metadata(fields, method, summary):
    metadata = json.loads(str(fields['metadata']))
    for key in ('kinematic_audit', 'loader_validation', 'smoothness_comparison',
                'smoothness_revision', 'cycle_refinement', 'knee_extension',
                'generation_command', 'accepted_visual_preview'):
        metadata.pop(key, None)
    metadata.update(method=method, revision=method,
        dynamic_stability='Not asserted; numerical candidate requiring full dynamics evaluation',
        refinement_reference=str(REFERENCE),
        refinement_reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        refinement_pose_change_limits={'joint_angle_deg':10., 'root_component_mm':30.,
                                      'root_quaternion_angle_deg':10.},
        refinement_summary=summary)
    return json.dumps(metadata,ensure_ascii=False)


def encode(fit, poses):
    return np.column_stack((poses[:, 0] - fit.root_x, poses[:, 1:3],
        Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_euler('xyz'), poses[:, fit.qadr]))


class Projection(ContactFit):
    def __init__(self, model, fields, lowpass):
        reference = fields['qpos'].astype(float)
        meta = json.loads(str(fields['metadata']))
        super().__init__(model, reference, reference, float(fields['frequency']),
            fields['foot_contact'], fields['foot_roll_phase'], fields['foot_roll_mode'],
            fields['cycle_translation'].astype(float), meta['target_stance_clearance_m'])
        self.reference_values = self.initial.copy()
        self.initial = encode(self, lowpass)
        self.nominal = self.initial.copy()
        self.prior_weights = np.r_[np.full(3, 300.), np.full(3, 40.), np.full(len(self.hinges), 40.)]
        self.curvature_weights = np.r_[np.full(3, 10000.), np.full(3, 600.), np.full(len(self.hinges), 600.)]
        self.deviation = np.r_[np.full(3, .03), np.full(3, np.deg2rad(10.)),
            np.full(len(self.hinges), np.deg2rad(10.))]
        # The original support transform fixes the world location of planted soles.
        # Swing targets follow the smooth seed, avoiding the original velocity spikes.
        self.state(self.reference_values.ravel())
        original_targets = []
        for i, data in enumerate(self.data):
            frame = []
            for side in range(2):
                body = self.constraint_bodies[i, side]
                rotation = data.xmat[body].reshape(3, 3).copy()
                frame.append((data.xpos[body] + rotation @ self.material[i, side], rotation))
            original_targets.append(frame)
        self.state(self.initial.ravel())
        for i, data in enumerate(self.data):
            for side in range(2):
                body = self.constraint_bodies[i, side]
                rotation = data.xmat[body].reshape(3, 3)
                if self.contact[i, side] > .5:
                    self.targets_p[i, side], self.targets_r[i, side] = original_targets[i][side]
                    self.position_weights[i, side] = 5000.
                    self.rotation_weights[i, side] = 200.
                else:
                    self.targets_p[i, side] = data.xpos[body] + rotation @ self.material[i, side]
                    self.targets_r[i, side] = rotation
                    self.position_weights[i, side] = 300.
                    self.rotation_weights[i, side] = 40.
                self.targets_toe[i, side] = self.nominal[i, 6 + self.mtp[side]]
        # Penalize all frequency components above the third stride harmonic.
        basis = np.column_stack([np.ones(self.count)] + [v for k in range(1, 4)
            for v in (np.cos(2*np.pi*k*np.arange(self.count)/self.count),
                      np.sin(2*np.pi*k*np.arange(self.count)/self.count))])
        self.highpass = np.eye(self.count) - basis @ np.linalg.pinv(basis)
        self.highpass_weights = np.r_[np.full(3, 5000.), np.full(3, 400.),
                                      np.full(len(self.hinges), 400.)]
        self.cached = None

    def bounds(self):
        # Deliberately do not inherit the original exact stance-knee extension lock.
        lower = np.maximum(np.tile(self.lower, (self.count, 1)), self.reference_values-self.deviation)
        upper = np.minimum(np.tile(self.upper, (self.count, 1)), self.reference_values+self.deviation)
        for column in self.knee_columns:
            hinge = self.hinges[column-6]
            upper[:, column] = np.minimum(self.model.jnt_range[hinge, 1]-.0001,
                                          self.reference_values[:, column]+self.deviation[column])
        return lower.ravel(), upper.ravel()

    def residual(self, flat):
        base = super().residual(flat)
        upper_gap = np.zeros((self.count, 2))
        for i in range(self.count):
            for side in range(2):
                if self.contact[i, side] > .5:
                    upper_gap[i, side] = 30000. * max(min(self.gaps[i][2*side:2*side+2])-.0008, 0.)
        high = (self.highpass @ self.values) * self.highpass_weights
        return np.r_[base, upper_gap.ravel(), high.ravel()]

    def jacobian(self, flat):
        base = super().jacobian(flat)
        vals, rr, cc = [], [], []
        for i, (data, mapping) in enumerate(zip(self.data, self.maps)):
            for side in range(2):
                k = 2*side + int(self.gaps[i][2*side+1] < self.gaps[i][2*side])
                if self.contact[i, side] > .5 and self.gaps[i][k] > .0008:
                    mujoco.mj_jac(self.model, data, self.jacp, None, self.bottoms[i][k], self.bodies[k])
                    derivative = 30000.*self.jacp[2]@mapping
                    indices = np.flatnonzero(derivative)
                    vals.extend(derivative[indices]); rr.extend(np.full(len(indices), 2*i+side)); cc.extend(i*self.columns+indices)
        gap = coo_matrix((vals,(rr,cc)),shape=(2*self.count,self.count*self.columns)).tocsr()
        ri, ci = np.nonzero(self.highpass)
        row = (ri[:,None]*self.columns+np.arange(self.columns)).ravel()
        col = (ci[:,None]*self.columns+np.arange(self.columns)).ravel()
        data = (self.highpass[ri,ci,None]*self.highpass_weights).ravel()
        high = coo_matrix((data,(row,col)),shape=(self.count*self.columns,self.count*self.columns)).tocsr()
        return vstack((base,gap,high),format='csr')


def summarize(model, fields, positions):
    frequency = float(fields['frequency']); translation = fields['cycle_translation'].astype(float)
    velocities = cyclic_velocities(model,positions,frequency,translation)
    acceleration = (np.roll(velocities,-1,axis=0)-np.roll(velocities,1,axis=0))*frequency/2
    original = fields['qpos'].astype(float)
    hinges = [j for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
    adr = model.jnt_qposadr[hinges]
    quat_before = Rotation.from_quat(original[:,[4,5,6,3]])
    quat_after = Rotation.from_quat(positions[:,[4,5,6,3]])
    meshes = [model.geom(name).id for name in ('foot_r','toe_r','foot_l','toe_l')]
    from scripts.optimize_walk_contacts import mesh_points
    bodies = [model.body(name).id for name in ('foot_r','toe_r','foot_l','toe_l')]
    vertices = [mesh_points(model,g) for g in meshes]
    required, gaps, support_gap = [], [], []
    d = mujoco.MjData(model)
    for i,(p,v,ac) in enumerate(zip(positions,velocities,acceleration)):
        d.qpos[:]=p; d.qvel[:]=v; mujoco.mj_forward(model,d)
        ma = np.empty(model.nv); mujoco.mj_mulM(model,d,ma,ac)
        required.append((ma+d.qfrc_bias-d.qfrc_passive)[:3])
        clearance = np.array([(d.xpos[b]+vv@d.xmat[b].reshape(3,3).T)[:,2].min()
                              for b,vv in zip(bodies,vertices)])
        gaps.append(clearance)
        support_gap.extend([min(clearance[2*s:2*s+2]) for s in range(2) if fields['foot_contact'][i,s]>.5])
    force = np.array(required)
    return {'max_joint_change_deg':float(np.rad2deg(abs(positions[:,adr]-original[:,adr])).max()),
        'max_root_position_component_change_mm':float(abs(positions[:,:3]-original[:,:3]).max()*1000),
        'max_root_quaternion_angle_change_deg':float(np.rad2deg((quat_before.inv()*quat_after).magnitude()).max()),
        'minimum_sole_clearance_m':float(np.min(gaps)), 'maximum_support_gap_m':float(np.max(support_gap)),
        'required_ground_force_min_N':force.min(0).tolist(), 'required_ground_force_max_N':force.max(0).tolist(),
        'negative_vertical_frames':np.flatnonzero(force[:,2]<0).tolist(),
        'pyramidal_friction_failed_frames':np.flatnonzero(abs(force[:,0])+abs(force[:,1])>force[:,2]).tolist(),
        'root_vx_range':velocities[:,0].min().item(),
        'root_vx_max':velocities[:,0].max().item(),
        'mean_forward_speed_m_s':float(np.mean(velocities[:,0])),
        'cadence_steps_per_minute':120./(len(positions)/frequency)}


def main():
    with np.load(REFERENCE) as archive:
        fields = {k:archive[k] for k in archive.files}
    model = mujoco.MjModel.from_xml_path(str(ROOT/'robot/MJCFmodelposition.xml'))
    positions = fields['qpos'].astype(float)
    hinges = [j for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
    adr = model.jnt_qposadr[hinges]
    count = len(positions); translation = fields['cycle_translation'].astype(float)
    values = np.column_stack((positions[:,:3],Rotation.from_quat(positions[:,[4,5,6,3]]).as_euler('xyz'),positions[:,adr]))
    values[:,0] -= np.arange(count)*translation[0]/count
    spectrum = np.fft.rfft(values,axis=0); spectrum[4:]=0.
    filtered = np.fft.irfft(spectrum,count,axis=0)
    seed = positions.copy(); seed[:,:3]=filtered[:,:3]; seed[:,0]+=np.arange(count)*translation[0]/count
    seed[:,3:7]=Rotation.from_euler('xyz',filtered[:,3:6]).as_quat()[:,[3,0,1,2]]
    seed[:,adr]=filtered[:,6:]; enforce_passive(model,seed)
    seed_fields=fields.copy();seed_fields['qpos']=seed.astype(np.float32)
    seed_fields['qvel']=cyclic_velocities(model,seed,float(fields['frequency']),translation).astype(np.float32)
    seed_summary=summarize(model,fields,seed)
    seed_fields['metadata']=artifact_metadata(fields,'Periodic three-harmonic warm start; contact not projected',seed_summary)
    np.savez_compressed(OUT/'fft3_seed.npz',**seed_fields)
    report={'reference':str(REFERENCE),'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            'fft3_seed':seed_summary}
    (OUT/'fft3_seed_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    fit=Projection(model,fields,seed);lower,upper=fit.bounds()
    initial=np.clip(fit.initial.ravel(),lower+1e-10,upper-1e-10)
    started=time.monotonic()
    result=least_squares(fit.residual,initial,jac=fit.jacobian,bounds=(lower,upper),max_nfev=300,
        ftol=1e-7,xtol=1e-7,gtol=1e-7,tr_options={'atol':1e-7,'btol':1e-7})
    candidate=fit.decode(result.x.reshape(count,fit.columns))
    report['projected']=summarize(model,fields,candidate)
    report['optimizer']={'message':result.message,'nfev':result.nfev,'residual_calls':fit.calls,
        'cost':result.cost,'elapsed_s':time.monotonic()-started}
    final_fields=fields.copy();final_fields['qpos']=candidate.astype(np.float32)
    final_fields['qvel']=cyclic_velocities(model,candidate,float(fields['frequency']),translation).astype(np.float32)
    final_fields['metadata']=artifact_metadata(fields,'Bounded foot-contact projection of the periodic three-harmonic warm start',report['projected'])
    np.savez_compressed(OUT/'fft3_contact_candidate.npz',**final_fields)
    (OUT/'fft3_contact_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
