"""Read-only force-capacity analysis of a prescribed walking reference."""
from __future__ import annotations
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import mujoco
import numpy as np
from scipy.optimize import linprog
from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg

OUTPUT = Path(sys.argv[2]).resolve(); OUTPUT.mkdir(parents=True,exist_ok=True)
MOTION = Path(sys.argv[1]).resolve()
MODEL = ROOT / 'robot/MJCFmodelposition.xml'
model_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
motion_sha = hashlib.sha256(MOTION.read_bytes()).hexdigest()
env = LytRobotAmpEnv(LytRobotEnvCfg(model_xml=str(MODEL), motion_file=str(MOTION),
                                  motion_loop=(True,), num_envs=1, domain_randomization=False,
                                  target_velocity=1.39, reset_noise_scale=0.))
m, ref = env.model, env.motion
names = [m.actuator(i).name for i in range(m.nu)]
capacities = -m.actuator_forcerange[:, 0].copy()
geom_ids = [m.geom(name).id for name in ('foot_r', 'toe_r', 'foot_l', 'toe_l')]
qacc = (np.roll(ref.reference_qvel, -1, axis=0) - np.roll(ref.reference_qvel, 1, axis=0)) * ref.frequency / 2


def system(qpos, qvel, acceleration, allowed_contact, threshold=.002):
    data = mujoco.MjData(m)
    data.qpos[:] = qpos
    mujoco.mj_normalizeQuat(m, data.qpos)
    data.qvel[:] = qvel
    mujoco.mj_forward(m, data)
    inertia = np.empty(m.nv)
    mujoco.mj_mulM(m, data, inertia, acceleration)
    required = inertia + data.qfrc_bias - data.qfrc_passive
    actuation = np.zeros((m.nv, m.nu))
    for index in range(m.nu):
        start, count = int(data.moment_rowadr[index]), int(data.moment_rownnz[index])
        cols = data.moment_colind[start:start + count]
        actuation[cols, index] = -data.actuator_moment[start:start + count]
    # Use MuJoCo's native contact edge Jacobians: their basis already carries
    # contact-frame orientation and the model's pyramidal friction coefficients.
    if mujoco.mj_isSparse(m):
        raise RuntimeError('Expected dense constraint Jacobian.')
    constraint_jacobian = data.efc_J.reshape(data.nefc, m.nv)
    ground_columns, ground_bounds = [], []
    floor = m.geom('floor').id
    for contact in data.contact:
        if contact.geom1 == floor and int(contact.geom2) in geom_ids:
            geom = int(contact.geom2)
        elif contact.geom2 == floor and int(contact.geom1) in geom_ids:
            geom = int(contact.geom1)
        else:
            continue
        if allowed_contact[geom_ids.index(geom) // 2] < .5 or contact.efc_address < 0:
            continue
        address = int(contact.efc_address)
        kind = int(data.efc_type[address])
        if kind == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL):
            count = 2 * (int(contact.dim) - 1)
        elif kind == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS):
            count = 1
        else:
            raise RuntimeError('This linear audit requires the model pyramidal contact cone.')
        ground_columns.append(constraint_jacobian[address:address+count].T)
        ground_bounds.extend([(0., None)] * count)
    ground = np.concatenate(ground_columns, axis=1) if ground_columns else np.zeros((m.nv, 0))
    inequalities = np.zeros((0, ground.shape[1]))
    passive_columns, passive_bounds = [], []
    constraint_jacobian = data.efc_J.reshape(data.nefc, m.nv)
    for index, kind in enumerate(data.efc_type):
        if int(kind) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
            passive_columns.append(constraint_jacobian[index, :, None])
            passive_bounds.append((None, None))
        elif int(kind) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT):
            passive_columns.append(constraint_jacobian[index, :, None])
            passive_bounds.append((0., None))
    passive = np.concatenate(passive_columns, axis=1) if passive_columns else np.zeros((m.nv, 0))
    gain = m.actuator_gainprm[:, 0]
    bias = (m.actuator_biasprm[:, 0] + m.actuator_biasprm[:, 1] * data.actuator_length
            + m.actuator_biasprm[:, 2] * data.actuator_velocity)
    geom_min = np.maximum(-(gain * m.actuator_ctrlrange[:, 1] + bias), 0.)
    geom_max = np.maximum(-(gain * m.actuator_ctrlrange[:, 0] + bias), 0.)
    return dict(required=required, actuation=actuation, ground=ground,
                ground_bounds=ground_bounds, ground_inequalities=inequalities,
                passive=passive, passive_bounds=passive_bounds,
                geom_min=geom_min, geom_max=geom_max, contacts=len(ground_columns),
                data=data)


def ground_feasible(state):
    ground = state['ground']
    if ground.shape[1] == 0:
        return False
    inequality = state['ground_inequalities']
    result = linprog(np.zeros(ground.shape[1]), A_ub=inequality if len(inequality) else None,
                     b_ub=np.zeros(len(inequality)) if len(inequality) else None, A_eq=ground[:6],
                     b_eq=state['required'][:6], bounds=state['ground_bounds'], method='highs')
    return bool(result.success)


def solve_forces(state, mode):
    matrix = np.concatenate((state['actuation'], state['ground'], state['passive']), axis=1)
    n = matrix.shape[1]
    if mode == 'current':
        lower = np.minimum(state['geom_min'], capacities)
        upper = np.minimum(state['geom_max'], capacities)
    elif mode == 'unlimited_force_current_stroke':
        lower, upper = state['geom_min'], state['geom_max']
    elif mode == 'ideal_force_control':
        lower, upper = np.zeros(m.nu), np.full(m.nu, np.inf)
    else:
        raise ValueError(mode)
    bounds = [(float(lo), None if not np.isfinite(hi) else float(hi))
              for lo, hi in zip(lower, upper)] + state['ground_bounds'] + state['passive_bounds']
    # Minimize the largest ratio T_i/current_capacity_i, specifying one
    # reproducible allocation among the many possible muscle-force distributions.
    eq = np.column_stack((matrix, np.zeros(m.nv)))
    ground_ineq = state['ground_inequalities']
    inequality = np.zeros((len(ground_ineq) + m.nu, n + 1))
    inequality[:len(ground_ineq), m.nu:m.nu + state['ground'].shape[1]] = ground_ineq
    for i in range(m.nu):
        inequality[len(ground_ineq) + i, i] = 1.
        inequality[len(ground_ineq) + i, -1] = -capacities[i]
    cost = np.zeros(n + 1)
    cost[-1] = 1.
    result = linprog(cost, A_ub=inequality, b_ub=np.zeros(len(inequality)),
                     A_eq=eq, b_eq=state['required'], bounds=bounds + [(0., None)], method='highs')
    if not result.success:
        return None
    residual = matrix @ result.x[:n] - state['required']
    if np.max(np.abs(residual)) > 1e-5:
        raise RuntimeError('Force solution failed the dynamics reconstruction check.')
    return dict(forces=result.x[:m.nu], ratio=float(result.x[-1]), residual=float(np.max(np.abs(residual))))


modes = ('current', 'unlimited_force_current_stroke', 'ideal_force_control')
rows, solved = [], {mode: [] for mode in modes}
for frame, (position, velocity, acceleration) in enumerate(zip(ref.reference_qpos, ref.reference_qvel, qacc)):
    state = system(position, velocity, acceleration, ref.reference_foot_contact[frame])
    root_force = state['required'][:3]
    row = dict(frame=frame, time_s=frame/ref.frequency, contacts=state['contacts'],
               required_ground_x_N=float(root_force[0]), required_ground_y_N=float(root_force[1]),
               required_ground_z_N=float(root_force[2]),
               optimistic_ground_wrench_feasible=ground_feasible(state))
    for mode in modes:
        answer = solve_forces(state, mode)
        row[mode + '_feasible'] = answer is not None
        row[mode + '_capacity_ratio'] = answer['ratio'] if answer else None
        if answer:
            solved[mode].append((frame, answer))
        for i, name in enumerate(names):
            row[mode + '_' + name + '_N'] = float(answer['forces'][i]) if answer else None
    rows.append(row)

summary = dict(motion=str(MOTION), motion_sha256=motion_sha, model=str(MODEL), model_sha256=model_sha,
               root_armature=m.dof_armature[:6].tolist(), robot_mass_kg=float(m.body_mass.sum()), frames=len(rows), frequency_hz=ref.frequency,
               cycle_duration_s=ref.frame_count/ref.frequency, speed_mps=float(ref.reference_root_lin_vel[:, 0].mean()),
               equation='M qacc + bias - passive = -actuator_moment.T T + J_ground.T F + J_passive.T lambda; T>=0, Fz>=0',
               muscle_allocation='Minimum largest fraction of the current nominal muscle capacities, independently at each frame.',
               assumptions=['Exact prescribed qpos, qvel and cyclic finite-difference qacc; model body masses/inertias, gravity and damping unchanged.',
                            'Only labeled stance feet supply ground force, using actual active MuJoCo floor-contact Jacobians.',
                            'MuJoCo native pyramidal edge force variables are nonnegative and include exact contact-frame friction directions.',
                            'No extra body support force, external tether, pushing tendon or active actuator on passive toe joints.',
                            'Contact forces are freely allocated inside native cones, ignoring the soft-contact force law: this is an optimistic necessary condition, not a walking simulation.',
                            'These are instantaneous necessary conditions; feasible frames do not establish stable closed-loop walking.'],
               required_ground_force_min_N=np.min([[r['required_ground_x_N'],r['required_ground_y_N'],r['required_ground_z_N']] for r in rows],axis=0).tolist(),
               required_ground_force_max_N=np.max([[r['required_ground_x_N'],r['required_ground_y_N'],r['required_ground_z_N']] for r in rows],axis=0).tolist(),
               negative_required_vertical_frames=[r['frame'] for r in rows if r['required_ground_z_N'] < -1e-5],
               native_contact_ground_feasible_frames=sum(r['optimistic_ground_wrench_feasible'] for r in rows),
               force_modes={}, muscles=[])
for mode in modes:
    solutions=solved[mode]
    summary['force_modes'][mode]=dict(feasible_frames=len(solutions),
                                    feasible_frame_indices=[frame for frame,_ in solutions],
                                    peak_minimized_capacity_ratio=max((answer['ratio'] for _,answer in solutions),default=None))
for i,name in enumerate(names):
    muscle=dict(name=name,current_max_tension_N=float(capacities[i]),
                stroke_min_m=float(m.actuator_ctrlrange[i,0]),stroke_max_m=float(m.actuator_ctrlrange[i,1]))
    for mode in modes:
        muscle[mode+'_peak_in_feasible_frames_N']=max((float(answer['forces'][i]) for _,answer in solved[mode]),default=None)
    summary['muscles'].append(muscle)
(OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUTPUT/'force_trace.csv').open('w') as file:
    writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
# Verify the input artifacts were not modified during analysis.
assert hashlib.sha256(MOTION.read_bytes()).hexdigest()==motion_sha
assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==model_sha
print(json.dumps(summary,indent=2),flush=True)
env.close()
