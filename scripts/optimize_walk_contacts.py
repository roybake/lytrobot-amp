#!/usr/bin/env python3
"""Fit a source-derived LYT gait cycle with foot contacts and natural knee motion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.repair_walk_reference import enforce_passive, periodic_fit


def cyclic_velocities(model, positions, frequency, translation):
    velocities = np.empty((len(positions), model.nv), dtype=np.float64)
    forward, reverse = np.empty(model.nv), np.empty(model.nv)
    for i, current in enumerate(positions):
        previous = positions[(i - 1) % len(positions)].copy()
        following = positions[(i + 1) % len(positions)].copy()
        if i == 0:
            previous[:2] -= translation
        if i == len(positions) - 1:
            following[:2] += translation
        mujoco.mj_differentiatePos(model, forward, 1 / frequency, current, following)
        mujoco.mj_differentiatePos(model, reverse, 1 / frequency, current, previous)
        velocities[i] = 0.5 * (forward - reverse)
    return velocities


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10. + u * (-15. + 6. * u))


def skew(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


def mesh_points(model, geom):
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[start:start + count].astype(float)
    vertices = vertices[ConvexHull(vertices).vertices]
    rotation = Rotation.from_quat(model.geom_quat[geom, [1, 2, 3, 0]]).as_matrix()
    return model.geom_pos[geom] + vertices @ rotation.T


def resample_poses(poses, count, translation=None):
    phase = np.arange(count) * len(poses) / count
    low = np.floor(phase).astype(int)
    high = (low + 1) % len(poses)
    blend = (phase - low)[:, None]
    first, second = poses[low].copy(), poses[high].copy()
    if translation is not None:
        second[high == 0, :2] += translation
    reverse = np.sum(first[:, 3:7] * second[:, 3:7], axis=1) < 0
    second[reverse, 3:7] *= -1
    result = (1 - blend) * first + blend * second
    result[:, 3:7] /= np.linalg.norm(result[:, 3:7], axis=1, keepdims=True)
    return result


def resample_contacts(contact, count):
    old_count = len(contact)
    phase = np.arange(count) * old_count / count
    active = np.zeros((count, 2), dtype=np.float32)
    stance_phase = np.full((count, 2), -1., dtype=np.float32)
    modes = np.zeros((count, 2), dtype=np.int8)
    for side in range(2):
        previous = contact[:, side] > .5
        strikes = np.flatnonzero(previous & ~np.roll(previous, 1))
        if len(strikes) != 1:
            raise ValueError('A walking cycle must have one stance interval per leg')
        relative = (phase - strikes[0]) % old_count
        stance = int(previous.sum())
        selected = relative < stance
        active[selected, side] = 1.
        normalized = np.clip(relative[selected] / (stance - old_count / count), 0., 1.)
        stance_phase[selected, side] = normalized
        modes[selected, side] = np.where(normalized < .2, 1, np.where(normalized >= .7, 3, 2))
    return active, stance_phase, modes


class ContactFit:
    def __init__(self, model, source, seed, frequency, contact, phases, modes, translation,
                 ground_clearance=.0005):
        self.model, self.source, self.seed = model, source, seed
        self.count, self.frequency = len(seed), frequency
        self.contact, self.phases, self.modes = contact, phases, modes
        self.translation = translation
        self.ground_clearance = ground_clearance
        self.hinges = np.array([j for j in range(model.njnt)
                               if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE])
        self.qadr, self.vadr = model.jnt_qposadr[self.hinges], model.jnt_dofadr[self.hinges]
        self.columns = 6 + len(self.hinges)
        self.bodies = [model.body(name).id for name in ('foot_r', 'toe_r', 'foot_l', 'toe_l')]
        self.hip_bodies = [model.body('femur_' + side).id for side in ('r', 'l')]
        self.knee_bodies = [model.body('tibia_' + side).id for side in ('r', 'l')]
        self.ankle_bodies = [model.body('talus_' + side).id for side in ('r', 'l')]
        self.meshes = [mesh_points(model, model.geom(name).id)
                       for name in ('foot_r', 'toe_r', 'foot_l', 'toe_l')]
        self.mtp = [int(np.flatnonzero(self.hinges == model.joint('mtp_angle_' + s).id)[0])
                    for s in ('r', 'l')]
        self.data = [mujoco.MjData(model) for _ in seed]
        neutral = self.data[0]
        neutral.qpos[:] = model.qpos0
        mujoco.mj_kinematics(model, neutral)
        self.rest_foot_lanes = neutral.xpos[self.bodies[::2], 1].copy()
        self.rest_knee_offsets = (neutral.xpos[self.knee_bodies, 1]
                                 - neutral.xpos[self.hip_bodies, 1]).copy()
        self.rest_ankle_offsets = (neutral.xpos[self.ankle_bodies, 1]
                                  - neutral.xpos[self.hip_bodies, 1]).copy()
        self.jacp, self.jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
        self.root_x = np.arange(self.count) * translation[0] / self.count
        self.targets_p = np.empty((self.count, 2, 3))
        self.targets_r = np.empty((self.count, 2, 3, 3))
        self.targets_toe = np.empty((self.count, 2))
        self.constraint_bodies = np.empty((self.count, 2), dtype=int)
        self.material = np.zeros((self.count, 2, 3))
        self.position_weights = np.empty((self.count, 2))
        self.rotation_weights = np.empty((self.count, 2))
        self._build_targets()
        source_joints = periodic_fit(source[:, self.qadr])
        source_joints = np.clip(source_joints, model.jnt_range[self.hinges, 0] + .0001,
                                model.jnt_range[self.hinges, 1] - .0001)
        self.nominal = np.column_stack((np.zeros(self.count), np.zeros(self.count),
                                       source[:, 2], np.zeros((self.count, 3)), source_joints))
        self.knee_columns = []
        self.extension_frames = []
        for side, suffix in enumerate(('r', 'l')):
            joint = model.joint('knee_angle_' + suffix).id
            column = 6 + int(np.flatnonzero(self.hinges == joint)[0])
            self.knee_columns.append(column)
            active = contact[:, side] > .5
            phase = phases[:, side]
            extend = smooth((phase - .15) / .20) * (1 - smooth((phase - .70) / .20))
            extend = np.where(active, extend, 0.)
            self.nominal[:, column] *= 1 - extend
            self.extension_frames.append(active & (phase >= .35) & (phase <= .70))
        self.prior_weights = np.r_[25., 25., 35., 12., 12., 12., np.full(len(self.hinges), 5.)]
        for side in ('r', 'l'):
            k = int(np.flatnonzero(self.hinges == model.joint('knee_angle_' + side).id)[0])
            self.prior_weights[6 + k] = 18.
            for prefix in ('hip_adduction_', 'hip_rotation_'):
                k = int(np.flatnonzero(self.hinges == model.joint(prefix + side).id)[0])
                self.nominal[:, 6 + k] = 0.
                self.prior_weights[6 + k] = 50.
        self.curvature_weights = np.r_[100., 100., 120., 30., 30., 30., np.full(len(self.hinges), 6.)]
        for column in self.knee_columns:
            self.curvature_weights[column] = 30.
        self.lower = np.r_[-.1, -.15, .65, -.3, -.35, -.2, model.jnt_range[self.hinges, 0] + .0001]
        self.upper = np.r_[.1, .15, .98, .3, .35, .2, model.jnt_range[self.hinges, 1] - .0001]
        for column in self.knee_columns:
            self.upper[column] = 0.
        eulers = Rotation.from_quat(seed[:, [4, 5, 6, 3]]).as_euler('xyz')
        self.initial = np.column_stack((seed[:, 0] - self.root_x, seed[:, 1], seed[:, 2],
                                       eulers, seed[:, self.qadr]))
        self.cached = None
        self.calls, self.started = 0, time.monotonic()

    def bounds(self):
        lower = np.tile(self.lower, (self.count, 1))
        upper = np.tile(self.upper, (self.count, 1))
        for column, frames in zip(self.knee_columns, self.extension_frames):
            lower[frames, column] = -1e-6
            upper[frames, column] = 0.
        return lower.ravel(), upper.ravel()

    def _build_targets(self):
        data = self.data[0]
        for side in range(2):
            foot, toe = self.bodies[2 * side:2 * side + 2]
            foot_mesh, toe_mesh = self.meshes[2 * side:2 * side + 2]
            joint = self.hinges[self.mtp[side]]
            axis, offset = self.model.jnt_axis[joint], self.model.body_pos[toe]
            bottom = foot_mesh[:, 2].min()
            def gap(angle):
                rot = Rotation.from_rotvec(axis * angle).as_matrix()
                return (offset + toe_mesh @ rot.T)[:, 2].min() - bottom
            grid = np.linspace(*self.model.jnt_range[joint], 101)
            roots = [brentq(gap, a, b) for a, b in zip(grid[:-1], grid[1:]) if gap(a) * gap(b) <= 0]
            flat_toe = min(roots, key=abs)
            active = self.contact[:, side] > .5
            strike = int(np.flatnonzero(active & ~np.roll(active, 1))[0])
            stance = int(active.sum())
            data.qpos[:] = self.seed[strike]
            mujoco.mj_kinematics(self.model, data)
            anchor = data.xpos[foot].copy()
            anchor[1] = self.rest_foot_lanes[side]
            anchor[2] = self.ground_clearance - bottom
            heel = foot_mesh[np.argmin(foot_mesh @ Rotation.from_rotvec(axis * .18).as_matrix()[2])]
            flat_r = np.eye(3)
            tip_rotation = Rotation.from_rotvec(axis * flat_toe).as_matrix()
            tip = toe_mesh[np.argmin(toe_mesh @ tip_rotation[2])]
            for i in range(self.count):
                phase_frame = (i - strike) % self.count
                base = anchor.copy()
                if i < strike:
                    base[:2] -= self.translation
                if active[i]:
                    phase = self.phases[i, side]
                    heel_angle = .18 * (1 - smooth(phase / .2))
                    toe_angle = .38 * smooth((phase - .7) / .3)
                    rotation = Rotation.from_rotvec(axis * (heel_angle - toe_angle)).as_matrix()
                    self.position_weights[i, side] = 3000.
                    self.rotation_weights[i, side] = 300.
                    if self.modes[i, side] == 1:
                        self.constraint_bodies[i, side] = foot
                        self.material[i, side] = heel
                        self.targets_p[i, side] = base + heel
                        self.targets_p[i, side, 2] = self.ground_clearance
                        self.targets_r[i, side] = rotation
                    elif phase >= .7:
                        self.constraint_bodies[i, side] = toe
                        self.material[i, side] = tip
                        self.targets_p[i, side] = base + offset + tip_rotation @ tip
                        self.targets_p[i, side, 2] = self.ground_clearance
                        self.targets_r[i, side] = tip_rotation
                    else:
                        self.constraint_bodies[i, side] = foot
                        self.targets_p[i, side] = base
                        self.targets_r[i, side] = flat_r
                    self.targets_toe[i, side] = flat_toe + toe_angle
                else:
                    u = (phase_frame - stance + 1) / (self.count - stance + 1)
                    u = float(np.clip(u, 0., 1.))
                    toe_off_rotation = Rotation.from_rotvec(-axis * .38).as_matrix()
                    toe_off_position = base + offset - toe_off_rotation @ offset
                    heel_rotation = Rotation.from_rotvec(axis * .18).as_matrix()
                    landing_base = base + np.r_[self.translation, 0.]
                    heel_position = landing_base + heel - heel_rotation @ heel
                    heel_position[2] = self.ground_clearance - (heel_rotation @ heel)[2]
                    blend = smooth(u)
                    self.constraint_bodies[i, side] = foot
                    self.targets_p[i, side] = ((1 - blend) * toe_off_position + blend * heel_position
                                               + [0., 0., .055 * np.sin(np.pi * u)**2])
                    self.targets_r[i, side] = Rotation.from_rotvec(axis * (-.38 * (1 - blend) + .18 * blend)).as_matrix()
                    self.targets_toe[i, side] = flat_toe + .38 * (1 - smooth(u))
                    self.position_weights[i, side] = 30.
                    self.rotation_weights[i, side] = 5.

    def decode(self, values):
        poses = self.seed.copy()
        poses[:, 0] = self.root_x + values[:, 0]
        poses[:, 1:3] = values[:, 1:3]
        poses[:, 3:7] = Rotation.from_euler('xyz', values[:, 3:6]).as_quat()[:, [3, 0, 1, 2]]
        poses[:, self.qadr] = values[:, 6:]
        enforce_passive(self.model, poses)
        return poses

    def state(self, flat):
        if self.cached is not None and np.array_equal(flat, self.cached):
            return
        self.values = flat.reshape(self.count, self.columns)
        self.poses = self.decode(self.values)
        self.maps, self.gaps, self.bottoms, self.bottom_local_points = [], [], [], []
        for pose, x, data in zip(self.poses, self.values, self.data):
            data.qpos[:] = pose
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            mapping = np.zeros((self.model.nv, self.columns))
            mapping[:3, :3] = np.eye(3)
            roll, pitch = x[3:5]
            mapping[3:6, 3:6] = [[1., 0., -np.sin(pitch)],
                [0., np.cos(roll), np.sin(roll) * np.cos(pitch)],
                [0., -np.sin(roll), np.cos(roll) * np.cos(pitch)]]
            mapping[self.vadr, np.arange(6, self.columns)] = 1.
            self.maps.append(mapping)
            gaps, points, local_points = [], [], []
            for body, mesh in zip(self.bodies, self.meshes):
                r = data.xmat[body].reshape(3, 3)
                local_point = mesh[np.argmin(mesh @ r[2])]
                point = data.xpos[body] + r @ local_point
                gaps.append(point[2])
                points.append(point)
                local_points.append(local_point)
            self.gaps.append(gaps)
            self.bottoms.append(points)
            self.bottom_local_points.append(local_points)
        self.cached = flat.copy()

    @property
    def rows(self):
        return 12 + self.columns + 2 + 4 + 8 + 8 + self.columns

    def residual(self, flat):
        self.state(flat)
        local = []
        for i, (x, data) in enumerate(zip(self.values, self.data)):
            rows = []
            for side in range(2):
                body = self.constraint_bodies[i, side]
                rotation = data.xmat[body].reshape(3, 3)
                point = data.xpos[body] + rotation @ self.material[i, side]
                rows.extend(self.position_weights[i, side] * (point - self.targets_p[i, side]))
                error = Rotation.from_matrix(self.targets_r[i, side].T @ rotation).as_rotvec()
                rows.extend(self.rotation_weights[i, side] * error)
            rows.extend(self.prior_weights * (x - self.nominal[i]))
            for side, column in enumerate(self.mtp):
                rows.append(3. * (x[6 + column] - self.targets_toe[i, side]))
            for k, gap in enumerate(self.gaps[i]):
                if self.contact[i, k // 2] > .5:
                    clearance = self.ground_clearance
                else:
                    clearance = .006
                rows.append(10000. * min(gap - clearance, 0.))
            for body in self.bodies:
                forward = data.xmat[body].reshape(3, 3)[:, 0]
                rows.append(200. * np.arctan2(forward[1], forward[0]))
            for side, hip in enumerate(self.hip_bodies):
                rows.append(500. * (data.xpos[self.knee_bodies[side], 1]
                                     - data.xpos[hip, 1] - self.rest_knee_offsets[side]))
                rows.append(500. * (data.xpos[self.ankle_bodies[side], 1]
                                     - data.xpos[hip, 1] - self.rest_ankle_offsets[side]))
            for k, body in enumerate(self.bodies):
                side = k // 2
                loaded = (self.contact[i, side] > .5
                          and self.gaps[i][k] <= min(self.gaps[i][2*side:2*side+2]) + .0015)
                if loaded:
                    local_point = self.bottom_local_points[i][k]
                    previous, following = self.data[(i-1) % self.count], self.data[(i+1) % self.count]
                    before = previous.xpos[body] + previous.xmat[body].reshape(3, 3) @ local_point
                    after = following.xpos[body] + following.xmat[body].reshape(3, 3) @ local_point
                    if i == 0:
                        before[:2] -= self.translation
                    if i == self.count - 1:
                        after[:2] += self.translation
                    rows.extend(80. * (after[:2] - before[:2]) * self.frequency / 2)
                else:
                    rows.extend((0., 0.))
            curvature = self.values[(i - 1) % self.count] - 2 * x + self.values[(i + 1) % self.count]
            rows.extend(self.curvature_weights * curvature)
            local.extend(rows)
        self.calls += 1
        if self.calls % 50 == 0:
            print(f'Evaluation {self.calls}: cost={np.dot(local, local)/2:.6f}, elapsed={time.monotonic()-self.started:.1f}s', flush=True)
        return np.asarray(local)

    def jacobian(self, flat):
        self.state(flat)
        blocks, indices_r, indices_c = [], [], []
        for i, (data, mapping) in enumerate(zip(self.data, self.maps)):
            block = []
            for side in range(2):
                body = self.constraint_bodies[i, side]
                r = data.xmat[body].reshape(3, 3)
                point = data.xpos[body] + r @ self.material[i, side]
                mujoco.mj_jac(self.model, data, self.jacp, self.jacr, point, body)
                phi = Rotation.from_matrix(self.targets_r[i, side].T @ r).as_rotvec()
                angle, s = np.linalg.norm(phi), skew(phi)
                coef = 1/12 if angle < 1e-5 else (1-angle/2/np.tan(angle/2))/angle**2
                inverse = np.eye(3) - .5*s + coef*s@s
                block.extend(self.position_weights[i, side] * self.jacp @ mapping)
                block.extend(self.rotation_weights[i, side] * inverse @ self.targets_r[i, side].T @ self.jacr @ mapping)
            block.extend(np.diag(self.prior_weights))
            for column in self.mtp:
                row = np.zeros(self.columns)
                row[6 + column] = 3.
                block.append(row)
            for k, (body, point, gap) in enumerate(zip(self.bodies, self.bottoms[i], self.gaps[i])):
                clearance = self.ground_clearance if self.contact[i, k // 2] > .5 else .006
                if gap < clearance:
                    mujoco.mj_jac(self.model, data, self.jacp, None, point, body)
                    block.append(10000. * self.jacp[2] @ mapping)
                else:
                    block.append(np.zeros(self.columns))
            for body in self.bodies:
                forward = data.xmat[body].reshape(3, 3)[:, 0]
                horizontal_norm = forward[0]**2 + forward[1]**2
                yaw_gradient = np.array([-forward[0]*forward[2]/horizontal_norm,
                                         -forward[1]*forward[2]/horizontal_norm, 1.])
                mujoco.mj_jacBody(self.model, data, self.jacp, self.jacr, body)
                block.append(200. * yaw_gradient @ self.jacr @ mapping)
            for side, hip in enumerate(self.hip_bodies):
                mujoco.mj_jacBody(self.model, data, self.jacp, None, hip)
                hip_y_jacobian = self.jacp[1].copy()
                for body in (self.knee_bodies[side], self.ankle_bodies[side]):
                    mujoco.mj_jacBody(self.model, data, self.jacp, None, body)
                    block.append(500. * (self.jacp[1] - hip_y_jacobian) @ mapping)
            slip_row = len(block)
            for k, body in enumerate(self.bodies):
                block.extend(np.zeros((2, self.columns)))
                side = k // 2
                loaded = (self.contact[i, side] > .5
                          and self.gaps[i][k] <= min(self.gaps[i][2*side:2*side+2]) + .0015)
                if loaded:
                    local_point = self.bottom_local_points[i][k]
                    for neighbor, sign in (((i-1) % self.count, -1.), ((i+1) % self.count, 1.)):
                        neighbor_data = self.data[neighbor]
                        point = neighbor_data.xpos[body] + neighbor_data.xmat[body].reshape(3, 3) @ local_point
                        mujoco.mj_jac(self.model, neighbor_data, self.jacp, None, point, body)
                        derivative = sign * 80. * self.frequency / 2 * self.jacp[:2] @ self.maps[neighbor]
                        rr, cc = np.nonzero(derivative)
                        blocks.extend(derivative[rr, cc])
                        indices_r.extend(i*self.rows + slip_row + 2*k + rr)
                        indices_c.extend(neighbor*self.columns + cc)
            curvature_row = len(block)
            block.extend(-2 * np.diag(self.curvature_weights))
            block = np.asarray(block)
            rr, cc = np.nonzero(block)
            blocks.extend(block[rr, cc]); indices_r.extend(i*self.rows+rr); indices_c.extend(i*self.columns+cc)
            for neighbor in ((i - 1) % self.count, (i + 1) % self.count):
                blocks.extend(self.curvature_weights)
                indices_r.extend(i*self.rows+curvature_row+np.arange(self.columns))
                indices_c.extend(neighbor*self.columns+np.arange(self.columns))
        return coo_matrix((blocks, (indices_r, indices_c)), shape=(self.count*self.rows, self.count*self.columns)).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'robot/reference_data/walk1.npz')
    parser.add_argument('--seed', type=Path, default=ROOT/'logs/data_audit/walk1_contact_cycle_candidate.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'logs/data_audit/walk1_natural_contact_candidate.npz')
    parser.add_argument('--forward-speed', type=float,
                        help='Requested world-X speed in m/s; defaults to the seed speed.')
    parser.add_argument('--cycle-duration', type=float,
                        help='Duration of one right/left stride in seconds; defaults to the seed duration.')
    parser.add_argument('--ground-clearance', type=float, default=.0005,
                        help='Target stance clearance of the STL soles above the floor, in metres.')
    parser.add_argument('--frames-per-cycle', type=int,
                        help='Number of poses per cycle; cycle duration and cadence are preserved.')
    parser.add_argument('--max-evaluations', type=int, default=400)
    args = parser.parse_args()
    if args.output.resolve() in (args.source.resolve(), args.seed.resolve()):
        parser.error('Output must be separate from the source and seed motion archives')
    model = mujoco.MjModel.from_xml_path(str(ROOT/'robot/MJCFmodelfinal.xml'))
    with np.load(args.seed, allow_pickle=True) as a:
        fields = {k:a[k] for k in a.files}
    metadata = json.loads(str(fields['metadata']))
    seed_period = len(fields['qpos']) / float(fields['frequency'])
    period = seed_period if args.cycle_duration is None else args.cycle_duration
    speed = float(metadata['mean_forward_velocity']) if args.forward_speed is None else args.forward_speed
    if not np.isfinite(period) or period <= 0 or not np.isfinite(speed) or speed <= 0:
        parser.error('forward-speed and cycle-duration must be finite and positive')
    if not np.isfinite(args.ground_clearance) or args.ground_clearance < 0:
        parser.error('ground-clearance must be finite and nonnegative')
    count = len(fields['qpos']) if args.frames_per_cycle is None else args.frames_per_cycle
    if count < 4:
        parser.error('frames-per-cycle must be at least 4')
    frequency = count / period
    translation = np.array([speed * period, 0.], dtype=np.float64)
    seed_positions = fields['qpos'].astype(float)
    seed_positions[:, 0] *= translation[0] / float(fields['cycle_translation'][0])
    first, last = metadata['source_frames']
    with np.load(args.source, allow_pickle=True) as a:
        original = a['qpos'][first:last+1].astype(float)
        names = a['joint_names'].tolist()
    source = np.broadcast_to(model.qpos0, (len(original), model.nq)).copy()
    source[:, :7] = original[:, :7]
    for j in range(1, model.njnt):
        if model.joint(j).name in names:
            source[:, model.jnt_qposadr[j]] = original[:, 6+names.index(model.joint(j).name)]
    enforce_passive(model, source)
    if count != len(seed_positions):
        seed_positions = resample_poses(seed_positions, count, translation)
        source = resample_poses(source, count)
        fields['foot_contact'], fields['foot_roll_phase'], fields['foot_roll_mode'] = resample_contacts(
            fields['foot_contact'], count
        )
    fit = ContactFit(model, source, seed_positions, frequency,
                     fields['foot_contact'], fields['foot_roll_phase'], fields['foot_roll_mode'],
                     translation, ground_clearance=args.ground_clearance)
    lower, upper = fit.bounds()
    initial = np.clip(fit.initial.ravel(), lower+1e-8, upper-1e-8)
    solution = least_squares(fit.residual, initial, jac=fit.jacobian, bounds=(lower, upper),
                             max_nfev=args.max_evaluations, ftol=1e-8, xtol=1e-8, gtol=1e-8,
                             tr_options={'atol':1e-9, 'btol':1e-9})
    qpos = fit.decode(solution.x.reshape(fit.count, fit.columns))
    qpos[:, :2] -= qpos[0, :2]
    qpos = qpos.astype(np.float32)
    translation = fit.translation.astype(np.float32)
    qvel = cyclic_velocities(model, qpos.astype(float), fit.frequency, translation.astype(float)).astype(np.float32)
    for key in ('accepted_visual_preview', 'kinematic_audit', 'loader_validation', 'contact_ik',
                'comparison_1p28', 'original_archive_sha256', 'validation_protocol', 'generation_command'):
        metadata.pop(key, None)
    metadata.update(source=str(args.source.resolve()), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    optimization_seed=str(args.seed.resolve()), optimization_seed_sha256=hashlib.sha256(args.seed.read_bytes()).hexdigest(),
                    mean_forward_velocity=speed, cycle_duration_s=period,
                    frequency_hz=fit.frequency, cycle_translation=translation.tolist(),
                    cycle_frame_count=count,
                    reference_kind=f'One source-derived LYT walking cycle, {count} unique frames, translating and looping',
                    cadence_steps_per_minute=120/period,
                    target_stance_clearance_m=args.ground_clearance,
                    knee_extension={'stance_phase_start':.35, 'stance_phase_end':.70,
                                    'target_joint_angle_rad':0., 'maximum_error_rad':1e-6,
                                    'entry_phase_start':.15, 'exit_phase_end':.90},
                    forward_leg_alignment={'foot_and_toe_heading_target_rad':0.,
                                           'foot_lane_y_m':fit.rest_foot_lanes.tolist(),
                                           'knee_lateral_offset_from_hip_m':fit.rest_knee_offsets.tolist(),
                                           'ankle_lateral_offset_from_hip_m':fit.rest_ankle_offsets.tolist()},
                    minimum_knee_flexion_rad=None, method='Whole-cycle STL contact optimization with source knee motion, free ankle roll and periodic curvature regularization',
                    cycle_refinement={'optimizer_success':bool(solution.success), 'message':solution.message, 'function_evaluations':solution.nfev, 'cost':solution.cost},
                    velocity_method='Current-pose bidirectional MuJoCo differences')
    fields.update(qpos=qpos, qvel=qvel, frequency=fit.frequency,
                  cycle_translation=translation, metadata=json.dumps(metadata, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fields)
    print(json.dumps({'output':str(args.output), 'solution':metadata['cycle_refinement']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
