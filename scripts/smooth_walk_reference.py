#!/usr/bin/env python3
"""Smooth an accepted walking reference with bounded pose changes and contacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_walk_contacts import ContactFit, cyclic_velocities


class ReferenceSmoother(ContactFit):
    def __init__(self, model, fields, strength, max_joint_deviation, max_root_deviation,
                 max_root_angle_deviation):
        reference = fields['qpos'].astype(float)
        metadata = json.loads(str(fields['metadata']))
        super().__init__(model, reference, reference, float(fields['frequency']),
                         fields['foot_contact'], fields['foot_roll_phase'], fields['foot_roll_mode'],
                         fields['cycle_translation'].astype(float),
                         ground_clearance=float(metadata['target_stance_clearance_m']))
        self.nominal = self.initial.copy()
        self.prior_weights = np.r_[np.full(3, 1000.), np.full(3, 100.),
                                   np.full(len(self.hinges), 100.)]
        self.curvature_weights = np.sqrt(strength) * self.prior_weights
        self.pose_limits = np.r_[np.full(3, max_root_deviation),
                                 np.full(3, max_root_angle_deviation),
                                 np.full(len(self.hinges), max_joint_deviation)]
        # Use the accepted body's actual transforms as the positional targets.
        self.state(self.initial.ravel())
        for i, data in enumerate(self.data):
            for side in range(2):
                body = self.constraint_bodies[i, side]
                rotation = data.xmat[body].reshape(3, 3)
                self.targets_p[i, side] = data.xpos[body] + rotation @ self.material[i, side]
                self.targets_r[i, side] = rotation
                self.position_weights[i, side] = 3000. if self.contact[i, side] > .5 else 750.
                self.rotation_weights[i, side] = 400. if self.contact[i, side] > .5 else 100.
                self.targets_toe[i, side] = self.initial[i, 6 + self.mtp[side]]

    def bounds(self):
        lower, upper = super().bounds()
        lower = np.maximum(lower, (self.initial - self.pose_limits).ravel())
        upper = np.minimum(upper, (self.initial + self.pose_limits).ravel())
        return lower, upper

    @property
    def rows(self):
        return super().rows + 2

    def residual(self, flat):
        base = super().residual(flat).reshape(self.count, super().rows)
        upper_gap = np.zeros((self.count, 2))
        for i in range(self.count):
            for side in range(2):
                if self.contact[i, side] > .5:
                    gap = min(self.gaps[i][2*side:2*side+2])
                    upper_gap[i, side] = 50000. * max(gap - .0008, 0.)
        return np.column_stack((base, upper_gap)).ravel()

    def jacobian(self, flat):
        base = super().jacobian(flat).tocoo()
        values, rows, columns = list(base.data), list(base.row), list(base.col)
        for i, (data, mapping) in enumerate(zip(self.data, self.maps)):
            for side in range(2):
                k = 2*side + int(self.gaps[i][2*side+1] < self.gaps[i][2*side])
                if self.contact[i, side] > .5 and self.gaps[i][k] > .0008:
                    body, point = self.bodies[k], self.bottoms[i][k]
                    mujoco.mj_jac(self.model, data, self.jacp, None, point, body)
                    derivative = 50000. * self.jacp[2] @ mapping
                    indices = np.flatnonzero(derivative)
                    values.extend(derivative[indices])
                    rows.extend(np.full(len(indices), i*self.rows + super().rows + side))
                    columns.extend(i*self.columns + indices)
        return coo_matrix((values, (rows, columns)), shape=base.shape).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strength', type=float, default=.25)
    parser.add_argument('--max-joint-deviation-deg', type=float, default=2.)
    parser.add_argument('--max-root-deviation-mm', type=float, default=3.)
    parser.add_argument('--max-root-angle-deviation-deg', type=float, default=.2,
                        help='Maximum change of each root Euler angle in degrees.')
    parser.add_argument('--max-evaluations', type=int, default=1000)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Output must be separate from the accepted reference')
    if args.strength <= 0 or min(args.max_joint_deviation_deg, args.max_root_deviation_mm,
                                args.max_root_angle_deviation_deg) <= 0:
        parser.error('Smoothing strength and pose-change limits must be positive')
    with np.load(args.input) as archive:
        fields = {key: archive[key] for key in archive.files}
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'robot/MJCFmodelfinal.xml'))
    smoother = ReferenceSmoother(model, fields, args.strength,
                                 np.deg2rad(args.max_joint_deviation_deg),
                                 args.max_root_deviation_mm / 1000.,
                                 np.deg2rad(args.max_root_angle_deviation_deg))
    lower, upper = smoother.bounds()
    initial = np.clip(smoother.initial.ravel(), lower + 1e-10, upper - 1e-10)
    solution = least_squares(smoother.residual, initial, jac=smoother.jacobian,
                             bounds=(lower, upper), max_nfev=args.max_evaluations,
                             ftol=1e-8, xtol=1e-8, gtol=1e-8,
                             tr_options={'atol':1e-9, 'btol':1e-9})
    positions = smoother.decode(solution.x.reshape(smoother.count, smoother.columns)).astype(np.float32)
    # Keep the reference's global origin so pose comparisons include translation.
    velocities = cyclic_velocities(model, positions.astype(float), smoother.frequency,
                                    fields['cycle_translation'].astype(float)).astype(np.float32)
    metadata = json.loads(str(fields['metadata']))
    for key in ('kinematic_audit', 'loader_validation', 'generation_command', 'accepted_visual_preview'):
        metadata.pop(key, None)
    metadata['smoothness_revision'] = {
        'accepted_reference': str(args.input.resolve()),
        'accepted_reference_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'strength': args.strength,
        'max_joint_deviation_deg': args.max_joint_deviation_deg,
        'max_root_deviation_mm': args.max_root_deviation_mm,
        'max_root_angle_deviation_deg': args.max_root_angle_deviation_deg,
        'root_angle_limit_definition': 'Maximum absolute change of each xyz Euler angle',
        'stance_gap_penalty_threshold_m': .0008,
        'optimizer_success': bool(solution.success),
        'message': solution.message, 'evaluations': solution.nfev, 'cost': solution.cost,
    }
    fields.update(qpos=positions, qvel=velocities, metadata=json.dumps(metadata, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fields)
    print(json.dumps({'output':str(args.output), 'smoothness_revision':metadata['smoothness_revision']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
