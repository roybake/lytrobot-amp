#!/usr/bin/env python3
"""Diagnose whether a stair-frame foot pose is reachable without rate limits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "src"), str(REPO / "multimotion/retargeting")]
from build_references import (  # noqa: E402
    FOOT_GEOM_NAMES, MAIN_JOINT_QPOS, TRAIN_STAIR_CFG,
    landmark_positions, sole_landmarks, terrain_geom_ids,
)
from lytrobot_amp.terrain import load_model  # noqa: E402


def inspect(frame: int, attempts: int, root_x: float | None = None) -> None:
    with np.load(Path(__file__).parent / "stair_up_down_candidate.npz") as archive:
        pose = np.asarray(archive["qpos"][frame], dtype=np.float64)
        target = np.asarray(archive["target_landmarks"][frame], dtype=np.float64)
    model = load_model(REPO / "robot/MJCFmodelposition.xml", TRAIN_STAIR_CFG)
    data = mujoco.MjData(model)
    _, bodies, points = sole_landmarks(model)
    terrain = terrain_geom_ids(model)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES]
    joint_by_qpos = {int(model.jnt_qposadr[j]): j for j in range(model.njnt)}
    joint_ids = [joint_by_qpos[int(a)] for a in MAIN_JOINT_QPOS]
    lower = np.r_[pose[:3] + [-0.35, -0.30, -0.30],
                  np.deg2rad([-15, -30]), model.jnt_range[joint_ids, 0]]
    upper = np.r_[pose[:3] + [0.35, 0.30, 0.30],
                  np.deg2rad([15, 30]), model.jnt_range[joint_ids, 1]]
    if root_x is not None:
        lower[0], upper[0] = root_x - 0.001, root_x + 0.001
    target_z = target[:, 2]
    scratch = np.zeros(6)

    def evaluate(v: np.ndarray) -> tuple[np.ndarray, float]:
        q = np.asarray(model.qpos0).copy()
        q[:3] = v[:3]
        roll, pitch = v[3:5]
        q[3:7] = (
            np.cos(roll / 2) * np.cos(pitch / 2),
            np.sin(roll / 2) * np.cos(pitch / 2),
            np.cos(roll / 2) * np.sin(pitch / 2),
            np.sin(roll / 2) * np.sin(pitch / 2),
        )
        q[MAIN_JOINT_QPOS] = v[5:]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        points_actual = landmark_positions(data, bodies, points)
        closest = min(
            mujoco.mj_geomDistance(model, data, int(g), int(foot), 0.25, scratch)
            for g in terrain for foot in feet
        )
        return points_actual - target, closest

    def residual(v: np.ndarray) -> np.ndarray:
        errors, dist = evaluate(v)
        return np.r_[
            (errors * [90, 90, 230]).ravel(),
            1000.0 * min(dist - 0.0003, 0),
            0.1 * (v[5:] - pose[MAIN_JOINT_QPOS]),
        ]

    # Root tilt from the free quaternion; zero is still a useful alternate seed.
    initial = np.r_[pose[:3], 0., 0., pose[MAIN_JOINT_QPOS]]
    rng = np.random.default_rng(12345 + frame)
    best = (float("inf"), None, None)
    for k in range(attempts):
        guess = initial.copy()
        guess[0] += ([0, 0.05, 0.1, 0.15, -0.05][k % 5]
                     if k < 10 else rng.uniform(-0.18, 0.18))
        guess[1:3] += rng.uniform(-0.07, 0.07, size=2)
        guess[3:5] = np.deg2rad(rng.uniform([-12, -26], [12, 12]))
        if k % 3 == 0:
            guess[5:] = rng.uniform(lower[5:], upper[5:])
        fit = least_squares(residual, np.clip(guess, lower, upper),
                            bounds=(lower, upper), max_nfev=250)
        errors, dist = evaluate(fit.x)
        max_z = float(np.max(np.abs(errors[:, 2])))
        max_xy = float(np.max(np.linalg.norm(errors[:, :2], axis=1)))
        score = max_z + 0.25 * max_xy + 20 * max(0, -dist)
        if score < best[0]:
            best = (score, (max_z, max_xy, dist), fit.x)
            print(frame, k, "z/xy/geom", best[1], "root", np.round(fit.x[:3], 3), flush=True)
        if max_z < 0.005 and max_xy < 0.02 and dist >= -1e-4:
            break
    print("best", frame, best[1], "limits unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frame", type=int)
    parser.add_argument("--attempts", type=int, default=15)
    parser.add_argument("--root-x", type=float)
    args = parser.parse_args()
    inspect(args.frame, args.attempts, args.root_x)
