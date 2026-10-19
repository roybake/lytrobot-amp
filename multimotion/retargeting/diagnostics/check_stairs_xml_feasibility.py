#!/usr/bin/env python3
"""Bounded, local IK diagnostics on the unchanged source stair XML.

This is a kinematic reachability search, not a proof of impossibility or a
reference-motion generator. The robot's original joint ranges are hard bounds.
Run with an environment providing numpy, scipy, and mujoco >= 3.3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "robot/MJCFmodelfinal.xml"
TERRAIN = REPO / "robot/terrains/stairs.xml"
DEFAULT_REPORT = Path(__file__).with_name("stairs_xml_20cm_feasibility.json")
FOOT_NAMES = ("foot_r", "toe_r", "foot_l", "toe_l")
JOINT_NAMES = tuple(f"{joint}_{side}" for side in ("r", "l") for joint in (
    "hip_flexion", "hip_adduction", "hip_rotation", "knee_angle",
    "ankle_angle", "subtalar_angle", "mtp_angle",
))


def load_scene() -> tuple[mujoco.MjModel, np.ndarray]:
    spec = mujoco.MjSpec.from_file(str(MODEL))
    names = []
    for element in ET.parse(TERRAIN).findall("worldbody/geom"):
        name = "audit_source_" + element.attrib["name"]
        names.append(name)
        spec.worldbody.add_geom(
            name=name,
            type={"box": mujoco.mjtGeom.mjGEOM_BOX,
                  "plane": mujoco.mjtGeom.mjGEOM_PLANE}[element.attrib["type"]],
            size=np.fromstring(element.attrib["size"], sep=" "),
            pos=np.fromstring(element.get("pos", "0 0 0"), sep=" "),
            contype=1, conaffinity=1, group=2,
        )
    model = spec.compile()
    return model, np.asarray([mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names])


def local_mesh_vertices(model: mujoco.MjModel, geom: int) -> np.ndarray:
    mesh = int(model.geom_dataid[geom])
    start = int(model.mesh_vertadr[mesh])
    return np.asarray(model.mesh_vert[start:start + model.mesh_vertnum[mesh]])


def sole_landmarks(model: mujoco.MjModel, feet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bodies = model.geom_bodyid[feet]
    points = []
    for name, geom in zip(FOOT_NAMES, feet, strict=True):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        vertices = model.geom_pos[geom] + local_mesh_vertices(model, geom) @ rotation.reshape(3, 3).T
        if name.startswith("foot"):
            vertices = vertices[vertices[:, 0] <= vertices[:, 0].min() + .25 * np.ptp(vertices[:, 0])]
        lowest = vertices[:, 2].min()
        point = vertices[vertices[:, 2] <= lowest + .0005].mean(axis=0)
        point[2] = lowest
        points.append(point)
    return bodies, np.asarray(points)


def world_landmarks(data: mujoco.MjData, bodies: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray([data.xpos[b] + data.xmat[b].reshape(3, 3) @ p
                       for b, p in zip(bodies, points, strict=True)])


def hull_equations(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        return None
    return ConvexHull(points).equations


def hull_margin(equations: np.ndarray | None, point: np.ndarray) -> float | None:
    """Positive is inside; negative is the largest violated half-plane distance."""
    if equations is None:
        return None
    return -float(np.max(equations[:, :2] @ point[:2] + equations[:, 2]))


def diagnose(attempts: int, max_nfev: int) -> dict:
    model, terrain_ids = load_scene()
    data = mujoco.MjData(model)
    feet = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_NAMES])
    joints = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES])
    addresses = model.jnt_qposadr[joints]
    bodies, local_points = sole_landmarks(model, feet)
    mujoco.mj_forward(model, data)
    neutral = world_landmarks(data, bodies, local_points).reshape(2, 2, 3)
    centres = neutral.mean(axis=1)
    offsets = neutral - centres[:, None]
    offsets[:, :, 2] = 0.
    foot_vertices = [data.geom_xpos[g] + local_mesh_vertices(model, g) @ data.geom_xmat[g].reshape(3, 3).T for g in feet]
    scratch = np.zeros(6)
    rng = np.random.default_rng(20260922)
    # Infer stair dimensions and positions from the user's file, not terrain.py.
    boxes = [g for g in ET.parse(TERRAIN).findall("worldbody/geom") if g.get("type") == "box"]
    boxes.sort(key=lambda g: float(g.attrib["pos"].split()[0]))
    first_pos = np.fromstring(boxes[0].attrib["pos"], sep=" ")
    first_size = np.fromstring(boxes[0].attrib["size"], sep=" ")
    last_pos = np.fromstring(boxes[-1].attrib["pos"], sep=" ")
    last_size = np.fromstring(boxes[-1].attrib["size"], sep=" ")
    front, depth, height = float(first_pos[0]), float(2 * first_size[0]), float(first_pos[2] + first_size[2])
    rear = front - depth
    cases = (
        ("ascent_adjacent_flat_feet", [front, rear], [height, 0.], [0., 0.], (0, 1)),
        ("ascent_swing_over_riser", [rear + .5625 * depth, rear], [height + .03, 0.], [0., 0.], (1,)),
        ("ascent_rear_heel_raised", [front, rear], [height, 0.], [0., .04], (0, 1)),
        ("descent_adjacent_flat_feet", [float(last_pos[0] + 2 * last_size[0]), float(last_pos[0])], [0., float(last_pos[2] + last_size[2])], [0., 0.], (0, 1)),
    )
    profiles = {"wide_posture_search": (15., 30., .20, False),
                "moderate_posture_and_support_com": (8., 15., .12, True)}
    results = {}
    for case_index, (case_name, xs, zs, heel_raise, support) in enumerate(cases):
        targets = np.asarray([offsets[side] + [xs[side], [-.075, .075][side], zs[side] + .003] for side in range(2)])
        for side in range(2):
            targets[side, 0, 2] += heel_raise[side]
        targets = targets.reshape(4, 3)
        planned_vertices = []
        for side in support:
            for i in (2 * side, 2 * side + 1):
                if heel_raise[side] > .02 and i % 2 == 0:
                    continue
                planned_vertices.extend(foot_vertices[i][:, :2] - centres[side, :2] + [xs[side], [-.075, .075][side]])
        planned_hull = hull_equations(np.asarray(planned_vertices))
        results[case_name] = {}
        for profile_name, (roll_bound, pitch_bound, y_bound, constrain_com) in profiles.items():
            lower = np.r_[[min(xs) - .55, -y_bound, min(zs) + .45], np.deg2rad([-roll_bound, -pitch_bound]), model.jnt_range[joints, 0]]
            upper = np.r_[[max(xs) + .45, y_bound, max(zs) + 1.1], np.deg2rad([roll_bound, pitch_bound]), model.jnt_range[joints, 1]]

            def install(v: np.ndarray) -> None:
                data.qpos[:] = model.qpos0
                data.qpos[:3] = v[:3]
                r, p = v[3:5]
                data.qpos[3:7] = (np.cos(r / 2) * np.cos(p / 2), np.sin(r / 2) * np.cos(p / 2), np.cos(r / 2) * np.sin(p / 2), np.sin(r / 2) * np.sin(p / 2))
                data.qpos[addresses] = v[5:]
                for side in ("r", "l"):
                    patella = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"patella_{side}_j1")
                    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"knee_angle_{side}")
                    data.qpos[model.jnt_qposadr[patella]] = .005 + .012 * data.qpos[model.jnt_qposadr[knee]]
                mujoco.mj_forward(model, data)

            def distances() -> np.ndarray:
                return np.asarray([min(mujoco.mj_geomDistance(model, data, int(t), int(f), .20, scratch) for t in terrain_ids) for f in feet])

            def residual(v: np.ndarray) -> np.ndarray:
                install(v)
                parts = [((world_landmarks(data, bodies, local_points) - targets) * [120., 120., 300.]).ravel(), 1000. * np.minimum(distances() - .0005, 0.), .01 * v[3:]]
                if constrain_com:
                    com = data.subtree_com[0, :2]
                    parts.append(200. * np.maximum(planned_hull[:, :2] @ com + planned_hull[:, 2] - .005, 0.))
                return np.concatenate(parts)

            best = None
            for attempt in range(attempts):
                seed = np.r_[[np.mean(xs) - .10, 0., .86 + np.mean(zs)], np.deg2rad([0., -min(20., pitch_bound)]), np.zeros(14)]
                if attempt:
                    seed = np.r_[[np.mean(xs) + rng.uniform(-.4, .15), rng.uniform(-y_bound, y_bound), min(zs) + rng.uniform(.68, 1.1)], np.deg2rad([rng.uniform(-roll_bound, roll_bound), rng.uniform(-pitch_bound, pitch_bound)]), rng.uniform(model.jnt_range[joints, 0], model.jnt_range[joints, 1])]
                solution = least_squares(residual, np.clip(seed, lower + 1e-8, upper - 1e-8), bounds=(lower, upper), max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                cost = float(np.linalg.norm(residual(solution.x)))
                if best is None or cost < best[0]:
                    best = (cost, solution.x.copy(), bool(solution.success))
            cost, v, converged = best
            install(v)
            com = data.subtree_com[0].copy()
            errors = world_landmarks(data, bodies, local_points) - targets
            actual_contact_vertices = []
            for side in support:
                for i in (2 * side, 2 * side + 1):
                    geom = feet[i]
                    vertices = data.geom_xpos[geom] + local_mesh_vertices(model, geom) @ data.geom_xmat[geom].reshape(3, 3).T
                    actual_contact_vertices.extend(vertices[np.abs(vertices[:, 2] - zs[side]) <= .010, :2])
            actual_hull = hull_equations(np.asarray(actual_contact_vertices))
            joint_values = v[5:]
            joint_margin = np.minimum(joint_values - model.jnt_range[joints, 0], model.jnt_range[joints, 1] - joint_values)
            closest = (np.inf, -1, -1)
            for geom in np.flatnonzero((model.geom_bodyid > 0) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)):
                for terrain in terrain_ids:
                    distance = mujoco.mj_geomDistance(model, data, int(geom), int(terrain), .20, scratch)
                    if distance < closest[0]:
                        closest = (float(distance), int(geom), int(terrain))
            result = {
                "maximum_foot_landmark_error_m": float(np.linalg.norm(errors, axis=1).max()),
                "landmark_errors_xyz_m": errors.tolist(),
                "minimum_foot_terrain_distance_m": float(distances().min()),
                "minimum_all_body_mesh_terrain_distance_m": closest[0],
                "closest_robot_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[closest[1]])),
                "closest_terrain_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, closest[2]),
                "root_xyz_m": v[:3].tolist(), "root_roll_pitch_deg": np.rad2deg(v[3:5]).tolist(),
                "subtree_com_xyz_m": com.tolist(),
                "planned_support_hull_margin_m": hull_margin(planned_hull, com),
                "actual_contact_hull_margin_m": hull_margin(actual_hull, com),
                "contact_hull_vertex_height_tolerance_m": .010,
                "joint_angles_deg": dict(zip(JOINT_NAMES, np.rad2deg(joint_values).tolist())),
                "joint_limit_violation_rad": float(np.maximum(-joint_margin, 0.).max()),
                "joints_within_half_degree_of_limit": [name for name, margin in zip(JOINT_NAMES, joint_margin, strict=True) if margin <= np.deg2rad(.5)],
                "root_roll_limit_active": bool(abs(v[3]) >= np.deg2rad(roll_bound - .1)),
                "root_pitch_limit_active": bool(abs(v[4]) >= np.deg2rad(pitch_bound - .1)),
                "best_weighted_residual_norm": cost, "optimizer_reported_convergence": converged,
                "qpos": data.qpos.tolist(), "target_landmarks": targets.tolist(),
                "support_sides_right_zero_left_one": list(support),
            }
            results[case_name][profile_name] = result
            print(f"{case_name}/{profile_name}: foot_error={result['maximum_foot_landmark_error_m']:.5f}m mesh_distance={closest[0]:.5f}m COM_margin={result['actual_contact_hull_margin_m']} roll/pitch={np.rad2deg(v[3:5]).round(2).tolist()}", flush=True)
    return {
        "purpose": "Local kinematic diagnostic only; candidates are not approved motion references.",
        "limitations": ["Multi-start local least-squares cannot prove global infeasibility.", "COM inside the support hull is a quasi-static proxy, not dynamic balance or actuator feasibility.", "Contact hull uses mesh vertices within 10 mm of the intended tread height; tilted/noncontact feet can leave no hull.", "All joint bounds are original model values. COM support is a weighted residual and must be checked after optimization.", "No temporal smoothness, speed, dynamics, or self-collision acceptance is claimed."],
        "model": str(MODEL.relative_to(REPO)), "terrain": str(TERRAIN.relative_to(REPO)),
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "terrain_sha256": hashlib.sha256(TERRAIN.read_bytes()).hexdigest(),
        "step_height_m": height, "step_depth_m": depth, "mujoco_version": mujoco.__version__,
        "attempts_per_case_profile": attempts, "max_nfev_per_attempt": max_nfev,
        "profiles": {name: {"roll_bound_deg": p[0], "pitch_bound_deg": p[1], "root_lateral_bound_m": p[2], "com_support_residual": p[3]} for name, p in profiles.items()},
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--max-nfev", type=int, default=150)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = diagnose(args.attempts, args.max_nfev)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved diagnostic report: {args.output}")


if __name__ == "__main__":
    main()
