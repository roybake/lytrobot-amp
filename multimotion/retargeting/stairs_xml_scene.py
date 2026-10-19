#!/usr/bin/env python3
"""Load and audit the unmodified robot on the exact supplied stair XML.

The source files are only read.  Terrain assets and world geometries are
merged in memory, replacing the display scene's floor.  Explicit geometry
distance queries include robot meshes whose contact masks are disabled.
MuJoCo computes mesh distances using its convex collision representation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "robot/MJCFmodel_stair.xml"
DEFAULT_TERRAIN = ROOT / "robot/terrains/stairs.xml"
FOOT_NAMES = ("foot_r", "toe_r", "foot_l", "toe_l")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expanded_xml(path: Path) -> ET.Element:
    """Expand includes and resolve file paths without changing robot values."""
    root = ET.parse(path).getroot()
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "include":
                included = _expanded_xml((path.parent / child.attrib["file"]).resolve())
                index = list(parent).index(child)
                parent.remove(child)
                for offset, node in enumerate(list(included)):
                    parent.insert(index + offset, node)
            elif "file" in child.attrib:
                child.set("file", str((path.parent / child.attrib["file"]).resolve()))
    return root


def scene_xml(model_xml: str | Path = DEFAULT_MODEL,
              terrain_xml: str | Path = DEFAULT_TERRAIN) -> str:
    """Return the exact robot and source terrain as a self-contained XML string."""
    robot = _expanded_xml(Path(model_xml).expanduser().resolve())
    terrain_path = Path(terrain_xml).expanduser().resolve()
    # Terrain XML attaches the robot for direct viewing. Merge only its
    # own assets and world geometry so the robot is included exactly once.
    terrain = ET.parse(terrain_path).getroot()
    compiler = terrain.find("compiler")
    mesh_dir = terrain_path.parent / compiler.get("meshdir", "")
    # Existing static geometries belong to the robot's display scene.
    # Keep all robot bodies, including their joints, tendons and actuators.
    for world in robot.findall("worldbody"):
        for child in list(world):
            if child.tag == "geom":
                world.remove(child)
    assets = ET.SubElement(robot, "asset")
    asset_names: dict[str, str] = {}
    for source in terrain.findall("asset"):
        for item in source:
            if item.tag == "model":
                continue
            copied = deepcopy(item)
            if "file" in copied.attrib:
                asset_dir = mesh_dir if item.tag in ("mesh", "hfield") else terrain_path.parent
                copied.set("file", str((asset_dir / copied.attrib["file"]).resolve()))
            if "name" in copied.attrib:
                name = copied.attrib["name"]
                asset_names[name] = "stairs_xml_" + name
                copied.set("name", asset_names[name])
            assets.append(copied)
    for node in assets.iter():
        for key in ("texture", "material", "mesh", "hfield"):
            if node.get(key) in asset_names:
                node.set(key, asset_names[node.attrib[key]])
    world = ET.SubElement(robot, "worldbody")
    for source in terrain.findall("worldbody"):
        for item in source:
            if item.tag == "attach":
                continue
            copied = deepcopy(item)
            for node in copied.iter():
                for key in ("texture", "material", "mesh", "hfield"):
                    if node.get(key) in asset_names:
                        node.set(key, asset_names[node.attrib[key]])
                if node.tag == "geom":
                    # Terrain has MuJoCo defaults, whereas the robot disables
                    # collision globally. Make the terrain defaults explicit.
                    for key, value in {"contype": "1", "conaffinity": "1",
                                       "condim": "3", "margin": "0",
                                       "friction": "1 0.005 0.0001"}.items():
                        if key not in node.attrib:
                            node.set(key, value)
                    node.set("group", "2")
                    if "rgba" not in node.attrib:
                        node.set("rgba", "0.5 0.5 0.5 1")
            world.append(copied)
    return ET.tostring(robot, encoding="unicode")


def load_stairs_scene(model_xml: str | Path = DEFAULT_MODEL,
                      terrain_xml: str | Path = DEFAULT_TERRAIN) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(scene_xml(model_xml, terrain_xml))


def read_stair_treads(terrain_xml: str | Path = DEFAULT_TERRAIN) -> list[dict]:
    """Read actual axis-aligned tread extents, in source world coordinates."""
    result = []
    for geom in ET.parse(terrain_xml).getroot().findall("./worldbody/geom"):
        if geom.get("type") != "box":
            continue
        if any(key in geom.attrib for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError("Tread extraction requires axis-aligned source boxes.")
        x, y, z = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
        sx, sy, sz = np.fromstring(geom.attrib["size"], sep=" ")
        result.append(dict(name=geom.get("name", ""), x_min=float(x-sx),
                           x_max=float(x+sx), y_min=float(y-sy), y_max=float(y+sy),
                           height=float(z+sz)))
    return sorted(result, key=lambda item: item["x_min"])


def terrain_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.flatnonzero(model.geom_bodyid == 0).astype(np.int32)


def robot_mesh_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.flatnonzero((model.geom_bodyid != 0) &
                         (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)).astype(np.int32)


def contralateral_geom_pairs(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """All opposite-leg meshes; no adjacent same-leg anatomy is compared."""
    sides = {"r": [], "l": []}
    for geom in robot_mesh_geom_ids(model):
        body_name = model.body(int(model.geom_bodyid[geom])).name
        for side in sides:
            if body_name.endswith("_" + side):
                sides[side].append(int(geom))
    return [(right, left) for right in sides["r"] for left in sides["l"]]


def _geom_name(model: mujoco.MjModel, geom: int) -> str:
    return model.geom(geom).name or model.body(int(model.geom_bodyid[geom])).name


def signed_clearances(model: mujoco.MjModel, data: mujoco.MjData,
                      threshold: float = 0.30) -> dict:
    """Query every robot mesh against terrain and every opposite-leg pair."""
    scratch = np.zeros(6, dtype=np.float64)
    terrain_min = self_min = float(threshold)
    terrain_pair = self_pair = None
    for robot_geom in robot_mesh_geom_ids(model):
        for terrain_geom in terrain_geom_ids(model):
            distance = float(mujoco.mj_geomDistance(model, data, int(robot_geom),
                                                   int(terrain_geom), threshold, scratch))
            if distance < terrain_min:
                terrain_min = distance
                terrain_pair = [_geom_name(model, int(robot_geom)), _geom_name(model, int(terrain_geom))]
    for right, left in contralateral_geom_pairs(model):
        distance = float(mujoco.mj_geomDistance(model, data, right, left, threshold, scratch))
        if distance < self_min:
            self_min, self_pair = distance, [_geom_name(model, right), _geom_name(model, left)]
    return dict(terrain_min_m=terrain_min, self_min_m=self_min,
                terrain_pair=terrain_pair, self_pair=self_pair)


def qvel_from_qpos(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    result = np.zeros((len(qpos), model.nv), dtype=np.float64)
    if len(qpos) < 2:
        return result
    for frame in range(len(qpos)):
        low, high = max(0, frame-1), min(len(qpos)-1, frame+1)
        mujoco.mj_differentiatePos(model, result[frame], (high-low)/fps, qpos[low], qpos[high])
    return result


def interpolate_pose(model: mujoco.MjModel, first: np.ndarray,
                     second: np.ndarray, blend: float) -> np.ndarray:
    velocity = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, velocity, 1.0, first, second)
    pose = np.array(first, dtype=np.float64, copy=True)
    mujoco.mj_integratePos(model, pose, velocity, float(blend))
    return pose


def _mesh_landmarks(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Bottom heel/toe landmarks, expressed in their actual STL body frames."""
    bodies, landmarks = [], []
    for name in FOOT_NAMES:
        geom = model.geom(name).id
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        vertices = np.asarray(model.mesh_vert[start:start+int(model.mesh_vertnum[mesh])], dtype=np.float64)
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, model.geom_quat[geom])
        vertices = vertices @ matrix.reshape(3, 3).T + model.geom_pos[geom]
        if name.startswith("foot"):
            vertices = vertices[vertices[:, 0] <= vertices[:, 0].min()+0.25*np.ptp(vertices[:, 0])]
        bottom = vertices[vertices[:, 2] <= vertices[:, 2].min()+0.0005].mean(axis=0)
        bottom[2] = vertices[:, 2].min()
        bodies.append(int(model.geom_bodyid[geom]))
        landmarks.append(bottom)
    return np.asarray(bodies), np.asarray(landmarks)


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return dict(max=float(values.max()), mean=float(values.mean()),
                p95=float(np.percentile(values, 95))) if values.size else dict(max=0.0, mean=0.0, p95=0.0)


def validate_reference(npz_path: str | Path, *, model_xml: str | Path = DEFAULT_MODEL,
                       terrain_xml: str | Path = DEFAULT_TERRAIN,
                       report_path: str | Path | None = None, subframes: int = 2,
                       penetration_tolerance: float = 1e-4) -> dict:
    """Audit saved poses plus interpolation samples; return a JSON-ready report.

    This is a kinematic audit, not a claim of dynamic tracking stability.
    ``subframes`` samples each open frame interval at evenly spaced times.
    """
    if subframes < 0:
        raise ValueError("subframes must be nonnegative")
    model = load_stairs_scene(model_xml, terrain_xml)
    data = mujoco.MjData(model)
    with np.load(npz_path, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        fps = float(archive["fps"])
        stored_qvel = np.asarray(archive["qvel"], dtype=np.float64) if "qvel" in archive.files else None
        contact_key = next((key for key in ("foot_contact", "foot_contacts", "contacts") if key in archive.files), None)
        contacts = np.asarray(archive[contact_key], dtype=bool) if contact_key else None
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) < 2:
        raise ValueError(f"Expected at least two poses with nq={model.nq}; got {qpos.shape}")
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Nonfinite poses or invalid fps")
    if contacts is not None and contacts.shape != (len(qpos), 2):
        raise ValueError(f"Expected contact labels (frames, right/left); got {contacts.shape}")
    if stored_qvel is not None and (stored_qvel.shape != (len(qpos), model.nv) or not np.isfinite(stored_qvel).all()):
        raise ValueError(f"Invalid qvel; expected shape {(len(qpos), model.nv)}")
    qvel = qvel_from_qpos(model, qpos, fps)
    bodies, local_points = _mesh_landmarks(model)
    landmarks = np.zeros((len(qpos), 4, 3))
    terrain_min = self_min = np.inf
    terrain_worst = self_worst = None
    terrain_failures = self_failures = samples = 0
    for frame in range(len(qpos)):
        blends = [0.0] if frame == len(qpos)-1 else np.arange(subframes+1)/(subframes+1)
        for blend in blends:
            data.qpos[:] = qpos[frame] if blend == 0 else interpolate_pose(model, qpos[frame], qpos[frame+1], float(blend))
            mujoco.mj_forward(model, data)
            metrics = signed_clearances(model, data)
            samples += 1
            if metrics["terrain_min_m"] < terrain_min:
                terrain_min = metrics["terrain_min_m"]
                terrain_worst = dict(frame=float(frame+blend), pair=metrics["terrain_pair"])
            if metrics["self_min_m"] < self_min:
                self_min = metrics["self_min_m"]
                self_worst = dict(frame=float(frame+blend), pair=metrics["self_pair"])
            terrain_failures += metrics["terrain_min_m"] < -penetration_tolerance
            self_failures += metrics["self_min_m"] < -penetration_tolerance
            if blend == 0:
                landmarks[frame] = np.asarray([data.xpos[body]+data.xmat[body].reshape(3, 3)@point
                                               for body, point in zip(bodies, local_points, strict=True)])
    violations = {}
    hinge_velocity_columns = []
    for joint in range(model.njnt):
        if model.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            address = int(model.jnt_qposadr[joint])
            hinge_velocity_columns.append(int(model.jnt_dofadr[joint]))
            if model.jnt_limited[joint]:
                low, high = model.jnt_range[joint]
                violation = float(max(0.0, low-qpos[:, address].min(), qpos[:, address].max()-high))
                if violation > 1e-6:
                    violations[model.joint(joint).name] = violation
    norm_error = float(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1)-1)))
    equality_errors = {}
    for side in ("r", "l"):
        patella = int(model.joint(f"patella_{side}_j1").qposadr[0])
        knee = int(model.joint(f"knee_angle_{side}").qposadr[0])
        equality_errors[side] = float(np.max(np.abs(
            qpos[:, patella] - (.005 + .012 * qpos[:, knee])
        )))
    # MuJoCo quaternions use wxyz. These closed-form angles are sufficient for
    # rejecting large pelvis lean in this yaw-aligned stair task.
    w, x, y, z = qpos[:, 3:7].T
    root_roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    root_pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1., 1.))
    slip = {}
    if contacts is not None:
        for side, name in enumerate(("right", "left")):
            foot = landmarks[:, 2*side:2*side+2].mean(axis=1)
            consecutive = contacts[1:, side] & contacts[:-1, side]
            speed = np.linalg.norm(np.diff(foot[:, :2], axis=0), axis=1)*fps
            slip[name] = _summary(speed[consecutive])
            runs = np.flatnonzero(np.diff(np.r_[False, contacts[:, side], False]))
            drifts = [np.linalg.norm(foot[start:end, :2]-foot[start, :2], axis=1).max()
                      for start, end in zip(runs[::2], runs[1::2], strict=True)]
            slip[name]["max_contact_bout_drift_m"] = float(max(drifts, default=0.0))
    root_speed = np.linalg.norm(qvel[:, :3], axis=1)
    joint_speed = np.abs(qvel[:, hinge_velocity_columns])
    report = dict(
        reference=str(Path(npz_path).resolve()), model_xml=str(Path(model_xml).resolve()),
        terrain_xml=str(Path(terrain_xml).resolve()), reference_sha256=sha256_file(npz_path),
        model_xml_sha256=sha256_file(model_xml), terrain_xml_sha256=sha256_file(terrain_xml),
        mujoco_version=mujoco.__version__, nq=model.nq, nv=model.nv, nu=model.nu,
        frame_count=len(qpos), fps=fps, duration_s=(len(qpos)-1)/fps,
        treads=read_stair_treads(terrain_xml), collision_sample_count=samples,
        interpolation_subframes=subframes, collision_method="MuJoCo signed geom distance; compiled STL convex hulls; masks ignored",
        robot_mesh_count=len(robot_mesh_geom_ids(model)), opposite_leg_pair_count=len(contralateral_geom_pairs(model)),
        penetration_tolerance_m=penetration_tolerance,
        terrain_min_signed_distance_m=float(terrain_min), terrain_worst=terrain_worst,
        self_min_signed_distance_m=float(self_min), self_worst=self_worst,
        terrain_penetrating_samples=int(terrain_failures), self_penetrating_samples=int(self_failures),
        joint_limit_violations=violations, quaternion_norm_max_error=norm_error,
        equality_max_error_m=equality_errors,
        root_roll_abs_max_deg=float(np.rad2deg(np.max(np.abs(root_roll)))),
        root_pitch_abs_max_deg=float(np.rad2deg(np.max(np.abs(root_pitch)))),
        root_speed_m_s=_summary(root_speed), joint_speed_rad_s=_summary(joint_speed.ravel()),
        root_mean_forward_speed_m_s=float((qpos[-1, 0]-qpos[0, 0])/((len(qpos)-1)/fps)),
        stance_landmark_planar_speed_m_s=slip,
        qvel_present=stored_qvel is not None,
        qvel_vs_central_difference_max_error=float(np.max(np.abs(stored_qvel-qvel))) if stored_qvel is not None else None,
        kinematic_checks_passed=bool(not terrain_failures and not self_failures and
                                     not violations and norm_error < 1e-5 and
                                     max(equality_errors.values()) < 1e-5),
        dynamics_validated=False,
    )
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--terrain-xml", type=Path, default=DEFAULT_TERRAIN)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--subframes", type=int, default=2)
    args = parser.parse_args()
    report = validate_reference(args.reference, model_xml=args.model_xml, terrain_xml=args.terrain_xml,
                                report_path=args.report, subframes=args.subframes)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["kinematic_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
