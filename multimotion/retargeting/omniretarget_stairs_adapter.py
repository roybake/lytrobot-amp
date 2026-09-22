"""Official OmniRetarget SQP with the exact LYT model and XML stair scene.

Only the boundary to the vendored optimizer is adapted here. In particular,
joint limits are indexed by qpos address (the upstream constructor assumes an
unnamed free joint), and collision queries include meshes whose simulation
collision masks are disabled. No robot joint limit is widened.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = ROOT / "squat_training/retargeting/omniretarget_project/src/holosoma_retargeting"
sys.path.insert(0, str(OFFICIAL))

from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from stairs_xml_scene import robot_mesh_geom_ids, terrain_geom_ids, contralateral_geom_pairs

DEMO_JOINTS = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
               "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "Spine", "Spine1"]
MAPPING = dict(zip(DEMO_JOINTS[:9], ["pelvis", "femur_r", "tibia_r", "talus_r", "toe_r",
                                    "femur_l", "tibia_l", "talus_l", "toe_l"]))


class ExactStairsRetargeter(InteractionMeshRetargeter):
    """Keep the official interaction mesh, SQP, joint and foot-stick costs."""

    def __init__(self, scene_xml: Path, *, nominal_weight: float = 2.0,
                 foot_tolerance: float = 0.001, clearance: float = 0.0,
                 enforce_joint_limits: bool = True):
        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        constants = SimpleNamespace(
            ROBOT_DOF=model.nq - 7, ROBOT_HEIGHT=0.875, ROBOT_NAME="lyt_exact_stairs",
            ROBOT_URDF_FILE=str(scene_xml), OBJECT_NAME="ground",
            FOOT_STICKING_LINKS=("foot_r", "toe_r", "foot_l", "toe_l"),
            MANUAL_LB={str(i): -1.0 for i in range(3, 7)},
            MANUAL_UB={str(i): 1.0 for i in range(3, 7)}, MANUAL_COST={},
            NOMINAL_TRACKING_INDICES=np.arange(model.nq),
            DEMO_JOINTS=DEMO_JOINTS, JOINTS_MAPPING=MAPPING,
        )
        super().__init__(task_constants=constants, object_urdf_path=None,
                         q_a_init_idx=-7, activate_joint_limits=enforce_joint_limits,
                         activate_foot_sticking=True, activate_obj_non_penetration=True,
                         step_size=0.10, collision_detection_threshold=0.025,
                         penetration_tolerance=-clearance,
                         foot_sticking_tolerance=foot_tolerance,
                         w_nominal_tracking_init=nominal_weight,
                         visualize=False, debug=False)
        # The original named free joint must not shift the 16 scalar bounds.
        self.q_a_lb = np.full(model.nq, -1.e6)
        self.q_a_ub = np.full(model.nq, 1.e6)
        self.q_a_lb[3:7], self.q_a_ub[3:7] = -1., 1.
        for j in range(model.njnt):
            if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                a = int(model.jnt_qposadr[j])
                if model.jnt_limited[j]:
                    self.q_a_lb[a], self.q_a_ub[a] = model.jnt_range[j]
        self.foot_links = {"right_heel": "foot_r", "right_toe": "toe_r",
                           "left_heel": "foot_l", "left_toe": "toe_l"}
        self._collision_pairs = [(int(a), int(b)) for a in robot_mesh_geom_ids(model)
                                 for b in terrain_geom_ids(model)]
        self._collision_pairs.extend(contralateral_geom_pairs(model))
        self._scratch_fromto = np.zeros(6)

    def _update_jacobians_and_phis_from_q(self, q):
        """Query actual convex STL collision geometry, independent of masks."""
        m, d = self.robot_model, self.robot_data
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        transform = self._build_transform_qdot_to_qvel_fast()
        jacobians, distances = {}, {}
        for ga, gb in self._collision_pairs:
            dist = mujoco.mj_geomDistance(m, d, ga, gb,
                                         self.collision_detection_threshold, self._scratch_fromto)
            if dist >= self.collision_detection_threshold:
                continue
            p1, p2 = self._scratch_fromto[:3], self._scratch_fromto[3:]
            delta = p1 - p2
            norm = float(np.linalg.norm(delta))
            if norm > 1.e-10:
                normal = np.sign(dist) * delta / norm
            elif m.geom_type[gb] == mujoco.mjtGeom.mjGEOM_PLANE:
                normal = d.geom_xmat[gb].reshape(3, 3)[:, 2]
            else:
                # Avoid silently dropping a constraint at exact contact.
                normal = d.geom_xpos[ga] - d.geom_xpos[gb]
                normal /= max(float(np.linalg.norm(normal)), 1.e-12)
            ja, jb = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, ja, None, p1, int(m.geom_bodyid[ga]))
            mujoco.mj_jac(m, d, jb, None, p2, int(m.geom_bodyid[gb]))
            jacobians[(ga, gb)] = normal @ (ja - jb) @ transform
            distances[(ga, gb)] = float(dist)
        return jacobians, distances


def stair_interaction_points(treads: list[dict]) -> np.ndarray:
    """Sample both tread surfaces and risers, including entry/exit ground."""
    points = []
    for t in treads:
        for x in (t["x_min"] + .01, (t["x_min"] + t["x_max"]) / 2, t["x_max"] - .01):
            for y in (-.25, .25):
                points.append((x, y, t["height"]))
    for x in (treads[0]["x_min"] - .4, treads[-1]["x_max"] + .4):
        for y in (-.25, .25):
            points.append((x, y, 0.))
    return np.asarray(points)
