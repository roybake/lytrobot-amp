"""Virtual six-channel plantar normal-force sensor (no MJCF changes)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import mujoco
import numpy as np

FOOT_FORCE_NAMES = (
    "left_heel_force", "left_forefoot_medial_force", "left_forefoot_lateral_force",
    "right_heel_force", "right_forefoot_medial_force", "right_forefoot_lateral_force",
)


@dataclass(frozen=True)
class FootForceSensorConfig:
    enabled: bool = True
    force_clip: float = 4.0
    noise_std: float = 0.0
    gain_randomization: float = 0.0
    bias_randomization: float = 0.0
    lowpass_alpha: float = 1.0
    delay_steps: int = 0
    distribution_sigma: float = 0.035

    def metadata(self) -> dict:
        return {"sensor_version": "virtual-plantar-force-v1", "names": FOOT_FORCE_NAMES,
                "channel_order": list(FOOT_FORCE_NAMES), "normalization":
                "clip((F_i/(sum(body_mass)*abs(gravity_z)/2)),0,force_clip)/force_clip",
                "config": asdict(self)}


class CpuFootForceSensor:
    """CPU adapter.  Each solver contact's continuous normal force is split
    smoothly among geometry-derived heel/medial/lateral anchors."""
    def __init__(self, model: mujoco.MjModel, contact_geom_ids, terrain_geom_ids, config=FootForceSensorConfig()):
        self.model, self.config = model, config
        self.terrain_geom_ids = set(map(int, terrain_geom_ids))
        # Existing convention is right geoms then left geoms; public output is left then right.
        self.side_geoms = (list(map(int, contact_geom_ids[2:])), list(map(int, contact_geom_ids[:2])))
        self.geom_side = {g: side for side, geoms in enumerate(self.side_geoms) for g in geoms}
        self.geom_body = {g: int(model.geom_bodyid[g]) for g in self.geom_side}
        self.mass = max(float(np.sum(model.body_mass)), 1e-6)
        self.force_ref = max(self.mass * abs(float(model.opt.gravity[2])) / 2.0, 1e-6)
        self.anchors = self._derive_anchors()

    def _derive_anchors(self):
        anchors = []
        for side, geoms in enumerate(self.side_geoms):
            body = self.geom_body[geoms[0]]
            points = []
            for g in geoms:
                # geom_pos and geom_size are model geometry, not magic world constants.
                p = np.asarray(self.model.geom_pos[g], dtype=np.float32)
                size = np.asarray(self.model.geom_size[g], dtype=np.float32)
                points.extend(((p[0]-size[0], p[1]), (p[0]+size[0], p[1])))
            xy = np.asarray(points, dtype=np.float32)
            xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
            ymid = float(xy[:, 1].mean()); spread = max(float(np.ptp(xy[:, 1])) * .35, .015)
            heel_x = xmin + .18 * (xmax-xmin); fore_x = xmin + .78 * (xmax-xmin)
            anchors.append(np.asarray(((heel_x, ymid), (fore_x, ymid + spread), (fore_x, ymid - spread)), np.float32))
        return anchors

    def raw(self, data: mujoco.MjData) -> np.ndarray:
        out = np.zeros((2, 3), np.float32)
        cf = np.zeros(6, np.float64)
        for i in range(data.ncon):
            c = data.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
            foot = g2 if g1 in self.terrain_geom_ids else g1 if g2 in self.terrain_geom_ids else -1
            side = self.geom_side.get(foot)
            if side is None: continue
            mujoco.mj_contactForce(self.model, data, i, cf)
            normal = max(float(cf[0]), 0.0)
            if normal == 0.0: continue
            body = self.geom_body[foot]
            rot = np.asarray(data.xmat[body], np.float32).reshape(3, 3)
            local = rot.T @ (np.asarray(c.pos, np.float32) - np.asarray(data.xpos[body], np.float32))
            d2 = np.sum((self.anchors[side] - local[:2]) ** 2, axis=1)
            logits = -d2 / (2.0 * max(self.config.distribution_sigma, 1e-5) ** 2)
            w = np.exp(logits - logits.max()); w /= max(float(w.sum()), 1e-8)
            out[side] += normal * w
        return out.reshape(6)

    def normalized(self, data: mujoco.MjData) -> np.ndarray:
        return np.nan_to_num(np.clip(self.raw(data) / self.force_ref, 0, self.config.force_clip) / self.config.force_clip).astype(np.float32)
