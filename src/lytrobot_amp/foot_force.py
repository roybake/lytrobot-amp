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

    def __post_init__(self) -> None:
        if self.force_clip <= 0:
            raise ValueError("force_clip must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if not 0 <= self.gain_randomization < 1:
            raise ValueError("gain_randomization must be in [0, 1)")
        if self.bias_randomization < 0:
            raise ValueError("bias_randomization must be non-negative")
        if not 0 < self.lowpass_alpha <= 1:
            raise ValueError("lowpass_alpha must be in (0, 1]")
        if not isinstance(self.delay_steps, int) or self.delay_steps < 0:
            raise ValueError("delay_steps must be a non-negative integer")
        if self.distribution_sigma <= 0:
            raise ValueError("distribution_sigma must be positive")

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
        # Anchors for each side are defined in the body frame that owns the
        # fixed foot geom (the first geom), never in the articulated toe body.
        self.side_body = tuple(self.geom_body[geoms[0]] for geoms in self.side_geoms)
        self.mass = max(float(np.sum(model.body_mass)), 1e-6)
        self.force_ref = max(self.mass * abs(float(model.opt.gravity[2])) / 2.0, 1e-6)
        self.anchors = self._derive_anchors()

    def _derive_anchors(self):
        """Build all anchors in the parent foot-body frame.

        The articulated toe geom is deliberately not treated as though its
        ``geom_pos`` were expressed in the foot body.  The stable plantar
        frame is the body owning the foot geom; the heel/forefoot anchors are
        derived from that body's support geom.  Runtime toe contacts are still
        transformed from world coordinates into this same frame.
        """
        anchors = []
        for side, geoms in enumerate(self.side_geoms):
            foot_geom = geoms[0]
            body = self.geom_body[foot_geom]
            points = []
            # geom_aabb is expressed in the geom frame and is meaningful for
            # meshes, unlike geom_size (which is not an XYZ half extent).
            centre = np.asarray(self.model.geom_aabb[foot_geom, :3], dtype=np.float32)
            half = np.asarray(self.model.geom_aabb[foot_geom, 3:], dtype=np.float32)
            geom_rot = np.asarray(self.model.geom_quat[foot_geom], dtype=np.float64)
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, geom_rot)
            rotation = rotation.reshape(3, 3).astype(np.float32)
            geom_pos = np.asarray(self.model.geom_pos[foot_geom], dtype=np.float32)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        point_geom = centre + np.asarray((sx * half[0], sy * half[1], sz * half[2]), np.float32)
                        point_foot = geom_pos + rotation @ point_geom
                        points.append(point_foot[:2])
            xy = np.asarray(points, dtype=np.float32)
            xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
            ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
            ymid = .5 * (ymin + ymax); spread = max((ymax-ymin) * .32, .015)
            heel_x = xmin + .18 * (xmax-xmin); fore_x = xmin + .78 * (xmax-xmin)
            # Left medial points toward world centre (-local Y); right medial
            # points toward world centre (+local Y).  Public semantics remain
            # mirrored even though the MJCF geom ordering is right-first.
            medial_sign = -1.0 if side == 0 else 1.0
            anchors.append(np.asarray(((heel_x, ymid),
                                       (fore_x, ymid + medial_sign * spread),
                                       (fore_x, ymid - medial_sign * spread)), np.float32))
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
            # A toe contact is still expressed in the stable parent foot
            # plantar frame, which is also the frame used by ``anchors``.
            body = self.side_body[side]
            rot = np.asarray(data.xmat[body], np.float32).reshape(3, 3)
            local = rot.T @ (np.asarray(c.pos, np.float32) - np.asarray(data.xpos[body], np.float32))
            d2 = np.sum((self.anchors[side] - local[:2]) ** 2, axis=1)
            logits = -d2 / (2.0 * max(self.config.distribution_sigma, 1e-5) ** 2)
            w = np.exp(logits - logits.max()); w /= max(float(w.sum()), 1e-8)
            out[side] += normal * w
        return out.reshape(6)

    def normalized(self, data: mujoco.MjData) -> np.ndarray:
        return np.nan_to_num(np.clip(self.raw(data) / self.force_ref, 0, self.config.force_clip) / self.config.force_clip).astype(np.float32)


class FootForceSensorDynamics:
    """Stateful gain/bias/noise/filter/delay model for six force channels."""

    def __init__(self, num_envs: int, config: FootForceSensorConfig, *, seed: int = 0,
                 torch_device=None):
        self.num_envs, self.config = int(num_envs), config
        self.rng = np.random.default_rng(seed)
        self._torch = torch_device is not None
        self._torch_device = torch_device
        if self._torch:
            import torch
            self._torch_rng = torch.Generator(device=torch_device)
            self._torch_rng.manual_seed(int(seed))
            self.gain, self.bias = self._sample_calibration_torch(self.num_envs)
            self.filtered = torch.zeros((num_envs, 6), device=torch_device)
            self.delay = torch.zeros((num_envs, config.delay_steps + 1, 6), device=torch_device)
            self.cursor = torch.zeros(num_envs, dtype=torch.long, device=torch_device)
        else:
            gain, bias = self._sample_calibration(self.num_envs)
            self.gain, self.bias = gain, bias
            self.filtered = np.zeros((num_envs, 6), np.float32)
            self.delay = np.zeros((num_envs, config.delay_steps + 1, 6), np.float32)
            self.cursor = np.zeros(num_envs, dtype=np.int64)

    def _sample_calibration(self, count):
        cfg = self.config
        gain = self.rng.uniform(
            1.0-cfg.gain_randomization, 1.0+cfg.gain_randomization, (count, 6)
        ).astype(np.float32)
        bias = self.rng.uniform(
            -cfg.bias_randomization, cfg.bias_randomization, (count, 6)
        ).astype(np.float32)
        return gain, bias

    def _sample_calibration_torch(self, count):
        import torch
        cfg = self.config
        gain = torch.ones((count, 6), dtype=torch.float32, device=self._torch_device)
        bias = torch.zeros((count, 6), dtype=torch.float32, device=self._torch_device)
        if cfg.gain_randomization:
            gain.uniform_(1.0-cfg.gain_randomization, 1.0+cfg.gain_randomization,
                          generator=self._torch_rng)
        if cfg.bias_randomization:
            bias.uniform_(-cfg.bias_randomization, cfg.bias_randomization,
                          generator=self._torch_rng)
        return gain, bias

    def _ids(self, env_ids=None):
        if env_ids is None:
            if self._torch:
                import torch
                return torch.arange(self.num_envs, device=self._torch_device)
            return np.arange(self.num_envs, dtype=np.int64)
        if self._torch:
            import torch
            return env_ids if torch.is_tensor(env_ids) else torch.as_tensor(
                env_ids, dtype=torch.long, device=self._torch_device
            )
        return np.asarray(env_ids, dtype=np.int64)

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        self.filtered[ids] = 0
        self.delay[ids] = 0
        self.cursor[ids] = 0
        if self._torch:
            gain, bias = self._sample_calibration_torch(len(ids))
            self.gain[ids] = gain
            self.bias[ids] = bias
        else:
            gain, bias = self._sample_calibration(len(ids))
            self.gain[ids] = gain
            self.bias[ids] = bias

    def apply(self, force, env_ids=None):
        cfg = self.config
        ids = self._ids(env_ids)
        value = force * self.gain[ids] + self.bias[ids]
        if cfg.noise_std:
            if self._torch:
                import torch
                value = value + torch.randn_like(value) * cfg.noise_std
            else:
                value = value + self.rng.normal(0, cfg.noise_std, value.shape).astype(np.float32)
        value = value.clip(min=0) if self._torch else np.clip(value, 0, None)
        self.filtered[ids] = (
            cfg.lowpass_alpha * value + (1.0-cfg.lowpass_alpha) * self.filtered[ids]
        )
        cursor = self.cursor[ids]
        self.delay[ids, cursor] = self.filtered[ids]
        read = (cursor + 1) % self.delay.shape[1]
        result = self.delay[ids, read]
        result = result.clone() if self._torch else result.copy()
        self.cursor[ids] = read
        return result
