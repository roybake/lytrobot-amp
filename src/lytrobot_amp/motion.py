"""Reference motion loading and robot-state feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

DEFAULT_COMMON_JOINTS: tuple[str, ...] = (
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
)

CONTACT_GEOM_NAMES: tuple[str, ...] = (
    "foot_r",
    "toe_r",
    "foot_l",
    "toe_l",
)
CONTACT_BODY_FALLBACKS: tuple[str, ...] = (
    "foot_r",
    "toe_r",
    "foot_l",
    "toe_l",
)
CONTACT_HEIGHT_THRESHOLD = 0.035
CONTACT_REFERENCE_PERCENTILE = 50.0
CONTACT_MIN_RUN_FRAMES = 10


def _filter_short_contact_runs(contact: np.ndarray, min_run_frames: int = CONTACT_MIN_RUN_FRAMES) -> np.ndarray:
    filtered = np.asarray(contact, dtype=bool).copy()
    if filtered.size == 0 or min_run_frames <= 1:
        return filtered

    start = 0
    while start < filtered.size:
        end = start + 1
        while end < filtered.size and filtered[end] == filtered[start]:
            end += 1
        if end - start < min_run_frames:
            left = filtered[start - 1] if start > 0 else None
            right = filtered[end] if end < filtered.size else None
            if left is not None and right is not None and left == right:
                filtered[start:end] = left
        start = end
    return filtered


def resolve_contact_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    """Resolve the foot/toe mesh geoms used for ground contact."""
    geom_ids: list[int] = []
    missing: list[str] = []
    for geom_name, body_name in zip(CONTACT_GEOM_NAMES, CONTACT_BODY_FALLBACKS, strict=True):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                body_geom_ids = np.nonzero(np.asarray(model.geom_bodyid) == body_id)[0]
                if len(body_geom_ids) > 0:
                    geom_id = int(body_geom_ids[0])
        if geom_id < 0:
            missing.append(f"{geom_name} or body {body_name}")
        else:
            geom_ids.append(int(geom_id))
    if missing:
        raise RuntimeError(f"Contact geom(s) missing from robot model: {missing}")
    return np.asarray(geom_ids, dtype=np.int32)


@dataclass(frozen=True)
class MotionState:
    qpos: np.ndarray
    qvel: np.ndarray
    indices: np.ndarray
    motion_ids: np.ndarray | None = None


@dataclass(frozen=True)
class MotionTargets:
    root_height: np.ndarray
    root_quat: np.ndarray
    root_lin_vel: np.ndarray
    root_ang_vel: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    tendon_ctrl: np.ndarray
    foot_pos: np.ndarray
    foot_vel: np.ndarray
    foot_contact: np.ndarray


def _canonical_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat.ndim == 1:
        if quat[0] < 0.0:
            quat *= -1.0
        return quat
    signs = np.where(quat[:, :1] < 0.0, -1.0, 1.0).astype(np.float32)
    return quat * signs


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = _canonical_quat(quat)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1.0e-6)


def _repair_loop_root_translation_velocity(
    qpos: np.ndarray,
    qvel: np.ndarray,
    frequency: float,
) -> np.ndarray:
    """Repair the horizontal root velocity at a translating loop seam.

    Older generated walking references differentiated the last and first root
    positions as if they occupied the same cycle.  Because walking advances in
    X, that produced roughly -20 m/s at both seam frames.  Recomputing the two
    boundary samples with the cycle displacement keeps old archives usable.
    """

    repaired = np.asarray(qvel, dtype=np.float32).copy()
    positions = np.asarray(qpos, dtype=np.float64)
    if positions.ndim != 2 or len(positions) < 3 or repaired.shape[1] < 2:
        return repaired

    root_steps = np.diff(positions[:, :2], axis=0)
    cycle_translation = (
        positions[-1, :2] - positions[0, :2] + np.median(root_steps, axis=0)
    )
    scale = 0.5 * float(frequency)
    repaired[0, :2] = (
        positions[1, :2] - (positions[-1, :2] - cycle_translation)
    ) * scale
    repaired[-1, :2] = (
        positions[0, :2] + cycle_translation - positions[-2, :2]
    ) * scale
    return repaired


class MotionReference:
    """Maps the walking npz onto the tendon-driven MuJoCo robot.

    The source motion names match the lower-body robot joints, but the robot has
    patella constraints, hard joint limits, and tendon position actuators.  The
    class therefore precomputes a robot-feasible reference: clipped joint states,
    root targets, local foot targets, contact labels, and tendon lengths in
    actuator order.
    """

    def __init__(
        self,
        motion_file: str | Path,
        model: mujoco.MjModel,
        common_joint_names: Sequence[str] = DEFAULT_COMMON_JOINTS,
        actuator_ids: Sequence[int] | np.ndarray | None = None,
        zero_root_xy: bool = True,
    ) -> None:
        self.motion_file = Path(motion_file).expanduser().resolve()
        if not self.motion_file.exists():
            raise FileNotFoundError(f"Motion file not found: {self.motion_file}")

        archive = np.load(self.motion_file, allow_pickle=True)
        self.source_qpos = np.asarray(archive["qpos"], dtype=np.float32)
        self.source_qvel = np.asarray(archive["qvel"], dtype=np.float32)
        self.frequency = float(np.asarray(archive["frequency"]).item())
        self.source_loop = (
            bool(np.asarray(archive["loop"]).item())
            if "loop" in archive.files
            else False
        )
        if self.source_loop:
            self.source_qvel = _repair_loop_root_translation_velocity(
                self.source_qpos,
                self.source_qvel,
                self.frequency,
            )
        self.frame_count = int(self.source_qpos.shape[0])
        self.zero_root_xy = zero_root_xy
        self.source_foot_contact: np.ndarray | None = None
        if "foot_contact" in archive.files:
            source_foot_contact = np.asarray(archive["foot_contact"], dtype=np.float32)
            expected_shape = (self.frame_count, 2)
            if source_foot_contact.shape != expected_shape:
                raise ValueError(
                    f"foot_contact must have shape {expected_shape}, got {source_foot_contact.shape}."
                )
            self.source_foot_contact = np.clip(source_foot_contact, 0.0, 1.0)

        source_joint_names = [str(name) for name in archive["joint_names"]]
        self.source_joint_to_index = {name: i for i, name in enumerate(source_joint_names)}

        self.model = model
        self.model_qpos0 = np.asarray(model.qpos0, dtype=np.float32).copy()

        robot_joint_names: list[str] = []
        robot_qpos_adrs: list[int] = []
        robot_qvel_adrs: list[int] = []
        source_qpos_cols: list[int] = []
        source_qvel_cols: list[int] = []

        for name in common_joint_names:
            source_index = self.source_joint_to_index.get(name)
            robot_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if source_index is None or robot_joint_id < 0:
                continue
            if model.jnt_type[robot_joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            robot_joint_names.append(name)
            robot_qpos_adrs.append(int(model.jnt_qposadr[robot_joint_id]))
            robot_qvel_adrs.append(int(model.jnt_dofadr[robot_joint_id]))
            source_qpos_cols.append(7 + source_index - 1)
            source_qvel_cols.append(6 + source_index - 1)

        if not robot_joint_names:
            raise RuntimeError("No common lower-body joints found between motion and robot model.")

        self.common_joint_names = tuple(robot_joint_names)
        self.robot_qpos_adrs = np.asarray(robot_qpos_adrs, dtype=np.int32)
        self.robot_qvel_adrs = np.asarray(robot_qvel_adrs, dtype=np.int32)
        self.source_qpos_cols = np.asarray(source_qpos_cols, dtype=np.int32)
        self.source_qvel_cols = np.asarray(source_qvel_cols, dtype=np.int32)

        self.root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.root_body_id < 0:
            raise RuntimeError("Body 'pelvis' not found in robot model.")
        self.contact_geom_ids = resolve_contact_geom_ids(model)
        self.actuator_ids = (
            np.asarray(actuator_ids, dtype=np.int32)
            if actuator_ids is not None
            else np.arange(model.nu, dtype=np.int32)
        )
        self.actuator_tendon_ids = np.asarray(model.actuator_trnid[self.actuator_ids, 0], dtype=np.int32)
        self.ctrl_low = np.asarray(model.actuator_ctrlrange[self.actuator_ids, 0], dtype=np.float32)
        self.ctrl_high = np.asarray(model.actuator_ctrlrange[self.actuator_ids, 1], dtype=np.float32)

        self.root_xy0 = self.source_qpos[0, :2].copy()
        self.foot_pos_dim = int(len(CONTACT_GEOM_NAMES) * 3)
        self.contact_dim = 2
        # AMP must distinguish genuine locomotion from an in-place leg swing.
        # Root linear/angular velocity contributes six motion-state dimensions
        # in addition to the pose, joint, and local-foot features.
        self.feature_dim = int(1 + 4 + 6 + 2 * len(self.common_joint_names) + self.foot_pos_dim)

        self._precompute_robot_reference()

    def _state_from_source(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        index = int(frame_index) % self.frame_count
        qpos = self.model_qpos0.copy()
        qvel = np.zeros(self.model.nv, dtype=np.float32)

        qpos[:7] = self.source_qpos[index, :7]
        if self.zero_root_xy:
            qpos[0:2] -= self.root_xy0
        qpos[3:7] = _normalize_quat(qpos[3:7])

        qvel[:6] = self.source_qvel[index, :6]
        qpos[self.robot_qpos_adrs] = self.source_qpos[index, self.source_qpos_cols]
        qvel[self.robot_qvel_adrs] = self.source_qvel[index, self.source_qvel_cols]
        self._clip_joint_ranges(qpos, qvel)
        return qpos.astype(np.float32), qvel.astype(np.float32)

    def _clip_joint_ranges(self, qpos: np.ndarray, qvel: np.ndarray | None = None) -> None:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if not self.model.jnt_limited[joint_id]:
                continue
            qadr = int(self.model.jnt_qposadr[joint_id])
            low, high = self.model.jnt_range[joint_id]
            value = float(qpos[qadr])
            clipped = float(np.clip(value, low, high))
            qpos[qadr] = clipped
            if qvel is None or clipped == value:
                continue
            dadr = int(self.model.jnt_dofadr[joint_id])
            if (value < low and qvel[dadr] < 0.0) or (value > high and qvel[dadr] > 0.0):
                qvel[dadr] = 0.0

    def _precompute_robot_reference(self) -> None:
        data = mujoco.MjData(self.model)
        qpos = np.zeros((self.frame_count, self.model.nq), dtype=np.float32)
        qvel = np.zeros((self.frame_count, self.model.nv), dtype=np.float32)
        tendon_ctrl = np.zeros((self.frame_count, len(self.actuator_ids)), dtype=np.float32)
        foot_pos = np.zeros((self.frame_count, len(CONTACT_GEOM_NAMES), 3), dtype=np.float32)
        foot_world_z = np.zeros((self.frame_count, len(CONTACT_GEOM_NAMES)), dtype=np.float32)

        for frame_index in range(self.frame_count):
            qpos[frame_index], qvel[frame_index] = self._state_from_source(frame_index)
            data.qpos[:] = qpos[frame_index]
            data.qvel[:] = qvel[frame_index]
            mujoco.mj_forward(self.model, data)

            tendon_lengths = np.asarray(data.ten_length[self.actuator_tendon_ids], dtype=np.float32)
            tendon_ctrl[frame_index] = np.clip(tendon_lengths, self.ctrl_low, self.ctrl_high)
            foot_world = np.asarray(data.geom_xpos[self.contact_geom_ids], dtype=np.float32)
            foot_world_z[frame_index] = foot_world[:, 2]
            foot_pos[frame_index] = self.foot_positions_local_from_data(data)

        dt = 1.0 / max(self.frequency, 1.0e-6)
        foot_vel = np.gradient(foot_pos, dt, axis=0, edge_order=1).astype(np.float32)
        right_side_z = np.min(foot_world_z[:, 0:2], axis=1)
        left_side_z = np.min(foot_world_z[:, 2:4], axis=1)
        right_threshold = np.percentile(right_side_z, CONTACT_REFERENCE_PERCENTILE)
        left_threshold = np.percentile(left_side_z, CONTACT_REFERENCE_PERCENTILE)
        right_contact = _filter_short_contact_runs(right_side_z <= right_threshold)
        left_contact = _filter_short_contact_runs(left_side_z <= left_threshold)
        foot_contact = np.stack((right_contact, left_contact), axis=1).astype(np.float32)
        if self.source_foot_contact is not None:
            # Locomotion references can keep the legacy height inference, while
            # stationary tasks such as squats explicitly declare both planted
            # feet throughout the cycle.
            foot_contact = self.source_foot_contact.copy()

        self.reference_qpos = qpos
        self.reference_qvel = qvel
        self.reference_root_height = qpos[:, 2].copy()
        self.reference_root_quat = _normalize_quat(qpos[:, 3:7])
        self.reference_root_lin_vel = qvel[:, 0:3].copy()
        self.reference_root_ang_vel = qvel[:, 3:6].copy()
        self.reference_joint_pos = qpos[:, self.robot_qpos_adrs].copy()
        self.reference_joint_vel = qvel[:, self.robot_qvel_adrs].copy()
        self.reference_tendon_ctrl = tendon_ctrl
        self.reference_foot_pos = foot_pos
        self.reference_foot_vel = foot_vel
        self.reference_foot_contact = foot_contact

        raw_features = self._features_from_parts(
            self.reference_root_height[:, None],
            self.reference_root_quat,
            self.reference_root_lin_vel,
            self.reference_root_ang_vel,
            self.reference_joint_pos,
            self.reference_joint_vel,
            self.reference_foot_pos.reshape(self.frame_count, -1),
        )
        self.feature_mean = raw_features.mean(axis=0).astype(np.float32)
        scale_floor = self._feature_scale_floor()
        self.feature_std = np.maximum(raw_features.std(axis=0), scale_floor).astype(np.float32)
        self.reference_features = self._normalize_features(raw_features)
        self.mirrored_reference_features = self._normalize_features(self._mirror_raw_features(raw_features))

    def _feature_scale_floor(self) -> np.ndarray:
        joint_count = len(self.common_joint_names)
        floors = np.concatenate(
            (
                np.full(1, 0.05, dtype=np.float32),
                np.full(4, 0.05, dtype=np.float32),
                np.full(3, 0.35, dtype=np.float32),
                np.full(3, 0.75, dtype=np.float32),
                np.full(joint_count, 0.12, dtype=np.float32),
                np.full(joint_count, 1.00, dtype=np.float32),
                np.full(self.foot_pos_dim, 0.05, dtype=np.float32),
            )
        )
        return floors

    def foot_positions_local_from_data(self, data: mujoco.MjData) -> np.ndarray:
        root_pos = np.asarray(data.xpos[self.root_body_id], dtype=np.float32)
        root_mat = np.asarray(data.xmat[self.root_body_id], dtype=np.float32).reshape(3, 3)
        foot_world = np.asarray(data.geom_xpos[self.contact_geom_ids], dtype=np.float32)
        return (foot_world - root_pos) @ root_mat

    def state_at(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        index = int(frame_index) % self.frame_count
        return self.reference_qpos[index].copy(), self.reference_qvel[index].copy()

    def sample_states(self, batch_size: int, rng: np.random.Generator) -> MotionState:
        indices = rng.integers(0, self.frame_count, size=batch_size, endpoint=False)
        return MotionState(
            qpos=self.reference_qpos[indices].copy(),
            qvel=self.reference_qvel[indices].copy(),
            indices=indices.astype(np.int32),
        )

    def _interp(self, values: np.ndarray, low_index: np.ndarray, high_index: np.ndarray, blend: np.ndarray) -> np.ndarray:
        low_values = values[low_index]
        high_values = values[high_index]
        while blend.ndim < low_values.ndim:
            blend = blend[..., None]
        return ((1.0 - blend) * low_values + blend * high_values).astype(np.float32)

    def targets_at_phase(self, phase: np.ndarray) -> MotionTargets:
        phase_arr = np.asarray(phase, dtype=np.float32)
        low_index = np.floor(phase_arr).astype(np.int32) % self.frame_count
        high_index = (low_index + 1) % self.frame_count
        blend = (phase_arr - np.floor(phase_arr)).astype(np.float32)

        root_quat = _normalize_quat(self._interp(self.reference_root_quat, low_index, high_index, blend))
        return MotionTargets(
            root_height=self._interp(self.reference_root_height, low_index, high_index, blend),
            root_quat=root_quat,
            root_lin_vel=self._interp(self.reference_root_lin_vel, low_index, high_index, blend),
            root_ang_vel=self._interp(self.reference_root_ang_vel, low_index, high_index, blend),
            joint_pos=self._interp(self.reference_joint_pos, low_index, high_index, blend),
            joint_vel=self._interp(self.reference_joint_vel, low_index, high_index, blend),
            tendon_ctrl=self._interp(self.reference_tendon_ctrl, low_index, high_index, blend),
            foot_pos=self._interp(self.reference_foot_pos, low_index, high_index, blend),
            foot_vel=self._interp(self.reference_foot_vel, low_index, high_index, blend),
            foot_contact=self._interp(self.reference_foot_contact, low_index, high_index, blend),
        )

    def _features_from_parts(
        self,
        root_height: np.ndarray,
        root_quat: np.ndarray,
        root_lin_vel: np.ndarray,
        root_ang_vel: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        foot_pos: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            (
                root_height,
                _normalize_quat(root_quat),
                root_lin_vel,
                root_ang_vel,
                joint_pos,
                joint_vel,
                foot_pos,
            ),
            axis=1,
        ).astype(np.float32)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.feature_mean) / self.feature_std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(normalized, -10.0, 10.0).astype(np.float32)

    def _mirror_raw_features(self, features: np.ndarray) -> np.ndarray:
        mirrored = np.asarray(features, dtype=np.float32).copy()
        joint_count = len(self.common_joint_names)
        root_lin_vel_start = 5
        root_ang_vel_start = root_lin_vel_start + 3
        joint_pos_start = root_ang_vel_start + 3
        joint_vel_start = joint_pos_start + joint_count
        foot_start = joint_vel_start + joint_count

        # Mirror root orientation across the sagittal plane: roll/yaw change sign.
        mirrored[:, 2] *= -1.0
        mirrored[:, 4] *= -1.0
        # Polar linear velocity flips only lateral Y; axial angular velocity
        # flips roll X and yaw Z under this reflection.
        mirrored[:, root_lin_vel_start + 1] *= -1.0
        mirrored[:, root_ang_vel_start + 0] *= -1.0
        mirrored[:, root_ang_vel_start + 2] *= -1.0

        half = joint_count // 2
        if joint_count == 2 * half:
            mirrored[:, joint_pos_start : joint_pos_start + half] = features[
                :, joint_pos_start + half : joint_pos_start + joint_count
            ]
            mirrored[:, joint_pos_start + half : joint_pos_start + joint_count] = features[
                :, joint_pos_start : joint_pos_start + half
            ]
            mirrored[:, joint_vel_start : joint_vel_start + half] = features[
                :, joint_vel_start + half : joint_vel_start + joint_count
            ]
            mirrored[:, joint_vel_start + half : joint_vel_start + joint_count] = features[
                :, joint_vel_start : joint_vel_start + half
            ]

        foot = features[:, foot_start:].reshape(features.shape[0], len(CONTACT_GEOM_NAMES), 3)
        mirrored_foot = foot[:, [2, 3, 0, 1], :].copy()
        mirrored_foot[:, :, 1] *= -1.0
        mirrored[:, foot_start:] = mirrored_foot.reshape(features.shape[0], -1)
        return mirrored.astype(np.float32)

    def features_from_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        foot_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        qpos_arr = np.asarray(qpos, dtype=np.float32)
        qvel_arr = np.asarray(qvel, dtype=np.float32)

        single = qpos_arr.ndim == 1
        if single:
            qpos_arr = qpos_arr[None, :]
            qvel_arr = qvel_arr[None, :]

        batch_size = qpos_arr.shape[0]
        if foot_pos is None:
            foot_pos_arr = np.zeros((batch_size, self.foot_pos_dim), dtype=np.float32)
        else:
            foot_pos_arr = np.asarray(foot_pos, dtype=np.float32).reshape(batch_size, self.foot_pos_dim)

        raw_features = self._features_from_parts(
            qpos_arr[:, 2:3],
            qpos_arr[:, 3:7],
            qvel_arr[:, 0:3],
            qvel_arr[:, 3:6],
            qpos_arr[:, self.robot_qpos_adrs],
            qvel_arr[:, self.robot_qvel_adrs],
            foot_pos_arr,
        )
        features = self._normalize_features(raw_features)
        if single:
            return features[0]
        return features

    def sample_features(
        self,
        batch_size: int,
        rng: np.random.Generator,
        mirror_prob: float = 0.5,
    ) -> np.ndarray:
        indices = rng.integers(0, self.frame_count, size=batch_size, endpoint=False)
        features = self.reference_features[indices].copy()
        prob = float(np.clip(mirror_prob, 0.0, 1.0))
        if prob > 0.0:
            mirror = rng.random(batch_size) < prob
            if np.any(mirror):
                features[mirror] = self.mirrored_reference_features[indices[mirror]]
        return features


class MotionLibrary:
    """Balanced retargeted clips for a single skill-conditioned AMP policy."""

    def __init__(
        self,
        motion_files: Sequence[str | Path],
        model: mujoco.MjModel,
        *,
        actuator_ids: Sequence[int] | np.ndarray | None = None,
        motion_weights: Sequence[float] | None = None,
        motion_names: Sequence[str] | None = None,
        motion_loop: Sequence[bool] | None = None,
    ) -> None:
        paths = tuple(Path(path).expanduser().resolve() for path in motion_files)
        if not paths:
            raise ValueError("MotionLibrary requires at least one clip.")
        self.motions = tuple(
            MotionReference(path, model, actuator_ids=actuator_ids) for path in paths
        )
        first = self.motions[0]
        for motion in self.motions[1:]:
            if motion.common_joint_names != first.common_joint_names:
                raise ValueError("All motion clips must map the same robot joints.")
            if motion.feature_dim != first.feature_dim:
                raise ValueError("All motion clips must have the same AMP feature dimension.")

        self.motion_files = paths
        self.motion_count = len(paths)
        self.motion_names = tuple(
            motion_names if motion_names is not None else [path.stem for path in paths]
        )
        if len(self.motion_names) != self.motion_count:
            raise ValueError("motion_names length must match motion_files.")
        self.motion_loop = np.asarray(
            motion_loop if motion_loop is not None else [True] * self.motion_count,
            dtype=bool,
        )
        if self.motion_loop.shape != (self.motion_count,):
            raise ValueError("motion_loop length must match motion_files.")
        weights = np.asarray(
            motion_weights if motion_weights is not None else np.ones(self.motion_count),
            dtype=np.float64,
        )
        if (
            weights.shape != (self.motion_count,)
            or np.any(weights < 0.0)
            or not np.any(weights > 0.0)
        ):
            raise ValueError("motion_weights must be non-negative and contain a positive value.")
        self.motion_weights = weights / weights.sum()

        self.model = model
        self.model_qpos0 = first.model_qpos0
        self.common_joint_names = first.common_joint_names
        self.robot_qpos_adrs = first.robot_qpos_adrs
        self.robot_qvel_adrs = first.robot_qvel_adrs
        self.root_body_id = first.root_body_id
        self.contact_geom_ids = first.contact_geom_ids
        self.actuator_ids = first.actuator_ids
        self.actuator_tendon_ids = first.actuator_tendon_ids
        self.foot_pos_dim = first.foot_pos_dim
        self.contact_dim = first.contact_dim
        self.feature_dim = first.feature_dim
        self.frame_counts = np.asarray(
            [motion.frame_count for motion in self.motions], dtype=np.int32
        )
        self.frequencies = np.asarray(
            [motion.frequency for motion in self.motions], dtype=np.float32
        )
        self.max_frame_count = int(self.frame_counts.max())

        raw_features = [
            motion._features_from_parts(
                motion.reference_root_height[:, None],
                motion.reference_root_quat,
                motion.reference_root_lin_vel,
                motion.reference_root_ang_vel,
                motion.reference_joint_pos,
                motion.reference_joint_vel,
                motion.reference_foot_pos.reshape(motion.frame_count, -1),
            )
            for motion in self.motions
        ]
        all_raw = np.concatenate(raw_features, axis=0)
        self.feature_mean = all_raw.mean(axis=0).astype(np.float32)
        self.feature_std = np.maximum(
            all_raw.std(axis=0), first._feature_scale_floor()
        ).astype(np.float32)
        self.reference_features = tuple(
            self._normalize_features(raw) for raw in raw_features
        )
        self.mirrored_reference_features = tuple(
            self._normalize_features(motion._mirror_raw_features(raw))
            for motion, raw in zip(self.motions, raw_features, strict=True)
        )

    @property
    def frame_count(self) -> int:
        return int(self.frame_counts[0])

    @property
    def frequency(self) -> float:
        return float(self.frequencies[0])

    @property
    def motion_file(self) -> Path:
        return self.motion_files[0]

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.feature_mean) / self.feature_std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(normalized, -10.0, 10.0).astype(np.float32)

    def foot_positions_local_from_data(self, data: mujoco.MjData) -> np.ndarray:
        return self.motions[0].foot_positions_local_from_data(data)

    def sample_motion_ids(
        self, batch_size: int, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.choice(
            self.motion_count, size=batch_size, p=self.motion_weights
        ).astype(np.int32)

    def sample_states(
        self,
        batch_size: int,
        rng: np.random.Generator,
        motion_ids: np.ndarray | None = None,
    ) -> MotionState:
        ids = (
            self.sample_motion_ids(batch_size, rng)
            if motion_ids is None
            else np.asarray(motion_ids, dtype=np.int32)
        )
        if ids.shape != (batch_size,):
            raise ValueError(f"Expected motion_ids shape {(batch_size,)}, got {ids.shape}.")
        indices = np.asarray(
            [rng.integers(0, self.frame_counts[motion_id]) for motion_id in ids],
            dtype=np.int32,
        )
        qpos = np.stack(
            [
                self.motions[motion_id].reference_qpos[index]
                for motion_id, index in zip(ids, indices, strict=True)
            ]
        )
        qvel = np.stack(
            [
                self.motions[motion_id].reference_qvel[index]
                for motion_id, index in zip(ids, indices, strict=True)
            ]
        )
        return MotionState(qpos=qpos, qvel=qvel, indices=indices, motion_ids=ids)

    def targets_at_phase(
        self,
        phase: np.ndarray,
        motion_ids: np.ndarray | None = None,
    ) -> MotionTargets:
        phase_arr = np.asarray(phase, dtype=np.float32)
        ids = (
            np.zeros(len(phase_arr), dtype=np.int32)
            if motion_ids is None
            else np.asarray(motion_ids, dtype=np.int32)
        )
        if ids.shape != phase_arr.shape:
            raise ValueError("motion_ids must have the same shape as phase.")
        targets = []
        for motion_id, value in zip(ids, phase_arr, strict=True):
            frame_count = self.frame_counts[motion_id]
            bounded = (
                np.mod(value, frame_count)
                if self.motion_loop[motion_id]
                else np.clip(value, 0.0, frame_count - 1.0)
            )
            targets.append(
                self.motions[motion_id].targets_at_phase(
                    np.asarray([bounded], dtype=np.float32)
                )
            )

        def join(name: str) -> np.ndarray:
            return np.concatenate([getattr(target, name) for target in targets], axis=0)

        return MotionTargets(
            root_height=join("root_height"),
            root_quat=join("root_quat"),
            root_lin_vel=join("root_lin_vel"),
            root_ang_vel=join("root_ang_vel"),
            joint_pos=join("joint_pos"),
            joint_vel=join("joint_vel"),
            tendon_ctrl=join("tendon_ctrl"),
            foot_pos=join("foot_pos"),
            foot_vel=join("foot_vel"),
            foot_contact=join("foot_contact"),
        )

    def features_from_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        foot_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        qpos_arr = np.asarray(qpos, dtype=np.float32)
        qvel_arr = np.asarray(qvel, dtype=np.float32)
        single = qpos_arr.ndim == 1
        if single:
            qpos_arr = qpos_arr[None, :]
            qvel_arr = qvel_arr[None, :]
        batch_size = len(qpos_arr)
        foot = (
            np.zeros((batch_size, self.foot_pos_dim), dtype=np.float32)
            if foot_pos is None
            else np.asarray(foot_pos, dtype=np.float32).reshape(batch_size, -1)
        )
        raw = self.motions[0]._features_from_parts(
            qpos_arr[:, 2:3],
            qpos_arr[:, 3:7],
            qvel_arr[:, 0:3],
            qvel_arr[:, 3:6],
            qpos_arr[:, self.robot_qpos_adrs],
            qvel_arr[:, self.robot_qvel_adrs],
            foot,
        )
        features = self._normalize_features(raw)
        return features[0] if single else features

    def sample_features(
        self,
        batch_size: int,
        rng: np.random.Generator,
        mirror_prob: float = 0.5,
        motion_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        motion_ids = (
            self.sample_motion_ids(batch_size, rng)
            if motion_ids is None
            else np.asarray(motion_ids, dtype=np.int32)
        )
        if motion_ids.shape != (batch_size,):
            raise ValueError(
                f"Expected motion_ids shape {(batch_size,)}, got {motion_ids.shape}."
            )
        output = np.empty((batch_size, self.feature_dim), dtype=np.float32)
        prob = float(np.clip(mirror_prob, 0.0, 1.0))
        for row, motion_id in enumerate(motion_ids):
            frame = int(rng.integers(0, self.frame_counts[motion_id]))
            source = (
                self.mirrored_reference_features[motion_id]
                if rng.random() < prob
                else self.reference_features[motion_id]
            )
            output[row] = source[frame]
        return output

    def sample_features_for_motion_ids(
        self,
        motion_ids: np.ndarray,
        rng: np.random.Generator,
        mirror_prob: float = 0.5,
    ) -> np.ndarray:
        """Sample expert features from an explicit motion-id batch."""

        ids = np.asarray(motion_ids, dtype=np.int32)
        return self.sample_features(
            int(ids.size), rng, mirror_prob=mirror_prob, motion_ids=ids
        )

    def packed(self, attribute: str) -> np.ndarray:
        """Pad a per-frame MotionReference attribute for GPU indexed lookup."""

        values = [np.asarray(getattr(motion, attribute)) for motion in self.motions]
        shape = (self.motion_count, self.max_frame_count) + values[0].shape[1:]
        result = np.empty(shape, dtype=values[0].dtype)
        for motion_id, value in enumerate(values):
            result[motion_id, : len(value)] = value
            result[motion_id, len(value) :] = value[-1]
        return result


def load_motion_library(
    motion_files: Sequence[str | Path],
    model: mujoco.MjModel,
    *,
    actuator_ids: Sequence[int] | np.ndarray | None = None,
    motion_weights: Sequence[float] | None = None,
    motion_names: Sequence[str] | None = None,
    motion_loop: Sequence[bool] | None = None,
    force_library: bool = False,
) -> MotionReference | MotionLibrary:
    files = tuple(motion_files)
    if len(files) == 1 and not force_library:
        return MotionReference(files[0], model, actuator_ids=actuator_ids)
    return MotionLibrary(
        files,
        model,
        actuator_ids=actuator_ids,
        motion_weights=motion_weights,
        motion_names=motion_names,
        motion_loop=motion_loop,
    )
