"""Physical consistency gates for the repaired reference and its training loader."""

import json
from pathlib import Path

import mujoco
import numpy as np

from lytrobot_amp.motion import MotionReference

ROOT = Path(__file__).resolve().parents[1]


def test_repaired_walk_matches_model_and_periodic_velocities():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot/MJCFmodelfinal.xml"))
    with np.load(ROOT / "robot/reference_data/walk1_mjcf_repaired.npz") as archive:
        qpos, qvel = archive["qpos"].astype(float), archive["qvel"].astype(float)
        translation, hz = archive["cycle_translation"], float(archive["frequency"])
        assert qpos.shape[1] == model.nq and len(qpos)/hz < 5.0
        assert qvel.shape == (len(qpos), model.nv)
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
        assert np.all(archive["foot_contact"].sum(1) >= 1)
        assert "dynamic_stability" in json.loads(str(archive["metadata"]))
    for joint in range(1, model.njnt):
        values = qpos[:, model.jnt_qposadr[joint]]
        assert np.all(values >= model.jnt_range[joint, 0] - 1e-6)
        assert np.all(values <= model.jnt_range[joint, 1] + 1e-6)
    for i in range(len(qpos)):
        before, after = qpos[(i-1) % len(qpos)].copy(), qpos[(i+1) % len(qpos)].copy()
        if i == 0:
            before[:2] -= translation
        if i == len(qpos)-1:
            after[:2] += translation
        velocity = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, velocity, 2/hz, before, after)
        np.testing.assert_allclose(velocity, qvel[i], atol=3e-6)


def test_training_loader_preserves_repaired_pose_and_passive_constraints():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot/MJCFmodelfinal.xml"))
    path = ROOT / "robot/reference_data/walk1_mjcf_repaired.npz"
    reference = MotionReference(path, model, zero_root_xy=False)
    with np.load(path) as archive:
        np.testing.assert_allclose(reference.reference_qpos, archive["qpos"], atol=1e-6)
        np.testing.assert_allclose(reference.reference_qvel, archive["qvel"], atol=1e-6)
    data = mujoco.MjData(model)
    for pose in reference.reference_qpos:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        equality = data.efc_pos[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        assert np.max(np.abs(equality)) < 1e-6
        foot_ids = set(reference.contact_geom_ids)
        floor = model.geom("floor").id
        contacts = [c for c in data.contact if (c.geom1 in foot_ids and c.geom2 == floor) or (c.geom2 in foot_ids and c.geom1 == floor)]
        assert any(c.dist <= 0.002 for c in contacts)
        assert min(c.dist for c in contacts) >= -0.0005


def test_legacy_none_metadata_still_loads(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot/MJCFmodelfinal.xml"))
    with np.load(ROOT / "robot/reference_data/walk1_mjcf_repaired.npz") as archive:
        fields = {key: archive[key] for key in archive.files}
    fields["metadata"] = np.asarray(None, dtype=object)
    path = tmp_path / "none_metadata.npz"
    np.savez_compressed(path, **fields)
    reference = MotionReference(path, model)
    assert not reference.enforce_passive_joint_constraints
