"""Regression checks for reference-to-MJCF contact consistency."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.audit_walk_contact import audit_reference, check_quality

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "robot/MJCFmodelposition.xml"
REFERENCE = ROOT / "robot/reference_data/walk1_mjcf_repaired.npz"


def test_canonical_walk_reference_has_feasible_ground_contacts() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    report = audit_reference(model, REFERENCE)
    check_quality(report)
    assert report["frames"] == 45
    assert abs(report["mean_forward_speed_mps"] - 0.87) < 1e-5


def test_audit_rejects_both_feet_above_floor(tmp_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    with np.load(REFERENCE, allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files if key != "metadata"}
    fields["qpos"] = fields["qpos"].copy()
    fields["qpos"][:, 2] += 0.10
    raised_path = tmp_path / "floating.npz"
    np.savez_compressed(raised_path, **fields)

    report = audit_reference(model, raised_path)
    with pytest.raises(RuntimeError, match="maximum_support_gap_m"):
        check_quality(report)


def test_audit_rejects_joint_layout_mismatch(tmp_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    with np.load(REFERENCE, allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files if key != "metadata"}
    fields["joint_names"] = fields["joint_names"][::-1]
    wrong_path = tmp_path / "wrong_order.npz"
    np.savez_compressed(wrong_path, **fields)

    with pytest.raises(ValueError, match="Joint names/order"):
        audit_reference(model, wrong_path)
