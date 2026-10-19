"""The 1.28 m/s reference must describe supported, natural robot motion."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.gait_constraints import NATURAL_STEP_INTERVAL_SECONDS, STEP_INTERVAL_SIGMA_SECONDS
from lytrobot_amp.motion import MotionReference
from scripts.repair_walk_reference import audit


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "robot/reference_data/walk1_mjcf_1p28_natural.npz"


def test_natural_reference_speed_support_no_slip_and_knees():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot/MJCFmodelfinal.xml"))
    with np.load(REFERENCE) as data:
        positions, velocities = data["qpos"].astype(float), data["qvel"].astype(float)
        frequency = float(data["frequency"])
        translation = data["cycle_translation"].astype(float)
        contact = data["foot_contact"]
        metadata = json.loads(str(data["metadata"]))
    duration = len(positions) / frequency
    assert duration == pytest.approx(.765)
    assert translation[0] / duration == pytest.approx(1.28, abs=1e-6)
    assert velocities[:, 0].mean() == pytest.approx(1.28, abs=1e-6)
    assert np.min(-positions[:, [10, 18]]) >= .25 - 1e-6
    assert metadata["cycle_refinement"]["optimizer_success"]
    metrics = audit(model, positions, velocities, frequency, contact, translation)
    assert metrics["joint_limit_max_excess"] <= 1e-6
    assert metrics["minimum_foot_clearance_m"] >= -.0005
    assert metrics["frames_without_foot_within_2mm"] == 0
    assert metrics["qvel_differentiation_rmse"] < 1e-5
    assert metrics["stance_interior_velocity_rms_m_s"] < .15
    assert metrics["swing_interior_contact_count"] == 0
    assert metrics["knee_lock_fraction"] == 0
    # The raw transition metric remains visible in the audit.
    assert metrics["contact_horizontal_velocity_rms_m_s"] >= metrics["stance_interior_velocity_rms_m_s"]
    original = ROOT / "robot/reference_data/walk1_mjcf.npz"
    import hashlib
    assert hashlib.sha256(original.read_bytes()).hexdigest() == metadata["source_sha256"]


def test_training_uses_original_fixed_cadence_reward_without_extra_actor_inputs():
    cfg = LytRobotEnvCfg(
        motion_file=str(ROOT / "robot/reference_data/walk1_mjcf.npz"), motion_loop=(True,), num_envs=1,
        task_reward_mode="stage1_reference", target_velocity=1.28,
        stand_start_prob=0., reset_noise_scale=0., domain_randomization=False,
        sensor_natural_gait_reward_coef=1., termination_mode="fall", episode_steps=0,
    )
    env = LytRobotAmpEnv(cfg)
    try:
        env.sensor_cadence_seen[:] = True
        env.sensor_step_interval_ema[:] = .4
        _, _, _, info = env.step(np.zeros((1, env.action_dim), dtype=np.float32))
        expected = np.exp(-abs(.4 - NATURAL_STEP_INTERVAL_SECONDS)
                          / STEP_INTERVAL_SIGMA_SECONDS)
        assert info["sensor_step_cadence"][0] == pytest.approx(expected)
        assert np.isfinite(env.observe()).all()
    finally:
        env.close()
