from pathlib import Path

import mujoco
import numpy as np

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.motion import MotionLibrary
from lytrobot_amp.terrain import StairTerrainCfg, stair_boxes, terrain_height_at


ROOT = Path(__file__).resolve().parents[1]
MOTIONS = ROOT / "robot" / "reference_data" / "multimotion"
NAMES = ("walk", "turn_left", "turn_right", "stair_up", "stair_down")


def motion_paths() -> tuple[str, ...]:
    return tuple(str(MOTIONS / f"{name}_omniretarget.npz") for name in NAMES)


def test_more_stair_layout_has_ascent_platform_and_descent() -> None:
    cfg = StairTerrainCfg(
        enabled=True,
        step_height=0.08,
        step_depth=0.30,
        step_count=5,
        start_x=-1.2,
    )
    boxes = stair_boxes(cfg)
    assert len(boxes) == 11
    assert [round(float(center[2] + size[2]), 2) for _, center, size in boxes[:5]] == [
        0.08,
        0.16,
        0.24,
        0.32,
        0.40,
    ]
    samples = np.asarray(
        [cfg.ascent_start + 0.15, cfg.top_start + 0.10, cfg.descent_start + 1.35]
    )
    np.testing.assert_allclose(terrain_height_at(samples, cfg), [0.08, 0.40, 0.08])


def test_motion_library_balances_clips_and_uses_shared_amp_normalization() -> None:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot" / "MJCFmodelposition.xml"))
    library = MotionLibrary(
        motion_paths(),
        model,
        motion_names=NAMES,
        motion_weights=(0.30, 0.10, 0.10, 0.25, 0.25),
        motion_loop=(True, False, False, False, False),
    )
    assert library.motion_count == 5
    assert library.feature_dim == 45
    assert library.frame_counts.tolist() == [48, 173, 161, 141, 157]
    assert [motion.frequency for motion in library.motions] == [40.0] * 5
    features = library.sample_features(128, np.random.default_rng(7))
    assert features.shape == (128, 45)
    assert np.isfinite(features).all()


def test_cpu_multimotion_environment_observation_and_step() -> None:
    cfg = LytRobotEnvCfg(
        motion_files=motion_paths(),
        motion_names=NAMES,
        motion_weights=(0.20, 0.20, 0.20, 0.20, 0.20),
        motion_loop=(True, False, False, False, False),
        num_envs=4,
        stair_terrain=True,
        stair_step_height=0.13,
        stair_step_depth=0.58,
        stair_step_count=3,
        stair_start_x=-0.95,
        stair_down_offset=0.25,
        motion_reset_prob=1.0,
        stand_start_prob=0.0,
        use_reference_velocity=True,
        domain_randomization=False,
        episode_timeout_jitter_steps=0,
    )
    env = LytRobotAmpEnv(cfg)
    try:
        assert env.obs_dim == 132
        assert env.last_obs.shape == (4, 132)
        assert env.amp_observe().shape == (4, 45)
        observation, reward, done, _ = env.step(
            np.zeros((4, env.action_dim), dtype=np.float32)
        )
        assert observation.shape == (4, 132)
        assert reward.shape == done.shape == (4,)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward).all()
    finally:
        env.close()


def test_retargeted_clips_have_no_leg_overlap_and_controlled_turn_rate() -> None:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot" / "MJCFmodelposition.xml"))
    data = mujoco.MjData(model)
    body_pairs = (
        ("femur_r", "femur_l"),
        ("tibia_r", "tibia_l"),
        ("foot_r", "foot_l"),
        ("foot_r", "toe_l"),
        ("toe_r", "foot_l"),
        ("toe_r", "toe_l"),
    )

    def body_geom(name: str) -> int:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return int(np.flatnonzero(np.asarray(model.geom_bodyid) == body_id)[0])

    pair_ids = [(body_geom(first), body_geom(second)) for first, second in body_pairs]
    fromto = np.zeros(6, dtype=np.float64)
    for name, path in zip(NAMES, motion_paths(), strict=True):
        with np.load(path) as archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            qvel = np.asarray(archive["qvel"], dtype=np.float64)
            frequency = float(archive["frequency"])
            roll_mode = np.asarray(archive["foot_roll_mode"], dtype=np.int8)
        assert frequency == 40.0
        assert set(np.unique(roll_mode)).issubset({0, 1, 2, 3})
        for pose in qpos:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            distances = [
                mujoco.mj_geomDistance(
                    model,
                    data,
                    first,
                    second,
                    0.20,
                    fromto,
                )
                for first, second in pair_ids
            ]
            assert min(distances) >= -1.0e-5, (name, min(distances))
        if name in {"turn_left", "turn_right"}:
            quat = qpos[:, 3:7]
            yaw = np.unwrap(
                np.arctan2(
                    2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                    1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
                )
            )
            yaw_change = float(np.rad2deg(yaw[-1] - yaw[0]))
            assert abs(yaw_change) <= 90.0 + 0.1
            assert yaw_change > 0.0 if name == "turn_left" else yaw_change < 0.0
            assert (len(qpos) - 1) / frequency >= 3.5
            assert np.rad2deg(np.max(np.abs(qvel[:, 5]))) <= 50.0
