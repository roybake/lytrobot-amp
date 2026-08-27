from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np

from sensor_training.config import SensorObservationConfig
from sensor_training.envs import (
    SensorHistoryCpuAmpEnv,
    _normalized_terrain_context,
)
from sensor_training.observation import HistoryMotionReference, resolve_imu_sensor_layout
from lytrobot_amp.env import LytRobotEnvCfg


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeMotion:
    frame_count = 12
    frequency = 100.0
    feature_dim = 2
    common_joint_names = ("joint_r", "joint_l")
    reference_features = np.arange(24, dtype=np.float32).reshape(12, 2)
    mirrored_reference_features = -reference_features


class _FixedRng:
    def uniform(self, _low: float, _high: float, size: tuple[int, int]) -> np.ndarray:
        return np.full(size, 11.0, dtype=np.float32)


class _FakeMotionLibrary:
    feature_dim = 2
    common_joint_names = ("joint_r", "joint_l")
    frame_counts = np.asarray([12, 8], dtype=np.int32)
    frequencies = np.asarray([100.0, 50.0], dtype=np.float32)
    reference_features = (
        np.arange(24, dtype=np.float32).reshape(12, 2),
        100.0 + np.arange(16, dtype=np.float32).reshape(8, 2),
    )
    mirrored_reference_features = tuple(-value for value in reference_features)

    @staticmethod
    def sample_motion_ids(batch_size: int, _rng: np.random.Generator) -> np.ndarray:
        return np.arange(batch_size, dtype=np.int32) % 2


class SensorObservationTests(unittest.TestCase):
    def test_privileged_terrain_targets_are_continuous_and_bounded(self) -> None:
        cfg = LytRobotEnvCfg(
            rough_terrain_height=0.05,
            slope_angle_deg=6.0,
            stair_step_height=0.20,
            stair_step_depth=0.22,
            gap_terrain_width=0.20,
            terrain_course_base_height=0.30,
        )
        contexts = {
            scenario: _normalized_terrain_context(scenario, cfg)
            for scenario in (
                "flat",
                "rough",
                "slope_up",
                "slope_down",
                "stair_up",
                "stair_down",
                "gap",
                "pit",
            )
        }

        self.assertTrue(all(np.isfinite(value).all() for value in contexts.values()))
        self.assertTrue(all(np.max(np.abs(value)) <= 1.0 for value in contexts.values()))
        self.assertGreater(contexts["slope_up"][0], 0.0)
        self.assertEqual(contexts["slope_up"][0], -contexts["slope_down"][0])
        self.assertGreater(contexts["stair_up"][2], 0.0)
        self.assertEqual(contexts["stair_up"][2], -contexts["stair_down"][2])
        self.assertGreater(contexts["rough"][1], 0.0)
        self.assertGreater(contexts["gap"][4], 0.0)
        self.assertGreater(contexts["pit"][5], 0.0)

    def test_default_history_means_current_plus_five_previous_frames(self) -> None:
        cfg = SensorObservationConfig()
        self.assertEqual(cfg.history_steps, 5)
        self.assertEqual(cfg.frame_count, 6)

    def test_imu_layout_contains_seven_accelerometers_and_seven_gyros(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"))
        layout = resolve_imu_sensor_layout(model, accelerometer_scale=9.81, gyroscope_scale=10.0)
        self.assertEqual(layout.dim, 42)
        self.assertEqual(len(layout.names), 14)
        self.assertEqual(layout.names[0:2], ("rootacc", "rootgyro"))

    def test_amp_reference_history_has_consecutive_frames_and_expected_shape(self) -> None:
        motion = HistoryMotionReference(_FakeMotion(), frame_count=6, policy_dt=0.02)
        features = motion.sample_features(8, _FixedRng(), mirror_prob=0.0)
        self.assertEqual(features.shape, (8, 12))
        sequence = features.reshape(8, 6, 2)
        frame_delta = sequence[:, 1:] - sequence[:, :-1]
        np.testing.assert_allclose(frame_delta, 4.0, atol=1.0e-5)

    def test_amp_interpolation_wraps_rounded_upper_boundary(self) -> None:
        table = np.arange(24, dtype=np.float32).reshape(12, 2)
        phases = np.asarray([12.0, np.nextafter(np.float32(12.0), np.float32(0.0)), -12.0])
        features = HistoryMotionReference._interpolate_table(table, phases)
        self.assertEqual(features.shape, (3, 2))
        np.testing.assert_allclose(features[[0, 2]], table[[0, 0]])
        self.assertTrue(np.isfinite(features).all())

    def test_amp_reference_history_supports_multiple_motion_clips(self) -> None:
        motion = HistoryMotionReference(
            _FakeMotionLibrary(),
            frame_count=6,
            policy_dt=0.02,
        )
        features = motion.sample_features(8, _FixedRng(), mirror_prob=0.0)
        self.assertEqual(features.shape, (8, 12))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features[0::2] < 100.0))
        self.assertTrue(np.all(features[1::2] >= 100.0))

    def test_actor_observation_contains_only_current_imu_and_tendon_frame(self) -> None:
        env = SensorHistoryCpuAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"),
                motion_file=str(REPO_ROOT / "robot" / "reference_data" / "walk1_mjcf.npz"),
                num_envs=1,
                target_velocity=1.0,
                forward_velocity_tracking_sigma=0.5,
                forward_velocity_reward_coef=3.0,
                forward_progress_reward_coef=2.0,
                forward_velocity_deficit_penalty_coef=4.0,
                forward_velocity_excess_penalty_coef=4.0,
                stand_start_prob=0.0,
                motion_reset_prob=0.0,
                reset_noise_scale=0.0,
            )
        )
        try:
            observation = env.reset()
            self.assertEqual(env.sensor_frame_dim, 56)
            self.assertEqual(env.obs_dim, 56)
            self.assertEqual(env.sensor_history_obs_dim, 336)
            self.assertEqual(observation.shape, (1, 56))
            self.assertEqual(
                env.observation_metadata["actor_input_fields"],
                ("imu", "tendon_length"),
            )
            self.assertEqual(env.observation_metadata["command_fields"], ())
            self.assertEqual(env.observation_metadata["command_dim"], 0)
            self.assertTrue(env.observation_metadata["strict_sensor_only"])
            self.assertFalse(
                env.observation_metadata["actor_uses_stacked_history"]
            )
            self.assertFalse(
                env.observation_metadata["privileged_training_context"][
                    "actor_input"
                ]
            )
            self.assertNotEqual(env.privileged_context_dim, 0)
            self.assertEqual(
                env.privileged_observe().shape,
                (1, env.privileged_context_dim),
            )
            observation_before_command_change = observation.copy()
            env.base_env.cfg.target_velocity = 3.0
            np.testing.assert_allclose(env.observe(), observation_before_command_change)
            action = np.zeros((1, env.action_dim), dtype=np.float32)
            _observation, _reward, _done, info = env.step(action)
            expected_deficit = np.clip(
                (3.0 - float(info["forward_velocity"][0])) / 3.0,
                0.0,
                2.0,
            )
            self.assertAlmostEqual(float(info["forward_velocity_deficit"][0]), expected_deficit, places=5)
            expected_excess = np.clip(
                (float(info["forward_velocity"][0]) - 3.0) / 3.0,
                0.0,
                2.0,
            )
            self.assertAlmostEqual(float(info["forward_velocity_excess"][0]), expected_excess, places=5)
            self.assertIn("termination", info)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
