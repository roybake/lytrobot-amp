from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np

from sensor_training.config import SensorObservationConfig
from sensor_training.envs import SensorHistoryCpuAmpEnv
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


class SensorObservationTests(unittest.TestCase):
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

    def test_actor_observation_appends_current_forward_velocity_command(self) -> None:
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
            self.assertEqual(env.obs_dim, 337)
            self.assertEqual(observation.shape, (1, 337))
            self.assertAlmostEqual(float(observation[0, -1]), 1.0)
            self.assertEqual(env.observation_metadata["command_fields"], ("target_forward_velocity",))
            action = np.zeros((1, env.action_dim), dtype=np.float32)
            _observation, _reward, _done, info = env.step(action)
            expected_deficit = np.clip(1.0 - float(info["forward_velocity"][0]), 0.0, 2.0)
            self.assertAlmostEqual(float(info["forward_velocity_deficit"][0]), expected_deficit, places=5)
            expected_excess = np.clip(float(info["forward_velocity"][0]) - 1.0, 0.0, 2.0)
            self.assertAlmostEqual(float(info["forward_velocity_excess"][0]), expected_excess, places=5)
            self.assertIn("termination", info)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
