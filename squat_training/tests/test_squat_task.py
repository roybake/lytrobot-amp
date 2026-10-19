from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from lytrobot_amp.env import LytRobotEnvCfg
from lytrobot_amp.motion import MotionReference
from squat_training.envs import SquatCpuAmpEnv, SquatRewardMixin, SquatSensorHistoryCpuAmpEnv


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = REPO_ROOT / "robot" / "MJCFmodelfinal.xml"
REFERENCE = REPO_ROOT / "squat_training" / "reference_data" / "squat_cmu_22_14_omniretarget.npz"
RAW_REFERENCE = REPO_ROOT / "squat_training" / "retargeting" / "results" / "cmu_22_14_squat_lyt_raw.npz"
RETARGET_MODEL = REPO_ROOT / "squat_training" / "retargeting" / "models" / "lyt" / "lyt_14dof.xml"
ORIGINAL_ROOT_DROP = 0.3104939866971852
SHALLOW_ROOT_MIN = 0.7225467562675476
SHALLOW_ROOT_MAX = 0.873674750328064


class SquatTaskTests(unittest.TestCase):
    def test_omniretarget_result_keeps_both_feet_planted(self) -> None:
        import mujoco

        with np.load(RAW_REFERENCE) as archive:
            qpos = np.asarray(archive["qpos"])
        model = mujoco.MjModel.from_xml_path(str(RETARGET_MODEL))
        data = mujoco.MjData(model)
        sole_names = ("right_sole", "right_toe_sole", "left_sole", "left_toe_sole")
        soles = np.zeros((len(qpos), len(sole_names), 3))
        for frame, pose in enumerate(qpos):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            soles[frame] = [data.body(name).xpos for name in sole_names]
        root_drop = float(np.ptp(qpos[:, 2]))
        self.assertGreater(root_drop, 0.12)
        self.assertLess(root_drop, 0.19)
        self.assertAlmostEqual(root_drop, ORIGINAL_ROOT_DROP * 0.5, delta=0.01)
        lowest_frame = int(np.argmin(qpos[:, 2]))
        self.assertGreater(lowest_frame, len(qpos) // 3)
        self.assertLess(lowest_frame, 2 * len(qpos) // 3)
        self.assertGreater(float(qpos[0, 2] - qpos[lowest_frame, 2]), 0.12)
        self.assertGreater(float(qpos[-1, 2] - qpos[lowest_frame, 2]), 0.12)
        self.assertLess(float(np.max(np.abs(soles[:, :, 2]))), 3.0e-4)
        self.assertLess(float(np.max(np.ptp(soles[:, :, :2], axis=0))), 0.01)

    def test_reference_archive_is_cyclic_and_contains_a_squat(self) -> None:
        self.assertTrue(REFERENCE.exists())
        with np.load(REFERENCE) as archive:
            qpos = np.asarray(archive["qpos"])
            qvel = np.asarray(archive["qvel"])
            foot_contact = np.asarray(archive["foot_contact"])
            metadata = str(archive["metadata"])
            self.assertEqual(qpos.shape[1], 21)
            self.assertEqual(qvel.shape, (len(qpos), 20))
            self.assertEqual(foot_contact.shape, (len(qpos), 2))
            self.assertAlmostEqual(float(archive["frequency"]), 40.0)
            self.assertEqual(len(qpos), 240)
            self.assertAlmostEqual(len(qpos) / float(archive["frequency"]), 6.0)
            self.assertTrue(np.isfinite(qpos).all())
            self.assertTrue(np.isfinite(qvel).all())
            np.testing.assert_allclose(foot_contact, 1.0)
            # The 240 samples cover [0, 6 s), so the final sample is just
            # before the standing pose at the next cyclic sample.
            self.assertLess(float(np.max(np.abs(qpos[0] - qpos[-1]))), 2.0e-3)
            self.assertEqual(int(np.argmin(qpos[:, 2])), 120)
            root_drop = float(np.ptp(qpos[:, 2]))
            self.assertGreater(root_drop, 0.12)
            self.assertLess(root_drop, 0.19)
            self.assertAlmostEqual(root_drop, ORIGINAL_ROOT_DROP * 0.5, delta=0.01)
            self.assertIn('"squat_depth_scale": 0.5', metadata)
            # First half is the squat, second half is the recovery to stand.
            self.assertGreater(float(qpos[0, 2] - qpos[120, 2]), 0.12)
            self.assertGreater(float(qpos[-1, 2] - qpos[120, 2]), 0.12)

    def test_actor_receives_sensor_history_and_internal_phase_clock(self) -> None:
        env = SquatSensorHistoryCpuAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(MODEL),
                motion_file=str(REFERENCE),
                num_envs=1,
                target_velocity=0.0,
                stand_start_prob=0.0,
                motion_reset_prob=0.0,
                reset_noise_scale=0.0,
                episode_timeout_jitter_steps=0,
            )
        )
        try:
            observation = env.reset()
            self.assertEqual(env.imu_layout.dim, 42)
            self.assertEqual(env.base_env.position_action_dim, 14)
            self.assertEqual(env.sensor_frame_dim, 56)
            self.assertEqual(env.obs_dim, 338)
            self.assertEqual(env.amp_obs_dim, 270)
            self.assertEqual(observation.shape, (1, 338))
            self.assertEqual(env.observation_metadata["command_dim"], 2)
            self.assertEqual(
                env.observation_metadata["policy_inputs"],
                ("imu", "tendon_length", "internal_phase"),
            )
            self.assertEqual(env.observation_metadata["posture_sensor_inputs"], ("imu", "tendon_length"))
            np.testing.assert_allclose(observation[0, -2:], (0.0, 1.0), atol=1.0e-6)
        finally:
            env.close()

    def test_explicit_double_support_labels_survive_motion_loading(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MODEL))
        motion = MotionReference(REFERENCE, model)
        np.testing.assert_allclose(motion.reference_foot_contact, 1.0)

    def test_squat_reward_does_not_penalize_reference_as_gait_phase_asymmetry(self) -> None:
        env = SquatCpuAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(MODEL),
                motion_file=str(REFERENCE),
                num_envs=1,
                target_velocity=0.0,
                reset_noise_scale=0.0,
            )
        )
        try:
            symmetry = []
            phase_symmetry = []
            zero_action = np.zeros(env.action_dim, dtype=np.float32)
            double_support = np.ones(2, dtype=np.float32)
            for foot_pos in env.motion.reference_foot_pos:
                values = env._symmetry_track(0, foot_pos, double_support, zero_action)
                symmetry.append(float(values[0]))
                phase_symmetry.append(float(values[3]))

            # The inherited walking phase metric is poor for this valid squat,
            # but the squat-specific score must remain close to one.
            self.assertLess(float(np.mean(phase_symmetry)), 0.30)
            self.assertGreater(float(np.mean(symmetry)), 0.90)
        finally:
            env.close()

    def test_squat_reward_excludes_swing_reward_and_makes_falls_negative(self) -> None:
        cfg = LytRobotEnvCfg()
        base_parts = {
            "root_height_value": SHALLOW_ROOT_MAX,
            "reference_root_height_value": SHALLOW_ROOT_MAX,
            "pose_track": 1.0,
            "height": 1.0,
            "root_orientation": 1.0,
            "root_vel": 1.0,
            "vel_track": 1.0,
            "distal_vel": 1.0,
            "foot_pos": 1.0,
            "stance_foot": 1.0,
            "contact": 1.0,
            "symmetry": 1.0,
            "torque_penalty": 0.0,
            "action_rate": 0.0,
            "distal_action_rate": 0.0,
            "missed_contact": 0.0,
            "termination": 0.0,
            "foot_vel": 0.0,
        }

        class RewardSource:
            def _compute_reward_done(self):
                return 0.0, False, False, dict(self.parts)

        class RewardHarness(SquatRewardMixin, RewardSource):
            pass

        harness = RewardHarness()
        harness.cfg = cfg
        harness.motion = type(
            "Motion",
            (),
            {
                "frame_count": 240,
                "reference_root_height": np.asarray([SHALLOW_ROOT_MIN, SHALLOW_ROOT_MAX]),
            },
        )()
        harness.motion_phase = np.asarray([0.0], dtype=np.float32)
        harness.parts = base_parts
        reward_without_swing, _, _, _ = harness._compute_reward_done()
        harness.parts = {**base_parts, "foot_vel": 1.0}
        reward_with_swing, _, _, _ = harness._compute_reward_done()
        self.assertAlmostEqual(float(reward_without_swing), float(reward_with_swing))

        harness.parts = {**base_parts, "foot_vel": 1.0, "termination": 1.0}
        fallen_reward, _, _, _ = harness._compute_reward_done()
        self.assertLess(float(fallen_reward), 0.0)

    def test_missed_stand_is_penalized_during_ascending_phase(self) -> None:
        cfg = LytRobotEnvCfg(symmetry_reward_coef=0.75)
        parts = {
            "root_height_value": 0.68,
            "reference_root_height_value": SHALLOW_ROOT_MIN,
            "pose_track": 0.0,
            "height": 0.0,
            "root_orientation": 1.0,
            "root_vel": 1.0,
            "vel_track": 1.0,
            "distal_vel": 1.0,
            "foot_pos": 1.0,
            "stance_foot": 1.0,
            "contact": 1.0,
            "symmetry": 1.0,
            "torque_penalty": 0.0,
            "action_rate": 0.0,
            "distal_action_rate": 0.0,
            "missed_contact": 0.0,
            "termination": 0.0,
        }

        class RewardSource:
            def _compute_reward_done(self):
                return 0.0, False, False, dict(self.parts)

        class RewardHarness(SquatRewardMixin, RewardSource):
            pass

        harness = RewardHarness()
        harness.cfg = cfg
        harness.motion = type(
            "Motion",
            (),
            {
                "frame_count": 240,
                "reference_root_height": np.asarray([SHALLOW_ROOT_MIN, SHALLOW_ROOT_MAX]),
            },
        )()
        harness.parts = parts
        harness.motion_phase = np.asarray([120.0], dtype=np.float32)
        bottom_reward, _, _, bottom_parts = harness._compute_reward_done()
        harness.motion_phase[:] = 220.0
        harness.parts["reference_root_height_value"] = SHALLOW_ROOT_MAX
        ascent_reward, _, _, ascent_parts = harness._compute_reward_done()

        self.assertEqual(float(bottom_parts["missed_stand"]), 0.0)
        self.assertGreater(float(ascent_parts["missed_stand"]), 0.9)
        self.assertLess(float(ascent_reward), float(bottom_reward) - 2.0)

    def test_bottom_depth_signal_rejects_a_shallow_squat(self) -> None:
        cfg = LytRobotEnvCfg(symmetry_reward_coef=0.75)
        parts = {
            "root_height_value": SHALLOW_ROOT_MAX,
            "reference_root_height_value": SHALLOW_ROOT_MIN,
            "pose_track": 0.0,
            "height": 0.0,
            "root_orientation": 1.0,
            "root_vel": 0.0,
            "vel_track": 0.0,
            "distal_vel": 0.0,
            "foot_pos": 1.0,
            "stance_foot": 1.0,
            "contact": 1.0,
            "symmetry": 1.0,
            "torque_penalty": 0.0,
            "action_rate": 0.0,
            "distal_action_rate": 0.0,
            "missed_contact": 0.0,
            "termination": 0.0,
        }

        class RewardSource:
            def _compute_reward_done(self):
                return 0.0, False, False, dict(self.parts)

        class RewardHarness(SquatRewardMixin, RewardSource):
            pass

        harness = RewardHarness()
        harness.cfg = cfg
        harness.motion = type(
            "Motion",
            (),
            {
                "frame_count": 240,
                "reference_root_height": np.asarray([SHALLOW_ROOT_MIN, SHALLOW_ROOT_MAX]),
            },
        )()
        harness.motion_phase = np.asarray([120.0], dtype=np.float32)
        harness.parts = parts
        shallow_reward, _, _, shallow_parts = harness._compute_reward_done()

        harness.parts = {**parts, "root_height_value": SHALLOW_ROOT_MIN, "height": 1.0}
        full_reward, _, _, full_parts = harness._compute_reward_done()

        self.assertGreater(float(shallow_parts["bottom_depth_deficit"]), 0.95)
        self.assertEqual(float(shallow_parts["bottom_depth_success"]), 0.0)
        self.assertLess(float(shallow_reward), float(full_reward) - 5.0)
        self.assertGreater(float(full_parts["bottom_depth_success"]), 0.95)

    def test_ascent_height_signal_rewards_partial_recovery_outside_gaussian_band(self) -> None:
        cfg = LytRobotEnvCfg(symmetry_reward_coef=0.75)
        parts = {
            "root_height_value": 0.60,
            "reference_root_height_value": SHALLOW_ROOT_MAX,
            "pose_track": 0.0,
            "height": 0.0,
            "root_orientation": 1.0,
            "root_vel": 0.0,
            "vel_track": 0.0,
            "distal_vel": 0.0,
            "foot_pos": 1.0,
            "stance_foot": 1.0,
            "contact": 1.0,
            "symmetry": 1.0,
            "torque_penalty": 0.0,
            "action_rate": 0.0,
            "distal_action_rate": 0.0,
            "missed_contact": 0.0,
            "termination": 0.0,
        }

        class RewardSource:
            def _compute_reward_done(self):
                return 0.0, False, False, dict(self.parts)

        class RewardHarness(SquatRewardMixin, RewardSource):
            pass

        harness = RewardHarness()
        harness.cfg = cfg
        harness.motion = type(
            "Motion",
            (),
            {
                "frame_count": 240,
                "reference_root_height": np.asarray([SHALLOW_ROOT_MIN, SHALLOW_ROOT_MAX]),
            },
        )()
        harness.motion_phase = np.asarray([220.0], dtype=np.float32)
        harness.parts = parts
        crouched_reward, _, _, crouched_parts = harness._compute_reward_done()
        harness.parts = {**parts, "root_height_value": 0.78}
        recovering_reward, _, _, recovering_parts = harness._compute_reward_done()

        # The Gaussian height score is deliberately identical and saturated;
        # the dense signal must still prefer upward recovery.
        self.assertGreater(float(recovering_reward), float(crouched_reward) + 3.0)
        self.assertGreater(
            float(crouched_parts["ascent_height_lag"]),
            float(recovering_parts["ascent_height_lag"]),
        )


if __name__ == "__main__":
    unittest.main()
