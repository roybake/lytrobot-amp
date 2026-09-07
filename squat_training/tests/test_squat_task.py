from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lytrobot_amp.env import LytRobotEnvCfg
from lytrobot_amp.motion import MotionReference
from squat_training.envs import SquatCpuAmpEnv, SquatRewardMixin, SquatSensorHistoryCpuAmpEnv
from squat_training.play import squat_play_defaults
from squat_training import train as squat_train
from squat_training.train import squat_training_defaults


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = REPO_ROOT / "robot" / "MJCFmodelposition.xml"
REFERENCE = REPO_ROOT / "squat_training" / "reference_data" / "squat_cmu_22_14_omniretarget.npz"
RAW_REFERENCE = REPO_ROOT / "squat_training" / "retargeting" / "results" / "cmu_22_14_squat_lyt_raw.npz"
RETARGET_MODEL = REPO_ROOT / "squat_training" / "retargeting" / "models" / "lyt" / "lyt_14dof.xml"
ORIGINAL_ROOT_DROP = 0.3104939866971852
SHALLOW_ROOT_MIN = 0.7225467562675476
SHALLOW_ROOT_MAX = 0.873674750328064


class SquatTaskTests(unittest.TestCase):
    def test_training_defaults_bound_recurrent_policy_updates(self) -> None:
        defaults = squat_training_defaults()

        self.assertEqual(defaults["--learning-rate"], "1e-4")
        self.assertEqual(defaults["--max-learning-rate"], "1e-4")
        self.assertEqual(defaults["--target-kl"], "0.01")
        self.assertEqual(defaults["--target-kl-stop-multiplier"], "1.5")
        self.assertEqual(defaults["--clip-coef"], "0.10")
        self.assertEqual(defaults["--epochs"], "2")
        self.assertEqual(defaults["--stand-start-prob"], "1.0")
        self.assertEqual(defaults["--motion-reset-prob"], "0.0")
        self.assertEqual(defaults["--min-upright-z"], "0.65")
        self.assertEqual(defaults["--squat-cycle-completion-bonus"], "5.0")

    def test_playback_starts_from_stable_standing(self) -> None:
        defaults = squat_play_defaults()

        self.assertEqual(defaults["--stand-start-prob"], "1.0")
        self.assertEqual(defaults["--target-velocity"], "0.0")
        self.assertNotIn("--record-dir", defaults)

    def test_training_requires_survival_after_a_completed_cycle(self) -> None:
        with (
            patch.object(sys, "argv", ["train.py"]),
            patch.object(squat_train.base_train, "main") as base_main,
        ):
            squat_train.main()

            base_main.assert_called_once_with()
            self.assertIn("--no-squat-terminate-on-cycle-completion", sys.argv)
            self.assertNotIn("--squat-terminate-on-cycle-completion", sys.argv)

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

    def test_actor_receives_only_current_imu_and_tendon_frame(self) -> None:
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
                squat_phase_feedback=True,
                squat_terminate_on_cycle_completion=True,
            )
        )
        try:
            observation = env.reset()
            self.assertEqual(env.imu_layout.dim, 42)
            self.assertEqual(env.base_env.position_action_dim, 14)
            self.assertEqual(env.sensor_frame_dim, 56)
            self.assertEqual(env.sensor_history_obs_dim, 336)
            self.assertEqual(env.obs_dim, 56)
            self.assertEqual(env.amp_obs_dim, 306)
            self.assertEqual(observation.shape, (1, 56))
            self.assertEqual(env.observation_metadata["command_dim"], 0)
            self.assertEqual(env.observation_metadata["command_fields"], ())
            self.assertEqual(
                env.observation_metadata["policy_inputs"],
                ("imu", "tendon_length"),
            )
            self.assertEqual(env.observation_metadata["posture_sensor_inputs"], ("imu", "tendon_length"))
            self.assertEqual(
                env.observation_metadata["phase_source"],
                "training_only_asymmetric_critic",
            )
            self.assertEqual(
                env.observation_metadata["phase_feedback_inputs"],
                ("tendon_length", "imu_gravity_direction"),
            )
            self.assertTrue(env.observation_metadata["phase_feedback"]["enabled"])
            self.assertEqual(
                len(env.observation_metadata["phase_feedback"]["depth_direction"]),
                14,
            )
            self.assertEqual(
                len(env.observation_metadata["phase_feedback"]["stand_tendon_length"]),
                14,
            )
            self.assertEqual(
                env.observation_metadata["phase_feedback"]["stall_threshold"],
                0.25,
            )
            self.assertEqual(
                env.observation_metadata["phase_feedback"]["max_stall_steps"],
                80,
            )
            self.assertEqual(
                env.observation_metadata["success_definition"],
                "completed_stand_squat_stand_cycle",
            )
            self.assertTrue(
                env.observation_metadata["phase_feedback"][
                    "terminate_on_cycle_completion"
                ]
            )
            self.assertTrue(env.observation_metadata["strict_sensor_only"])
            self.assertFalse(
                env.observation_metadata["privileged_training_context"]["actor_input"]
            )
            self.assertEqual(
                env.privileged_observe().shape,
                (1, env.privileged_context_dim),
            )
        finally:
            env.close()

    def test_phase_feedback_blocks_turns_until_sensor_posture_is_reached(self) -> None:
        import mujoco

        env = SquatCpuAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(MODEL),
                motion_file=str(REFERENCE),
                num_envs=1,
                target_velocity=0.0,
                stand_start_prob=0.0,
                motion_reset_prob=0.0,
                reset_noise_scale=0.0,
                episode_timeout_jitter_steps=0,
                squat_phase_feedback=True,
            )
        )
        try:
            data = env.datas[0]
            stand_qpos, stand_qvel = env.motion.state_at(0)
            bottom_qpos, bottom_qvel = env.motion.state_at(120)

            data.qpos[:] = stand_qpos
            data.qvel[:] = stand_qvel
            mujoco.mj_forward(env.model, data)
            env.motion_phase[:] = 119.5
            blocked_bottom = env._squat_feedback_phase_scale()
            self.assertEqual(float(blocked_bottom[0]), 0.0)
            self.assertLess(float(env.squat_depth_sensor_progress[0]), 0.15)
            self.assertEqual(float(env._motion_phase_progress()[0]), 0.0)
            # Feedback must affect only clock advancement, not the shared
            # stand-start target interpolation used by rewards and controls.
            self.assertEqual(float(env._stand_start_progress()[0]), 1.0)

            data.qpos[:] = bottom_qpos
            data.qvel[:] = bottom_qvel
            mujoco.mj_forward(env.model, data)
            allowed_bottom = env._squat_feedback_phase_scale()
            self.assertGreater(float(allowed_bottom[0]), 0.99)
            self.assertGreater(float(env.squat_depth_sensor_progress[0]), 0.85)

            env.motion_phase[:] = 239.5
            blocked_stand = env._squat_feedback_phase_scale()
            self.assertEqual(float(blocked_stand[0]), 0.0)

            data.qpos[:] = stand_qpos
            data.qvel[:] = stand_qvel
            mujoco.mj_forward(env.model, data)
            allowed_stand = env._squat_feedback_phase_scale()
            self.assertGreater(float(allowed_stand[0]), 0.99)
            self.assertLess(float(env.squat_depth_sensor_progress[0]), 0.15)
        finally:
            env.close()

    def test_cycle_state_requires_stand_bottom_stand_and_resets_stall_budget(self) -> None:
        env = SquatCpuAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(MODEL),
                motion_file=str(REFERENCE),
                num_envs=1,
                target_velocity=0.0,
                motion_reset_prob=0.0,
                reset_noise_scale=0.0,
                squat_phase_feedback=True,
            )
        )
        try:
            completed, episode_success = env._update_squat_cycle_state(
                np.asarray([0.05], dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
            )
            self.assertEqual(float(completed), 0.0)
            self.assertEqual(float(episode_success), 0.0)
            self.assertEqual(int(env.squat_cycle_stage[0]), 1)

            completed, _ = env._update_squat_cycle_state(
                np.asarray([0.90], dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
            )
            self.assertEqual(float(completed), 0.0)
            self.assertEqual(int(env.squat_cycle_stage[0]), 2)

            env.squat_phase_stall_steps[0] = 7
            completed, episode_success = env._update_squat_cycle_state(
                np.asarray([0.05], dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
            )
            self.assertEqual(float(completed), 1.0)
            self.assertEqual(float(episode_success), 1.0)
            self.assertEqual(int(env.squat_cycle_stage[0]), 1)
            self.assertEqual(int(env.squat_cycle_count[0]), 1)
            self.assertEqual(int(env.squat_phase_stall_steps[0]), 0)
        finally:
            env.close()

    def test_completed_cycle_immediately_ends_as_success(self) -> None:
        import mujoco

        env = SquatCpuAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(MODEL),
                motion_file=str(REFERENCE),
                num_envs=1,
                target_velocity=0.0,
                stand_start_prob=0.0,
                motion_reset_prob=0.0,
                reset_noise_scale=0.0,
                episode_timeout_jitter_steps=0,
                squat_phase_feedback=True,
                squat_terminate_on_cycle_completion=True,
            )
        )
        try:
            stand_qpos, _stand_qvel = env.motion.state_at(0)
            env.datas[0].qpos[:] = stand_qpos
            env.datas[0].qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.datas[0])
            env.motion_phase[:] = 239.0
            env.squat_cycle_stage[:] = 2

            _obs, _reward, done, info = env.step(
                np.zeros((1, env.action_dim), dtype=np.float32)
            )

            self.assertEqual(float(done[0]), 1.0)
            self.assertEqual(float(info["time_outs"][0]), 1.0)
            self.assertEqual(float(info["successes"][0]), 1.0)
            self.assertEqual(float(info["squat_cycle_completed"][0]), 1.0)
        finally:
            env.close()

    def test_completed_cycle_does_not_end_default_survival_episode(self) -> None:
        import mujoco

        env = SquatCpuAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(MODEL),
                motion_file=str(REFERENCE),
                num_envs=1,
                target_velocity=0.0,
                stand_start_prob=0.0,
                motion_reset_prob=0.0,
                reset_noise_scale=0.0,
                episode_timeout_jitter_steps=0,
                squat_phase_feedback=True,
                squat_terminate_on_cycle_completion=False,
            )
        )
        try:
            stand_qpos, _stand_qvel = env.motion.state_at(0)
            env.datas[0].qpos[:] = stand_qpos
            env.datas[0].qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.datas[0])
            env.motion_phase[:] = 239.0
            env.squat_cycle_stage[:] = 2

            _obs, _reward, done, info = env.step(
                np.zeros((1, env.action_dim), dtype=np.float32)
            )

            self.assertEqual(float(done[0]), 0.0)
            self.assertEqual(float(info["time_outs"][0]), 0.0)
            self.assertEqual(float(info["successes"][0]), 0.0)
            self.assertEqual(float(info["squat_cycle_completed"][0]), 1.0)
            self.assertEqual(float(info["squat_episode_success"][0]), 1.0)
        finally:
            env.close()

    def test_stalled_phase_cannot_earn_positive_reward_and_times_out_as_failure(self) -> None:
        cfg = LytRobotEnvCfg(
            symmetry_reward_coef=0.75,
            squat_phase_max_stall_steps=2,
            squat_phase_stall_penalty=0.25,
        )
        parts = {
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
            "ctrl_track": 1.0,
            "ctrl_penalty": 0.0,
            "action_rate": 0.0,
            "distal_action": 0.0,
            "distal_action_rate": 0.0,
            "missed_contact": 0.0,
            "termination": 0.0,
        }

        class RewardSource:
            def _compute_reward_done(self):
                return (
                    0.0,
                    bool(getattr(self, "base_done", False)),
                    bool(getattr(self, "base_timed_out", False)),
                    dict(self.parts),
                )

        class RewardHarness(SquatRewardMixin, RewardSource):
            pass

        harness = RewardHarness()
        harness.cfg = cfg
        harness.motion = type(
            "Motion",
            (),
            {
                "frame_count": 240,
                "reference_root_height": np.asarray(
                    [SHALLOW_ROOT_MIN, SHALLOW_ROOT_MAX]
                ),
            },
        )()
        harness.motion_phase = np.asarray([60.0], dtype=np.float32)
        harness.parts = parts
        harness._squat_phase_feedback_ready = True
        harness.squat_phase_feedback_scale = np.asarray([0.0], dtype=np.float32)
        harness.squat_phase_stalled = np.asarray([True])
        harness.squat_phase_stall_steps = np.asarray([1], dtype=np.int32)
        harness.squat_cycle_stage = np.asarray([0], dtype=np.int32)
        harness.squat_cycle_count = np.asarray([0], dtype=np.int32)
        harness.squat_depth_sensor_progress = np.asarray([0.5], dtype=np.float32)
        harness._squat_feedback_measurements = lambda: (
            np.asarray([0.5], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        )

        reward, done, timed_out, reward_parts = harness._compute_reward_done()
        self.assertLess(float(reward), 0.0)
        self.assertFalse(done)
        self.assertFalse(timed_out)
        self.assertEqual(float(reward_parts["phase_stalled"]), 1.0)
        self.assertEqual(float(reward_parts["phase_stall_timeout"]), 0.0)

        harness.squat_phase_stall_steps[:] = 2
        reward, done, timed_out, reward_parts = harness._compute_reward_done()
        self.assertLess(float(reward), -10.0)
        self.assertTrue(done)
        self.assertFalse(timed_out)
        self.assertEqual(float(reward_parts["phase_stall_timeout"]), 1.0)
        self.assertEqual(float(reward_parts["termination"]), 1.0)

        harness.squat_phase_feedback_scale[:] = 1.0
        harness.squat_phase_stalled[:] = False
        harness.squat_phase_stall_steps[:] = 0
        harness.base_done = True
        harness.base_timed_out = True
        reward, done, timed_out, reward_parts = harness._compute_reward_done()
        self.assertLess(float(reward), -10.0)
        self.assertTrue(done)
        self.assertFalse(timed_out)
        self.assertEqual(float(reward_parts["incomplete_squat_cycle"]), 1.0)

    def test_squat_success_requires_completed_cycle_on_successful_end(self) -> None:
        class StepSource:
            def step(self, _action):
                return (
                    np.zeros((2, 1), dtype=np.float32),
                    np.zeros(2, dtype=np.float32),
                    np.ones(2, dtype=np.float32),
                    {
                        "time_outs": np.ones(2, dtype=np.float32),
                        "squat_episode_success": np.asarray(
                            [1.0, 0.0], dtype=np.float32
                        ),
                    },
                )

        class StepHarness(SquatRewardMixin, StepSource):
            pass

        _, _, _, info = StepHarness().step(np.zeros((2, 1), dtype=np.float32))
        np.testing.assert_array_equal(info["successes"], [1.0, 0.0])

    def test_torch_stall_timeout_matches_warp_vector_semantics(self) -> None:
        cfg = LytRobotEnvCfg(
            symmetry_reward_coef=0.75,
            squat_phase_max_stall_steps=2,
            squat_phase_stall_penalty=0.25,
            squat_terminate_on_cycle_completion=True,
        )
        scalar_parts = {
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
            "ctrl_track": 1.0,
            "ctrl_penalty": 0.0,
            "action_rate": 0.0,
            "distal_action": 0.0,
            "distal_action_rate": 0.0,
            "missed_contact": 0.0,
            "termination": 0.0,
        }

        class RewardSource:
            def _compute_reward_done(self):
                parts = {
                    key: torch.full((2,), value, dtype=torch.float32)
                    for key, value in self.scalar_parts.items()
                }
                return (
                    torch.zeros(2),
                    torch.zeros(2, dtype=torch.bool),
                    torch.zeros(2, dtype=torch.bool),
                    parts,
                )

        class RewardHarness(SquatRewardMixin, RewardSource):
            pass

        harness = RewardHarness()
        harness.cfg = cfg
        harness.motion = type(
            "Motion",
            (),
            {
                "frame_count": 240,
                "reference_root_height": np.asarray(
                    [SHALLOW_ROOT_MIN, SHALLOW_ROOT_MAX]
                ),
            },
        )()
        harness.motion_phase = torch.asarray([60.0, 60.0])
        harness.scalar_parts = scalar_parts
        harness._squat_phase_feedback_ready = True
        harness.squat_phase_feedback_scale = torch.asarray([0.0, 1.0])
        harness.squat_phase_stalled = torch.asarray([True, False])
        harness.squat_phase_stall_steps = torch.asarray([2, 0])
        harness.squat_cycle_stage = torch.asarray([0, 2], dtype=torch.long)
        harness.squat_cycle_count = torch.zeros(2, dtype=torch.long)
        harness.squat_depth_sensor_progress = torch.full((2,), 0.5)
        harness._squat_feedback_measurements = lambda: (
            torch.asarray([0.5, 0.05]),
            torch.ones(2),
        )

        reward, done, timed_out, reward_parts = harness._compute_reward_done()
        self.assertLess(float(reward[0]), -10.0)
        self.assertGreater(float(reward[1]), 0.0)
        torch.testing.assert_close(done, torch.asarray([True, True]))
        torch.testing.assert_close(timed_out, torch.asarray([False, True]))
        torch.testing.assert_close(
            reward_parts["phase_stall_timeout"],
            torch.asarray([1.0, 0.0]),
        )
        torch.testing.assert_close(
            reward_parts["squat_cycle_completed"],
            torch.asarray([0.0, 1.0]),
        )

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
            "ctrl_track": 1.0,
            "ctrl_penalty": 0.0,
            "action_rate": 0.0,
            "distal_action": 0.0,
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
            "ctrl_track": 0.0,
            "ctrl_penalty": 0.0,
            "action_rate": 0.0,
            "distal_action": 0.0,
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
            "ctrl_track": 0.0,
            "ctrl_penalty": 0.0,
            "action_rate": 0.0,
            "distal_action": 0.0,
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
            "ctrl_track": 0.0,
            "ctrl_penalty": 0.0,
            "action_rate": 0.0,
            "distal_action": 0.0,
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
