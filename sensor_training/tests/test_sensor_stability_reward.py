"""Tests for the phase-independent sensor task reward."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sensor_training.envs import SensorHistoryCpuAmpEnv, sensor_stability_task_reward
from lytrobot_amp.env import LytRobotEnvCfg
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv


def _info(
    *,
    speed: float,
    progress: float,
    deficit: float,
    termination: float = 0.0,
    knee_flexion: float = 1.0,
):
    return {
        "sensor_upright": np.asarray([1.0], dtype=np.float32),
        "sensor_height": np.asarray([1.0], dtype=np.float32),
        "stable_forward_velocity_track": np.asarray([speed], dtype=np.float32),
        "stable_forward_progress": np.asarray([progress], dtype=np.float32),
        "forward_velocity_deficit": np.asarray([deficit], dtype=np.float32),
        "gait_progress": np.ones(1, dtype=np.float32),
        "knee_flexion_track": np.asarray([knee_flexion], dtype=np.float32),
        "torque_penalty": np.zeros(1, dtype=np.float32),
        "action_rate": np.zeros(1, dtype=np.float32),
        "distal_action_rate": np.zeros(1, dtype=np.float32),
        "termination": np.asarray([termination], dtype=np.float32),
    }


class SensorStabilityRewardTests(unittest.TestCase):
    def test_cpu_sensor_reward_does_not_depend_on_reference_root_pose(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env = SensorHistoryCpuAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(repo_root / "robot" / "MJCFmodelfinal.xml"),
                motion_file=str(
                    repo_root / "robot" / "reference_data" / "walk1_mjcf_repaired.npz"
                ),
                num_envs=1,
                sensor_stability_reward=True,
                stand_start_prob=0.0,
                motion_reset_prob=0.0,
                reset_noise_scale=0.0,
                domain_randomization=False,
            )
        )
        try:
            env.reset()
            base = env.base_env
            data = base.datas[0]
            targets = base._targets_at_current_phase()
            changed_targets = replace(
                targets,
                root_quat=np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                root_height=np.asarray([1.3], dtype=np.float32),
            )
            foot_pos = base._foot_positions_local(data)
            foot_world_pos = base._foot_positions_world(data)
            foot_geom_contact = base._foot_geom_contacts(data)
            foot_contact = base._side_contacts_from_geom(foot_geom_contact)

            def evaluate(reference):
                _reward, done, _timed_out, info = base._compute_reward_done(
                    data,
                    np.zeros(base.action_dim, dtype=np.float32),
                    np.zeros(base.action_dim, dtype=np.float32),
                    0,
                    0.5,
                    reference,
                    foot_pos,
                    np.zeros_like(foot_pos),
                    np.zeros_like(foot_world_pos),
                    foot_geom_contact,
                    foot_contact,
                )
                sensor_reward = sensor_stability_task_reward(
                    {key: np.asarray([value], dtype=np.float32) for key, value in info.items()},
                    fall_penalty=25.0,
                    stall_penalty=10.0,
                )
                return done, info, sensor_reward

            first_done, first_info, first_reward = evaluate(targets)
            second_done, second_info, second_reward = evaluate(changed_targets)
            self.assertEqual(first_done, second_done)
            self.assertAlmostEqual(
                first_info["stable_forward_velocity_track"],
                second_info["stable_forward_velocity_track"],
            )
            self.assertAlmostEqual(first_info["sensor_height"], second_info["sensor_height"])
            np.testing.assert_allclose(first_reward, second_reward)
        finally:
            env.close()

    def test_walking_outscores_standing(self) -> None:
        walking = sensor_stability_task_reward(
            _info(speed=1.0, progress=1.0, deficit=0.0),
            fall_penalty=25.0,
            stall_penalty=10.0,
        )
        standing = sensor_stability_task_reward(
            _info(speed=0.0, progress=0.0, deficit=1.0),
            fall_penalty=25.0,
            stall_penalty=10.0,
        )

        self.assertGreater(float(walking[0]), float(standing[0]))
        self.assertLess(float(standing[0]), 0.0)

    def test_fall_overrides_all_dense_reward(self) -> None:
        reward = sensor_stability_task_reward(
            _info(speed=1.0, progress=1.0, deficit=0.0, termination=1.0),
            fall_penalty=25.0,
            stall_penalty=10.0,
        )
        self.assertEqual(float(reward[0]), -25.0)

    def test_flexed_knees_outscore_the_locked_knee_shortcut(self) -> None:
        flexed = sensor_stability_task_reward(
            _info(speed=1.0, progress=1.0, deficit=0.0, knee_flexion=1.0),
            fall_penalty=25.0,
            stall_penalty=10.0,
            knee_gait_reward_coef=1.0,
        )
        locked = sensor_stability_task_reward(
            _info(speed=1.0, progress=1.0, deficit=0.0, knee_flexion=0.0),
            fall_penalty=25.0,
            stall_penalty=10.0,
            knee_gait_reward_coef=1.0,
        )

        self.assertAlmostEqual(float(flexed[0] - locked[0]), 1.0)

    def test_short_equal_steps_lose_to_full_equal_steps(self) -> None:
        short = _info(speed=1.0, progress=1.0, deficit=0.0)
        full = _info(speed=1.0, progress=1.0, deficit=0.0)
        for info, stride_score in ((short, 0.15), (full, 1.0)):
            info["sensor_mirror"] = np.asarray([1.0], dtype=np.float32)
            info["sensor_stride_length"] = np.asarray([stride_score], dtype=np.float32)
        short_reward = sensor_stability_task_reward(
            short, fall_penalty=25.0, stall_penalty=10.0,
            natural_gait_reward_coef=8.0,
        )
        full_reward = sensor_stability_task_reward(
            full, fall_penalty=25.0, stall_penalty=10.0,
            natural_gait_reward_coef=8.0,
        )
        self.assertGreater(float(full_reward[0] - short_reward[0]), 1.0)

    def test_overspeed_reduces_sensor_reward(self) -> None:
        normal = _info(speed=1.0, progress=1.0, deficit=0.0)
        fast = _info(speed=1.0, progress=1.0, deficit=0.0)
        normal["forward_velocity_excess"] = np.asarray([0.0], dtype=np.float32)
        fast["forward_velocity_excess"] = np.asarray([0.5], dtype=np.float32)
        normal_reward = sensor_stability_task_reward(
            normal, fall_penalty=25.0, stall_penalty=10.0,
            velocity_excess_penalty_coef=2.0,
        )
        fast_reward = sensor_stability_task_reward(
            fast, fall_penalty=25.0, stall_penalty=10.0,
            velocity_excess_penalty_coef=2.0,
        )
        self.assertAlmostEqual(float(normal_reward[0] - fast_reward[0]), 1.0)

    def test_torch_inputs_stay_on_torch(self) -> None:
        info = {key: torch.as_tensor(value) for key, value in _info(
            speed=1.0, progress=1.0, deficit=0.0
        ).items()}
        reward = sensor_stability_task_reward(
            info,
            fall_penalty=25.0,
            stall_penalty=10.0,
        )
        self.assertTrue(torch.is_tensor(reward))
        self.assertEqual(tuple(reward.shape), (1,))

    def test_stall_receives_an_explicit_terminal_penalty(self) -> None:
        info = _info(speed=0.0, progress=0.0, deficit=1.0)
        info["insufficient_progress"] = np.asarray([1.0], dtype=np.float32)
        reward = sensor_stability_task_reward(
            info,
            fall_penalty=25.0,
            stall_penalty=10.0,
        )
        self.assertEqual(float(reward[0]), -10.0)

    def test_warp_fast_path_preserves_sensor_reward_inputs(self) -> None:
        env = LytRobotWarpAmpEnv.__new__(LytRobotWarpAmpEnv)
        env.qpos = torch.zeros((2, 10), dtype=torch.float32)
        env.qpos[:, 2] = 0.9
        env.qpos[:, 3] = 1.0
        env.qvel = torch.zeros((2, 9), dtype=torch.float32)
        env.ctrl = torch.zeros((2, 2), dtype=torch.float32)
        env.xmat = torch.zeros((2, 1, 3, 3), dtype=torch.float32)
        env.xmat[:, 0, 2, 2] = 1.0
        env.pelvis_body_id = 0
        env.nominal_root_height = 0.9
        env.more_terrain_cfg = SimpleNamespace(
            serial_course=False,
            course_base_height=0.0,
        )
        env.has_distal_action = True
        env.distal_position_action_mask = torch.tensor([False, True])
        env.robot_qpos_adrs = torch.tensor([7, 8])
        env.knee_joint_indices = torch.tensor([0, 1])
        env.episode_step = torch.ones(2, dtype=torch.long)
        env.episode_timeout_step = torch.full((2,), 100, dtype=torch.long)
        env.cfg = SimpleNamespace(
            forward_velocity_tracking_sigma=0.25,
            episode_steps=100,
            target_velocity=0.87,
            fall_height=0.45,
            max_height=1.55,
            min_upright_z=0.45,
            fall_penalty=2.0,
            sensor_knee_flexion_target=0.25,
            sensor_knee_gait_reward_coef=1.0,
        )
        targets = {
            "root_quat": env.qpos[:, 3:7].clone(),
            "root_height": torch.full((2,), 0.9),
        }

        fast_reward, dones, time_outs, info = env._compute_sensor_reward_done(
            torch.zeros((2, 2)),
            torch.zeros((2, 2)),
            torch.full((2,), 0.87),
            targets,
        )
        wrapped_reward = sensor_stability_task_reward(
            info,
            fall_penalty=25.0,
            stall_penalty=10.0,
            knee_gait_reward_coef=1.0,
        )

        torch.testing.assert_close(fast_reward, wrapped_reward)
        self.assertFalse(bool(dones.any()))
        self.assertFalse(bool(time_outs.any()))

    def test_warp_sensor_reward_does_not_depend_on_reference_root_pose(self) -> None:
        env = LytRobotWarpAmpEnv.__new__(LytRobotWarpAmpEnv)
        env.qpos = torch.zeros((1, 10), dtype=torch.float32)
        env.qpos[:, 2] = 0.9
        env.qpos[:, 3] = 1.0
        env.qvel = torch.zeros((1, 9), dtype=torch.float32)
        env.ctrl = torch.zeros((1, 2), dtype=torch.float32)
        env.xmat = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        env.xmat[:, 0, 2, 2] = 1.0
        env.pelvis_body_id = 0
        env.nominal_root_height = 0.9
        env.more_terrain_cfg = SimpleNamespace(
            serial_course=False,
            course_base_height=0.0,
        )
        env.has_distal_action = False
        env.distal_position_action_mask = torch.tensor([False, False])
        env.robot_qpos_adrs = torch.tensor([7, 8])
        env.knee_joint_indices = torch.tensor([0, 1])
        env.episode_step = torch.ones(1, dtype=torch.long)
        env.episode_timeout_step = torch.full((1,), 100, dtype=torch.long)
        env.cfg = SimpleNamespace(
            forward_velocity_tracking_sigma=0.25,
            episode_steps=100,
            target_velocity=0.87,
            fall_height=0.45,
            max_height=1.55,
            min_upright_z=0.45,
            fall_penalty=2.0,
            sensor_knee_flexion_target=0.25,
            sensor_knee_gait_reward_coef=1.0,
        )
        first_targets = {
            "root_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "root_height": torch.tensor([0.9]),
        }
        second_targets = {
            "root_quat": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "root_height": torch.tensor([1.3]),
        }

        first = env._compute_sensor_reward_done(
            torch.zeros((1, 2)),
            torch.zeros((1, 2)),
            torch.full((1,), 0.87),
            first_targets,
        )
        second = env._compute_sensor_reward_done(
            torch.zeros((1, 2)),
            torch.zeros((1, 2)),
            torch.full((1,), 0.87),
            second_targets,
        )

        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        torch.testing.assert_close(first[3]["sensor_upright"], second[3]["sensor_upright"])
        torch.testing.assert_close(first[3]["sensor_height"], second[3]["sensor_height"])


if __name__ == "__main__":
    unittest.main()
