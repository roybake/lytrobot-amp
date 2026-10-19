"""Regressions for stationary stair policies and incorrect support targets."""

import json
from pathlib import Path
import unittest

import numpy as np
import torch

from lytrobot_amp.mimickit_deepmimic import deepmimic_reward
from lytrobot_amp.env import LytRobotEnvCfg
from sensor_training.envs import SensorHistoryCpuAmpEnv, SensorHistoryWarpAmpEnv

ROOT = Path(__file__).resolve().parents[2]


class StairRouteRegressions(unittest.TestCase):
    def test_route_deviation_terminates_cpu_and_warp_without_timeout_bootstrap(self):
        cfg = LytRobotEnvCfg(
            model_xml=str(ROOT / 'robot/MJCFmodel_stair.xml'),
            motion_files=(str(ROOT / 'robot/reference_data/multimotion/stairs_supported.npz'),),
            motion_names=('stairs',), motion_loop=(False,), skill_conditioning=False,
            deepmimic_reward=True, deepmimic_track_root_xy=True,
            deepmimic_route_tracking=True, sensor_phase_observation=True,
            sensor_root_imu_only=True, sensor_gait_command_enabled=False,
            stair_terrain=True, terrain_embedded_in_model=True,
            preserve_motion_world_xy=True, reference_root_x_offset=-.39,
            stair_fixed_start=True, stair_phase_reset_prob=0,
            num_envs=1, motion_reset_prob=0, stand_start_prob=0,
            reset_noise_scale=0, action_noise_std=0, domain_randomization=False,
            randomize_episode_seed=False, episode_timeout_jitter_steps=0,
            episode_steps=1,
        )
        backends = [('cpu', lambda: SensorHistoryCpuAmpEnv(cfg))]
        if torch.cuda.is_available():
            backends.append(('warp', lambda: SensorHistoryWarpAmpEnv(cfg, device='cuda')))
        for name, make_env in backends:
            with self.subTest(backend=name):
                env = make_env()
                try:
                    obs = env.reset()
                    self.assertEqual(obs.shape, (1, 210))
                    if name == 'cpu':
                        env.base_env.datas[0].qpos[0] += 1.0
                        action = np.zeros((1, 14), dtype=np.float32)
                    else:
                        env.base_env.qpos[:, 0] += 1.0
                        action = torch.zeros(1, 14, device='cuda')
                    _, reward, done, info = env.step(action)
                    self.assertTrue(bool(done[0]))
                    self.assertEqual(float(info['deepmimic_route_failure'][0]), 1.)
                    self.assertEqual(float(info['time_outs'][0]), 0.)
                    self.assertEqual(float(info['termination'][0]), 1.)
                    self.assertTrue(bool(np.isfinite(float(reward[0]))))
                finally:
                    env.close()

    def test_standing_cannot_collect_local_pose_credit_far_behind_reference(self):
        quat = torch.tensor([[1., 0., 0., 0.]])
        target_velocity = torch.tensor([[0.4, 0., 0.]])
        arguments = dict(
            root_height=torch.ones(1), root_quat=quat, root_ang_vel=torch.zeros(1, 3),
            joint_pos=torch.zeros(1, 14), joint_vel=torch.zeros(1, 14),
            key_pos_root_local=torch.zeros(1, 4, 3),
            target_root_height=torch.ones(1), target_root_quat=quat,
            target_root_lin_vel=target_velocity, target_root_ang_vel=torch.zeros(1, 3),
            target_joint_pos=torch.zeros(1, 14), target_joint_vel=torch.zeros(1, 14),
            target_key_pos_root_local=torch.zeros(1, 4, 3),
            target_root_xy=torch.tensor([[1.2, 0.]]), route_tracking=True,
        )
        tracking, _ = deepmimic_reward(**arguments,
            root_xy=arguments['target_root_xy'], root_lin_vel=target_velocity)
        standing, _ = deepmimic_reward(**arguments,
            root_xy=torch.zeros(1, 2), root_lin_vel=torch.zeros(1, 3))
        self.assertGreater(float(tracking), .99)
        self.assertLess(float(standing), .01)
        # At the right location, moving backward must still earn much less.
        backward, _ = deepmimic_reward(**arguments,
            root_xy=arguments['target_root_xy'], root_lin_vel=-target_velocity)
        self.assertLess(float(backward), .01)

    def test_support_repair_preserves_the_fixed_human_joint_and_xy_motion(self):
        original = np.load(ROOT / 'robot/reference_data/multimotion/stairs.npz')
        repaired = np.load(ROOT / 'robot/reference_data/multimotion/stairs_supported.npz')
        np.testing.assert_array_equal(original['qpos'][:, :2], repaired['qpos'][:, :2])
        np.testing.assert_array_equal(original['qpos'][:, 3:], repaired['qpos'][:, 3:])
        self.assertLess(np.max(np.abs(original['qpos'][:, 2]-repaired['qpos'][:, 2])), .025)
        self.assertTrue(np.all(repaired['foot_contact'].sum(axis=1) >= 1))
        report = json.loads((ROOT / 'robot/reference_data/multimotion/stairs_supported.audit.json').read_text())
        self.assertEqual(report['unsupported_frames'], 0)
        self.assertTrue(report['contact_geometry_pass'])
        self.assertTrue(report['alignment_pass'])


if __name__ == '__main__':
    unittest.main()
