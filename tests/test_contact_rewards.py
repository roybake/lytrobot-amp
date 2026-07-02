from __future__ import annotations

import unittest

import numpy as np

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg


class ContactRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = LytRobotAmpEnv.__new__(LytRobotAmpEnv)
        self.env.cfg = LytRobotEnvCfg()

    def test_side_contacts_preserve_heel_toe_granularity(self) -> None:
        geom_contact = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(
            self.env._side_contacts_from_geom(geom_contact),
            np.asarray([1.0, 1.0], dtype=np.float32),
        )

    def test_stance_reward_uses_world_horizontal_velocity(self) -> None:
        world_vel = np.zeros((4, 3), dtype=np.float32)
        geom_contact = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ref_contact = np.asarray([1.0, 0.0], dtype=np.float32)

        track, speed = self.env._stance_foot_track(world_vel, geom_contact, ref_contact)
        self.assertAlmostEqual(track, 1.0)
        self.assertAlmostEqual(speed, 0.0)

        world_vel[0, 0] = self.env.cfg.stance_foot_vel_sigma
        track, speed = self.env._stance_foot_track(world_vel, geom_contact, ref_contact)
        self.assertAlmostEqual(track, float(np.exp(-1.0)), places=6)
        self.assertAlmostEqual(speed, self.env.cfg.stance_foot_vel_sigma, places=6)

    def test_missing_expected_contact_has_zero_stance_score(self) -> None:
        track, speed = self.env._stance_foot_track(
            np.zeros((4, 3), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        self.assertAlmostEqual(track, 0.0)
        self.assertAlmostEqual(speed, 0.0)

    def test_local_velocity_tracking_ignores_stance_geometries(self) -> None:
        foot_vel = np.zeros((4, 3), dtype=np.float32)
        ref_foot_vel = np.zeros((4, 3), dtype=np.float32)
        foot_vel[0:2] = 10.0
        track = self.env._swing_foot_vel_track(
            foot_vel,
            ref_foot_vel,
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        self.assertAlmostEqual(track, 1.0)


if __name__ == "__main__":
    unittest.main()
