from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.motion import CONTACT_GEOM_NAMES, resolve_contact_geom_ids


REPO_ROOT = Path(__file__).resolve().parents[1]


class ContactRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = LytRobotAmpEnv.__new__(LytRobotAmpEnv)
        self.env.cfg = LytRobotEnvCfg()

    def test_model_uses_only_position_actuators(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"))
        actuator_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            for actuator_id in range(model.nu)
        }
        self.assertEqual(model.nu, 14)
        self.assertFalse(any(name is not None and name.endswith("_force") for name in actuator_names))
        np.testing.assert_allclose(model.actuator_gainprm[:, 0], -model.actuator_biasprm[:, 1])
        self.assertTrue(np.all(model.actuator_gainprm[:, 0] > 0.0))

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

    def test_stride_asymmetry_dominates_sensor_symmetry_score(self) -> None:
        self.env.cfg = LytRobotEnvCfg(
            symmetry_ema_decay=0.0,
            symmetry_foot_weight=0.85,
            symmetry_contact_weight=0.10,
            symmetry_action_weight=0.05,
        )
        self.env.foot_x_min = np.asarray([[-0.05, -0.35]], dtype=np.float32)
        self.env.foot_x_max = np.asarray([[0.05, 0.05]], dtype=np.float32)
        self.env.foot_stride_ema = np.zeros((1, 2), dtype=np.float32)
        self.env.contact_balance_ema = np.zeros((1, 2), dtype=np.float32)
        self.env.action_energy_ema = np.zeros((1, 2), dtype=np.float32)
        self.env.right_action_indices = np.asarray([], dtype=np.int32)
        self.env.left_action_indices = np.asarray([], dtype=np.int32)
        foot_pos = np.asarray(
            [
                [0.05, -0.05, 0.0],
                [0.05, -0.05, 0.0],
                [-0.05, 0.05, 0.0],
                [-0.05, 0.05, 0.0],
            ],
            dtype=np.float32,
        )

        symmetry, foot_symmetry, stride_symmetry, phase_symmetry, lateral_symmetry, contact_balance, action_balance = (
            self.env._symmetry_track(
                0,
                foot_pos,
                np.asarray([1.0, 1.0], dtype=np.float32),
                np.zeros(14, dtype=np.float32),
            )
        )

        self.assertLess(stride_symmetry, 1.0e-5)
        self.assertAlmostEqual(phase_symmetry, 1.0)
        self.assertAlmostEqual(lateral_symmetry, 1.0)
        self.assertAlmostEqual(contact_balance, 1.0)
        self.assertAlmostEqual(action_balance, 1.0)
        self.assertLess(foot_symmetry, 0.20)
        self.assertLess(symmetry, 0.30)

    def test_foot_mesh_geoms_provide_contact_collision(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"))
        contact_ids = resolve_contact_geom_ids(model)
        self.assertEqual(len(contact_ids), 4)
        self.assertEqual(
            {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
                for geom_id in contact_ids
            },
            set(CONTACT_GEOM_NAMES),
        )
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            collision_enabled = bool(model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
            self.assertEqual(collision_enabled, name == "floor" or name in CONTACT_GEOM_NAMES)
            if name in CONTACT_GEOM_NAMES:
                self.assertEqual(model.geom_type[geom_id], mujoco.mjtGeom.mjGEOM_MESH)
                mesh_id = int(model.geom_dataid[geom_id])
                self.assertGreaterEqual(mesh_id, 0)
                self.assertEqual(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
