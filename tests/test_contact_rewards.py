from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from lytrobot_amp.env import (
    LytRobotAmpEnv,
    LytRobotEnvCfg,
    configure_robot_collision_profile,
)
from lytrobot_amp.motion import CONTACT_GEOM_NAMES, resolve_contact_geom_ids
from lytrobot_amp.terrain import STRICT_CONTACT_MARGIN


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

    def test_contacting_foot_slip_has_explicit_bounded_penalty(self) -> None:
        world_vel = np.zeros((4, 3), dtype=np.float32)
        geom_contact = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

        self.assertAlmostEqual(
            self.env._foot_slip_penalty(world_vel, geom_contact),
            0.0,
        )

        world_vel[0, 0] = (
            self.env.cfg.foot_slip_velocity_threshold
            + self.env.cfg.stance_foot_vel_sigma
        )
        # One of the two contacting geoms violates the threshold by one scale.
        self.assertAlmostEqual(
            self.env._foot_slip_penalty(world_vel, geom_contact),
            0.5,
            places=6,
        )

        world_vel[:, 0] = 100.0
        self.assertAlmostEqual(
            self.env._foot_slip_penalty(world_vel, geom_contact),
            4.0,
            places=6,
        )

    def test_warp_and_cpu_slip_penalties_match(self) -> None:
        from lytrobot_amp.env_warp import LytRobotWarpAmpEnv

        world_vel = np.asarray(
            [
                [0.20, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.08, 0.08, 0.0],
            ],
            dtype=np.float32,
        )
        geom_contact = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        expected = self.env._foot_slip_penalty(world_vel, geom_contact)
        warp_env = LytRobotWarpAmpEnv.__new__(LytRobotWarpAmpEnv)
        warp_env.cfg = self.env.cfg

        actual = warp_env._foot_slip_penalty(
            torch.from_numpy(world_vel[None, ...]),
            torch.from_numpy(geom_contact[None, ...]),
        )

        self.assertAlmostEqual(float(actual[0]), expected, places=6)

    def test_warp_contact_aggregation_ignores_stale_capacity_tail(self) -> None:
        from lytrobot_amp.env_warp import LytRobotWarpAmpEnv

        warp_env = LytRobotWarpAmpEnv.__new__(LytRobotWarpAmpEnv)
        warp_env.num_envs = 2
        warp_env.device = torch.device("cpu")
        warp_env.model = SimpleNamespace(ngeom=7)
        warp_env.contact_geom = torch.tensor(
            [[0, 2], [3, 1], [1, 5], [6, 1], [0, 4], [0, 0]],
            dtype=torch.int32,
        )
        warp_env.contact_worldid = torch.tensor(
            [0, 0, 1, 1, 1, -1], dtype=torch.int32
        )
        warp_env.contact_dim = torch.tensor([3, 3, 3, 3, 3, 0], dtype=torch.int32)
        warp_env.contact_dist = torch.tensor(
            [-0.002, -0.001, -0.003, -0.004, -0.5, 0.0],
            dtype=torch.float32,
        )
        warp_env.contact_count = torch.tensor([4], dtype=torch.int32)
        warp_env.contact_slot_ids = torch.arange(6, dtype=torch.int32)
        warp_env.geom_is_terrain = torch.tensor(
            [True, True, False, False, False, False, False]
        )
        warp_env.geom_foot_index = torch.tensor(
            [-1, -1, 0, 1, 2, 3, -1], dtype=torch.long
        )

        contacts, penetration = warp_env._terrain_contact_state()

        torch.testing.assert_close(
            contacts,
            torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        )
        torch.testing.assert_close(penetration, torch.tensor([0.002, 0.004]))

    def test_warp_reset_penetration_stops_forwarding_once_clear(self) -> None:
        from lytrobot_amp.env_warp import LytRobotWarpAmpEnv

        warp_env = LytRobotWarpAmpEnv.__new__(LytRobotWarpAmpEnv)
        warp_env.qpos = torch.zeros((2, 7), dtype=torch.float32)
        penetrations = iter(
            (
                torch.tensor([0.0, 0.02]),
                torch.tensor([0.0, 0.0]),
            )
        )
        forward_calls = 0

        def penetration_depths() -> torch.Tensor:
            return next(penetrations)

        def warp_forward() -> None:
            nonlocal forward_calls
            forward_calls += 1

        warp_env._terrain_penetration_depths = penetration_depths
        warp_env._warp_forward = warp_forward
        warp_env._resolve_terrain_reset_penetration(torch.tensor([1]))

        self.assertEqual(forward_calls, 1)
        self.assertGreater(float(warp_env.qpos[1, 2]), 0.02)
        self.assertEqual(float(warp_env.qpos[0, 2]), 0.0)

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

    def test_only_full_foot_meshes_collide_with_static_terrain(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"))
        configure_robot_collision_profile(model, "feet")
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
            if model.geom_bodyid[geom_id] == 0:
                self.assertEqual(model.geom_contype[geom_id], 1)
                self.assertEqual(model.geom_conaffinity[geom_id], 0)
            else:
                self.assertEqual(model.geom_contype[geom_id], 0)
                self.assertEqual(
                    model.geom_conaffinity[geom_id],
                    1 if name in CONTACT_GEOM_NAMES else 0,
                )
            self.assertEqual(model.geom_condim[geom_id], 3)
            self.assertAlmostEqual(model.geom_margin[geom_id], STRICT_CONTACT_MARGIN)
            if name in CONTACT_GEOM_NAMES:
                self.assertEqual(model.geom_type[geom_id], mujoco.mjtGeom.mjGEOM_MESH)
                mesh_id = int(model.geom_dataid[geom_id])
                self.assertGreaterEqual(mesh_id, 0)
                self.assertEqual(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
                    name,
                )

    def test_standing_pose_contacts_all_four_full_foot_meshes_with_capacity_headroom(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            str(REPO_ROOT / "robot" / "MJCFmodelposition.xml")
        )
        configure_robot_collision_profile(model, "feet")
        model.opt.iterations = 10
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)

        contacted_robot_geoms = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for contact in data.contact[: data.ncon]
            for geom_id in (int(contact.geom1), int(contact.geom2))
            if model.geom_bodyid[geom_id] != 0
        }
        self.assertEqual(contacted_robot_geoms, set(CONTACT_GEOM_NAMES))
        self.assertLess(data.ncon, 32)
        self.assertLess(data.nefc, 128)

    def test_full_body_collision_is_model_default_and_restores_every_robot_geom(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            str(REPO_ROOT / "robot" / "MJCFmodelposition.xml")
        )
        robot_geom_ids = np.flatnonzero(model.geom_bodyid != 0)
        np.testing.assert_array_equal(
            model.geom_conaffinity[robot_geom_ids],
            np.ones(len(robot_geom_ids), dtype=model.geom_conaffinity.dtype),
        )

        configure_robot_collision_profile(model, "feet")
        self.assertEqual(int(model.geom_conaffinity[robot_geom_ids].sum()), 4)
        configure_robot_collision_profile(model, "full_body")
        np.testing.assert_array_equal(
            model.geom_conaffinity[robot_geom_ids],
            np.ones(len(robot_geom_ids), dtype=model.geom_conaffinity.dtype),
        )


if __name__ == "__main__":
    unittest.main()
