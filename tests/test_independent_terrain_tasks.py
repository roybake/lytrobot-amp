from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.motion import MotionLibrary
from lytrobot_amp.terrain import (
    MoreTerrainCfg,
    StairTerrainCfg,
    load_model,
    standalone_rough_field_placements,
    terrain_spawn_offset,
    terrain_height_from_model,
)
from scripts import train
from scripts.train_flat import default_train_args as flat_args
from scripts.train_rough import default_train_args as rough_args
from scripts.train_slope_down import default_train_args as slope_down_args
from scripts.train_slope_up import default_train_args as slope_up_args


ROOT = Path(__file__).resolve().parents[1]


def _values(args: list[str], option: str) -> list[str]:
    start = args.index(option) + 1
    end = start
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return args[start:end]


def test_four_independent_launchers_have_disjoint_sensor_only_contracts() -> None:
    launchers = {
        "independent_flat": flat_args(),
        "independent_rough": rough_args(),
        "independent_slope_up": slope_up_args(),
        "independent_slope_down": slope_down_args(),
    }
    expected_motion_names = {
        "independent_flat": "walk",
        "independent_rough": "rough_walk",
        "independent_slope_up": "slope_up",
        "independent_slope_down": "slope_down",
    }
    for task_name, args in launchers.items():
        with patch.object(sys, "argv", ["train.py", *args]):
            parsed = train.parse_args()
        assert parsed.training_task == task_name
        assert parsed.motion_names == [expected_motion_names[task_name]]
        assert len(parsed.motion_files) == 1
        assert parsed.motion_weights == [1.0]
        assert parsed.terrain_types == [task_name.removeprefix("independent_")]
        assert parsed.serial_terrain_course is False
        assert parsed.distillation_coef == 0.0
        assert parsed.teacher_checkpoint is None
        assert parsed.reference_ctrl_blend == 0.0
        assert parsed.skill_conditioning is False
        assert parsed.amp_task_only is True
        assert parsed.standalone_terrain_field is True
        assert parsed.motion_loop == ["true"]
        assert parsed.use_reference_velocity is False

    log_roots = {
        _values(args, "--log-root")[0] for args in launchers.values()
    }
    assert len(log_roots) == 4


def test_standalone_slope_spawn_offsets_are_contact_calibrated() -> None:
    more = MoreTerrainCfg(
        enabled=True,
        terrain_types=("slope_up",),
        standalone_field=True,
    )
    stairs = StairTerrainCfg(enabled=False)
    np.testing.assert_allclose(
        terrain_spawn_offset("slope_up", more, stairs),
        [0.60, more.slope_lane_y, 0.04],
    )
    np.testing.assert_allclose(
        terrain_spawn_offset("slope_down", more, stairs),
        [-0.20, more.slope_lane_y, 0.06],
    )


def test_standalone_slope_geometry_matches_uphill_and_downhill_references() -> None:
    for kind in ("slope_up", "slope_down"):
        env = LytRobotAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(ROOT / "robot" / "MJCFmodelposition.xml"),
                motion_files=(
                    str(ROOT / "robot" / "reference_data" / "multimotion" / f"{kind}_omniretarget.npz"),
                ),
                motion_names=(kind,),
                motion_weights=(1.0,),
                motion_loop=(False,),
                skill_conditioning=False,
                terrain_types=(kind,),
                num_envs=1,
                seed=7,
                motion_reset_prob=1.0,
                stand_start_prob=0.0,
                reset_noise_scale=0.0,
                domain_randomization=False,
                episode_timeout_jitter_steps=0,
            )
        )
        try:
            ramp_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_GEOM, f"more_{kind}"
            )
            assert ramp_id >= 0
            env.reset()
            x = np.asarray([0.0, 1.85, 3.70], dtype=np.float32)
            y = np.full_like(x, 3.0)
            heights = terrain_height_from_model(env.model, env.datas[0], x, y)
            if kind == "slope_up":
                assert heights[0] < heights[1] < heights[2]
            else:
                assert heights[0] > heights[1] > heights[2]
            observation, reward, done, _info = env.step(
                np.zeros((1, env.action_dim), dtype=np.float32)
            )
            assert observation.shape == (1, env.obs_dim)
            assert np.isfinite(observation).all()
            assert np.isfinite(reward).all()
            assert np.isfinite(done).all()
        finally:
            env.close()


def test_standalone_slope_reference_contacts_are_physically_aligned() -> None:
    for kind in ("slope_up", "slope_down"):
        env = LytRobotAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(ROOT / "robot" / "MJCFmodelposition.xml"),
                motion_files=(
                    str(ROOT / "robot" / "reference_data" / "multimotion" / f"{kind}_omniretarget.npz"),
                ),
                motion_names=(kind,),
                motion_weights=(1.0,),
                motion_loop=(True,),
                skill_conditioning=False,
                terrain_types=(kind,),
                standalone_terrain_field=True,
                num_envs=1,
                seed=7,
                motion_reset_prob=1.0,
                stand_start_prob=0.0,
                reset_noise_scale=0.0,
                domain_randomization=False,
                episode_timeout_jitter_steps=0,
            )
        )
        try:
            # Supplying one loop flag must not fall back to MotionReference,
            # whose phase handling cannot represent non-looping semantics.
            assert isinstance(env.motion, MotionLibrary)
            motion = env.motion.motions[0]
            offset = terrain_spawn_offset(kind, env.more_terrain_cfg, env.stair_cfg)
            simulated_contacts = []
            max_penetration = 0.0
            nonfoot_penetrating_contacts = 0
            data = env.datas[0]
            for qpos, qvel in zip(
                motion.reference_qpos, motion.reference_qvel, strict=True
            ):
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                data.qpos[:3] += offset
                mujoco.mj_forward(env.model, data)
                simulated_contacts.append(env._foot_contacts(data))
                max_penetration = max(
                    max_penetration, env._terrain_penetration_depth(data)
                )
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    geom1 = int(contact.geom1)
                    geom2 = int(contact.geom2)
                    other = (
                        geom2
                        if geom1 in env.terrain_geom_ids
                        else geom1 if geom2 in env.terrain_geom_ids else None
                    )
                    if (
                        other is not None
                        and other not in env.contact_geom_id_to_index
                        and float(contact.dist) <= 0.0
                    ):
                        nonfoot_penetrating_contacts += 1

            simulated = np.asarray(simulated_contacts, dtype=np.float32)
            mismatch = np.mean(
                np.abs(simulated - motion.reference_foot_contact), axis=0
            )
            airborne = np.mean(np.max(simulated, axis=1) < 0.5)
            assert np.max(mismatch) < 0.10
            assert airborne < 0.10
            assert max_penetration <= 1.0e-5
            assert nonfoot_penetrating_contacts == 0
        finally:
            env.close()


def test_independent_terrain_fields_cover_the_runtime_training_area() -> None:
    field_cfg = dict(
        standalone_field=True,
        standalone_field_x_min=-2.0,
        standalone_field_x_max=22.0,
        standalone_field_y_min=-10.0,
        standalone_field_y_max=14.0,
    )

    rough_cfg = MoreTerrainCfg(
        enabled=True,
        terrain_types=("rough",),
        **field_cfg,
    )
    rough_placements = standalone_rough_field_placements(rough_cfg)
    assert len(rough_placements) == 40
    rough_model = load_model(
        ROOT / "robot" / "MJCFmodelposition.xml",
        StairTerrainCfg(enabled=False),
        rough_cfg,
    )
    rough_names = [
        mujoco.mj_id2name(rough_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        for geom_id in range(rough_model.ngeom)
    ]
    assert sum(name.startswith("standalone_field_rough_") for name in rough_names) == 40
    rough_field_ids = [
        geom_id
        for geom_id, name in enumerate(rough_names)
        if name.startswith("standalone_field_rough_")
    ]
    x_min = min(float(rough_model.geom_pos[geom_id, 0] - rough_model.geom_size[geom_id, 0]) for geom_id in rough_field_ids)
    x_max = max(float(rough_model.geom_pos[geom_id, 0] + rough_model.geom_size[geom_id, 0]) for geom_id in rough_field_ids)
    y_min = min(float(rough_model.geom_pos[geom_id, 1] - rough_model.geom_size[geom_id, 1]) for geom_id in rough_field_ids)
    y_max = max(float(rough_model.geom_pos[geom_id, 1] + rough_model.geom_size[geom_id, 1]) for geom_id in rough_field_ids)
    assert x_min <= rough_cfg.standalone_field_x_min
    assert x_max >= rough_cfg.standalone_field_x_max
    assert y_min <= rough_cfg.standalone_field_y_min
    assert y_max >= rough_cfg.standalone_field_y_max

    for kind in ("slope_up", "slope_down"):
        cfg = MoreTerrainCfg(enabled=True, terrain_types=(kind,), **field_cfg)
        model = load_model(
            ROOT / "robot" / "MJCFmodelposition.xml",
            StairTerrainCfg(enabled=False),
            cfg,
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            for geom_id in range(model.ngeom)
        ]
        assert names.count(f"standalone_field_{kind}") == 1
        heights = terrain_height_from_model(
            model,
            data,
            np.asarray([-1.0, 0.0, 3.0, 10.0, 21.0]),
            np.full(5, 3.0),
        )
        if kind == "slope_up":
            assert np.all(np.diff(heights) > 0.0)
        else:
            assert np.all(np.diff(heights) < 0.0)
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        assert float(model.geom_pos[floor_id, 2]) < float(np.min(heights))
