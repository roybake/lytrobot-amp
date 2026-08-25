import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from lytrobot_amp.training_defaults import (
    ORIGINAL_RETARGETED_WALK_SPEED_MPS,
    TRAINING_TARGET_SPEED_MPS,
)
from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.terrain import (
    MoreTerrainCfg,
    StairTerrainCfg,
    load_model,
    more_boxes,
    terrain_height_from_model,
    terrain_kind_from_motion,
    terrain_spawn_offset,
)


ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = ROOT / "robot" / "reference_data" / "multimotion"
TERRAIN_DIR = ROOT / "robot" / "terrains" / "more_discolrl"
NAMES = (
    "walk",
    "turn_left",
    "turn_right",
    "rough_walk",
    "pit_walk",
    "gap_cross",
    "stair_up",
    "stair_down",
)


def motion_paths() -> tuple[str, ...]:
    return tuple(str(MOTION_DIR / f"{name}_omniretarget.npz") for name in NAMES)


def test_more_source_files_are_copied_with_heightfield_and_licenses() -> None:
    expected = {
        "gap.xml",
        "pit.xml",
        "rough.xml",
        "stairs.xml",
        "height_field.png",
        "height_field_lyt.png",
        "terrain_source.py",
        "LICENSE-CC-BY-NC-4.0.md",
        "LICENSE-BSD-3-Clause.txt",
    }
    assert expected.issubset({path.name for path in TERRAIN_DIR.iterdir()})
    assert (TERRAIN_DIR / "height_field.png").stat().st_size > 1_000


def test_scene_uses_more_xml_visual_palette() -> None:
    scene = ET.parse(ROOT / "robot" / "scene.xml").getroot()
    visual_rgba = scene.find("./visual/rgba")
    assert visual_rgba is not None
    assert visual_rgba.attrib["haze"] == "0.15 0.25 0.35 1"

    skybox = scene.find("./asset/texture[@type='skybox']")
    assert skybox is not None
    assert skybox.attrib["rgb1"] == "0.3 0.5 0.7"
    assert skybox.attrib["rgb2"] == "0 0 0"

    checker = scene.find("./asset/texture[@name='groundplane']")
    assert checker is not None
    assert checker.attrib["rgb1"] == "0.2 0.3 0.4"
    assert checker.attrib["rgb2"] == "0.1 0.2 0.3"
    assert checker.attrib["markrgb"] == "0.8 0.8 0.8"

    material = scene.find("./asset/material[@name='groundplane']")
    assert material is not None
    assert material.attrib["texrepeat"] == "5 5"
    assert material.attrib["reflectance"] == "0"
    assert material.attrib["specular"] == "0"
    assert material.attrib["shininess"] == "0"
    assert material.attrib["metallic"] == "0"
    assert material.attrib["roughness"] == "1"

    floor = scene.find("./worldbody/geom[@name='floor']")
    assert floor is not None
    assert floor.attrib["material"] == "groundplane"
    assert floor.attrib["rgba"] == "0.5 0.5 0.5 1"


def test_more_scene_contains_gap_pit_rough_and_stairs() -> None:
    more = MoreTerrainCfg(enabled=True)
    stairs = StairTerrainCfg(
        enabled=True,
        step_height=0.13,
        step_depth=0.58,
        step_count=3,
        start_x=-0.95,
    )
    boxes = more_boxes(more)
    assert sum(name.startswith("more_gap") for name, _, _ in boxes) == 3
    assert sum(name.startswith("more_pit") for name, _, _ in boxes) == 1
    gap_centers = [
        float(center[0])
        for name, center, _ in boxes
        if name.startswith("more_gap")
    ]
    np.testing.assert_allclose(np.diff(gap_centers), [2.20, 2.20])

    model = load_model(ROOT / "robot" / "MJCFmodelposition.xml", stairs, more)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nhfield == 1
    terrain_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
        and (name.startswith("more_") or name.startswith("stair_"))
    ]
    np.testing.assert_allclose(
        model.geom_rgba[terrain_geom_ids],
        np.tile([0.5, 0.5, 0.5, 1.0], (len(terrain_geom_ids), 1)),
    )
    material_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_MATERIAL, "more_matte_terrain"
    )
    assert material_id >= 0
    np.testing.assert_allclose(model.mat_rgba[material_id], [0.5, 0.5, 0.5, 1.0])
    texture_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_TEXTURE, "more_terrain_checker"
    )
    assert texture_id >= 0
    assert model.mat_texid[material_id, 0] == texture_id
    np.testing.assert_allclose(model.mat_texrepeat[material_id], [5.0, 5.0])
    assert model.mat_reflectance[material_id] == 0.0
    assert model.mat_specular[material_id] == 0.0
    assert model.mat_shininess[material_id] == 0.0
    assert model.mat_metallic[material_id] == 0.0
    assert model.mat_roughness[material_id] == 1.0
    assert (int(model.hfield_nrow[0]), int(model.hfield_ncol[0])) == (32, 64)
    heights = terrain_height_from_model(
        model,
        data,
        np.asarray([2.0, 3.1, 4.2, 2.0, 2.0, 0.70]),
        np.asarray(
            [
                more.gap_lane_y,
                more.gap_lane_y,
                more.gap_lane_y,
                more.pit_lane_y,
                more.rough_lane_y,
                stairs.center_y,
            ]
        ),
    )
    np.testing.assert_allclose(heights[[0, 1, 2, 3, 5]], [0.30, 0.0, 0.30, 0.30, 0.13])
    assert 0.0 < heights[4] <= more.rough_height + 1.0e-6


def test_motion_names_route_to_matching_terrain_lanes() -> None:
    more = MoreTerrainCfg(enabled=True)
    stairs = StairTerrainCfg(enabled=True)
    assert [terrain_kind_from_motion(name) for name in NAMES] == [
        "flat",
        "flat",
        "flat",
        "rough",
        "pit",
        "gap",
        "stairs",
        "stairs",
    ]
    np.testing.assert_allclose(
        terrain_spawn_offset("gap", more, stairs),
        [more.gap_spawn_x, more.gap_lane_y, 0.0],
    )
    np.testing.assert_allclose(
        terrain_spawn_offset("rough", more, stairs),
        [more.rough_spawn_x, more.rough_lane_y, 0.0],
    )


def test_terrain_references_pass_contact_and_collision_audits() -> None:
    with np.load(MOTION_DIR / "walk_omniretarget.npz") as archive:
        walk_frames = len(archive["qpos"])
    expected_frames = {
        "rough_walk": 3 * walk_frames,
        "pit_walk": 3 * walk_frames,
        "gap_cross": 4 * walk_frames,
    }
    for task, frame_count in expected_frames.items():
        with np.load(MOTION_DIR / f"{task}_omniretarget.npz") as archive:
            qpos = np.asarray(archive["qpos"])
            qvel = np.asarray(archive["qvel"])
            metadata = json.loads(str(archive["metadata"]))
        assert len(qpos) == frame_count
        assert qvel.shape[0] == frame_count
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
        assert metadata["minimum_cross_leg_geometry_distance"] >= -1.0e-5
        assert metadata["support_clearance_min"] >= -0.003
        assert metadata["support_clearance_max"] <= 0.032
        assert metadata["maximum_roll_target_error"] <= 0.011
        if task == "gap_cross":
            assert metadata["unsupported_gap_contact_frames"] == 0


def test_forward_locomotion_references_preserve_original_retargeted_speed() -> None:
    for task in ("walk", "rough_walk", "pit_walk", "gap_cross"):
        with np.load(MOTION_DIR / f"{task}_omniretarget.npz") as archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            frequency = float(archive["frequency"])
            loop = bool(np.asarray(archive["loop"]).item())
            metadata = json.loads(str(archive["metadata"]))
        duration = (len(qpos) - (0 if loop else 1)) / frequency
        displacement = float(qpos[-1, 0] - qpos[0, 0])
        if loop:
            displacement += float(np.median(np.diff(qpos[:, 0])))
        speed = displacement / duration
        np.testing.assert_allclose(speed, TRAINING_TARGET_SPEED_MPS, atol=0.02)
        assert metadata["target_forward_speed_mps"] == TRAINING_TARGET_SPEED_MPS
        if task == "walk":
            np.testing.assert_allclose(
                metadata["demonstrated_forward_speed_mps"],
                ORIGINAL_RETARGETED_WALK_SPEED_MPS,
                atol=0.02,
            )
        else:
            assert metadata["derived_from"] == "walk_omniretarget.npz"


def test_multimotion_references_have_no_single_frame_ik_impulses() -> None:
    paths = (
        ROOT / "robot" / "reference_data" / "walk_cmu_69_01_omniretarget.npz",
        MOTION_DIR / "stair_up_omniretarget.npz",
        MOTION_DIR / "stair_down_omniretarget.npz",
    )
    for path in paths:
        with np.load(path) as archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            qvel = np.asarray(archive["qvel"], dtype=np.float64)
        assert np.max(np.abs(np.diff(qpos[:, 7:], axis=0))) <= 0.80
        assert np.max(np.abs(qvel[:, 6:])) <= 18.0


def test_cpu_environment_loads_and_steps_all_eight_skills() -> None:
    env = LytRobotAmpEnv(
        LytRobotEnvCfg(
            motion_files=motion_paths(),
            motion_names=NAMES,
            motion_weights=(0.15, 0.075, 0.075, 0.15, 0.15, 0.15, 0.125, 0.125),
            motion_loop=(True, False, False, False, False, False, False, False),
            terrain_types=("gap", "pit", "rough", "stairs"),
            num_envs=4,
            stair_step_height=0.13,
            stair_step_depth=0.58,
            stair_step_count=3,
            stair_start_x=-0.95,
            stair_down_offset=0.25,
            motion_reset_prob=1.0,
            stand_start_prob=0.0,
            use_reference_velocity=True,
            domain_randomization=False,
            episode_timeout_jitter_steps=0,
        )
    )
    try:
        assert env.motion_count == 8
        assert env.obs_dim == 135
        assert env.model.nhfield == 1
        observation, reward, done, _ = env.step(
            np.zeros((4, env.action_dim), dtype=np.float32)
        )
        assert observation.shape == (4, 135)
        assert reward.shape == done.shape == (4,)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward).all()
    finally:
        env.close()
