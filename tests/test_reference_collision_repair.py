import numpy as np

from lytrobot_amp.env import LytRobotAmpEnv
from scripts.audit_independent_references import launcher_config
from scripts.repair_independent_references import interpolate_pose, required_lift, set_pose, verify_and_render


def test_translation_interpolation_unwraps_both_directions():
    poses = np.zeros((2, 7))
    poses[:, 3] = 1
    poses[1, 0] = 0.5
    period = np.array([1.0, 0.0, 0.1])
    assert interpolate_pose(poses, 1.5, period)[0] == 0.75
    assert interpolate_pose(poses, -0.5, period)[0] == -0.25
    np.testing.assert_allclose(interpolate_pose(poses, 3.5, period)[:3],
                               interpolate_pose(poses, 1.5, period)[:3] + period)


def test_actual_rough_collision_projection_preserves_joints_and_margins():
    cfg = launcher_config("rough")
    from scripts.independent_terrain_task import REFERENCE_DIR
    cfg.motion_files = (str(REFERENCE_DIR / "rough_walk_omniretarget.npz"),)
    env = LytRobotAmpEnv(cfg)
    try:
        pose = env.motion.motions[0].reference_qpos[115].astype(float)
        original = pose.copy()
        margins = env.model.geom_margin.copy()
        set_pose(env, pose, "rough")
        assert env._terrain_penetration_depth(env.datas[0]) > 0.01
        lift = required_lift(env, pose, "rough")
        assert 0 < lift < 0.05
        np.testing.assert_array_equal(pose, original)
        pose[2] += lift
        set_pose(env, pose, "rough")
        assert env._terrain_penetration_depth(env.datas[0]) <= 1e-8
        np.testing.assert_array_equal(env.model.geom_margin, margins)
        np.testing.assert_array_equal(pose[7:], original[7:])
    finally:
        env.close()


def test_saved_rough_reference_has_no_penetration_through_four_tile_cycles(tmp_path):
    from pathlib import Path

    cfg = launcher_config("rough")
    result = verify_and_render("rough", Path(cfg.motion_files[0]), tmp_path, 0, False)
    assert result["finite"]
    assert result["loop_translation_m"] == [5.0, 0.0, 0.0]
    assert abs(result["mean_forward_speed_mps"] - cfg.target_velocity) < .001
    for placement in result["placements"]:
        assert placement["max_penetration_m"] <= 1e-6
        assert placement["nonfoot_penetrating_contacts"] == 0
        assert placement["no_support_within_3mm_fraction"] <= .05
