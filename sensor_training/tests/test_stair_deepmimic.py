"""Contract checks for the sensor-only stair imitation entry point."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch

from lytrobot_amp.env import LytRobotEnvCfg
from lytrobot_amp.motion import resolve_contact_geom_ids
from sensor_training.envs import SensorHistoryCpuAmpEnv
from sensor_training.networks import SensorActorCritic
from lytrobot_amp.mimickit_deepmimic import deepmimic_reward
from sensor_training.stair_deepmimic import (
    SensorStairDeepMimicConfig,
    SensorStairDeepMimicTrainer,
)
from sensor_training.train_stairs_deepmimic import (
    EXACT_STAIR_EPISODE_STEPS,
    build_train_args,
)
from scripts.train import load_best_checkpoint_quality
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_entry_uses_sensor_trainer_and_explicit_imitation() -> None:
    args = build_train_args("exact", ["--num-envs", "8"])
    assert args.count("--num-envs") == 1
    assert args[args.index("--num-envs") + 1] == "8"
    assert "--deepmimic-reward" in args
    assert "--no-amp" in args
    assert "--use-reference-velocity" in args
    assert "--no-use-reference-velocity" not in args
    assert "--deepmimic-track-root-xy" in args
    assert args[args.index("--deepmimic-root-xy-penalty") + 1] == "0.0"
    assert "--deepmimic-route-tracking" in args
    assert "--no-trajectory-feedback" in args
    assert "--no-sensor-gait-command" in args
    assert "--sensor-root-imu-only" in args
    assert args[args.index("--stair-phase-reset-prob") + 1] == "0.9"
    assert args[args.index("--lcp-weight") + 1] == "0.0"
    assert "--sensor-phase-observation" in args
    assert args[args.index("--task-reward-lerp") + 1] == "1.0"
    assert args[args.index("--motion-files") + 1].endswith("stairs_supported.npz")
    assert args[args.index("--model-xml") + 1].endswith("robot/terrains/stairs.xml")
    assert args[args.index("--reference-root-x-offset") + 1] == "-0.39"
    assert args[args.index("--log-root") + 1] == str(REPO_ROOT / "logs")
    assert args[args.index("--log-group") + 1] == "stairs"
    assert "--run-name" not in args
    assert "--terrain-embedded-in-model" in args
    assert "--preserve-motion-world-xy" in args
    assert args[args.index("--episode-steps") + 1] == str(EXACT_STAIR_EPISODE_STEPS)


def test_stair_deepmimic_defaults_to_4096_envs() -> None:
    for direction in ("exact", "up", "down", "up_down"):
        args = build_train_args(direction)
        assert args[args.index("--num-envs") + 1] == "4096"


def test_stair_run_name_cannot_override_timestamp() -> None:
    with pytest.raises(SystemExit, match="时间戳"):
        build_train_args("exact", ["--run-name", "manual-name"])


def test_deepmimic_best_checkpoint_uses_imitation_reward(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "completed_episode_count,rollout_episode_success_rate,"
        "rollout_mean_task_reward,mean_episode_length\n"
        "64,0.0,0.72,300\n"
        "64,0.0,0.81,180\n",
        encoding="utf-8",
    )
    assert load_best_checkpoint_quality(
        metrics, min_completed=32, deepmimic_ppo=True
    ) == (0.81, 180.0)
    assert load_best_checkpoint_quality(
        metrics, min_completed=32, deepmimic_ppo=False
    ) == (0.0, 300.0)


def test_stair_best_checkpoint_prioritizes_full_route_success(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "completed_episode_count,rollout_episode_success_rate,"
        "rollout_mean_task_reward,mean_episode_length\n"
        "64,0.0,0.81,300\n"
        "64,0.05,0.70,2400\n",
        encoding="utf-8",
    )
    assert load_best_checkpoint_quality(
        metrics, min_completed=32, deepmimic_ppo=True, stair_route=True
    ) == (0.05, 0.70)


def test_exact_stair_scene_and_motion_cpu_smoke() -> None:
    model_xml = REPO_ROOT / "robot" / "terrains" / "stairs.xml"
    motion_file = REPO_ROOT / "robot" / "reference_data" / "multimotion" / "stairs.npz"
    compiled = mujoco.MjModel.from_xml_path(str(model_xml))
    assert sum(
        mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_GEOM, index) == "stair_up_1"
        for index in range(compiled.ngeom)
    ) == 1
    terrain_names = {
        mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_GEOM, index)
        for index in range(compiled.ngeom)
        if int(compiled.geom_bodyid[index]) == 0
    }
    assert {
        "floor",
        "stair_up_1",
        "stair_up_2",
        "stair_up_3",
        "top_platform",
        "stair_down_1",
        "stair_down_2",
        "stair_down_3",
    }.issubset(terrain_names)
    contact_geom_ids = resolve_contact_geom_ids(compiled)
    assert tuple(
        mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_GEOM, int(index))
        for index in contact_geom_ids
    ) == ("foot_r", "toe_r", "foot_l", "toe_l")
    # The rigid stair feet retain one physical box per side.  Toe markers are
    # end-effectors; anterior points on each box are classified as toe contact.
    assert compiled.geom_contype[contact_geom_ids[[0, 2]]].tolist() == [1, 1]
    assert compiled.geom_contype[contact_geom_ids[[1, 3]]].tolist() == [0, 0]

    env = SensorHistoryCpuAmpEnv(
        LytRobotEnvCfg(
            model_xml=str(model_xml),
            motion_files=(str(motion_file),),
            motion_names=("stairs",),
            motion_loop=(False,),
            skill_conditioning=False,
            deepmimic_reward=True,
            stair_terrain=True,
            terrain_embedded_in_model=True,
            preserve_motion_world_xy=True,
            reference_root_x_offset=-0.39,
            stair_fixed_start=True,
            stair_step_count=3,
            stair_top_length=2.19646,
            num_envs=1,
            stand_start_prob=0.0,
            motion_reset_prob=0.0,
            reset_noise_scale=0.0,
            action_noise_std=0.0,
            domain_randomization=False,
            episode_timeout_jitter_steps=0,
            reference_ctrl_blend=1.0,
        )
    )
    try:
        observation = env.reset()
        source = np.load(motion_file, allow_pickle=True)
        np.testing.assert_allclose(
            env.base_env.motion.reference_qpos[0, :2],
            source["qpos"][0, :2] + np.asarray([-0.39, 0.0]),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            env.base_env.datas[0].qpos[:2], env.base_env.motion.reference_qpos[0, :2],
            atol=1.0e-6,
        )
        assert observation.shape == (1, env.obs_dim)
        assert np.isfinite(observation).all()
        assert env.base_env._foot_geom_contacts(env.base_env.datas[0]).tolist() == [
            1.0, 0.0, 1.0, 0.0,
        ]
        observation, reward, done, info = env.step(
            np.zeros((1, env.action_dim), dtype=np.float32)
        )
        assert np.isfinite(observation).all()
        assert np.isfinite(reward).all()
        assert np.isfinite(done).all()
        assert np.isfinite(info["deepmimic_pose"]).all()
    finally:
        env.close()


def test_lcp_loss_reaches_sensor_actor() -> None:
    policy = SensorActorCritic(
        obs_dim=12, action_dim=2, sensor_frame_dim=12,
        actor_history_frames=1, recurrent_hidden_dim=8,
        actor_hidden_dims=(8,), critic_hidden_dims=(8,),
    )
    trainer = SensorStairDeepMimicTrainer.__new__(SensorStairDeepMimicTrainer)
    trainer.cfg = SensorStairDeepMimicConfig(lcp_weight=0.002)
    trainer.actor_critic = policy
    obs = torch.randn(2, 3, 12)
    hidden = policy.initial_state(3)
    starts = torch.zeros(2, 3)
    loss = trainer._actor_auxiliary_loss(obs, hidden, starts, torch.arange(3))
    assert loss is not None and torch.isfinite(loss) and loss.item() >= 0.0
    loss.backward()
    assert any(p.grad is not None for p in policy.actor.parameters())


def test_root_xy_error_reduces_deepmimic_reward() -> None:
    zeros3 = torch.zeros(1, 3)
    zeros14 = torch.zeros(1, 14)
    zeros43 = torch.zeros(1, 4, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    inputs = dict(
        root_height=torch.ones(1), root_quat=quat,
        root_lin_vel=zeros3, root_ang_vel=zeros3,
        joint_pos=zeros14, joint_vel=zeros14,
        key_pos_root_local=zeros43,
        target_root_height=torch.ones(1), target_root_quat=quat,
        target_root_lin_vel=zeros3, target_root_ang_vel=zeros3,
        target_joint_pos=zeros14, target_joint_vel=zeros14,
        target_key_pos_root_local=zeros43,
        target_root_xy=torch.zeros(1, 2), root_xy_penalty=0.5,
    )
    on_path, _ = deepmimic_reward(**inputs, root_xy=torch.zeros(1, 2))
    off_path, terms = deepmimic_reward(**inputs, root_xy=torch.tensor([[1.0, 0.0]]))
    assert on_path.item() > off_path.item() + 0.5
    assert terms["root_xy_error"].item() == 1.0


def test_stair_actor_observes_sensors_previous_action_and_deepmimic_phase() -> None:
    cfg = LytRobotEnvCfg(
        model_xml=str(REPO_ROOT / "robot" / "terrains" / "stairs.xml"),
        motion_files=(str(REPO_ROOT / "robot" / "reference_data" / "multimotion" / "stairs.npz"),),
        motion_names=("stairs",), motion_loop=(False,),
        stair_terrain=True, terrain_embedded_in_model=True,
        preserve_motion_world_xy=True, stair_fixed_start=True,
        reference_root_x_offset=-0.39,
        stair_phase_reset_prob=1.0, trajectory_feedback=False,
        sensor_gait_command_enabled=False, sensor_root_imu_only=True,
        sensor_phase_observation=True,
        deepmimic_reward=True, deepmimic_track_root_xy=False,
        num_envs=8, stand_start_prob=0.0, motion_reset_prob=0.0,
        reset_noise_scale=0.0, domain_randomization=False,
    )
    env = SensorHistoryCpuAmpEnv(cfg)
    try:
        obs = env.reset()
        assert obs.shape == (8, 210)
        assert np.all(env.base_env.motion_phase >= 0)
        assert np.max(env.base_env.motion_phase) > 800
        assert np.any(np.mod(env.base_env.motion_phase, 32) != 0)
        assert np.all(env.base_env.stair_remaining_distance >= 0)
        assert np.any(env.base_env.stair_remaining_distance > 0)
        metadata = env.observation_metadata
        assert metadata["imu_sensor_names"] == ("rootacc", "rootgyro")
        assert metadata["actor_input_fields"] == (
            "imu", "tendon_length", "previous_action", "phase",
        )
        assert metadata["hardware_sensor_fields"] == ("imu", "tendon_length")
        assert metadata["command_dim"] == 0
        assert metadata["strict_sensor_only"]
        assert metadata["actor_frame_dim"] == 35
        before = env._normalized_sensor_frame().copy()
        env.base_env.motion_phase[:] = 123.0
        after = env._normalized_sensor_frame()
        np.testing.assert_array_equal(after[:, :-1], before[:, :-1])
        assert np.any(after[:, -1] != before[:, -1])
    finally:
        env.close()
