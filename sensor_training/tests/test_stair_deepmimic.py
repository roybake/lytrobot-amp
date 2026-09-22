"""Contract checks for the sensor-only stair imitation entry point."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch

from lytrobot_amp.env import LytRobotEnvCfg
from sensor_training.envs import SensorHistoryCpuAmpEnv
from sensor_training.networks import SensorActorCritic
from sensor_training.stair_deepmimic import (
    SensorStairDeepMimicConfig,
    SensorStairDeepMimicTrainer,
)
from sensor_training.train_stairs_deepmimic import (
    EXACT_STAIR_EPISODE_STEPS,
    build_train_args,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_entry_uses_sensor_trainer_and_explicit_imitation() -> None:
    args = build_train_args("exact", ["--num-envs", "8"])
    assert "--deepmimic-reward" in args
    assert "--no-amp" in args
    assert args.index("--no-use-reference-velocity") > args.index("--use-reference-velocity")
    assert args[args.index("--task-reward-lerp") + 1] == "1.0"
    assert args[args.index("--motion-files") + 1].endswith("stairs.npz")
    assert args[args.index("--model-xml") + 1].endswith("robot/terrains/stairs.xml")
    assert "--terrain-embedded-in-model" in args
    assert "--preserve-motion-world-xy" in args
    assert args[args.index("--episode-steps") + 1] == str(EXACT_STAIR_EPISODE_STEPS)


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
            env.base_env.motion.reference_qpos[0, :2], source["qpos"][0, :2]
        )
        assert observation.shape == (1, env.obs_dim)
        assert np.isfinite(observation).all()
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
