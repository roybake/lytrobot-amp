"""Walking training spans PPO updates and ends an episode only on a fall."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
import torch

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
from lytrobot_amp.env_mjlab import LytRobotMjlabAmpEnv
from lytrobot_amp.ppo_amp import PPOAMPConfig, PPOAMPTrainer
from scripts.play import load_onnx_metadata, make_env_cfg, parse_args, OnnxPolicy
from sensor_training import gait_evaluation
from sensor_training.gait_evaluation import evaluate_gait_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def continuous_cfg():
    return LytRobotEnvCfg(
        model_xml=str(ROOT / 'robot/MJCFmodelfinal.xml'),
        motion_file=str(ROOT / 'robot/reference_data/walk1_mjcf_repaired.npz'),
        num_envs=1, seed=42, domain_randomization=False, reset_noise_scale=0.0,
        stand_start_prob=1.0, motion_reset_prob=0.0, action_noise_std=0.0,
        episode_steps=1, episode_timeout_jitter_steps=240, motion_loop=(False,),
        termination_mode='fall', min_forward_progress_speed=100.0,
        forward_progress_grace_steps=0, forward_progress_window_steps=1,
    )


@pytest.fixture(params=['cpu', 'warp', 'mjlab'])
def continuous_env(request, continuous_cfg):
    if request.param != 'cpu' and not torch.cuda.is_available():
        pytest.skip('CUDA is required for the GPU physics backends')
    factories = {'cpu': LytRobotAmpEnv, 'warp': LytRobotWarpAmpEnv,
                 'mjlab': LytRobotMjlabAmpEnv}
    env = factories[request.param](replace(continuous_cfg))
    try:
        yield env
    finally:
        env.close()


def zero_action(env):
    if getattr(env, 'is_torch_env', False):
        return torch.zeros(env.num_envs, env.action_dim, device=env.device)
    return np.zeros((env.num_envs, env.action_dim), dtype=np.float32)


def test_training_crosses_time_limit_reference_end_and_progress_guard(continuous_env):
    env = continuous_env
    assert env.cfg.episode_steps == 0
    assert env.cfg.episode_timeout_jitter_steps == 0
    assert env.cfg.motion_loop == (True,)
    env.stand_start[:] = False
    env.episode_step[:] = 6000
    env.motion_phase[:] = env.motion.frame_count - 0.25
    _, _, done, info = env.step(zero_action(env))
    assert not bool(done[0])
    assert float(info['time_outs'][0]) == 0.0
    assert int(env.episode_step[0]) == 6001
    assert float(env.motion_phase[0]) < 1.0


def test_training_starts_from_standing(continuous_env):
    env = continuous_env
    env.reset()
    assert bool(env.stand_start[0])
    assert float(env.motion_phase[0]) == env.cfg.stand_start_motion_phase
    if getattr(env, 'is_torch_env', False):
        qpos = env.qpos[0].detach().cpu().numpy()
        qvel = env.qvel[0].detach().cpu().numpy()
    else:
        qpos, qvel = env.datas[0].qpos, env.datas[0].qvel
    np.testing.assert_allclose(qpos, env.motion.model_qpos0, atol=1e-7)
    np.testing.assert_array_equal(qvel, 0.)


def test_training_resets_at_existing_posture_fall_threshold(continuous_env):
    env = continuous_env
    if getattr(env, 'is_torch_env', False):
        env.qpos[:, 3:7] = torch.tensor([np.sqrt(.5), 0., np.sqrt(.5), 0.],
                                      device=env.device)
        env.qvel.zero_()
        env._warp_forward()
    else:
        data = env.datas[0]
        data.qpos[3:7] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
        data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, data)
    _, _, done, info = env.step(zero_action(env))
    assert bool(done[0])
    assert float(info['time_outs'][0]) == 0.0
    assert float(info['termination'][0]) == 1.0
    assert int(env.episode_step[0]) == 0


def test_ppo_updates_without_ending_episode_and_export_evaluates_for_finite_time(
    continuous_cfg, tmp_path, monkeypatch,
):
    torch.set_num_threads(1)
    training_cfg = replace(continuous_cfg, stand_start_prob=1., motion_reset_prob=0.)
    env = LytRobotAmpEnv(training_cfg)
    cfg = PPOAMPConfig(num_steps=4, update_epochs=1, num_minibatches=1,
                       amp_enabled=False, actor_hidden_dims=(16,),
                       critic_hidden_dims=(16,), init_std=1e-4, min_action_std=1e-4)
    trainer = PPOAMPTrainer(env, cfg, 'cpu', tmp_path)
    with torch.no_grad():
        for parameter in trainer.actor_critic.actor.parameters():
            parameter.zero_()
    try:
        trainer.train_iteration()
        trainer.train_iteration()
        assert trainer.global_step == 8
        assert int(env.episode_step[0]) == 8
        pt_path, onnx_path = trainer.save(tmp_path / 'model_final.pt', iteration=1)
        metadata = load_onnx_metadata(onnx_path)
        assert metadata['env_config']['episode_steps'] == 0
        assert metadata['env_config']['termination_mode'] == 'fall'
        assert metadata['env_config']['stand_start_prob'] == 1.
        assert metadata['env_config']['motion_reset_prob'] == 0.
        assert trainer.load(pt_path) == 2
        assert env.cfg.episode_steps == 0
        monkeypatch.setattr(sys, 'argv', ['play.py', '--checkpoint', str(onnx_path), '--no-viewer'])
        playback_cfg = make_env_cfg(parse_args(), metadata)
        saved_cfg = json.loads(json.dumps(asdict(training_cfg)))
        assert json.loads(json.dumps(asdict(playback_cfg))) == saved_cfg
        playback_env = LytRobotAmpEnv(playback_cfg)
        training_env = LytRobotAmpEnv(replace(training_cfg))
        try:
            training_obs, playback_obs = training_env.reset(), playback_env.reset()
            np.testing.assert_array_equal(training_obs, playback_obs)
            assert training_env.stand_start[0] and playback_env.stand_start[0]
            np.testing.assert_array_equal(playback_env.datas[0].qpos, playback_env.motion.model_qpos0)
            np.testing.assert_array_equal(playback_env.datas[0].qvel, 0.)
            initial_obs = playback_obs.copy()
            policy = OnnxPolicy(onnx_path)
            for _ in range(200):
                action = policy.act_inference(playback_obs)
                training_obs, training_reward, training_done, _ = training_env.step(action)
                playback_obs, playback_reward, playback_done, _ = playback_env.step(action)
                np.testing.assert_array_equal(training_obs, playback_obs)
                np.testing.assert_array_equal(training_reward, playback_reward)
                np.testing.assert_array_equal(training_done, playback_done)

            evaluated = []
            def capture_evaluation(cfg):
                result = LytRobotAmpEnv(cfg)
                reset = result.reset
                def capture_reset(*args):
                    obs = reset(*args)
                    evaluated.append((asdict(cfg), obs.copy()))
                    return obs
                result.reset = capture_reset
                return result
            monkeypatch.setattr(gait_evaluation, 'LytRobotAmpEnv', capture_evaluation)
            metrics = evaluate_gait_checkpoint(onnx_path, seconds=.05)
            evaluation_cfg = replace(training_cfg, randomize_episode_seed=False,
                reset_noise_scale=0., action_noise_std=0., obs_noise_std=0.,
                domain_randomization=False, domain_randomization_scale=0.)
            assert json.loads(json.dumps(evaluated[0][0])) == json.loads(json.dumps(asdict(evaluation_cfg)))
            np.testing.assert_array_equal(evaluated[0][1], initial_obs)
        finally:
            playback_env.close()
            training_env.close()
        assert metrics['eval/gait_duration_s'] == pytest.approx(.05)
        assert metrics['eval/gait_survived'] == 1.0
    finally:
        env.close()
