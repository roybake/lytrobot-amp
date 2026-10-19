"""Continuous evaluation supports long non-looping references without resets."""

from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from lytrobot_amp.env import LytRobotEnvCfg
from sensor_training import gait_evaluation as evaluation

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkpoint_environment(monkeypatch):
    cfg = LytRobotEnvCfg(
        motion_file=str(ROOT / 'robot/reference_data/walk1_mjcf_repaired.npz'),
        motion_loop=(False,),
        stand_start_phase_delay_steps=0,
        stand_start_transition_min_steps=1,
        stand_start_transition_max_steps=1,
    )
    metadata = {'env_config': asdict(cfg)}
    monkeypatch.setattr(evaluation, 'load_onnx_metadata', lambda path: metadata)
    environments = []

    def track_environment(factory):
        def create(*args):
            env = factory(*args)
            original_close = env.close
            def capture_close():
                base = env.base_env if hasattr(env, 'base_env') else env
                env.evaluated_qpos = [data.qpos.copy() for data in base.datas]
                original_close()
            env.close = mock.Mock(wraps=capture_close)
            environments.append(env)
            return env
        return create

    monkeypatch.setattr(evaluation, 'LytRobotAmpEnv', track_environment(evaluation.LytRobotAmpEnv))
    monkeypatch.setattr(evaluation, 'SensorHistoryCpuAmpEnv', track_environment(evaluation.SensorHistoryCpuAmpEnv))

    class ZeroPolicy:
        def __init__(self, path):
            pass

        def validate_dimensions(self, obs_dim, action_dim):
            self.action_dim = action_dim

        def act_inference(self, obs, episode_start):
            return np.zeros((len(obs), self.action_dim), dtype=np.float32)

    monkeypatch.setattr(evaluation, 'OnnxPolicy', ZeroPolicy)
    return metadata, environments


@pytest.mark.parametrize('sensor', [False, True])
def test_non_looping_reference_completes_requested_duration(checkpoint_environment, sensor):
    metadata, environments = checkpoint_environment
    if sensor:
        metadata['sensor_policy_architecture'] = {'recurrent_hidden_dim': 16}
    metrics = evaluation.evaluate_gait_checkpoint('unused.onnx', seconds=.1)
    env = environments[0]
    base = env.base_env if sensor else env
    assert not base.motion.source_loop
    assert metadata['env_config']['motion_loop'] == (False,)
    assert metrics['eval/gait_survived'] == 1.
    assert metrics['eval/gait_duration_s'] == pytest.approx(.1)
    env.close.assert_called_once()


def test_non_looping_reference_end_cannot_pass_as_continuous_gait(checkpoint_environment):
    metadata, environments = checkpoint_environment
    metadata['env_config']['stand_start_motion_phase'] = 43.
    with pytest.raises(RuntimeError, match='Unexpected reset before the continuous evaluation time limit'):
        evaluation.evaluate_gait_checkpoint('unused.onnx', seconds=.1)
    assert not environments[0].motion.source_loop
    environments[0].close.assert_called_once()


def test_explicit_force_loop_still_allows_crossing_reference_end(checkpoint_environment):
    metadata, environments = checkpoint_environment
    metadata['env_config']['stand_start_motion_phase'] = 43.
    metrics = evaluation.evaluate_gait_checkpoint('unused.onnx', seconds=.1, force_loop=True)
    assert environments[0].motion.source_loop
    assert metadata['env_config']['motion_loop'] == (False,)
    assert metrics['eval/gait_survived'] == 1.
    environments[0].close.assert_called_once()


def test_non_looping_teacher_student_suite(checkpoint_environment, tmp_path):
    _, environments = checkpoint_environment
    teacher = tmp_path / 'teacher.onnx'
    teacher.touch()
    evaluation._cached_teacher_metrics.cache_clear()
    try:
        metrics = evaluation.evaluate_gait_suite('student.onnx', seconds=.1, teacher_checkpoint=teacher)
    finally:
        evaluation._cached_teacher_metrics.cache_clear()
    assert metrics['eval/gait_seed_count'] == 3.
    assert metrics['eval/gait_survived'] == 1.
    assert metrics['teacher/gait_survived'] == 1.
    assert metrics['student/gait_survived'] == 1.
    assert len(environments) == 6
    for env in environments:
        assert not env.motion.source_loop
        env.close.assert_called_once()


def test_evaluation_fixes_randomness_before_environment_construction(checkpoint_environment):
    metadata, environments = checkpoint_environment
    metadata['env_config'].update(
        num_envs=32, seed=2026, randomize_episode_seed=True,
        reset_noise_scale=.013, action_noise_std=.04, domain_randomization_scale=.25,
        episode_steps=900, episode_timeout_jitter_steps=17,
    )
    evaluation.evaluate_gait_checkpoint('unused.onnx', seconds=.1, seed=123)
    expected = replace(LytRobotEnvCfg(**metadata['env_config']), num_envs=1, seed=123,
        randomize_episode_seed=False, reset_noise_scale=0., action_noise_std=0., obs_noise_std=0.,
        domain_randomization=False, domain_randomization_scale=0.)
    assert asdict(environments[0].cfg) == asdict(expected)


def test_nominal_evaluation_repeats_with_random_training_metadata(checkpoint_environment):
    metadata, environments = checkpoint_environment
    metadata['env_config'].update(randomize_episode_seed=True, reset_noise_scale=.02,
        action_noise_std=.04, obs_noise_std=.03, motion_loop=(True,),
        stand_start_phase_delay_steps=25, stand_start_transition_min_steps=80,
        stand_start_transition_max_steps=160)
    first = evaluation.evaluate_gait_checkpoint('unused.onnx', seconds=2., seed=42)
    second = evaluation.evaluate_gait_checkpoint('unused.onnx', seconds=2., seed=42)
    assert first == second
    np.testing.assert_array_equal(environments[0].stand_start_transition_steps,
                                  environments[1].stand_start_transition_steps)
    np.testing.assert_array_equal(environments[0].evaluated_qpos[0],
                                  environments[1].evaluated_qpos[0])
