"""Teacher uses the user's basic Stage1 recipe; selection rewards progress."""
from __future__ import annotations
from pathlib import Path
import runpy
import sys
from unittest import mock

from scripts import train
from sensor_training import train_serial_distillation as serial
from sensor_training.gait_evaluation import gait_checkpoint_quality, reference_fidelity_metrics, FIDELITY_LIMITS

ROOT = Path(__file__).resolve().parents[2]


def effective_args(command):
    original_run = runpy.run_path
    captured = []
    with mock.patch.object(sys, 'argv', command[1:]), mock.patch.object(runpy, 'run_path', lambda *a, **k: captured.extend(sys.argv)):
        original_run(str(ROOT/'scripts/train_stage1.py'), run_name='__main__')
    with mock.patch.object(sys, 'argv', captured):
        return vars(train.parse_args())


def test_final_teacher_training_arguments_match_basic_stage1():
    with mock.patch.object(sys, 'argv', ['pipeline', '--final-sensor-gait']):
        args = serial.parse_args()
    assert args.num_envs == 1024 and args.student_num_envs == 1024
    command = serial._phase_command('teacher_stage1', args, serial._phase_paths('10-02-10:00:00'), resume_checkpoint=None)
    baseline = [sys.executable, str(ROOT/'scripts/train_stage1.py'), '--device', 'cuda', '--sim-backend', 'warp',
                '--num-envs', '1024', '--max-iterations', '50000']
    actual, expected = effective_args(command), effective_args(baseline)
    evaluation_logging = {'log_dir', 'wandb_log_interval', 'evaluate_gait', 'gait_fidelity_selection',
                          'gait_evaluation_seconds', 'gait_evaluation_seeds', 'stop_on_gait_acceptance'}
    for key in expected.keys() - evaluation_logging:
        assert actual[key] == expected[key], key
    assert actual['amp_reward_coef'] == 1.2
    assert actual['task_reward_lerp'] == .5
    assert actual['reference_joint_position_reward_coef'] == 1.4
    assert actual['reference_joint_velocity_reward_coef'] == .45
    assert actual['termination_mode'] == 'fall'
    assert actual['episode_steps'] == 0
    assert actual['motion_loop'] == ['true']
    assert actual['stand_start_prob'] == 1.
    assert actual['motion_reset_prob'] == 0.
    assert actual['forward_velocity_reward_coef'] == 2.
    assert actual['forward_velocity_deficit_penalty_coef'] == 2.5
    assert actual['learning_rate'] == expected['learning_rate']
    assert '--teacher-action-loss-coef' not in command
    assert actual['gait_fidelity_selection'] and actual['gait_evaluation_seconds'] == 30


def test_failed_checkpoint_quality_does_not_freeze_at_first_saved_model():
    prefix = {'eval/reference_fidelity_accepted': 0., 'eval/gait_survived': 0.,
              'eval/gait_duration_s': 1.61, 'eval/reference_fidelity_score': .95}
    walking = dict(prefix, **{'eval/gait_duration_s': 24.81, 'eval/reference_fidelity_score': .1})
    survived = dict(walking, **{'eval/gait_survived': 1., 'eval/gait_duration_s': 30.})
    accepted = dict(survived, **{'eval/reference_fidelity_accepted': 1., 'eval/reference_fidelity_score': .01})
    assert gait_checkpoint_quality(prefix, fidelity=True) < gait_checkpoint_quality(walking, fidelity=True)
    assert gait_checkpoint_quality(walking, fidelity=True) < gait_checkpoint_quality(survived, fidelity=True)
    assert gait_checkpoint_quality(survived, fidelity=True) < gait_checkpoint_quality(accepted, fidelity=True)
    metrics = {key: maximum*.5 for key, maximum in FIDELITY_LIMITS.items()}
    metrics.update({'eval/gait_accepted': 0., 'eval/episode_completion': 0.,
                    'eval/swing_contact_fraction': 0.})
    score = reference_fidelity_metrics(metrics)
    assert score['eval/reference_fidelity_accepted'] == 0.
    assert score['eval/reference_fidelity_score'] > 0.


def test_handoff_rejects_stale_early_best_even_when_both_old_scores_are_zero(tmp_path):
    for name in ('model_best', 'model_final'):
        (tmp_path/(name+'.pt')).touch(); (tmp_path/(name+'.onnx')).touch()
    def evaluate(path, **kwargs):
        return {'eval/reference_fidelity_accepted': 0., 'eval/reference_fidelity_score': 0.,
                'eval/gait_survived': 0., 'eval/gait_duration_s': 1.61 if path.stem == 'model_best' else 24.81}
    with mock.patch('sensor_training.gait_evaluation.evaluate_gait_suite', evaluate):
        selected, quality = serial.select_teacher_handoff(tmp_path)
    assert selected.name == 'model_final.pt' and quality['eval/gait_duration_s'] == 24.81
