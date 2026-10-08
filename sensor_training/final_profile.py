"""Production-oriented flat human-gait recipe; legacy defaults remain explicit."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'robot/reference_data/walk1_mjcf.npz'

FINAL_SENSOR_OPTIONS = {
    'include_tendon_velocity': True,
    'include_tendon_tracking_error': True,
    'tendon_velocity_filter_alpha': .25,
    'tendon_velocity_scale': 1.0,
    'tendon_tracking_error_scale': .05,
    'tendon_measurement_enabled': True,
    'tendon_measurement_noise_std': .0001,
    'tendon_measurement_dr_noise_std': .0005,
    'tendon_measurement_bias_m': 0.,
    'tendon_measurement_bias_range_m': .002,
    'tendon_measurement_scale_error': .01,
    'tendon_measurement_quantization_m': .00005,
    'tendon_measurement_delay_max_ticks': 2,
    'tendon_measurement_clip_margin_m': .02,
    'imu_measurement_noise_std': .001,
    'imu_measurement_dr_noise_std': .01,
    'imu_measurement_bias_range': .02,
    'imu_measurement_delay_max_ticks': 2,
    'recurrent_hidden_dim': 256,
    'state_estimator_enabled': True,
    'state_estimator_include_foot_position': True,
    'actor_use_estimated_state': True,
    'state_joint_position_loss_weight': 1.,
    'state_joint_velocity_loss_weight': .5,
    'state_root_velocity_loss_weight': .5,
    'state_contact_loss_weight': .5,
    'state_foot_position_loss_weight': .5,
    'future_prediction_enabled': False,
}


def apply_final_profile(args):
    if args.num_envs is None:
        args.num_envs = 1024
    if args.student_num_envs is None:
        # Student rollout capacity is independent of the basic Teacher batch.
        args.student_num_envs = 1024
    for key, value in FINAL_SENSOR_OPTIONS.items():
        setattr(args, key, value)
    args.student_root_imu_only = False
    args.teacher_quality_gate = True
    args.include_robust_stage = True
    if args.student_num_steps is None:
        args.student_num_steps = 96
    if args.student_num_steps not in (96, 128):
        raise ValueError('Final sensor gait uses BPTT 96 (default) or 128.')


def final_teacher_evaluation_flags():
    """Evaluation does not alter the basic scripts/train_stage1.py recipe."""
    return ['--evaluate-gait', '--gait-fidelity-selection',
            '--gait-evaluation-seconds', '30', '--gait-evaluation-seeds', '42', '123', '2026',
            '--no-stop-on-gait-acceptance']


def final_training_flags(*, student=False, robust=False, iterations=50000):
    flags = ['--motion-file', str(REFERENCE), '--target-velocity', '1.28',
             '--task-reward-mode', 'stage1_reference', '--action-residual-scale', '.5',
             '--amp-reward-coef', '.3', '--task-reward-lerp', '.9',
             '--sensor-natural-gait-reward-coef', '.5', '--reference-gait-style-reward-coef', '4.',
             '--reference-joint-position-reward-coef', '4.',
             '--reference-joint-velocity-reward-coef', '1.5',
             '--reference-root-velocity-reward-coef', '1.5',
             '--reference-foot-position-reward-coef', '2.5',
             '--gait-fidelity-selection', '--gait-evaluation-seconds', '30',
             '--gait-evaluation-seeds', '42', '123', '2026',
             '--no-stop-on-gait-acceptance', '--obs-noise-std', '0.0',
             '--no-trajectory-feedback', '--no-sensor-phase-observation', '--no-use-moe']
    if student:
        flags += ['--no-sensor-stability-reward', '--sensor-knee-gait-reward-coef', '0.',
                  '--teacher-action-loss-scale', '1.0',
                  '--teacher-action-loss-coef', '2.0' if robust else '5.0',
                  '--teacher-action-loss-final-coef', '.5' if robust else '1.0',
                  '--teacher-action-loss-decay-iterations', str(min(12000, max(1, iterations*3//4))),
                  '--teacher-action-probability', '.2' if robust else '.8',
                  '--teacher-action-probability-final', '0.',
                  '--teacher-action-probability-decay-iterations', str(min(2000 if robust else 4000, max(1, iterations//3)))]
    if robust:
        flags += ['--domain-randomization', '--dr-curriculum-mode', 'linear',
                  '--dr-curriculum-iterations', str(max(1, iterations//2)),
                  '--warp-global-domain-randomization', '--global-dr-resample-interval', '250',
                  '--body-mass-scale-min', '.85', '--body-mass-scale-max', '1.15',
                  '--payload-mass-min', '0.', '--payload-mass-max', '2.',
                  '--ground-friction-min', '.65', '--ground-friction-max', '1.4',
                  '--gravity-min', '9.6', '--gravity-max', '10.05',
                  '--tendon-length-scale-min', '.98', '--tendon-length-scale-max', '1.02',
                  '--actuator-strength-scale-min', '.9', '--actuator-strength-scale-max', '1.1',
                  '--action-delay-min-ms', '0.', '--action-delay-max-ms', '20.',
                  '--push-force-min', '0.', '--push-force-max', '20.']
    else:
        flags += ['--no-domain-randomization', '--action-noise-std', '0.0']
    return flags


def final_checkpoint_errors(checkpoint, *, require_no_takeover=True):
    """Validate deployment architecture and semantics, not just tensor sizes."""
    errors = []
    architecture = checkpoint.get('sensor_policy_architecture', {})
    for key in ('recurrent_hidden_dim', 'state_estimator_enabled',
                'state_estimator_include_foot_position', 'actor_use_estimated_state', 'future_prediction_enabled'):
        if architecture.get(key) != FINAL_SENSOR_OPTIONS[key]:
            errors.append(f'architecture {key} must be {FINAL_SENSOR_OPTIONS[key]}')
    observation = checkpoint.get('sensor_observation', {})
    config = observation.get('config', {})
    for key in ('include_tendon_velocity', 'include_tendon_tracking_error', 'tendon_measurement_enabled'):
        if config.get(key) is not True:
            errors.append(f'observation {key} must be enabled')
    if observation.get('tendon_signal_semantics') != 'shared_cached_measurement_v1':
        errors.append('checkpoint does not use a shared measured tendon signal')
    if observation.get('actor_frame_dim') != 144 or checkpoint.get('obs_dim') != 864 or checkpoint.get('action_dim') != 14:
        errors.append('final observation/action dimensions must be 864/14 (six 144D frames)')
    if architecture.get('actor_history_frames') != 6 or architecture.get('recurrent_input_dim') != 864:
        errors.append('final GRU must consume the short six-frame sensor history')
    from sensor_training.state_estimation import state_layout, STATE_SCALES
    state = checkpoint.get('state_estimation', {})
    if state.get('target_layout') != state_layout(True):
        errors.append('state target layout must contain exactly 45 values including feet')
    if state.get('target_scales') != STATE_SCALES:
        errors.append('state normalization must match the physical 45D representation')
    if observation.get('imu_dim_per_frame') != 42:
        errors.append('final policy requires all seven installed IMUs (42 values)')
    env = checkpoint.get('env_config', {})
    for key, value in (('reference_ctrl_blend', 1.), ('decimation', 2),
                       ('sensor_root_imu_only', False), ('trajectory_feedback', False),
                       ('sensor_phase_observation', False)):
        if env.get(key, False if isinstance(value, bool) else None) != value:
            errors.append(f'env {key} must be {value}')
    if Path(env.get('motion_file', '')).resolve() != REFERENCE.resolve():
        errors.append('checkpoint reference must be walk1_mjcf.npz')
    if require_no_takeover and checkpoint.get('distillation', {}).get('teacher_behavior_probability', 1.) != 0.:
        errors.append('Teacher takeover has not reached zero')
    return errors
