"""Explicit sensor CLI extensions, isolated from the privileged training CLI."""
from __future__ import annotations
import argparse
from dataclasses import asdict, fields
from functools import partial
import sys

from .config import SensorObservationConfig
from .ppo_config import SensorPPOAMPConfig

OPTIONAL_POLICY_OPTIONS = ('gamma', 'gae_lambda', 'init_std', 'entropy_coef', 'amp_startup_gate')
STAIR_CURRICULUM_OPTIONS = ('stair_rsi_warmup_iterations', 'stair_rsi_ramp_iterations')

OBS_OPTIONS = ('history_steps', 'include_tendon_velocity', 'include_tendon_tracking_error',
               'tendon_velocity_scale', 'tendon_velocity_filter_alpha', 'tendon_tracking_error_scale',
               *tuple(field.name for field in fields(SensorObservationConfig)
                      if field.name.startswith(('tendon_measurement_', 'imu_measurement_'))))
POLICY_OPTIONS = ('recurrent_hidden_dim', 'state_estimator_enabled',
                  'state_estimator_include_foot_position', 'actor_use_estimated_state',
                  'state_joint_position_loss_weight', 'state_joint_velocity_loss_weight',
                  'state_root_velocity_loss_weight', 'state_contact_loss_weight',
                  'state_foot_position_loss_weight', 'future_prediction_enabled',
                  'future_prediction_steps', 'future_prediction_coef', 'privileged_context_loss_coef')


def _history_steps_from_frame_count(value: str) -> int:
    frame_count = int(value)
    if frame_count < 1:
        raise argparse.ArgumentTypeError('frame-count must be at least one')
    return frame_count - 1


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be non-negative')
    return result


def option_parser(ppo_config_type=SensorPPOAMPConfig):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    defaults = {**asdict(SensorObservationConfig()), **asdict(ppo_config_type())}
    for name in OBS_OPTIONS + POLICY_OPTIONS:
        value = defaults[name]
        kwargs = {'default': value}
        if isinstance(value, bool):
            kwargs['action'] = argparse.BooleanOptionalAction
        elif isinstance(value, tuple):
            kwargs.update(type=int, nargs='+')
        else:
            kwargs['type'] = type(value)
        if name == 'history_steps':
            history = parser.add_mutually_exclusive_group()
            history.add_argument('--history-steps', **{**kwargs, 'default': None})
            history.add_argument('--frame-count', dest='history_steps',
                                 type=_history_steps_from_frame_count,
                                 default=None,
                                 help='Total sensor frames, including the current frame.')
        else:
            parser.add_argument('--' + name.replace('_', '-'), **kwargs)
    for name in OPTIONAL_POLICY_OPTIONS:
        parser.add_argument('--'+name.replace('_', '-'), default=None,
                            **({'action': argparse.BooleanOptionalAction} if name == 'amp_startup_gate'
                               else {'type': float}))
    # Keep stair-only flags scoped to the stair trainer's configuration.
    for name in STAIR_CURRICULUM_OPTIONS:
        if name in defaults:
            parser.add_argument('--'+name.replace('_', '-'), type=_nonnegative_int,
                                default=None,
                                help=f'Stair curriculum iterations (default: {defaults[name]}).')
    return parser


def install_training_options(base_train):
    """Attach explicit observation config and preserve config/argv recording."""
    parser = option_parser(base_train.PPOAMPConfig)
    if '--help' in sys.argv or '-h' in sys.argv:
        print('Sensor state estimation and observation options:')
        parser.print_help()
    options, remaining = parser.parse_known_args(sys.argv[1:])
    if options.history_steps is None:
        options.history_steps = SensorObservationConfig().history_steps
    if options.recurrent_hidden_dim < 1:
        parser.error('recurrent-hidden-dim must be positive')
    options.future_prediction_steps = tuple(options.future_prediction_steps)
    observation = SensorObservationConfig(**{key: getattr(options, key) for key in OBS_OPTIONS})
    sys.argv[1:] = remaining
    original_parse = base_train.parse_args
    def parse_args():
        args = original_parse()
        vars(args).update(vars(options))
        return args
    base_train.parse_args = parse_args
    original_config = base_train.PPOAMPConfig
    def config(**kwargs):
        cfg = original_config(**kwargs)
        for key in POLICY_OPTIONS:
            setattr(cfg, key, getattr(options, key))
        for key in OPTIONAL_POLICY_OPTIONS + STAIR_CURRICULUM_OPTIONS:
            if getattr(options, key, None) is not None:
                setattr(cfg, key, getattr(options, key))
        return cfg
    base_train.PPOAMPConfig = config
    for key in ('LytRobotAmpEnv', 'LytRobotWarpAmpEnv', 'LytRobotMjlabAmpEnv'):
        constructor = getattr(base_train, key)
        if constructor is not None:
            setattr(base_train, key, partial(constructor, observation_cfg=observation))


def observation_config_from_metadata(metadata):
    saved = dict(metadata.get('sensor_observation', {}).get('config', {}))
    if 'gait_lookahead_seconds' in saved:
        saved['gait_lookahead_seconds'] = tuple(saved['gait_lookahead_seconds'])
    allowed = {field.name for field in fields(SensorObservationConfig)}
    return SensorObservationConfig(**{key: value for key, value in saved.items() if key in allowed})


def estimator_kwargs(architecture):
    """Missing fields mean the legacy network, never a changed dimension."""
    defaults = dict(state_estimator_enabled=False, state_estimator_include_foot_position=False,
                    actor_use_estimated_state=False, future_prediction_enabled=False,
                    future_prediction_steps=(1, 3, 5), future_prediction_dim=0)
    return {key: architecture.get(key, value) for key, value in defaults.items()}
