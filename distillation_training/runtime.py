"""Shared simulator construction, tensors, variants, and checkpoint schema."""
from __future__ import annotations

from pathlib import Path
import torch

from lytrobot_amp.env import LytRobotEnvCfg
from .config import (
    ACTION_DIM,
    HIDDEN_DIM,
    PHASE_FORCE,
    PRIVILEGED_DIM,
    PRIVILEGED_FIELDS,
    RECURRENT_STATE_DIM,
    SENSOR_VARIANTS,
    SensorVariantSpec,
    sensor_variant_spec,
)
from .envs import (
    DistillationForceCpuEnv,
    DistillationForceWarpEnv,
    DistillationNoForceCpuEnv,
    DistillationNoForceWarpEnv,
)


ROOT = Path(__file__).resolve().parents[1]


def add_runtime_args(parser, *, steps=96, sensor_variant_default=None):
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--num-steps', type=int, default=steps)
    parser.add_argument('--max-iterations', type=int, default=1000)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--sim-backend', choices=('cpu', 'warp'), default='warp')
    parser.add_argument('--sensor-variant', choices=tuple(SENSOR_VARIANTS), default=sensor_variant_default)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--output', default=None)


def selected_variant(args) -> SensorVariantSpec:
    value = getattr(args, 'sensor_variant', None) or PHASE_FORCE.name
    return sensor_variant_spec(value)


def make_env(args):
    if args.sim_backend == 'warp' and args.device != 'cuda':
        raise ValueError('Warp requires --device cuda')
    spec = selected_variant(args)
    args.sensor_variant = spec.name
    cfg = LytRobotEnvCfg(
        model_xml=str(ROOT/'robot/MJCFmodelposition.xml'),
        motion_file=str(ROOT/'robot/reference_data/walk1_mjcf.npz'),
        num_envs=args.num_envs,
        seed=args.seed,
        reference_ctrl_blend=0.0,
        domain_randomization=False,
        sensor_stability_reward=True,
        stand_start_prob=1.0,
        motion_reset_prob=0.0,
        stand_start_transition_min_steps=120,
        stand_start_transition_max_steps=120,
    )
    if spec.use_foot_force:
        env_type = DistillationForceWarpEnv if args.sim_backend == 'warp' else DistillationForceCpuEnv
    else:
        env_type = DistillationNoForceWarpEnv if args.sim_backend == 'warp' else DistillationNoForceCpuEnv
    env = env_type(cfg, args.device) if args.sim_backend == 'warp' else env_type(cfg)
    validate_runtime_schema(env, getattr(args, 'num_steps', getattr(args, 'steps', 0)), spec)
    return env, cfg


def tensor(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def env_step(env, action):
    sent = action if env.is_torch_env else action.detach().cpu().numpy()
    obs, reward, done, info = env.step(sent)
    return tensor(obs, action.device), tensor(reward, action.device), tensor(done, action.device), info


def dimension_metadata(spec: SensorVariantSpec) -> dict:
    return {
        'sensor_variant': spec.name,
        'sensor_dim': spec.sensor_dim,
        'estimator_input_dim': spec.estimator_input_dim,
        'action_dim': ACTION_DIM,
        'privileged_dim': PRIVILEGED_DIM,
        'hidden_dim': HIDDEN_DIM,
        'recurrent_state_dim': RECURRENT_STATE_DIM,
    }


def sensor_metadata(env, spec: SensorVariantSpec | None = None):
    spec = spec or sensor_variant_spec(
        PHASE_FORCE.name if env.obs_dim == PHASE_FORCE.sensor_dim else 'phase_no_force'
    )
    data = dict(env.observation_metadata)
    data.update(dimension_metadata(spec))
    data['privileged_schema'] = {
        'fields': list(PRIVILEGED_FIELDS), 'dim': PRIVILEGED_DIM, 'actor_input': False
    }
    data['foot_force_dim'] = 6 if spec.use_foot_force else 0
    data['amp_actor_input'] = False
    return data


def validate_runtime_schema(env, bptt_steps, spec: SensorVariantSpec):
    metadata = env.observation_metadata
    fields = tuple(metadata.get('actor_input_fields', ()))
    values = {
        'Actor observation dim': env.obs_dim,
        'IMU': env.imu_layout.dim,
        'Tendon': env.base_env.position_action_dim,
        'Foot force': len(metadata.get('foot_force', {}).get('channel_order', ())),
        'Phase': metadata.get('phase_dim', 0),
        'Estimator input': spec.estimator_input_dim,
        'GRU hidden': HIDDEN_DIM,
        'Privileged target': PRIVILEGED_DIM,
        'Recurrent state': RECURRENT_STATE_DIM,
        'BPTT': bptt_steps,
        'Control rate': round(1.0/env.base_env.dt),
        'reference_ctrl_blend': env.base_env.cfg.reference_ctrl_blend,
        'reference tendon actor input': any('reference' in str(field) for field in fields),
        'privileged actor input': bool(metadata.get('privileged_actor_input', True)),
    }
    expected = {
        'Actor observation dim': spec.sensor_dim,
        'IMU': 42,
        'Tendon': 14,
        'Foot force': 6 if spec.use_foot_force else 0,
        'Phase': 2,
        'Estimator input': spec.estimator_input_dim,
        'GRU hidden': 128,
        'Privileged target': 20,
        'Recurrent state': 142,
        'Control rate': 100,
        'reference_ctrl_blend': 0.0,
        'reference tendon actor input': False,
        'privileged actor input': False,
    }
    for key, value in expected.items():
        if values[key] != value:
            raise ValueError(f"schema mismatch: {key}={values[key]!r}, expected {value!r}")
    print('\n'.join(f'{key}: {value}' for key, value in values.items()))


_DIMENSION_KEYS = set(dimension_metadata(PHASE_FORCE))
TEACHER_KEYS = {
    'shared_actor', 'critic', 'median_critic', 'log_std', 'optimizer',
    'ppo_config', 'env_config', 'sensor_metadata', 'foot_force_metadata',
    'privileged_schema', 'phase_clock', 'iteration', 'global_step', *_DIMENSION_KEYS,
}
STUDENT_KEYS = {
    'shared_actor', 'history_estimator', 'log_std', 'sensor_metadata',
    'privileged_schema', 'phase_clock', 'iteration', 'global_step', *_DIMENSION_KEYS,
}


def checkpoint_variant(checkpoint: dict, kind: str = 'checkpoint') -> SensorVariantSpec:
    name = checkpoint.get('sensor_variant')
    if name is None:
        raise ValueError(f'{kind} checkpoint missing sensor_variant')
    spec = sensor_variant_spec(name)
    expected = dimension_metadata(spec)
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        rendered = ', '.join(f'{key}={actual!r} (expected {wanted!r})' for key, (actual, wanted) in mismatches.items())
        raise ValueError(f'{kind} checkpoint dimension metadata mismatch: {rendered}')
    metadata = checkpoint.get('sensor_metadata', {})
    if (metadata.get('actor_frame_dim') != spec.sensor_dim
        or metadata.get('phase_dim') != 2
        or metadata.get('sensor_variant') != spec.name
        or metadata.get('estimator_input_dim') != spec.estimator_input_dim):
        raise ValueError(f'{kind} checkpoint sensor metadata does not match {spec.name}')
    return spec


def validate_checkpoint(path, kind, expected_variant: str | SensorVariantSpec | None = None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    required = TEACHER_KEYS if kind == 'teacher' else STUDENT_KEYS
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f'{kind} checkpoint missing keys: {sorted(missing)}')
    spec = checkpoint_variant(checkpoint, kind)
    if expected_variant is not None:
        expected = sensor_variant_spec(expected_variant)
        if spec != expected:
            raise ValueError(
                f'{kind} checkpoint variant {spec.name!r} does not match expected {expected.name!r}'
            )
    return checkpoint


def inherit_checkpoint_variant(args, *checkpoints: tuple[str, dict]) -> SensorVariantSpec:
    specs = [checkpoint_variant(checkpoint, label) for label, checkpoint in checkpoints]
    first = specs[0]
    for spec in specs[1:]:
        if spec != first:
            raise ValueError(f'cross-variant checkpoint load rejected: {first.name} != {spec.name}')
    cli = getattr(args, 'sensor_variant', None)
    if cli is not None and sensor_variant_spec(cli) != first:
        raise ValueError(
            f'--sensor-variant {cli!r} does not match checkpoint variant {first.name!r}'
        )
    args.sensor_variant = first.name
    return first
