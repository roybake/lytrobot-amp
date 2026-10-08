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
DEFAULT_MOTION_FILE = ROOT / 'robot/reference_data/walk1_mjcf.npz'
TASK_REWARD_MODES = ('gated_reference', 'stage1_reference', 'sensor_stability')
LOCOMOTION_REWARD_PROFILE = 'gated_forward_gait_v5'
BACKWARD_VELOCITY_PENALTY = 8.0


def add_runtime_args(parser, *, steps=96, sensor_variant_default=None):
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--num-steps', type=int, default=steps)
    parser.add_argument('--max-iterations', type=int, default=1000)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--sim-backend', choices=('cpu', 'warp'), default='warp')
    parser.add_argument('--sensor-variant', choices=tuple(SENSOR_VARIANTS), default=sensor_variant_default)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--task-reward-mode', choices=TASK_REWARD_MODES, default='gated_reference')
    parser.add_argument('--motion-file', default=str(DEFAULT_MOTION_FILE))
    parser.add_argument('--target-velocity', type=float, default=1.28)
    parser.add_argument('--allow-legacy-reward-checkpoint', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', default='lytrobot-distillation')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-run-name', default=None)
    parser.add_argument('--wandb-tags', default='lytrobot,distillation,teacher')
    parser.add_argument('--wandb-log-interval', type=int, default=1)
    parser.add_argument('--output', default=None)


def resolve_training_output(args, stage: str) -> None:
    """Place flat-gait checkpoints in Stage1; validate explicit outputs."""
    from lytrobot_amp.training_paths import new_run_dir, validate_run_name
    output = Path(args.output or f"{stage}.pt").expanduser()
    role = "student" if stage.startswith("student") else "teacher"
    if output.parent == Path("."):
        run = new_run_dir(ROOT / "logs" / "distillation")
        output = run / "stage1" / role / output.name
    else:
        run = output.parent.parent.parent if output.parent.name in ("teacher", "student") else output.parent
        validate_run_name(run.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    if role == "student":
        from lytrobot_amp.training_paths import snapshot_teacher
        snapshot_teacher(getattr(args, "teacher_checkpoint", None), run / "stage1" / "teacher")
    args.output = str(output.resolve())
    args.wandb_run_name = args.wandb_run_name or f"{run.name}-{role}"
    print(f"[INFO] training output={args.output}")


def selected_variant(args) -> SensorVariantSpec:
    value = getattr(args, 'sensor_variant', None) or PHASE_FORCE.name
    return sensor_variant_spec(value)


def make_env(args):
    if args.sim_backend == 'warp' and args.device != 'cuda':
        raise ValueError('Warp requires --device cuda')
    spec = selected_variant(args)
    args.sensor_variant = spec.name
    task_reward_mode = getattr(args, 'task_reward_mode', 'gated_reference')
    if task_reward_mode not in TASK_REWARD_MODES:
        raise ValueError(f'unsupported task_reward_mode: {task_reward_mode!r}')
    cfg = LytRobotEnvCfg(
        model_xml=str(ROOT/'robot/MJCFmodelposition.xml'),
        motion_file=str(Path(getattr(args, 'motion_file', DEFAULT_MOTION_FILE)).resolve()),
        num_envs=args.num_envs,
        seed=args.seed,
        target_velocity=float(getattr(args, 'target_velocity', 0.87)),
        task_reward_mode=('stage1_reference' if task_reward_mode == 'gated_reference'
                          else task_reward_mode),
        reference_ctrl_blend=0.0,
        domain_randomization=False,
        sensor_stability_reward=task_reward_mode == 'sensor_stability',
        forward_velocity_reward_coef=2.0,
        forward_progress_reward_coef=1.0,
        forward_velocity_deficit_penalty_coef=2.5,
        fall_penalty=60.0,
        min_forward_progress_speed=0.12,
        forward_progress_grace_steps=200,
        forward_progress_window_steps=100,
        forward_progress_termination_penalty=10.0,
        symmetry_reward_coef=1.2,
        symmetry_foot_pos_sigma=0.08,
        side_foot_pos_coef=0.0,
        step_gap_coef=0.0,
        swing_advance_coef=0.0,
        foot_vel_reward_coef=0.35,
        stance_foot_reward_coef=1.0,
        contact_reward_coef=0.70,
        missed_contact_penalty_coef=0.80,
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
    env.task_reward_mode = task_reward_mode
    validate_runtime_schema(env, getattr(args, 'num_steps', getattr(args, 'steps', 0)), spec)
    return env, cfg


def tensor(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def env_step(env, action):
    sent = action if env.is_torch_env else action.detach().cpu().numpy()
    obs, reward, done, info = env.step(sent)
    task_reward = tensor(reward, action.device)
    if getattr(env, 'task_reward_mode', None) == 'gated_reference':
        task_reward = gated_reference_reward(task_reward, info, env.base_env.cfg.fall_penalty)
    return tensor(obs, action.device), task_reward, tensor(done, action.device), info


def gated_reference_reward(reward: torch.Tensor, info: dict, fall_penalty: float) -> torch.Tensor:
    """Keep imitation credit only when the robot is upright; penalize backing up."""
    device = reward.device
    upright = tensor(info['sensor_upright'], device)
    height = tensor(info['sensor_height'], device)
    forward_velocity = tensor(info['forward_velocity'], device)
    termination = tensor(info['termination'], device)
    stable_fraction = ((torch.minimum(upright, height) - 0.5) * 2.0).clamp(0.0, 1.0)
    shaped = (
        reward.clamp_min(0.0) * stable_fraction
        + reward.clamp_max(0.0)
        - BACKWARD_VELOCITY_PENALTY * (-forward_velocity).clamp(0.0, 2.0)
    )
    return torch.where(termination > 0.5, torch.full_like(shaped, -float(fall_penalty)), shaped)


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
    data['reference_tendon_actor_input'] = False
    return data


def task_metadata(env) -> dict:
    """Training-task identity shared by Teacher and every Student stage."""
    cfg = env.base_env.cfg
    motion_file = str(Path(cfg.motion_file).resolve())
    return {
        'task_reward_mode': getattr(env, 'task_reward_mode', 'gated_reference'),
        'motion_file': motion_file,
        'motion_file_stem': Path(motion_file).stem,
        'reference_frame_count': int(env.motion.frame_count),
        'reference_frequency': float(env.motion.frequency),
        'target_velocity': float(cfg.target_velocity),
        'reference_ctrl_blend': float(cfg.reference_ctrl_blend),
        'action_residual_scale': float(cfg.action_residual_scale),
        'control_dt': float(env.base_env.dt),
        'decimation': int(cfg.decimation),
        'stage1_reward_compatible': getattr(env, 'task_reward_mode', None) == 'stage1_reference',
        'locomotion_reward_profile': LOCOMOTION_REWARD_PROFILE,
        'reference_tendon_actor_input': False,
    }


def validate_runtime_schema(env, bptt_steps, spec: SensorVariantSpec):
    metadata = env.observation_metadata
    fields = tuple(metadata.get('actor_input_fields', ()))
    forbidden = {
        'reference_qpos', 'reference_qvel', 'reference_ctrl', 'future_reference',
        'foot_contact_truth', 'joint_truth', 'motion_frame_id', 'amp_feature',
    }
    leaked = sorted(forbidden.intersection(map(str, fields)))
    if leaked:
        raise ValueError(f'forbidden Teacher Actor inputs: {leaked}')
    privileged = env.privileged_schema
    if tuple(privileged.get('fields', ())) != tuple(PRIVILEGED_FIELDS) or privileged.get('dim') != 20:
        raise ValueError(f'Teacher privilege must be exactly the declared 20D schema: {privileged}')
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
    'privileged_schema', 'phase_clock', 'iteration', 'global_step',
    'task_reward_mode', 'motion_file', 'motion_file_stem', 'reference_frame_count',
    'reference_frequency', 'target_velocity', 'reference_ctrl_blend',
    'action_residual_scale', 'control_dt', 'decimation', 'stage1_reward_compatible',
    'locomotion_reward_profile',
    *_DIMENSION_KEYS,
}
STUDENT_KEYS = {
    'shared_actor', 'history_estimator', 'log_std', 'sensor_metadata',
    'privileged_schema', 'phase_clock', 'iteration', 'global_step',
    'task_reward_mode', 'motion_file', 'motion_file_stem', 'reference_frame_count',
    'reference_frequency', 'target_velocity', 'reference_ctrl_blend',
    'action_residual_scale', 'control_dt', 'decimation', 'stage1_reward_compatible',
    'locomotion_reward_profile',
    *_DIMENSION_KEYS,
}

TASK_IDENTITY_KEYS = (
    'task_reward_mode', 'motion_file_stem', 'reference_frame_count',
    'reference_frequency', 'target_velocity',
    'reference_ctrl_blend', 'action_residual_scale', 'control_dt', 'decimation',
    'locomotion_reward_profile',
)


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
    privilege = checkpoint.get('privileged_schema', {})
    if (tuple(privilege.get('fields', ())) != tuple(PRIVILEGED_FIELDS)
        or privilege.get('dim') != PRIVILEGED_DIM):
        raise ValueError(f'{kind} checkpoint privilege schema is not the restricted 20D schema')
    actor_fields = tuple(metadata.get('actor_input_fields', ()))
    if any('reference' in str(field) for field in actor_fields) or metadata.get('amp_actor_input'):
        raise ValueError(f'{kind} checkpoint contains forbidden Actor input metadata')
    return spec


def validate_checkpoint(path, kind, expected_variant: str | SensorVariantSpec | None = None,
                        *, allow_legacy_reward_checkpoint: bool = False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    required = TEACHER_KEYS if kind == 'teacher' else STUDENT_KEYS
    missing = required - set(checkpoint)
    if missing:
        task_missing = missing.intersection(TASK_IDENTITY_KEYS)
        if task_missing and allow_legacy_reward_checkpoint:
            print('[WARN] DANGEROUS legacy reward checkpoint accepted explicitly; '
                  f'missing task identity metadata: {sorted(task_missing)}')
            missing = missing - set(TASK_IDENTITY_KEYS) - {
                'motion_file', 'reference_frame_count', 'reference_frequency',
                'stage1_reward_compatible',
            }
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
    complete = [(label, checkpoint) for label, checkpoint in checkpoints
                if all(key in checkpoint for key in TASK_IDENTITY_KEYS)]
    if complete:
        first_label, first_checkpoint = complete[0]
        for label, checkpoint in complete[1:]:
            mismatches = {
                key: (first_checkpoint[key], checkpoint[key])
                for key in TASK_IDENTITY_KEYS
                if checkpoint[key] != first_checkpoint[key]
            }
            if mismatches:
                raise ValueError(
                    f'task mismatch between {first_label} and {label}: {mismatches}'
                )
        requested = {
            'task_reward_mode': getattr(args, 'task_reward_mode', 'gated_reference'),
            'motion_file_stem': Path(getattr(args, 'motion_file', DEFAULT_MOTION_FILE)).stem,
            'target_velocity': float(getattr(args, 'target_velocity', 0.87)),
            'reference_ctrl_blend': 0.0,
            'action_residual_scale': 0.5,
            'control_dt': 0.01,
            'decimation': 2,
            'locomotion_reward_profile': LOCOMOTION_REWARD_PROFILE,
        }
        mismatches = {
            key: (first_checkpoint[key], value)
            for key, value in requested.items()
            if first_checkpoint[key] != value
        }
        if mismatches:
            raise ValueError(f'checkpoint task does not match requested runtime: {mismatches}')
    return first
