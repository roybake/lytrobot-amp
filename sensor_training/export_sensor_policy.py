#!/usr/bin/env python3
"""Export a strict, pure-sensor recurrent checkpoint using the existing exporter."""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import onnx
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from sensor_training.networks import SensorActorCritic
from sensor_training.experiment_options import estimator_kwargs
from sensor_training.trainer import SensorHistoryPPOAMPTrainer
from sensor_training.final_profile import final_checkpoint_errors


def load_policy(checkpoint):
    architecture = checkpoint['sensor_policy_architecture']
    config = checkpoint['ppo_amp_config']
    kwargs = {key: architecture[key] for key in ('sensor_frame_dim', 'actor_history_frames',
              'recurrent_hidden_dim', 'use_moe', 'num_experts', 'gate_hidden_dims', 'expert_hidden_dims',
              'residual_scale', 'privileged_context_dim') if key in architecture}
    kwargs['use_moe'] = bool(architecture.get('use_moe', True))
    kwargs.update({key: config[key] for key in ('actor_hidden_dims', 'critic_hidden_dims',
                  'init_std', 'min_action_std', 'max_action_std') if key in config})
    policy = SensorActorCritic(checkpoint['obs_dim'], checkpoint['action_dim'],
                               **kwargs, **estimator_kwargs(architecture))
    policy.load_state_dict(checkpoint['actor_critic'], strict=True)
    return policy.eval()


def compute_profile(policy):
    modules = [policy.recurrent, policy.actor]
    if policy.state_estimator_enabled:
        modules += [getattr(policy, name+'_head') for name in ('joint_position', 'joint_velocity',
                    'root_velocity', 'contact')]
        if policy.state_estimator_include_foot_position:
            modules += [policy.foot_position_head]
    if policy.use_moe:
        modules += [policy.gate, policy.experts]
    parameters = sum(parameter.numel() for module in modules for parameter in module.parameters())
    gru = 3*policy.recurrent_hidden_dim*(policy.recurrent_input_dim+policy.recurrent_hidden_dim)
    linear = sum(layer.weight.numel() for module in modules[1:] for layer in module.modules() if isinstance(layer, nn.Linear))
    return {'deployment_parameters': parameters, 'macs_per_step': gru+linear,
            'approx_flops_per_step': 2*(gru+linear), 'fp32_weight_bytes': 4*parameters,
            'control_hz': 100, 'macs_per_second_at_100hz': 100*(gru+linear)}


def export_checkpoint(checkpoint_path, output, *, require_final=True):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if require_final:
        errors = final_checkpoint_errors(checkpoint)
        if errors:
            raise ValueError('Deployment export rejected: '+ '; '.join(errors))
    policy = load_policy(checkpoint)
    # The existing method needs only the reconstructed Actor and observation
    # dimension. No physics, true state, Teacher file or optimizer is required.
    exporter = SensorHistoryPPOAMPTrainer.__new__(SensorHistoryPPOAMPTrainer)
    exporter.actor_critic = policy
    exporter.env = SimpleNamespace(obs_dim=checkpoint['obs_dim'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exporter._export_onnx(output, checkpoint)
    model = onnx.load(output)
    onnx.checker.check_model(model)
    if [i.name for i in model.graph.input] != ['observation', 'recurrent_state', 'episode_start']:
        raise RuntimeError('Unexpected external ONNX inputs.')
    if [i.name for i in model.graph.output] != ['action', 'next_recurrent_state']:
        raise RuntimeError('Unexpected ONNX outputs.')
    import json
    output.with_suffix('.deployment.json').write_text(json.dumps({
        'onnx': str(output), 'compute': compute_profile(policy),
        'sensor_observation': checkpoint['sensor_observation'],
        'state_estimation': checkpoint['state_estimation'],
        'sensor_policy_architecture': checkpoint['sensor_policy_architecture'],
        'control': {key: checkpoint['env_config'][key] for key in
                    ('reference_ctrl_blend', 'action_residual_scale', 'timestep', 'decimation')},
    }, indent=2)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-legacy', action='store_true', help='Export another compatible sensor architecture explicitly.')
    args = parser.parse_args()
    result = export_checkpoint(args.checkpoint, args.output, require_final=not args.allow_legacy)
    print(f'[EXPORT] {result}')


if __name__ == '__main__':
    main()
