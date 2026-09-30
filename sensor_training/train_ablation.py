#!/usr/bin/env python3
"""A–G sensor-state ablations with identical rewards, resets and physics budget."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from sensor_training.training_defaults import stage_policy_rollout_envs


def training_command(experiment, seed, bptt, iterations, teacher=None, device='cuda', backend='warp'):
    if experiment not in 'ABCDEFG' or len(experiment) != 1:
        raise ValueError('Experiment must be A–G.')
    distill = experiment in 'FG'
    if distill and not teacher:
        raise ValueError('F/G require --teacher-checkpoint.')
    entry = 'train_distill_stage1.py' if distill else 'train_stage1.py'
    command = [sys.executable, str(ROOT/'sensor_training'/entry),
               '--device', device, '--sim-backend', backend,
               '--seed', str(seed), '--num-steps', str(bptt),
               '--num-envs', str(stage_policy_rollout_envs(bptt, 2)),
               '--decimation', '2', '--max-iterations', str(iterations),
               '--log-dir', str(ROOT/'logs'/'sensor_ablation'/f'{experiment}_s{seed}_bptt{bptt}'),
               '--motion-file', str(ROOT/'robot/reference_data/walk1_omniretarget.npz'),
               '--motion-loop', 'true', '--reference-ctrl-blend', '1.0',
               '--action-residual-scale', '.5', '--target-velocity', '.87',
               '--task-reward-mode', 'stage1_reference', '--no-sensor-stability-reward',
               '--no-domain-randomization', '--sensor-root-imu-only',
               '--episode-steps', '2000', '--episode-timeout-jitter-steps', '0',
               '--stand-start-prob', '.7', '--motion-reset-prob', '1.0',
               '--stand-start-motion-phase', '0', '--stand-start-phase-delay-steps', '25',
               '--stand-start-transition-min-steps', '80', '--stand-start-transition-max-steps', '160',
               '--startup-action-ramp-steps', '20', '--fall-penalty', '10',
               '--min-forward-progress-speed', '.1', '--forward-progress-grace-steps', '400',
               '--forward-progress-window-steps', '100', '--forward-progress-termination-penalty', '2',
               '--reference-gait-style-reward-coef', '4', '--sensor-natural-gait-reward-coef', '2',
               '--sensor-hop-penalty-coef', '1', '--sensor-tiptoe-penalty-coef', '1',
               '--sensor-stance-slip-penalty-coef', '.5', '--sensor-sync-forward-penalty-coef', '1',
               '--sensor-knee-gait-reward-coef', '0',
               '--forward-velocity-reward-coef', '2', '--forward-velocity-deficit-penalty-coef', '2.5',
               '--symmetry-reward-coef', '1.2', '--symmetry-foot-pos-sigma', '.08',
               '--side-foot-pos-coef', '0', '--step-gap-coef', '0', '--swing-advance-coef', '0',
               '--amp-reward-coef', '1.2', '--task-reward-lerp', '.75',
               '--learning-rate', '1e-4', '--min-learning-rate', '1e-5', '--max-learning-rate', '1e-4',
               '--target-kl', '.010', '--target-kl-stop-multiplier', '1.5',
               '--min-action-std', '.03', '--max-action-std', '.25',
               '--gamma', '.995', '--gae-lambda', '.97', '--init-std', '.15', '--no-amp-startup-gate',
               '--privileged-context-loss-coef', '.20', '--no-use-moe',
               '--recurrent-hidden-dim', '128' if experiment in 'AB' else '256',
               '--compile-policy-chunk-steps', '8', '--evaluate-gait', '--save-interval', '500']
    if experiment != 'A':
        command += ['--include-tendon-velocity', '--include-tendon-tracking-error']
    if experiment in 'DEFG':
        command += ['--state-estimator-enabled']
    if experiment in 'EFG':
        command += ['--actor-use-estimated-state']
    if distill:
        command += ['--teacher-checkpoint', str(teacher),
                    '--teacher-action-loss-coef', '1.0', '--teacher-action-loss-final-coef', '.3',
                    '--teacher-action-loss-scale', '.1', '--teacher-action-loss-decay-iterations', '12000',
                    '--teacher-action-probability', '.8', '--teacher-action-probability-final', '0',
                    '--teacher-action-probability-decay-iterations', '4000']
    if experiment == 'G':
        command += ['--future-prediction-enabled', '--future-prediction-steps', '1', '3', '5',
                    '--future-prediction-coef', '.1']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=list('ABCDEFG'), default='F')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2026])
    parser.add_argument('--bptt', type=int, choices=[64, 96, 128], default=96)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--teacher-checkpoint')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error('iterations must be positive')
    for seed in args.seeds:
        try:
            command = training_command(args.experiment, seed, args.bptt, args.iterations, args.teacher_checkpoint)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps({'experiment': args.experiment, 'seed': seed,
                          'physics_steps_per_iteration': 393216, 'command': command}, indent=2), flush=True)
        if not args.dry_run:
            run_dir = Path(command[command.index('--log-dir') + 1])
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / 'params/config.json').exists():
                raise FileExistsError(f'{run_dir} already contains a run; choose a new seed/BPTT or resume explicitly.')
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
