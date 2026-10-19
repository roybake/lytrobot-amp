"""An unbounded update schedule exits successfully only after gait acceptance."""

import csv
import sys

import pytest
import torch

from scripts import train
from scripts.play import load_onnx_metadata


def test_unbounded_training_requires_an_acceptance_stop(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train.py', '--max-iterations', '0'])
    with pytest.raises(ValueError, match='requires --evaluate-gait'):
        train.main()


def test_unbounded_training_continues_after_failure_and_stops_when_accepted(
    monkeypatch, tmp_path,
):
    from sensor_training import gait_evaluation

    torch.set_num_threads(1)
    monkeypatch.setattr(sys, 'argv', [
        'train.py', '--device', 'cpu', '--sim-backend', 'cpu',
        '--num-envs', '1', '--num-steps', '4', '--max-iterations', '0',
        '--termination-mode', 'fall', '--no-domain-randomization',
        '--no-episode-random-seed', '--reset-noise-scale', '0',
        '--no-wandb', '--no-amp', '--save-interval', '1',
        '--evaluate-gait', '--gait-fidelity-selection',
        '--gait-evaluation-seconds', '0.1', '--stop-on-gait-acceptance',
        '--log-dir', str(tmp_path),
    ])
    calls = []

    def measured_quality(path, **kwargs):
        calls.append(path)
        assert kwargs['seconds'] == .1
        kwargs['trace_path'].write_text('time_s,forward_velocity\n0.01,1.28\n')
        accepted = float(len(calls) == 2)
        return {
            'eval/gait_accepted': accepted,
            'eval/gait_survived': accepted,
            'eval/gait_duration_s': .1 if accepted else .05,
            'eval/gait_score': accepted,
            'eval/reference_fidelity_accepted': accepted,
            'eval/reference_fidelity_score': accepted,
            'eval/gait_seed_count': 3.,
        }

    monkeypatch.setattr(gait_evaluation, 'evaluate_gait_suite', measured_quality)
    train.main()
    assert len(calls) == 2
    with (tmp_path / 'metrics.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert [int(float(row['iteration'])) for row in rows] == [0, 1]
    final = load_onnx_metadata(tmp_path / 'model_final.onnx')
    best = load_onnx_metadata(tmp_path / 'model_best.onnx')
    assert final['iteration'] == best['iteration'] == 1
    assert '[GAIT_ACCEPTED]' in (tmp_path / 'train.log').read_text()
