from argparse import Namespace
import json

import onnx
import pytest
import torch

from distillation_training.config import (
    ACTION_DIM,
    HIDDEN_DIM,
    PHASE_FORCE,
    PHASE_NO_FORCE,
    PRIVILEGED_DIM,
    RECURRENT_STATE_DIM,
)
from distillation_training.networks import HistoryEstimator, SharedActor, StudentExport
from distillation_training.play import export_onnx
from distillation_training.runtime import dimension_metadata, inherit_checkpoint_variant


@pytest.mark.parametrize('spec', (PHASE_NO_FORCE, PHASE_FORCE))
def test_variant_network_and_estimator_dimensions(spec):
    actor = SharedActor(spec.sensor_dim)
    estimator = HistoryEstimator(spec.sensor_dim)
    export = StudentExport(actor, estimator, spec.sensor_dim)
    sensor = torch.zeros(2, spec.sensor_dim)
    action, state = export(
        sensor, torch.zeros(2, ACTION_DIM), torch.zeros(2, RECURRENT_STATE_DIM), torch.ones(2)
    )
    assert estimator.estimator_input_dim == spec.estimator_input_dim
    assert action.shape == (2, ACTION_DIM)
    assert state.shape == (2, RECURRENT_STATE_DIM)
    assert (PRIVILEGED_DIM, HIDDEN_DIM, RECURRENT_STATE_DIM) == (20, 128, 142)


def _checkpoint(spec):
    return {
        **dimension_metadata(spec),
        'sensor_metadata': {'actor_frame_dim': spec.sensor_dim, 'phase_dim': 2,
          'sensor_variant': spec.name, 'estimator_input_dim': spec.estimator_input_dim},
    }


@pytest.mark.parametrize(
    ('teacher', 'student'),
    ((PHASE_FORCE, PHASE_NO_FORCE), (PHASE_NO_FORCE, PHASE_FORCE)),
)
def test_cross_variant_teacher_student_load_is_rejected(teacher, student):
    with pytest.raises(ValueError, match='cross-variant'):
        inherit_checkpoint_variant(
            Namespace(sensor_variant=None),
            ('teacher', _checkpoint(teacher)),
            ('student', _checkpoint(student)),
        )


@pytest.mark.parametrize('spec', (PHASE_NO_FORCE, PHASE_FORCE))
def test_onnx_observation_and_metadata_follow_checkpoint_variant(tmp_path, spec):
    path = tmp_path / f'{spec.name}.onnx'
    export_onnx(
        SharedActor(spec.sensor_dim), HistoryEstimator(spec.sensor_dim), path,
        phase_clock_metadata={'frame_count': 35200}, sensor_variant=spec,
    )
    graph = onnx.load(path)
    inputs = {
        item.name: [dim.dim_param or dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in graph.graph.input
    }
    assert inputs['observation'] == ['batch', spec.sensor_dim]
    metadata = json.loads(
        {item.key: item.value for item in graph.metadata_props}['lytrobot_amp.metadata']
    )
    assert metadata['sensor_variant'] == spec.name
    assert metadata['estimator_input_dim'] == spec.estimator_input_dim
    assert metadata['foot_force_dim'] == (6 if spec.use_foot_force else 0)
