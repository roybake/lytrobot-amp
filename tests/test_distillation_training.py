import torch
import argparse
import json
import onnx

from distillation_training.config import (SENSOR_DIM,PRIVILEGED_DIM,
  ESTIMATOR_INPUT_DIM,HIDDEN_DIM,RECURRENT_STATE_DIM,PRIVILEGED_FIELDS)
from distillation_training.networks import SharedActor,HistoryEstimator,StudentExport
from distillation_training.runtime import add_runtime_args
from distillation_training.play import export_onnx
from foot_sensor_training.phase_clock import DeployablePhaseClock


def test_teacher_student_and_privileged_schema():
    assert (SENSOR_DIM,PRIVILEGED_DIM,ESTIMATOR_INPUT_DIM,HIDDEN_DIM,RECURRENT_STATE_DIM)==(64,20,92,128,142)
    lowered=" ".join(PRIVILEGED_FIELDS).lower()
    for forbidden in ("phase","reference","future","ref_ctrl","terrain_id"):
        assert forbidden not in lowered
    actor=SharedActor();sensor=torch.zeros(3,64);truth=torch.zeros(3,20)
    assert actor.mean(sensor,truth).shape==(3,14)


def test_student_deployment_has_only_four_inputs_and_142d_state():
    export=StudentExport(SharedActor(),HistoryEstimator()).eval()
    action,state=export(torch.zeros(2,64),torch.zeros(2,14),torch.zeros(2,142),torch.ones(2))
    assert action.shape==(2,14) and state.shape==(2,142)


def test_oracle_mean_is_shared_actor_mean():
    teacher=SharedActor();oracle=SharedActor();oracle.load_state_dict(teacher.state_dict())
    sensor=torch.randn(4,64);truth=torch.randn(4,20)
    torch.testing.assert_close(teacher.mean(sensor,truth),oracle.mean(sensor,truth))


def test_student_bptt_default_is_96():
    parser=argparse.ArgumentParser();add_runtime_args(parser)
    assert parser.parse_args([]).num_steps==96


def test_student_onnx_schema_and_phase_metadata(tmp_path):
    path=tmp_path/"student.onnx"
    clock=DeployablePhaseClock(35200,40,.01)
    export_onnx(SharedActor(),HistoryEstimator(),path,clock.metadata)
    graph=onnx.load(path)
    shapes={item.name:[dim.dim_param or dim.dim_value for dim in item.type.tensor_type.shape.dim] for item in graph.graph.input}
    assert shapes=={"observation":["batch",64],"previous_action":["batch",14],
      "recurrent_state":["batch",142],"episode_start":["batch"]}
    metadata=json.loads({p.key:p.value for p in graph.metadata_props}["lytrobot_amp.metadata"])
    assert metadata["history_estimator_input_dim"]==92
    assert metadata["phase_fields"]==["gait_phase_sin","gait_phase_cos"]
    assert metadata["phase_clock"]["transition_schedule"]["type"]=="delayed_smoothstep"
