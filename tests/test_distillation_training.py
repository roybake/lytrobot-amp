import torch
import argparse
import json
import onnx

from distillation_training.config import (SENSOR_DIM,PRIVILEGED_DIM,
  ESTIMATOR_INPUT_DIM,HIDDEN_DIM,RECURRENT_STATE_DIM,PRIVILEGED_FIELDS)
from distillation_training.networks import SharedActor,HistoryEstimator,StudentExport
from distillation_training.runtime import add_runtime_args, gated_reference_reward
from distillation_training.play import export_onnx
from distillation_training.amp import AMPConfig, AMPTrainingModule
from foot_sensor_training.phase_clock import DeployablePhaseClock


def test_teacher_student_and_privileged_schema():
    assert (SENSOR_DIM,PRIVILEGED_DIM,ESTIMATOR_INPUT_DIM,HIDDEN_DIM,RECURRENT_STATE_DIM)==(64,20,92,128,142)
    lowered=" ".join(PRIVILEGED_FIELDS).lower()
    for forbidden in ("phase","reference","future","ref_ctrl","terrain_id"):
        assert forbidden not in lowered
    actor=SharedActor(64);sensor=torch.zeros(3,64);truth=torch.zeros(3,20)
    assert actor.mean(sensor,truth).shape==(3,14)


def test_student_deployment_has_only_four_inputs_and_142d_state():
    export=StudentExport(SharedActor(64),HistoryEstimator(64),64).eval()
    action,state=export(torch.zeros(2,64),torch.zeros(2,14),torch.zeros(2,142),torch.ones(2))
    assert action.shape==(2,14) and state.shape==(2,142)


def test_oracle_mean_is_shared_actor_mean():
    teacher=SharedActor(64);oracle=SharedActor(64);oracle.load_state_dict(teacher.state_dict())
    sensor=torch.randn(4,64);truth=torch.randn(4,20)
    torch.testing.assert_close(teacher.mean(sensor,truth),oracle.mean(sensor,truth))


def test_teacher_exploration_and_reward_defaults_prioritize_walking():
    actor = SharedActor(58)
    actor_std = actor.distribution(torch.zeros(2, 58), torch.zeros(2, 20)).stddev
    torch.testing.assert_close(actor_std, torch.full_like(actor_std, 0.35))
    with torch.no_grad():
        actor.log_std.fill_(2.0)
    assert actor.distribution(torch.zeros(2, 58), torch.zeros(2, 20)).stddev.max() <= 0.60001
    actor.clamp_log_std_()
    assert actor.log_std.exp().max() <= 0.60001

    amp = AMPTrainingModule(45, 'cpu', AMPConfig())
    combined = amp.combine_reward(torch.tensor([-2.0]), torch.tensor([1.0]))
    torch.testing.assert_close(combined, torch.tensor([-1.8]))


def test_gated_reference_reward_rejects_tilted_backward_motion_and_falls():
    reward = torch.tensor([10.0, 10.0, 10.0])
    info = {
        'sensor_upright': torch.tensor([1.0, 0.55, 1.0]),
        'sensor_height': torch.tensor([1.0, 0.8, 1.0]),
        'forward_velocity': torch.tensor([0.8, -0.6, 0.8]),
        'termination': torch.tensor([0.0, 0.0, 1.0]),
    }
    shaped = gated_reference_reward(reward, info, fall_penalty=60.0)
    torch.testing.assert_close(shaped, torch.tensor([10.0, -3.8, -60.0]), atol=1e-6, rtol=0)


def test_student_bptt_default_is_96():
    parser=argparse.ArgumentParser();add_runtime_args(parser)
    assert parser.parse_args([]).num_steps==96


def test_student_onnx_schema_and_phase_metadata(tmp_path):
    path=tmp_path/"student.onnx"
    clock=DeployablePhaseClock(35200,40,.01)
    export_onnx(SharedActor(64),HistoryEstimator(64),path,clock.metadata)
    graph=onnx.load(path)
    shapes={item.name:[dim.dim_param or dim.dim_value for dim in item.type.tensor_type.shape.dim] for item in graph.graph.input}
    assert shapes=={"observation":["batch",64],"previous_action":["batch",14],
      "recurrent_state":["batch",142],"episode_start":["batch"]}
    metadata=json.loads({p.key:p.value for p in graph.metadata_props}["lytrobot_amp.metadata"])
    assert metadata["history_estimator_input_dim"]==92
    assert metadata["phase_fields"]==["gait_phase_sin","gait_phase_cos"]
    assert metadata["phase_clock"]["transition_schedule"]["type"]=="delayed_smoothstep"
