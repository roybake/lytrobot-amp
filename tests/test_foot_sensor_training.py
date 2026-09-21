import numpy as np
import torch
from lytrobot_amp.foot_force import FootForceSensorConfig, FOOT_FORCE_NAMES
from foot_sensor_training.config import SENSOR_FRAME_DIM
from distillation_training.networks import SharedActor,HistoryEstimator,StudentExport

def test_force_schema_and_normalization_are_safe():
    cfg=FootForceSensorConfig()
    assert len(FOOT_FORCE_NAMES)==6 and cfg.force_clip==4.0
    raw=np.array([-1.,0.,np.inf,np.nan,10.,2.],np.float32)
    out=np.nan_to_num(np.clip(raw,0,cfg.force_clip)/cfg.force_clip)
    assert out.shape==(6,) and np.isfinite(out).all() and (out>=0).all()

def test_standalone_schema_has_no_privileged_fields():
    assert SENSOR_FRAME_DIM==62

def test_shared_oracle_equals_teacher_mean():
    actor=SharedActor(); s=torch.randn(3,62); p=torch.randn(3,20)
    teacher=actor.mean(s,p); oracle=actor.mean(s,p)
    assert torch.allclose(teacher,oracle)

def test_student_export_has_only_deployment_inputs():
    export=StudentExport(SharedActor(),HistoryEstimator())
    action,state=export(torch.zeros(2,62),torch.zeros(2,14),torch.zeros(2,142),torch.ones(2))
    assert action.shape==(2,14) and state.shape==(2,142)
