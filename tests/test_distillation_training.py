from distillation_training.config import PRIVILEGED_FIELDS,PRIVILEGED_DIM,ESTIMATOR_INPUT_DIM

def test_privileged_schema_forbids_reference_leakage():
    text=' '.join(PRIVILEGED_FIELDS).lower()
    assert PRIVILEGED_DIM==20 and ESTIMATOR_INPUT_DIM==90
    for forbidden in ('phase','reference','qpos','qvel','foot_pos','contact'):
        assert forbidden not in text

def test_control_semantics_are_declared_current_tendon():
    from distillation_training.config import DistillationConfig
    assert DistillationConfig().reference_ctrl_blend==0.0
