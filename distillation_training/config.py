from dataclasses import dataclass
from lytrobot_amp.foot_force import FootForceSensorConfig

SENSOR_DIM=62; ACTION_DIM=14; PRIVILEGED_FIELDS=("root_linear_velocity_body_x","root_linear_velocity_body_y","root_linear_velocity_body_z","projected_gravity_x","projected_gravity_y","projected_gravity_z",*(f"tendon_velocity_{i}" for i in range(14)))
PRIVILEGED_DIM=20; ESTIMATOR_INPUT_DIM=90; HIDDEN_DIM=128
@dataclass(frozen=True)
class DistillationConfig:
    reference_ctrl_blend: float=0.0
    foot_force: FootForceSensorConfig=FootForceSensorConfig()
