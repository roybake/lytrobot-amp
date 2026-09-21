from __future__ import annotations

from dataclasses import dataclass

from lytrobot_amp.foot_force import FootForceSensorConfig


ACTION_DIM = 14
PRIVILEGED_DIM = 20
HIDDEN_DIM = 128
RECURRENT_STATE_DIM = HIDDEN_DIM + ACTION_DIM


@dataclass(frozen=True)
class SensorVariantSpec:
    name: str
    sensor_dim: int
    estimator_input_dim: int
    use_foot_force: bool


PHASE_NO_FORCE = SensorVariantSpec("phase_no_force", 58, 86, False)
PHASE_FORCE = SensorVariantSpec("phase_force", 64, 92, True)
SENSOR_VARIANTS = {
    PHASE_NO_FORCE.name: PHASE_NO_FORCE,
    PHASE_FORCE.name: PHASE_FORCE,
}


def sensor_variant_spec(value: str | SensorVariantSpec) -> SensorVariantSpec:
    if isinstance(value, SensorVariantSpec):
        return value
    try:
        return SENSOR_VARIANTS[value]
    except KeyError as exc:
        raise ValueError(
            f"unknown sensor variant {value!r}; expected one of {tuple(SENSOR_VARIANTS)}"
        ) from exc


# Compatibility aliases for callers that still mean the default Route B.
# Variant-aware runtime/network code uses SensorVariantSpec instead.
SENSOR_DIM = PHASE_FORCE.sensor_dim
ESTIMATOR_INPUT_DIM = PHASE_FORCE.estimator_input_dim

PRIVILEGED_FIELDS = (
    "root_linear_velocity_body_x",
    "root_linear_velocity_body_y",
    "root_linear_velocity_body_z",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    *(f"tendon_velocity_{i}" for i in range(ACTION_DIM)),
)


@dataclass(frozen=True)
class DistillationConfig:
    reference_ctrl_blend: float = 0.0
    foot_force: FootForceSensorConfig = FootForceSensorConfig()
