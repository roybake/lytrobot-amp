"""Shared terrain and motion defaults used by training entrypoints."""

from __future__ import annotations


# Deployment control contract. MuJoCo integrates at 200 Hz, while newly trained
# policies are evaluated every two physics steps at 100 Hz. Playback may still
# load old checkpoints at their saved control frequency for visual inspection.
SIMULATION_TIMESTEP_S = 0.005
CONTROL_FREQUENCY_HZ = 100.0
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQUENCY_HZ
CONTROL_DECIMATION = 2


def control_frequency_hz(timestep: float, decimation: int) -> float:
    """Return policy frequency for a MuJoCo timestep/decimation pair."""

    period = float(timestep) * int(decimation)
    if period <= 0.0:
        raise ValueError("timestep and decimation must define a positive control period.")
    return 1.0 / period


def require_100hz_control(timestep: float, decimation: int, *, source: str) -> None:
    """Fail closed when a training config is not the deployment clock."""

    frequency = control_frequency_hz(timestep, decimation)
    if abs(frequency - CONTROL_FREQUENCY_HZ) > 1.0e-6:
        raise ValueError(
            f"{source} control frequency is {frequency:g} Hz; this project requires "
            f"exactly {CONTROL_FREQUENCY_HZ:g} Hz "
            f"(timestep={SIMULATION_TIMESTEP_S:g}, decimation={CONTROL_DECIMATION})."
        )


# Preserve the native timing of CMU subject 69 trial 01 after scaling and
# retargeting it to the LYT model.
ORIGINAL_RETARGETED_WALK_SPEED_MPS = 0.8622949262039437
TRAINING_TARGET_SPEED_MPS = ORIGINAL_RETARGETED_WALK_SPEED_MPS
# This profile deliberately has its own reference directory and must not leak
# into the ordinary multi-motion or stair tasks.
NO_STAIRS_FAST_TARGET_SPEED_MPS = 5.0 / 3.6

ROUGH_TERRAIN_HEIGHT_M = 0.05
SLOPE_ANGLE_DEG = 6.0
MULTIMOTION_STAIR_STEP_HEIGHT_M = 0.13
MULTIMOTION_STAIR_STEP_DEPTH_M = 0.22

# The dedicated stair curriculum starts with the easier geometry and switches
# to a harder geometry when domain randomization is enabled in Stage 2.
STAIR_STAGE1_STEP_HEIGHT_M = 0.13
STAIR_STAGE1_STEP_DEPTH_M = 0.32
STAIR_STAGE2_STEP_HEIGHT_M = 0.20
STAIR_STAGE2_STEP_DEPTH_M = 0.22
