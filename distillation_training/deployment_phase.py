"""Controller-side construction of the exact phase clock used in training."""
from __future__ import annotations

from foot_sensor_training.phase_clock import DeployablePhaseClock


def phase_clock_from_metadata(metadata: dict, *, num_envs: int = 1, torch_device=None):
    """Build a deployment clock from checkpoint/ONNX ``phase_clock`` metadata."""
    if not metadata.get("enabled") or not metadata.get("deployable"):
        raise ValueError("checkpoint does not contain an enabled deployable phase clock")
    schedule=metadata.get("transition_schedule",{})
    if schedule.get("type")!="delayed_smoothstep":
        raise ValueError(f"unsupported transition schedule: {schedule!r}")
    return DeployablePhaseClock(
        frame_count=float(metadata["frame_count"]),frequency=float(metadata["frequency"]),
        control_dt=float(metadata["control_dt"]),start_phase=float(metadata["start_phase"]),
        num_envs=num_envs,
        stand_start_phase_delay_steps=int(metadata["stand_start_phase_delay_steps"]),
        stand_start_transition_steps=int(schedule["transition_steps"]),
        torch_device=torch_device)


def write_phase_channels(observation,clock: DeployablePhaseClock):
    """Write ``[sin(theta), cos(theta)]`` to the fixed final two channels."""
    observation[..., -2:]=clock.observe()
    return observation
