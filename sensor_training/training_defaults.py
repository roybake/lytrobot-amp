"""Time-consistent recurrent rollout defaults for sensor policies."""

from __future__ import annotations

from collections.abc import Sequence


PHYSICS_TIMESTEP_SECONDS = 0.005
# A 640 ms truncated-gradient window is long enough to cover the stance or
# swing half of the reference gait.  The GRU state itself still persists
# across rollouts, while halving the sequential rollout depth lets the GPU run
# twice as many environments per launch.  This is materially faster than the
# old 1.28 s / 768-world shape on launch-bound consumer GPUs.
TARGET_BPTT_SECONDS = 0.64
# Distillation receives a dense privileged action target at every policy step,
# so it needs less truncated-gradient depth than pure on-policy learning.  The
# recurrent state still persists across rollout boundaries.
DISTILLATION_BPTT_SECONDS = 0.48
TARGET_TRANSITIONS_PER_ROLLOUT = 98_304
# The privileged Stage1 recipe that produced the accepted gait simulated
# ``4096 worlds x 24 steps x decimation 4 = 393,216`` physics steps per PPO
# update.  Matching that budget at the 100 Hz sensor control rate needs
# ``3072 worlds x 64 steps x decimation 2``: the same number of gradient
# updates per simulated second, but twice the transitions per update because
# the sensor policy acts twice as often.  Measured on the RTX 3060 the larger
# shape sustains ~39k env-steps/s instead of ~21k, so the same learning curve
# costs roughly 1.8x less wall-clock time.
STAGE_POLICY_PHYSICS_STEPS_PER_ROLLOUT = 393_216
DEFAULT_SENSOR_DECIMATION = 2
REQUIRED_SENSOR_CONTROL_HZ = 100


def stage_policy_rollout_envs(num_steps: int, decimation: int) -> int:
    """Return the world count that matches the privileged physics budget."""

    per_env_physics = max(1, int(num_steps) * int(decimation))
    return max(1, round(STAGE_POLICY_PHYSICS_STEPS_PER_ROLLOUT / per_env_physics))


def validate_sensor_decimation(decimation: int) -> int:
    """Require the deployment-aligned 100 Hz sensor control rate."""

    decimation = int(decimation)
    if decimation != DEFAULT_SENSOR_DECIMATION:
        raise ValueError(
            "Sensor-policy training requires 100 Hz control "
            f"(--decimation {DEFAULT_SENSOR_DECIMATION}); got --decimation {decimation}."
        )
    return decimation


def recurrent_rollout_shape(decimation: int) -> tuple[int, int]:
    """Return ``(num_steps, num_envs)`` with a frequency-invariant horizon."""

    decimation = int(decimation)
    if decimation <= 0:
        raise ValueError("decimation must be positive.")
    num_steps = max(
        1,
        round(TARGET_BPTT_SECONDS / (PHYSICS_TIMESTEP_SECONDS * decimation)),
    )
    num_envs = max(1, round(TARGET_TRANSITIONS_PER_ROLLOUT / num_steps))
    return num_steps, num_envs


def distillation_rollout_shape(decimation: int) -> tuple[int, int]:
    """Return the higher-parallelism rollout shape for Teacher distillation."""

    decimation = validate_sensor_decimation(decimation)
    num_steps = max(
        1,
        round(DISTILLATION_BPTT_SECONDS / (PHYSICS_TIMESTEP_SECONDS * decimation)),
    )
    num_envs = max(1, round(TARGET_TRANSITIONS_PER_ROLLOUT / num_steps))
    return num_steps, num_envs


def policy_steps_for_seconds(seconds: float, decimation: int) -> int:
    """Convert a wall-clock duration to policy steps at a control rate."""

    if seconds <= 0.0:
        raise ValueError("seconds must be positive.")
    if decimation <= 0:
        raise ValueError("decimation must be positive.")
    return max(1, round(seconds / (PHYSICS_TIMESTEP_SECONDS * decimation)))


def cli_int(args: Sequence[str], name: str, default: int) -> int:
    """Read an integer from ``--option value`` or ``--option=value``."""

    for index, arg in enumerate(args):
        if arg == name:
            if index + 1 >= len(args):
                raise ValueError(f"{name} requires an integer value.")
            return int(args[index + 1])
        prefix = f"{name}="
        if arg.startswith(prefix):
            return int(arg[len(prefix) :])
    return int(default)


def recurrent_cli_defaults(args: Sequence[str]) -> tuple[int, int, int]:
    """Resolve decimation, BPTT steps and env count from partial CLI args."""

    decimation = validate_sensor_decimation(
        cli_int(args, "--decimation", DEFAULT_SENSOR_DECIMATION)
    )
    default_steps, _default_envs = recurrent_rollout_shape(decimation)
    num_steps = cli_int(args, "--num-steps", default_steps)
    if num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    num_envs = max(1, round(TARGET_TRANSITIONS_PER_ROLLOUT / num_steps))
    return decimation, num_steps, num_envs


def distillation_cli_defaults(args: Sequence[str]) -> tuple[int, int, int]:
    """Resolve required-rate distillation defaults from partial CLI args."""

    decimation = validate_sensor_decimation(
        cli_int(args, "--decimation", DEFAULT_SENSOR_DECIMATION)
    )
    default_steps, _default_envs = distillation_rollout_shape(decimation)
    num_steps = cli_int(args, "--num-steps", default_steps)
    if num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    num_envs = max(1, round(TARGET_TRANSITIONS_PER_ROLLOUT / num_steps))
    return decimation, num_steps, num_envs
