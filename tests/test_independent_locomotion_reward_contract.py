from __future__ import annotations

import math

from scripts.independent_terrain_task import (
    INDEPENDENT_TERRAIN_SPECS,
    default_train_args,
)


def _value(args: list[str], option: str) -> float:
    return float(args[args.index(option) + 1])


def test_stationary_supported_pose_is_not_a_reward_local_optimum() -> None:
    """A clean two-foot stand must lose even before the discriminator learns."""

    for key in INDEPENDENT_TERRAIN_SPECS:
        args = default_train_args(key)
        target_velocity = _value(args, "--target-velocity")
        sigma = _value(args, "--forward-velocity-tracking-sigma")
        velocity_coef = _value(args, "--forward-velocity-reward-coef")
        deficit_coef = _value(args, "--forward-velocity-deficit-penalty-coef")
        contact_coef = _value(args, "--contact-reward-coef")
        task_lerp = _value(args, "--task-reward-lerp")
        amp_coef = _value(args, "--amp-reward-coef")

        stationary_velocity_track = math.exp(
            -(target_velocity**2) / (sigma**2)
        )
        stationary_task_reward = (
            0.05
            + 1.20  # upright
            + 0.80  # physical height
            + velocity_coef * stationary_velocity_track
            - deficit_coef
            + contact_coef  # supported by at least one foot
        )
        # At discriminator logit zero, AMP reward is -log(sigmoid(0)) = log(2).
        uncertain_amp_total_reward = (
            task_lerp * stationary_task_reward
            + (1.0 - task_lerp) * amp_coef * math.log(2.0)
        )

        assert stationary_task_reward < 0.0
        assert uncertain_amp_total_reward < 0.0


def test_target_speed_has_a_large_positive_margin_over_standing() -> None:
    for key in INDEPENDENT_TERRAIN_SPECS:
        args = default_train_args(key)
        velocity_coef = _value(args, "--forward-velocity-reward-coef")
        progress_coef = _value(args, "--forward-progress-reward-coef")
        contact_coef = _value(args, "--contact-reward-coef")
        deficit_coef = _value(args, "--forward-velocity-deficit-penalty-coef")

        target_speed_task_reward = (
            0.05
            + 1.20
            + 0.80
            + velocity_coef
            + progress_coef
            + contact_coef
        )
        stationary_upper_bound_without_velocity_track = (
            0.05 + 1.20 + 0.80 + contact_coef - deficit_coef
        )
        assert target_speed_task_reward - stationary_upper_bound_without_velocity_track > 8.0


def test_training_exposes_deployment_rest_state_without_overpenalizing_steps() -> None:
    for key in INDEPENDENT_TERRAIN_SPECS:
        args = default_train_args(key)
        assert _value(args, "--motion-reset-prob") == 0.50
        assert _value(args, "--stand-start-prob") == 0.0
        assert _value(args, "--foot-slip-velocity-threshold") >= 0.08
        assert _value(args, "--foot-slip-penalty-coef") <= 0.50
