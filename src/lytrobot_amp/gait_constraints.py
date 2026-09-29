"""Causal, phase-free gait scores for the sensor-policy reward."""

from __future__ import annotations

import torch
import numpy as np


# The walking reference has a much longer swing than the 0.17 m shuffle in
# the failed sensor run.  This conservative floor rewards real steps without
# requiring the actor to observe a reference phase or rewarding overshoot.
MIN_NATURAL_STRIDE_METERS = 0.45
STRIDE_SHORTFALL_SIGMA_METERS = 0.20
NATURAL_STEP_INTERVAL_SECONDS = 0.75
STEP_INTERVAL_SIGMA_SECONDS = 0.25
NATURAL_SWING_CLEARANCE_METERS = 0.03
SWING_CLEARANCE_SIGMA_METERS = 0.03
MIN_SWING_DURATION_SECONDS = 0.05


def natural_gait_bonus(
    scores, gait_progress, speed_track, stability, knee_flexion_track, cfg
):
    """Add measured walking quality to a reference-tracking task reward.

    Every term vanishes during the standing command. The speed and stability
    gates keep a stationary shuffle or a falling stride from earning style
    reward. This is shared by CPU and Warp full-state training paths.
    """
    clip = torch.clamp if torch.is_tensor(gait_progress) else np.clip
    quality_names = (
        "sensor_stride_length", "sensor_step_cadence", "sensor_swing_clearance",
        "sensor_foot_roll", "sensor_alternation", "sensor_mirror",
    )
    quality = len(quality_names) / sum(
        1.0 / clip(scores[name], 0.05, 1.0) for name in quality_names
    )
    progress = clip(gait_progress, 0.0, 1.0)
    speed = clip(speed_track, 0.0, 1.0)
    stable = clip(stability, 0.0, 1.0)
    return stable * progress * (
        float(cfg.sensor_natural_gait_reward_coef) * speed * quality
        + float(cfg.sensor_knee_gait_reward_coef) * (clip(knee_flexion_track, 0.0, 1.0) - 1.0)
        - float(cfg.sensor_hop_penalty_coef) * scores["sensor_hop"]
        - float(cfg.sensor_tiptoe_penalty_coef)
        * (scores["sensor_tiptoe"] + scores["sensor_roll_violation"])
        - float(cfg.sensor_stance_slip_penalty_coef) * scores["sensor_stance_slip"]
        - float(cfg.sensor_sync_forward_penalty_coef) * scores["sensor_synchronous_forward"]
    )


def sensor_gait_scores(
    env, zones: torch.Tensor, foot_x: torch.Tensor,
    lateral_sum: torch.Tensor | None = None,
    stance_speed: torch.Tensor | None = None,
    foot_forward_speed: torch.Tensor | None = None,
    foot_height: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Score mirrored steps and heel/sole/toe roll from actual ground contacts.

    ``zones`` is [world, side, heel/sole/toe].  The state belongs to the
    environment so it survives PPO rollout boundaries but is cleared on reset.
    No reference phase or privileged reference pose is consulted.
    """

    heel, sole, toe = zones.unbind(dim=2)
    contact = zones.any(dim=2)
    previous = env.sensor_last_foot_contact
    raw_touchdown = contact & ~previous
    liftoff = ~contact & previous
    min_swing_steps = max(2, round(MIN_SWING_DURATION_SECONDS / max(float(env.dt), 1.0e-6)))
    touchdown = raw_touchdown & (env.sensor_swing_steps >= min_swing_steps)
    env.sensor_swing_steps.copy_(torch.where(
        contact,
        torch.zeros_like(env.sensor_swing_steps),
        env.sensor_swing_steps + 1,
    ))
    stage = torch.where(contact, env.sensor_foot_roll_stage, 0)
    bad_touchdown = touchdown & ~heel
    premature_toe = toe & ~heel & ~sole & (stage < 2)

    # Each foot must first strike with its heel, then load the sole, before a
    # toe-only push-off is considered valid.  A toe-first contact cannot jump
    # directly to a completed roll stage.
    stage = torch.where(heel & (stage == 0), 1, stage)
    stage = torch.where(sole & (stage >= 1), 2, stage)
    stage = torch.where(toe & (stage >= 2), 3, stage)
    env.sensor_foot_roll_stage.copy_(stage)
    env.sensor_last_foot_contact.copy_(contact)
    toe_only = toe & ~heel & ~sole
    env.sensor_toe_only_steps.copy_(
        torch.where(toe_only, env.sensor_toe_only_steps + 1, 0)
    )

    roll_track = torch.where(
        contact,
        0.35 * heel.float()
        + 0.35 * sole.float()
        + 0.15 * toe.float()
        + 0.15 * (stage >= 2).float()
        + 0.50 * (toe & (stage >= 3)).float(),
        torch.full_like(heel, 0.5, dtype=torch.float32),
    ).clamp(0.0, 1.0).mean(dim=1)
    roll_violation = (bad_touchdown | premature_toe).float().mean(dim=1)
    tiptoe = (toe_only & ((stage < 3) | (env.sensor_toe_only_steps > 15))).float().mean(dim=1)
    hop = (~contact.any(dim=1)).float()

    # Single support is the usual walking state; double support is permitted
    # at transfer.  Synchronous touchdown or repeated same-side touchdowns
    # are penalized even if average left/right contact time is equal.
    single_support = contact[:, 0] ^ contact[:, 1]
    double_support = contact[:, 0] & contact[:, 1]
    both_touchdown = touchdown[:, 0] & touchdown[:, 1]
    one_touchdown = touchdown[:, 0] ^ touchdown[:, 1]
    touchdown_side = torch.where(touchdown[:, 0], 0, 1)
    repeated_side = one_touchdown & (env.sensor_last_touchdown_side == touchdown_side)
    env.sensor_time_since_touchdown.add_(float(env.dt))
    valid_interval = one_touchdown & env.sensor_touchdown_seen & ~repeated_side
    measured_interval = env.sensor_time_since_touchdown
    first_interval = valid_interval & ~env.sensor_cadence_seen
    env.sensor_step_interval_ema.copy_(torch.where(
        first_interval,
        measured_interval,
        torch.where(
            valid_interval,
            0.8 * env.sensor_step_interval_ema + 0.2 * measured_interval,
            env.sensor_step_interval_ema,
        ),
    ))
    env.sensor_cadence_seen.copy_(env.sensor_cadence_seen | valid_interval)
    env.sensor_touchdown_seen.copy_(env.sensor_touchdown_seen | one_touchdown)
    env.sensor_time_since_touchdown.copy_(torch.where(
        one_touchdown,
        torch.zeros_like(env.sensor_time_since_touchdown),
        env.sensor_time_since_touchdown,
    ))
    cadence = torch.where(
        env.sensor_cadence_seen,
        torch.exp(-torch.abs(env.sensor_step_interval_ema - NATURAL_STEP_INTERVAL_SECONDS)
                  / STEP_INTERVAL_SIGMA_SECONDS),
        torch.full_like(env.sensor_step_interval_ema, 0.5),
    )
    alternation = (
        single_support.float()
        + 0.5 * double_support.float()
        - 0.3 * both_touchdown.float()
        - 0.3 * repeated_side.float()
    ).clamp(0.0, 1.0)
    env.sensor_single_support_ema.copy_(
        0.99 * env.sensor_single_support_ema + 0.01 * single_support.float()
    )
    single_support_coverage = (env.sensor_single_support_ema / 0.15).clamp(0.0, 1.0)
    alternation = 0.5 * alternation + 0.5 * single_support_coverage
    stance_slip = (
        (contact.float() * ((stance_speed - 0.15) / 0.5).clamp(0.0, 2.0)).mean(dim=1)
        if stance_speed is not None else torch.zeros_like(alternation)
    )
    if foot_height is not None:
        swing_height = torch.where(
            contact[:, 0] & ~contact[:, 1],
            foot_height[:, 1] - foot_height[:, 0],
            foot_height[:, 0] - foot_height[:, 1],
        ).clamp(min=0.0)
        clearance_track = torch.exp(
            -((swing_height - NATURAL_SWING_CLEARANCE_METERS)
              / SWING_CLEARANCE_SIGMA_METERS).square()
        )
        swing_clearance = torch.where(
            single_support,
            clearance_track,
            torch.full_like(clearance_track, 0.5),
        )
    else:
        swing_clearance = torch.full_like(alternation, 0.5)
    if foot_forward_speed is not None:
        vx_right, vx_left = foot_forward_speed.unbind(dim=1)
        moving = (vx_right.abs() + vx_left.abs()) > 0.25
        motion_mirror = torch.where(
            moving,
            ((vx_right - vx_left).abs() / (vx_right.abs() + vx_left.abs() + 0.10)).clamp(0.0, 1.0),
            torch.full_like(vx_right, 0.5),
        )
        synchronous_forward = (
            double_support.float()
            * ((torch.minimum(vx_right, vx_left) - 0.20) / 0.5).clamp(0.0, 2.0)
        )
    else:
        motion_mirror = torch.full_like(alternation, 0.5)
        synchronous_forward = torch.zeros_like(alternation)
    env.sensor_last_touchdown_side.copy_(
        torch.where(one_touchdown, touchdown_side, env.sensor_last_touchdown_side)
    )

    env.sensor_liftoff_x.copy_(
        torch.where(liftoff, foot_x, env.sensor_liftoff_x)
    )
    env.sensor_had_liftoff.copy_(env.sensor_had_liftoff | liftoff)
    measured_step = touchdown & env.sensor_had_liftoff
    stride = (foot_x - env.sensor_liftoff_x).clamp(0.0, 0.6)
    env.sensor_stride_ema.copy_(
        torch.where(
            measured_step,
            0.7 * env.sensor_stride_ema + 0.3 * stride,
            env.sensor_stride_ema,
        )
    )
    env.sensor_stride_seen.copy_(env.sensor_stride_seen | measured_step)
    env.sensor_contact_duty_ema.copy_(
        0.98 * env.sensor_contact_duty_ema + 0.02 * contact.float()
    )
    both_strides_seen = env.sensor_stride_seen.all(dim=1)
    mean_stride = env.sensor_stride_ema.mean(dim=1)
    stride_shortfall = (MIN_NATURAL_STRIDE_METERS - mean_stride).clamp(min=0.0)
    stride_length = torch.where(
        both_strides_seen,
        torch.exp(-(stride_shortfall / STRIDE_SHORTFALL_SIGMA_METERS).square()),
        torch.full_like(mean_stride, 0.5),
    )
    stride_match = torch.exp(
        -((env.sensor_stride_ema[:, 0] - env.sensor_stride_ema[:, 1]) / 0.15).square()
    )
    duty_match = torch.exp(
        -((env.sensor_contact_duty_ema[:, 0] - env.sensor_contact_duty_ema[:, 1]) / 0.25).square()
    )
    lateral_match = (
        torch.exp(-(lateral_sum / 0.08).square())
        if lateral_sum is not None else torch.ones_like(stride_match)
    )
    mirror = torch.where(
        both_strides_seen,
        0.40 * stride_match + 0.10 * duty_match + 0.20 * lateral_match + 0.30 * motion_mirror,
        0.15 + 0.20 * lateral_match + 0.30 * motion_mirror,
    )
    return {
        "sensor_mirror": mirror,
        "sensor_stride_length": stride_length,
        "sensor_step_cadence": cadence,
        "sensor_step_interval": env.sensor_step_interval_ema.clone(),
        "sensor_swing_clearance": swing_clearance,
        "sensor_lateral_mirror": lateral_match,
        "sensor_motion_mirror": motion_mirror,
        "sensor_foot_roll": roll_track,
        "sensor_roll_violation": roll_violation,
        "sensor_alternation": alternation,
        "sensor_hop": hop,
        "sensor_tiptoe": tiptoe,
        "sensor_stance_slip": stance_slip,
        "sensor_synchronous_forward": synchronous_forward,
        "sensor_single_support_coverage": single_support_coverage,
        "sensor_left_stride": env.sensor_stride_ema[:, 0].clone(),
        "sensor_right_stride": env.sensor_stride_ema[:, 1].clone(),
    }


def sensor_gait_scores_numpy(
    env, env_id: int, zones: np.ndarray, foot_x: np.ndarray,
    lateral_sum: float = 0.0,
    stance_speed: np.ndarray | None = None,
    foot_forward_speed: np.ndarray | None = None,
    foot_height: np.ndarray | None = None,
) -> dict[str, float]:
    """CPU equivalent of the batched gait scores for validation and fallback."""

    heel, sole, toe = (zones[:, i].astype(bool) for i in range(3))
    contact = np.any(zones, axis=1)
    previous = env.sensor_last_foot_contact[env_id].copy()
    raw_touchdown = contact & ~previous
    liftoff = ~contact & previous
    min_swing_steps = max(2, round(MIN_SWING_DURATION_SECONDS / max(float(env.dt), 1.0e-6)))
    touchdown = raw_touchdown & (env.sensor_swing_steps[env_id] >= min_swing_steps)
    env.sensor_swing_steps[env_id] = np.where(
        contact, 0, env.sensor_swing_steps[env_id] + 1
    )
    stage = np.where(contact, env.sensor_foot_roll_stage[env_id], 0)
    bad_touchdown = touchdown & ~heel
    premature_toe = toe & ~heel & ~sole & (stage < 2)
    stage = np.where(heel & (stage == 0), 1, stage)
    stage = np.where(sole & (stage >= 1), 2, stage)
    stage = np.where(toe & (stage >= 2), 3, stage)
    env.sensor_foot_roll_stage[env_id] = stage
    env.sensor_last_foot_contact[env_id] = contact
    toe_only = toe & ~heel & ~sole
    env.sensor_toe_only_steps[env_id] = np.where(
        toe_only, env.sensor_toe_only_steps[env_id] + 1, 0
    )

    roll = float(np.mean(np.where(
        contact,
        0.35 * heel + 0.35 * sole + 0.15 * toe + 0.15 * (stage >= 2)
        + 0.50 * (toe & (stage >= 3)),
        0.5,
    ).clip(0.0, 1.0)))
    violation = float(np.mean(bad_touchdown | premature_toe))
    tiptoe = float(np.mean(toe_only & ((stage < 3) | (env.sensor_toe_only_steps[env_id] > 15))))
    hop = float(not np.any(contact))

    single_support = bool(contact[0] ^ contact[1])
    double_support = bool(contact[0] & contact[1])
    both_touchdown = bool(touchdown[0] & touchdown[1])
    one_touchdown = bool(touchdown[0] ^ touchdown[1])
    touchdown_side = 0 if touchdown[0] else 1
    repeated_side = one_touchdown and env.sensor_last_touchdown_side[env_id] == touchdown_side
    env.sensor_time_since_touchdown[env_id] += float(env.dt)
    valid_interval = one_touchdown and env.sensor_touchdown_seen[env_id] and not repeated_side
    if valid_interval:
        measured_interval = float(env.sensor_time_since_touchdown[env_id])
        if env.sensor_cadence_seen[env_id]:
            env.sensor_step_interval_ema[env_id] = (
                0.8 * env.sensor_step_interval_ema[env_id] + 0.2 * measured_interval
            )
        else:
            env.sensor_step_interval_ema[env_id] = measured_interval
        env.sensor_cadence_seen[env_id] = True
    if one_touchdown:
        env.sensor_touchdown_seen[env_id] = True
        env.sensor_time_since_touchdown[env_id] = 0.0
    cadence = (
        float(np.exp(-abs(env.sensor_step_interval_ema[env_id] - NATURAL_STEP_INTERVAL_SECONDS)
                     / STEP_INTERVAL_SIGMA_SECONDS))
        if env.sensor_cadence_seen[env_id] else 0.5
    )
    alternation = float(np.clip(
        float(single_support) + 0.5 * float(double_support)
        - 0.3 * float(both_touchdown) - 0.3 * float(repeated_side),
        0.0, 1.0,
    ))
    env.sensor_single_support_ema[env_id] = (
        0.99 * env.sensor_single_support_ema[env_id] + 0.01 * float(single_support)
    )
    single_support_coverage = float(np.clip(env.sensor_single_support_ema[env_id] / 0.15, 0.0, 1.0))
    alternation = 0.5 * alternation + 0.5 * single_support_coverage
    stance_slip = (
        float(np.mean(contact * np.clip((stance_speed - 0.15) / 0.5, 0.0, 2.0)))
        if stance_speed is not None else 0.0
    )
    if foot_height is not None and single_support:
        swing_height = (
            float(foot_height[1] - foot_height[0])
            if contact[0] else float(foot_height[0] - foot_height[1])
        )
        swing_height = max(swing_height, 0.0)
        swing_clearance = float(np.exp(
            -(((swing_height - NATURAL_SWING_CLEARANCE_METERS)
               / SWING_CLEARANCE_SIGMA_METERS) ** 2)
        ))
    else:
        swing_clearance = 0.5
    if foot_forward_speed is not None:
        vr, vl = (float(v) for v in foot_forward_speed)
        moving = abs(vr) + abs(vl) > 0.25
        motion_mirror = float(np.clip(abs(vr - vl) / (abs(vr) + abs(vl) + 0.10), 0.0, 1.0)) if moving else 0.5
        synchronous_forward = (
            float(double_support) * float(np.clip((min(vr, vl) - 0.20) / 0.5, 0.0, 2.0))
        )
    else:
        motion_mirror = 0.5
        synchronous_forward = 0.0
    if one_touchdown:
        env.sensor_last_touchdown_side[env_id] = touchdown_side

    env.sensor_liftoff_x[env_id] = np.where(liftoff, foot_x, env.sensor_liftoff_x[env_id])
    env.sensor_had_liftoff[env_id] |= liftoff
    measured_step = touchdown & env.sensor_had_liftoff[env_id]
    stride = np.clip(foot_x - env.sensor_liftoff_x[env_id], 0.0, 0.6)
    env.sensor_stride_ema[env_id] = np.where(
        measured_step,
        0.7 * env.sensor_stride_ema[env_id] + 0.3 * stride,
        env.sensor_stride_ema[env_id],
    )
    env.sensor_stride_seen[env_id] |= measured_step
    env.sensor_contact_duty_ema[env_id] = (
        0.98 * env.sensor_contact_duty_ema[env_id] + 0.02 * contact.astype(np.float32)
    )
    lateral_match = float(np.exp(-((lateral_sum / 0.08) ** 2)))
    if np.all(env.sensor_stride_seen[env_id]):
        mean_stride = float(np.mean(env.sensor_stride_ema[env_id]))
        stride_shortfall = max(MIN_NATURAL_STRIDE_METERS - mean_stride, 0.0)
        stride_length = float(np.exp(-((stride_shortfall / STRIDE_SHORTFALL_SIGMA_METERS) ** 2)))
        stride_match = np.exp(-((np.diff(env.sensor_stride_ema[env_id])[0] / 0.15) ** 2))
        duty_match = np.exp(-((np.diff(env.sensor_contact_duty_ema[env_id])[0] / 0.25) ** 2))
        mirror = float(0.40 * stride_match + 0.10 * duty_match + 0.20 * lateral_match + 0.30 * motion_mirror)
    else:
        stride_length = 0.5
        mirror = 0.15 + 0.20 * lateral_match + 0.30 * motion_mirror
    return {
        "sensor_mirror": mirror,
        "sensor_stride_length": stride_length,
        "sensor_step_cadence": cadence,
        "sensor_step_interval": float(env.sensor_step_interval_ema[env_id]),
        "sensor_swing_clearance": swing_clearance,
        "sensor_lateral_mirror": lateral_match,
        "sensor_motion_mirror": motion_mirror,
        "sensor_foot_roll": roll,
        "sensor_roll_violation": violation,
        "sensor_alternation": alternation,
        "sensor_hop": hop,
        "sensor_tiptoe": tiptoe,
        "sensor_stance_slip": stance_slip,
        "sensor_synchronous_forward": synchronous_forward,
        "sensor_single_support_coverage": single_support_coverage,
        "sensor_left_stride": float(env.sensor_stride_ema[env_id, 0]),
        "sensor_right_stride": float(env.sensor_stride_ema[env_id, 1]),
    }
