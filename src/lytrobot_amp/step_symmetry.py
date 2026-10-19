"""Measure consecutive left/right landing steps, in right/left foot order."""

from __future__ import annotations

import math

import torch


MIN_CONTACT_SECONDS = 0.05
MIN_SWING_SECONDS = 0.05


class StepLengthTracker:
    """One causal measurement shared by CPU, GPU training and evaluation.

    A landing is confirmed after continuous contact, but its distance is saved
    at first contact. Short contact breaks cannot create extra steps. The
    opposite foot must support the landing; simultaneous landings never form
    a walking pair. State is retained across rollouts and cleared on reset.
    """

    def __init__(self, num_envs: int, dt: float, device="cpu"):
        self.device = torch.device(device)
        self.min_air = max(1, math.ceil(MIN_SWING_SECONDS / dt - 1e-8))
        # First contact is sample 1; confirmation follows a full 50 ms later.
        self.min_contact = 1 + max(1, math.ceil(MIN_CONTACT_SECONDS / dt - 1e-8))
        self.state = {}
        for name in ("air", "stance"):
            self.state[name] = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        for name in ("pending", "observed_contact"):
            self.state[name] = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)
        for name in ("landing", "lengths"):
            self.state[name] = torch.zeros((num_envs, 2), device=device)
        for name in ("clock", "last_time", "interval", "pair_age"):
            self.state[name] = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.state["last_side"] = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        for name in ("last_length", "error"):
            self.state[name] = torch.zeros(num_envs, device=device)
        for name in ("pair_seen", "positive"):
            self.state[name] = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=slice(None)):
        for name, value in self.state.items():
            value[env_ids] = -1 if name == "last_side" else 0

    def update(self, contact, foot_x, env_ids=slice(None), *, active=True):
        contact = torch.as_tensor(contact, dtype=torch.bool, device=self.device)
        foot_x = torch.as_tensor(foot_x, dtype=torch.float32, device=self.device)
        active = torch.as_tensor(active, dtype=torch.bool, device=self.device).reshape(-1)
        s = {name: value[env_ids] for name, value in self.state.items()}
        clock = s["clock"] + 1
        raw_landing = contact & (s["air"] >= self.min_air) & s["observed_contact"]
        simultaneous = raw_landing.all(dim=1)
        candidate = raw_landing & contact.flip(1) & ~simultaneous[:, None]
        gap = foot_x - foot_x.flip(1)
        landing = torch.where(candidate, gap, s["landing"])
        pending = contact & (s["pending"] | candidate)
        air = torch.where(contact, 0, s["air"] + 1)
        stance = torch.where(contact, s["stance"] + 1, 0)
        confirmed = pending & (stance == self.min_contact)
        single = confirmed[:, 0] ^ confirmed[:, 1]
        side = torch.where(confirmed[:, 0], 0, 1)
        length = torch.where(confirmed[:, 0], landing[:, 0], landing[:, 1])
        previous_side = torch.where(simultaneous, -1, s["last_side"])
        paired = single & (previous_side >= 0) & (previous_side != side)
        pair_lengths = torch.stack((
            torch.where(side == 0, length, s["last_length"]),
            torch.where(side == 1, length, s["last_length"]),
        ), dim=1)
        denominator = pair_lengths.abs().mean(dim=1).clamp_min(1e-6)
        error = (pair_lengths[:, 0] - pair_lengths[:, 1]).abs() / denominator
        lengths = torch.where(confirmed, landing, s["lengths"])
        positive = torch.where(paired, (pair_lengths > 1e-6).all(dim=1), s["positive"])
        interval = torch.where(paired, clock - s["last_time"], s["interval"])
        pair_age = torch.where(paired, 0, s["pair_age"] + 1)
        broken_sequence = simultaneous | (single & (previous_side == side))
        pair_seen = (s["pair_seen"] & ~broken_sequence) | paired
        current_error = torch.where(paired, error, s["error"])
        fresh = pair_seen & (pair_age <= 2 * interval) & ~simultaneous
        updates = dict(
            air=air, stance=stance, pending=pending, landing=landing, lengths=lengths,
            clock=clock, last_time=torch.where(single, clock, s["last_time"]),
            interval=interval, pair_age=pair_age,
            last_side=torch.where(single, side, previous_side),
            last_length=torch.where(single, length, s["last_length"]),
            pair_seen=pair_seen, positive=positive, error=current_error,
            observed_contact=s["observed_contact"] | contact,
        )
        for name, value in updates.items():
            mask = active[:, None] if value.ndim == 2 else active
            self.state[name][env_ids] = torch.where(mask, value, -1 if name == "last_side" else 0)
        metrics = {
            "right_step_length_m": lengths[:, 0],
            "left_step_length_m": lengths[:, 1],
            "right_step_event": confirmed[:, 0].float(),
            "left_step_event": confirmed[:, 1].float(),
            "step_pair_event": paired.float(),
            "step_length_relative_error": current_error,
            "step_pair_positive": positive.float(),
            "step_pair_fresh": fresh.float(),
        }
        return {name: torch.where(active, value, 0.) for name, value in metrics.items()}


def step_symmetry_reward(metrics, gait_progress, speed_track, stability):
    """Equal forward steps earn speed-gated credit; unequal pairs incur cost."""
    return gait_progress * stability * metrics["step_pair_fresh"] * (
        speed_track * metrics["step_pair_positive"]
        - metrics["step_length_relative_error"] ** 2
    )
