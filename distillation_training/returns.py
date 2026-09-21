"""Return computation shared by Teacher and recurrent Student PPO."""
from __future__ import annotations

import torch


def compute_gae_with_time_limit_bootstrap(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    time_outs: torch.Tensor,
    timeout_bootstrap_values: torch.Tensor,
    final_next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (
        rewards.shape
        == values.shape
        == dones.shape
        == time_outs.shape
        == timeout_bootstrap_values.shape
    ):
        raise ValueError("all rollout tensors must have identical [T, N] shapes")
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(final_next_value)
    for step in reversed(range(rewards.shape[0])):
        next_values = final_next_value if step == rewards.shape[0] - 1 else values[step + 1]
        next_values = torch.where(
            time_outs[step] > 0.0, timeout_bootstrap_values[step], next_values
        )
        terminal = torch.clamp(dones[step] - time_outs[step], min=0.0, max=1.0)
        next_non_terminal = 1.0 - terminal
        delta = rewards[step] + float(gamma) * next_values * next_non_terminal - values[step]
        last_gae = (
            delta
            + float(gamma) * float(gae_lambda) * next_non_terminal * last_gae
        )
        advantages[step] = last_gae
    return advantages, advantages + values
