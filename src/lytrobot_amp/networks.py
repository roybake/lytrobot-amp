"""Neural networks used by the PPO-AMP trainer."""

from __future__ import annotations

from collections.abc import Iterable
import math

import torch
from torch import nn
from torch.distributions import Normal


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    activation: type[nn.Module] = nn.ELU,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: tuple[int, ...] = (256, 128),
        critic_hidden_dims: tuple[int, ...] = (256, 128),
        init_std: float = 0.8,
        min_action_std: float = 0.05,
        max_action_std: float = 0.80,
    ) -> None:
        super().__init__()
        self.actor = build_mlp(obs_dim, actor_hidden_dims, action_dim, output_activation=nn.Tanh())
        self.critic = build_mlp(obs_dim, critic_hidden_dims, 1)
        self.min_action_std = float(max(min_action_std, 1.0e-4))
        self.max_action_std = float(max(max_action_std, self.min_action_std))
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_std)).log())

    def _clamped_log_std(self) -> torch.Tensor:
        return torch.clamp(
            self.log_std,
            min=math.log(self.min_action_std),
            max=math.log(self.max_action_std),
        )

    @torch.no_grad()
    def clamp_log_std_(self) -> None:
        self.log_std.copy_(self._clamped_log_std())

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = torch.exp(self._clamped_log_std()).expand_as(mean)
        return Normal(mean, std)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        if action is None:
            action = torch.clamp(dist.sample(), -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.actor(obs), -1.0, 1.0)


class AmpDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, 1, activation=nn.LeakyReLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def reward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        reward = -torch.log(torch.clamp(torch.sigmoid(-logits), min=1.0e-4))
        return torch.clamp(reward, 0.0, 5.0)
