"""Neural networks used by the BPO-AMP trainer."""

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
    """Actor-critic with likelihoods computed from raw Gaussian samples."""

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
        self.median_critic = build_mlp(obs_dim, critic_hidden_dims, 1)
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
            # The environments clip the executed command to actuator limits.
            # Keep the sampled action itself here so PPO/BPO likelihood ratios
            # are computed under the distribution that actually produced it.
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    def median_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.median_critic(obs).squeeze(-1)

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.actor(obs), -1.0, 1.0)


class AmpDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.condition_dim = max(int(condition_dim), 0)
        self.net = build_mlp(
            self.input_dim + self.condition_dim,
            hidden_dims,
            1,
            activation=nn.LeakyReLU,
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.condition_dim:
            if condition is None:
                raise ValueError("Conditional AMP requires motion/terrain condition IDs.")
            if condition.ndim == 1:
                condition = torch.nn.functional.one_hot(
                    condition.to(torch.long), num_classes=self.condition_dim
                ).to(dtype=x.dtype)
            if condition.shape != (x.shape[0], self.condition_dim):
                raise ValueError(
                    "AMP condition shape must be "
                    f"{(x.shape[0], self.condition_dim)}, got {tuple(condition.shape)}."
                )
            x = torch.cat((x, condition.to(device=x.device, dtype=x.dtype)), dim=-1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def reward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward(x, condition)
        reward = -torch.log(torch.clamp(torch.sigmoid(-logits), min=1.0e-4))
        return torch.clamp(reward, 0.0, 5.0)
