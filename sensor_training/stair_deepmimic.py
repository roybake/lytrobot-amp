"""DeepMimic stair policy trained from deployable sensor observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .ppo_config import SensorPPOAMPConfig
from .trainer import SensorHistoryPPOAMPTrainer


@dataclass
class SensorStairDeepMimicConfig(SensorPPOAMPConfig):
    """PPO configuration kept free of non-DeepMimic auxiliary objectives."""

    lcp_weight: float = 0.0
    privileged_context_loss_coef: float = 0.0


class SensorStairDeepMimicTrainer(SensorHistoryPPOAMPTrainer):
    """Add MimicKit's observation-gradient policy regularizer to sensor PPO."""

    def _actor_auxiliary_loss(
        self,
        obs: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        env_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        del env_indices
        if self.cfg.lcp_weight <= 0.0:
            return None
        # One recurrent step is enough to regularize the actor's local
        # sensitivity while keeping the second-order update tractable.
        sensor_obs = obs[0].detach().requires_grad_(True)
        hidden = initial_hidden.detach()
        next_hidden = self.actor_critic._next_hidden(
            sensor_obs, hidden, episode_starts[0]
        )
        dist, _gate, _residual = self.actor_critic._distribution_from_hidden(next_hidden)
        action = dist.sample().detach()
        log_prob = dist.log_prob(action).sum(dim=-1)
        gradient = torch.autograd.grad(
            log_prob.sum(), sensor_obs, create_graph=True
        )[0]
        lcp_loss = gradient.square().sum(dim=-1).mean()
        return float(self.cfg.lcp_weight) * lcp_loss
