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
    # At 100 Hz the discount horizon of gamma=.99 is only about one second.
    gamma: float = 0.999
    gae_lambda: float = 0.97
    init_std: float = 0.20
    stair_rsi_initial_probability: float = 0.90
    stair_rsi_final_probability: float = 0.20
    stair_rsi_warmup_iterations: int = 2000
    stair_rsi_ramp_iterations: int = 10000


class SensorStairDeepMimicTrainer(SensorHistoryPPOAMPTrainer):
    """Add MimicKit's observation-gradient policy regularizer to sensor PPO."""

    def train_iteration(self) -> dict[str, float]:
        if self.env.cfg.deepmimic_route_tracking:
            iteration = self.global_step / (self.env.num_envs * self.cfg.num_steps)
            fraction = min(1.0, max(0.0,
                (iteration - self.cfg.stair_rsi_warmup_iterations)
                / max(self.cfg.stair_rsi_ramp_iterations, 1)))
            probability = ((1-fraction) * self.cfg.stair_rsi_initial_probability
                           + fraction * self.cfg.stair_rsi_final_probability)
            self.env.cfg.stair_phase_reset_prob = probability
        metrics = super().train_iteration()
        metrics["stair_phase_reset_probability"] = self.env.cfg.stair_phase_reset_prob
        return metrics

    def evaluate_fixed_route(self) -> dict[str, float]:
        from .stair_evaluation import evaluate_fixed_stair_route
        return evaluate_fixed_stair_route(self.actor_critic, self.env.cfg)

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
