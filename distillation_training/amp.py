"""Shared AMP reward and discriminator training for Teacher and Student RL."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch.nn import functional as F

from lytrobot_amp.networks import AmpDiscriminator


@dataclass(frozen=True)
class AMPConfig:
    enabled: bool = True
    task_reward_weight: float = 0.5
    amp_reward_weight: float = 0.6
    discriminator_learning_rate: float = 3e-5
    gradient_penalty_coef: float = 2.0
    minibatch_size: int = 1024
    label_smoothing: float = 0.2
    observation_noise_std: float = 0.05
    mirror_reference_probability: float = 0.5

    def __post_init__(self) -> None:
        if self.task_reward_weight < 0 or self.amp_reward_weight < 0:
            raise ValueError('reward weights must be non-negative')
        if self.discriminator_learning_rate <= 0 or self.minibatch_size <= 0:
            raise ValueError('AMP learning rate and minibatch size must be positive')
        if self.gradient_penalty_coef < 0 or self.observation_noise_std < 0:
            raise ValueError('AMP penalty and noise must be non-negative')
        if not 0 <= self.label_smoothing < .5:
            raise ValueError('AMP label smoothing must be in [0, 0.5)')
        if not 0 <= self.mirror_reference_probability <= 1:
            raise ValueError('AMP mirror probability must be in [0, 1]')


def add_amp_args(parser) -> None:
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--task-reward-weight', type=float, default=0.5)
    parser.add_argument('--amp-reward-weight', type=float, default=0.6)
    parser.add_argument('--amp-discriminator-learning-rate', type=float, default=3e-5)
    parser.add_argument('--amp-gradient-penalty-coef', type=float, default=2.0)
    parser.add_argument('--amp-minibatch-size', type=int, default=1024)
    parser.add_argument('--amp-label-smoothing', type=float, default=0.2)
    parser.add_argument('--amp-observation-noise-std', type=float, default=0.05)
    parser.add_argument('--amp-mirror-reference-probability', type=float, default=0.5)
    parser.add_argument('--freeze-amp-discriminator', action='store_true')


def config_from_args(args) -> AMPConfig:
    return AMPConfig(
        enabled=bool(args.amp),
        task_reward_weight=float(args.task_reward_weight),
        amp_reward_weight=float(args.amp_reward_weight),
        discriminator_learning_rate=float(args.amp_discriminator_learning_rate),
        gradient_penalty_coef=float(args.amp_gradient_penalty_coef),
        minibatch_size=int(args.amp_minibatch_size),
        label_smoothing=float(args.amp_label_smoothing),
        observation_noise_std=float(args.amp_observation_noise_std),
        mirror_reference_probability=float(args.amp_mirror_reference_probability),
    )


class AMPTrainingModule:
    """The one AMP implementation used by both feed-forward training stages."""

    def __init__(self, feature_dim: int, device, config: AMPConfig,
                 reference_motion: str | None = None) -> None:
        self.device = torch.device(device)
        self.feature_dim = int(feature_dim)
        self.config = config
        self.reference_motion = reference_motion
        self.discriminator = AmpDiscriminator(self.feature_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=config.discriminator_learning_rate
        )

    @property
    def metadata(self) -> dict:
        return {
            **asdict(self.config),
            "feature_dim": self.feature_dim,
            "actor_input": False,
            "training_only": True,
            "reference_motion": self.reference_motion,
            "gradient_penalty": "mean(sum(grad(D(x))**2))",
        }

    def load(self, checkpoint: dict, *, load_optimizer: bool = False) -> None:
        state = checkpoint.get('discriminator')
        if state is None:
            raise ValueError('checkpoint does not contain an AMP discriminator')
        self.discriminator.load_state_dict(state)
        if load_optimizer and checkpoint.get('discriminator_optimizer') is not None:
            self.optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

    @torch.no_grad()
    def policy_reward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.config.enabled:
            return torch.zeros(features.shape[0], device=features.device)
        return self.discriminator.reward(features)

    def combine_reward(
        self, task_reward: torch.Tensor, amp_reward: torch.Tensor
    ) -> torch.Tensor:
        if not self.config.enabled:
            return task_reward
        return (
            self.config.task_reward_weight * task_reward
            + self.config.amp_reward_weight * amp_reward
        )

    def sample_expert(self, motion, count: int, rng: np.random.Generator) -> torch.Tensor:
        reference = motion.sample_features(
            int(count), rng, mirror_prob=self.config.mirror_reference_probability
        )
        return torch.as_tensor(reference, dtype=torch.float32, device=self.device)

    def _noisy(self, batch: torch.Tensor) -> torch.Tensor:
        if self.config.observation_noise_std <= 0:
            return batch
        return batch + self.config.observation_noise_std * torch.randn_like(batch)

    def discriminator_loss(
        self, agent: torch.Tensor, expert: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        agent = agent.detach()
        logits_agent = self.discriminator(self._noisy(agent))
        logits_expert = self.discriminator(self._noisy(expert))
        smoothing = float(np.clip(self.config.label_smoothing, 0.0, 0.45))
        bce = 0.5 * (
            F.binary_cross_entropy_with_logits(logits_agent, torch.full_like(logits_agent, smoothing))
            + F.binary_cross_entropy_with_logits(
                logits_expert, torch.full_like(logits_expert, 1.0 - smoothing)
            )
        )
        if self.config.gradient_penalty_coef > 0:
            penalty_batch = torch.cat((agent, expert), 0).detach().requires_grad_(True)
            gradients = torch.autograd.grad(
                self.discriminator(penalty_batch).sum(), penalty_batch,
                create_graph=True, retain_graph=True, only_inputs=True,
            )[0]
            gradient_penalty = gradients.pow(2).sum(1).mean()
        else:
            gradient_penalty = torch.zeros((), device=self.device)
        total = bce + self.config.gradient_penalty_coef * gradient_penalty
        metrics = {
            'discriminator_loss': bce.detach(),
            'agent_logit': logits_agent.detach().mean(),
            'expert_logit': logits_expert.detach().mean(),
            'gradient_penalty': gradient_penalty.detach(),
        }
        return total, metrics

    def update(
        self, agent_features: torch.Tensor, motion, rng: np.random.Generator, *, freeze: bool = False
    ) -> dict[str, torch.Tensor]:
        if not self.config.enabled:
            zero = torch.zeros((), device=self.device)
            return {key: zero for key in ('discriminator_loss','agent_logit','expert_logit','gradient_penalty')}
        flat = agent_features.reshape(-1, self.feature_dim)
        count = min(max(1, self.config.minibatch_size), flat.shape[0])
        indices = torch.randperm(flat.shape[0], device=self.device)[:count]
        agent = flat[indices]
        expert = self.sample_expert(motion, count, rng)
        total, metrics = self.discriminator_loss(agent, expert)
        if not bool(torch.isfinite(total)):
            raise RuntimeError('AMP discriminator loss became NaN/Inf')
        if not freeze:
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError('AMP discriminator gradient became NaN/Inf')
            self.optimizer.step()
        return metrics
