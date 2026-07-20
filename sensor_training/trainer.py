"""BPO-AMP trainer extension that records sensor-observation metadata."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from lytrobot_amp.ppo_amp import PPOAMPTrainer
from lytrobot_amp.networks import AmpDiscriminator

from .networks import SensorActorCritic
from .ppo_config import SensorPPOAMPConfig


class SensorHistoryPPOAMPTrainer(PPOAMPTrainer):
    def __init__(
        self,
        env,
        cfg: SensorPPOAMPConfig,
        device: str,
        log_dir: str | Path,
    ) -> None:
        # This initialization is intentionally local instead of changing the
        # shared lytrobot_amp trainer used by position-control training.
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.rng = np.random.default_rng(cfg.seed + 1000)

        self.actor_critic = SensorActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
            init_std=cfg.init_std,
            min_action_std=cfg.min_action_std,
            max_action_std=cfg.max_action_std,
        ).to(self.device)
        self.discriminator = AmpDiscriminator(
            input_dim=env.amp_obs_dim,
            hidden_dims=cfg.discriminator_hidden_dims,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=cfg.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=cfg.discriminator_learning_rate
        )

        self.obs = env.reset()
        self.global_step = 0
        self.episode_reward_sums = torch.zeros(env.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        window = max(1, int(cfg.episode_metric_window))
        self.completed_episode_rewards: deque[float] = deque(maxlen=window)
        self.completed_episode_lengths: deque[int] = deque(maxlen=window)

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> int:
        checkpoint = torch.load(Path(path), map_location=self.device)
        saved_metadata = checkpoint.get("sensor_observation")
        if saved_metadata is None:
            raise ValueError(
                "Checkpoint has no sensor_observation metadata; use a checkpoint produced by sensor_training."
            )
        current_metadata = self.env.observation_metadata
        if saved_metadata != current_metadata:
            raise ValueError(
                "Checkpoint sensor observation layout does not match the current sensor_training layout."
            )
        return super().load(path, load_optimizers=load_optimizers)

    def _checkpoint_payload(self, iteration: int) -> dict:
        checkpoint = super()._checkpoint_payload(iteration)
        checkpoint.update({
            "sensor_observation": self.env.observation_metadata,
        })
        return checkpoint
