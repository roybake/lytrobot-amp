"""PPO/BPO config type scoped to sensor-only policy training."""

from __future__ import annotations

from dataclasses import dataclass

from lytrobot_amp.ppo_amp import PPOAMPConfig


@dataclass
class SensorPPOAMPConfig(PPOAMPConfig):
    """BPO defaults for sensor-history policies."""

    target_kl: float = 0.025
    target_kl_stop_multiplier: float = 2.0
    recurrent_hidden_dim: int = 128
    privileged_context_loss_coef: float = 0.20
    compile_policy: bool = True
    fused_optimizer: bool = True
