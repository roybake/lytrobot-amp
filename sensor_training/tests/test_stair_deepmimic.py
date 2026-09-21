"""Contract checks for the sensor-only stair imitation entry point."""

from __future__ import annotations

import torch

from sensor_training.networks import SensorActorCritic
from sensor_training.stair_deepmimic import (
    SensorStairDeepMimicConfig,
    SensorStairDeepMimicTrainer,
)
from sensor_training.train_stairs_deepmimic import build_train_args


def test_entry_uses_sensor_trainer_and_explicit_imitation() -> None:
    args = build_train_args("up_down", ["--num-envs", "8"])
    assert "--deepmimic-reward" in args
    assert "--no-amp" in args
    assert args.index("--no-use-reference-velocity") > args.index("--use-reference-velocity")
    assert args[args.index("--task-reward-lerp") + 1] == "1.0"
    assert args[args.index("--motion-files") + 1].endswith(
        "stair_up_down_omniretarget.npz"
    )


def test_lcp_loss_reaches_sensor_actor() -> None:
    policy = SensorActorCritic(
        obs_dim=12, action_dim=2, sensor_frame_dim=12,
        actor_history_frames=1, recurrent_hidden_dim=8,
        actor_hidden_dims=(8,), critic_hidden_dims=(8,),
    )
    trainer = SensorStairDeepMimicTrainer.__new__(SensorStairDeepMimicTrainer)
    trainer.cfg = SensorStairDeepMimicConfig(lcp_weight=0.002)
    trainer.actor_critic = policy
    obs = torch.randn(2, 3, 12)
    hidden = policy.initial_state(3)
    starts = torch.zeros(2, 3)
    loss = trainer._actor_auxiliary_loss(obs, hidden, starts, torch.arange(3))
    assert loss is not None and torch.isfinite(loss) and loss.item() >= 0.0
    loss.backward()
    assert any(p.grad is not None for p in policy.actor.parameters())
