"""Autoreset must not change the value target of a timed-out episode."""

from types import SimpleNamespace

import torch
from torch import nn

from lytrobot_amp.ppo_amp import PPOAMPConfig, PPOAMPTrainer


class _TimeoutEnv:
    num_envs = 1
    obs_dim = 1
    action_dim = 1
    amp_obs_dim = 1
    is_torch_env = True

    def __init__(self) -> None:
        self.cfg = SimpleNamespace()
        self.steps = 0
        self.state = torch.zeros(1, 1)

    def reset(self):
        self.state = torch.zeros(1, 1)
        return self.state.clone()

    def observe(self):
        return self.state.clone()

    def amp_observe(self):
        return torch.zeros(1, 1)

    def step(self, action):
        del action
        self.steps += 1
        info = {
            "episode_step": torch.ones(1),
            "time_outs": torch.tensor([float(self.steps == 1)]),
            "successes": torch.zeros(1),
            "termination": torch.zeros(1),
        }
        if self.steps == 1:
            self.state = torch.tensor([[2.0]])
            info.update(self.terminal_observer(torch.tensor([0])))
            info["terminal_env_ids"] = torch.tensor([0])
            self.state = torch.tensor([[-5.0]])
            return self.state.clone(), torch.tensor([1.0]), torch.ones(1), info
        return self.state.clone(), torch.tensor([1000.0]), torch.zeros(1), info


def test_amp_timeout_uses_pre_reset_value_and_stops_gae_trace():
    env = _TimeoutEnv()
    cfg = PPOAMPConfig(
        num_steps=2,
        amp_enabled=False,
        reward_scale=1.0,
        task_reward_lerp=1.0,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        discriminator_hidden_dims=(8,),
    )
    trainer = PPOAMPTrainer(env, cfg, "cpu", ".")
    trainer.actor_critic.critic = nn.Linear(1, 1)
    with torch.no_grad():
        trainer.actor_critic.critic.weight.fill_(1.0)
        trainer.actor_critic.critic.bias.zero_()

    rollout = trainer.collect_rollout()

    torch.testing.assert_close(
        rollout["timeout_bootstrap_values"][0], torch.tensor([2.0])
    )
    torch.testing.assert_close(
        rollout["advantages"][0], torch.tensor([1.0 + cfg.gamma * 2.0])
    )
    torch.testing.assert_close(
        rollout["advantages"][1],
        torch.tensor([1000.0 + cfg.gamma * -5.0 - -5.0]),
    )
