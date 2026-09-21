"""Variant-aware shared-head teacher/student policy networks."""
from __future__ import annotations

import torch
from torch import nn

from sensor_training.networks import build_mlp
from .config import ACTION_DIM, HIDDEN_DIM, PRIVILEGED_DIM


class SharedActor(nn.Module):
    def __init__(
        self,
        sensor_dim: int,
        privileged_dim: int = PRIVILEGED_DIM,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self.sensor_dim = int(sensor_dim)
        self.privileged_dim = int(privileged_dim)
        self.action_dim = int(action_dim)
        self.sensor_encoder = build_mlp(self.sensor_dim, (128,), 128)
        self.privileged_encoder = build_mlp(self.privileged_dim, (64,), 64)
        self.actor_head = build_mlp(192, (256, 128), self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def mean(self, sensor, privileged):
        return torch.tanh(
            self.actor_head(
                torch.cat(
                    (self.sensor_encoder(sensor), self.privileged_encoder(privileged)), 1
                )
            )
        )

    def distribution(self, sensor, privileged):
        return torch.distributions.Normal(
            self.mean(sensor, privileged),
            self.log_std.exp().expand(sensor.shape[0], -1),
        )

    def forward(self, sensor, privileged, deterministic=False):
        distribution = self.distribution(sensor, privileged)
        action = distribution.mean if deterministic else distribution.rsample()
        return action, distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1)


class TeacherCritic(nn.Module):
    def __init__(
        self,
        sensor_dim: int,
        privileged_dim: int = PRIVILEGED_DIM,
    ):
        super().__init__()
        input_dim = int(sensor_dim) + int(privileged_dim)
        self.value_net = build_mlp(input_dim, (256, 128), 1)
        self.median_net = build_mlp(input_dim, (256, 128), 1)

    def forward(self, sensor, privileged):
        value_input = torch.cat((sensor, privileged), 1)
        return self.value_net(value_input).squeeze(-1), self.median_net(value_input).squeeze(-1)


class HistoryEstimator(nn.Module):
    def __init__(
        self,
        sensor_dim: int,
        policy_dt: float = 0.01,
        tendon_velocity_scale: float = 10.0,
    ):
        super().__init__()
        self.sensor_dim = int(sensor_dim)
        self.estimator_input_dim = self.sensor_dim + ACTION_DIM + ACTION_DIM
        self.gru = nn.GRUCell(self.estimator_input_dim, HIDDEN_DIM)
        self.out = build_mlp(HIDDEN_DIM, (128,), PRIVILEGED_DIM)
        self.policy_dt = float(policy_dt)
        self.tendon_velocity_scale = float(tendon_velocity_scale)

    def forward(self, sensor, previous_action, previous_tendon, hidden, episode_start=None):
        if episode_start is not None:
            hidden = hidden * (1 - episode_start.reshape(-1, 1).to(hidden.dtype))
        tendon_velocity = (sensor[:, 42:56] - previous_tendon) / (
            self.policy_dt * self.tendon_velocity_scale
        )
        hidden = self.gru(torch.cat((sensor, previous_action, tendon_velocity), 1), hidden)
        return self.out(hidden), hidden


class StudentExport(nn.Module):
    def __init__(self, actor, estimator, sensor_dim: int):
        super().__init__()
        self.actor = actor
        self.estimator = estimator
        self.sensor_dim = int(sensor_dim)
        if actor.sensor_dim != self.sensor_dim or estimator.sensor_dim != self.sensor_dim:
            raise ValueError("StudentExport actor/estimator sensor dimensions do not match")

    def forward(self, observation, previous_action, recurrent_state, episode_start):
        privileged, hidden = self.estimator(
            observation,
            previous_action,
            recurrent_state[:, HIDDEN_DIM:],
            recurrent_state[:, :HIDDEN_DIM],
            episode_start,
        )
        action = self.actor.mean(observation, privileged)
        return action, torch.cat((hidden, observation[:, 42:56]), 1)
