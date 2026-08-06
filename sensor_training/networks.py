"""Recurrent proprioceptive policy used by the sensor-training variant."""

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


class SensorActorCritic(nn.Module):
    """GRU actor-critic with a base MLP and softly weighted residual experts.

    The environment keeps its existing stacked sensor observation for AMP and
    checkpoint metadata.  The recurrent policy consumes only the newest frame;
    its hidden state carries causal information across policy steps instead of
    repeatedly re-encoding overlapping fixed histories.
    """

    is_recurrent = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: tuple[int, ...] = (256, 128),
        critic_hidden_dims: tuple[int, ...] = (256, 128),
        init_std: float = 0.8,
        min_action_std: float = 0.05,
        max_action_std: float = 0.80,
        *,
        sensor_frame_dim: int | None = None,
        recurrent_hidden_dim: int = 128,
        num_experts: int = 4,
        gate_hidden_dims: tuple[int, ...] = (128, 64),
        expert_hidden_dims: tuple[int, ...] = (128, 64),
        residual_scale: float = 0.35,
        privileged_context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.sensor_frame_dim = int(sensor_frame_dim or obs_dim)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.num_experts = int(num_experts)
        self.residual_scale = float(residual_scale)
        self.privileged_context_dim = int(privileged_context_dim)
        if not 0 < self.sensor_frame_dim <= self.obs_dim:
            raise ValueError("sensor_frame_dim must be in [1, obs_dim].")
        if self.recurrent_hidden_dim < 1:
            raise ValueError("recurrent_hidden_dim must be positive.")
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive.")
        if self.residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative.")
        if self.privileged_context_dim < 0:
            raise ValueError("privileged_context_dim must be non-negative.")

        self.recurrent = nn.GRUCell(self.sensor_frame_dim, self.recurrent_hidden_dim)
        # Keep the public ``actor`` name for checkpoint/export compatibility.
        self.actor = build_mlp(
            self.recurrent_hidden_dim,
            actor_hidden_dims,
            self.action_dim,
        )
        self.gate = build_mlp(
            self.recurrent_hidden_dim,
            gate_hidden_dims,
            self.num_experts,
        )
        self.experts = nn.ModuleList(
            build_mlp(
                self.recurrent_hidden_dim,
                expert_hidden_dims,
                self.action_dim,
                output_activation=nn.Tanh(),
            )
            for _ in range(self.num_experts)
        )
        critic_input_dim = self.recurrent_hidden_dim + self.privileged_context_dim
        self.critic = build_mlp(critic_input_dim, critic_hidden_dims, 1)
        self.median_critic = build_mlp(critic_input_dim, critic_hidden_dims, 1)
        self.context_estimator = (
            build_mlp(
                self.recurrent_hidden_dim,
                (64,),
                self.privileged_context_dim,
                output_activation=nn.Tanh(),
            )
            if self.privileged_context_dim > 0
            else None
        )
        self.min_action_std = float(max(min_action_std, 1.0e-4))
        self.max_action_std = float(max(max_action_std, self.min_action_std))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(init_std)).log())

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            (int(batch_size), self.recurrent_hidden_dim),
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def _clamped_log_std(self) -> torch.Tensor:
        return torch.clamp(
            self.log_std,
            min=math.log(self.min_action_std),
            max=math.log(self.max_action_std),
        )

    @torch.no_grad()
    def clamp_log_std_(self) -> None:
        self.log_std.copy_(self._clamped_log_std())

    def _next_hidden(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor | None,
    ) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected observations [batch, {self.obs_dim}], got {tuple(obs.shape)}.")
        if hidden_state.shape != (obs.shape[0], self.recurrent_hidden_dim):
            raise ValueError(
                "Expected recurrent state "
                f"{(obs.shape[0], self.recurrent_hidden_dim)}, got {tuple(hidden_state.shape)}."
            )
        if episode_start is not None:
            reset = episode_start.to(device=obs.device, dtype=obs.dtype).reshape(-1, 1)
            hidden_state = hidden_state * (1.0 - torch.clamp(reset, 0.0, 1.0))
        current_frame = obs[:, -self.sensor_frame_dim :]
        return self.recurrent(current_frame, hidden_state)

    def _actor_outputs(
        self,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_logits = self.actor(hidden_state)
        gate_weights = torch.softmax(self.gate(hidden_state), dim=-1)
        expert_actions = torch.stack(
            tuple(expert(hidden_state) for expert in self.experts),
            dim=1,
        )
        residual = torch.sum(gate_weights.unsqueeze(-1) * expert_actions, dim=1)
        mean = torch.tanh(base_logits + self.residual_scale * residual)
        return mean, gate_weights, residual

    def _distribution_from_hidden(
        self,
        hidden_state: torch.Tensor,
    ) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        mean, gate_weights, residual = self._actor_outputs(hidden_state)
        std = torch.exp(self._clamped_log_std()).expand_as(mean)
        return Normal(mean, std), gate_weights, residual

    def _distribution(self, obs: torch.Tensor) -> Normal:
        """Stateless compatibility distribution for one-shot callers."""

        hidden = self.initial_state(obs.shape[0], device=obs.device, dtype=obs.dtype)
        next_hidden = self._next_hidden(obs, hidden, None)
        return self._distribution_from_hidden(next_hidden)[0]

    def _critic_features(
        self,
        hidden_state: torch.Tensor,
        privileged_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.privileged_context_dim == 0:
            return hidden_state
        if privileged_context is None:
            privileged_context = torch.zeros(
                (hidden_state.shape[0], self.privileged_context_dim),
                dtype=hidden_state.dtype,
                device=hidden_state.device,
            )
        if privileged_context.shape != (
            hidden_state.shape[0],
            self.privileged_context_dim,
        ):
            raise ValueError(
                "Expected privileged context "
                f"{(hidden_state.shape[0], self.privileged_context_dim)}, "
                f"got {tuple(privileged_context.shape)}."
            )
        return torch.cat(
            (
                hidden_state,
                privileged_context.to(
                    device=hidden_state.device,
                    dtype=hidden_state.dtype,
                ),
            ),
            dim=-1,
        )

    def predict_context(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.context_estimator is None:
            return hidden_state.new_empty((hidden_state.shape[0], 0))
        return self.context_estimator(hidden_state)

    def recurrent_action_and_value(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        privileged_context: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        next_hidden = self._next_hidden(obs, hidden_state, episode_start)
        dist, gate_weights, residual = self._distribution_from_hidden(next_hidden)
        if action is None:
            # Keep raw Gaussian samples for an unbiased likelihood ratio.  The
            # actuator environment performs the physical command clipping.
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        critic_features = self._critic_features(next_hidden, privileged_context)
        value = self.critic(critic_features).squeeze(-1)
        median_value = self.median_critic(critic_features).squeeze(-1)
        return (
            action,
            log_prob,
            entropy,
            value,
            median_value,
            next_hidden,
            gate_weights,
            residual,
        )

    def evaluate_sequence(
        self,
        obs: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        actions: torch.Tensor,
        privileged_context: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Evaluate a time-major rollout with truncated BPTT."""

        if obs.ndim != 3:
            raise ValueError(f"Expected time-major observations, got {tuple(obs.shape)}.")
        hidden = initial_hidden
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        median_values: list[torch.Tensor] = []
        gate_weights: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        context_predictions: list[torch.Tensor] = []
        for step in range(obs.shape[0]):
            (
                _action,
                log_prob,
                entropy,
                value,
                median_value,
                hidden,
                weights,
                residual,
            ) = self.recurrent_action_and_value(
                obs[step],
                hidden,
                episode_starts[step],
                action=actions[step],
                privileged_context=(
                    None
                    if privileged_context is None
                    else privileged_context[step]
                ),
            )
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            median_values.append(median_value)
            gate_weights.append(weights)
            residuals.append(residual)
            context_predictions.append(self.predict_context(hidden))
        return (
            torch.stack(log_probs),
            torch.stack(entropies),
            torch.stack(values),
            torch.stack(median_values),
            hidden,
            torch.stack(gate_weights),
            torch.stack(residuals),
            torch.stack(context_predictions),
        )

    # Stateless compatibility methods are useful for unit tests and one-shot
    # consumers.  Rollout and deployment use the recurrent methods above.
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.initial_state(obs.shape[0], device=obs.device, dtype=obs.dtype)
        result = self.recurrent_action_and_value(obs, hidden, action=action)
        return result[0], result[1], result[2], result[3]

    def median_value(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_state(obs.shape[0], device=obs.device, dtype=obs.dtype)
        next_hidden = self._next_hidden(obs, hidden, None)
        return self.median_critic(
            self._critic_features(next_hidden, None)
        ).squeeze(-1)

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_state(obs.shape[0], device=obs.device, dtype=obs.dtype)
        action, _hidden = self.act_inference_recurrent(obs, hidden)
        return action

    @torch.no_grad()
    def act_inference_recurrent(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_hidden = self._next_hidden(obs, hidden_state, episode_start)
        mean, _gate_weights, _residual = self._actor_outputs(next_hidden)
        return torch.clamp(mean, -1.0, 1.0), next_hidden


class RecurrentSensorActorExport(nn.Module):
    """ONNX-facing deterministic actor with explicit recurrent state I/O."""

    def __init__(self, policy: SensorActorCritic) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        observation: torch.Tensor,
        recurrent_state: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reset = episode_start.to(
            device=observation.device,
            dtype=observation.dtype,
        ).reshape(-1, 1)
        recurrent_state = recurrent_state * (
            1.0 - torch.clamp(reset, 0.0, 1.0)
        )
        next_hidden = self.policy.recurrent(
            observation[:, -self.policy.sensor_frame_dim :],
            recurrent_state,
        )
        action, _weights, _residual = self.policy._actor_outputs(next_hidden)
        return torch.clamp(action, -1.0, 1.0), next_hidden
