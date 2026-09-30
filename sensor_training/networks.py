"""Recurrent proprioceptive policy used by the sensor-training variant."""

from __future__ import annotations

from collections.abc import Iterable
import math

import torch
from torch import nn
from torch.distributions import Normal

from .state_estimation import state_layout


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
    """GRU actor-critic, optionally with softly weighted residual experts.

    The recurrent policy consumes a fixed window of causal sensor frames and
    carries additional information between policy steps in its hidden state.
    Both inputs are deployable: the frame window is maintained by the control
    process and the hidden state is explicit in the exported ONNX interface.
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
        actor_history_frames: int = 1,
        recurrent_hidden_dim: int = 128,
        use_moe: bool = False,
        num_experts: int = 4,
        gate_hidden_dims: tuple[int, ...] = (128, 64),
        expert_hidden_dims: tuple[int, ...] = (128, 64),
        residual_scale: float = 0.35,
        privileged_context_dim: int = 0,
        state_estimator_enabled: bool = False,
        state_estimator_include_foot_position: bool = False,
        actor_use_estimated_state: bool = False,
        future_prediction_enabled: bool = False,
        future_prediction_steps: tuple[int, ...] = (1, 3, 5),
        future_prediction_dim: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.sensor_frame_dim = int(sensor_frame_dim or obs_dim)
        self.actor_history_frames = int(actor_history_frames)
        self.recurrent_input_dim = self.sensor_frame_dim * self.actor_history_frames
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.use_moe = bool(use_moe)
        self.num_experts = int(num_experts)
        self.residual_scale = float(residual_scale)
        self.privileged_context_dim = int(privileged_context_dim)
        if not 0 < self.sensor_frame_dim <= self.obs_dim:
            raise ValueError("sensor_frame_dim must be in [1, obs_dim].")
        if self.actor_history_frames < 1:
            raise ValueError("actor_history_frames must be positive.")
        if self.recurrent_input_dim > self.obs_dim:
            raise ValueError(
                "sensor_frame_dim * actor_history_frames must not exceed obs_dim."
            )
        if self.recurrent_hidden_dim < 1:
            raise ValueError("recurrent_hidden_dim must be positive.")
        if self.use_moe and self.num_experts < 1:
            raise ValueError("num_experts must be positive.")
        if self.residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative.")
        if self.privileged_context_dim < 0:
            raise ValueError("privileged_context_dim must be non-negative.")

        self.state_estimator_enabled = bool(state_estimator_enabled)
        self.actor_use_estimated_state = bool(actor_use_estimated_state)
        self.state_estimator_include_foot_position = bool(state_estimator_include_foot_position)
        self.future_prediction_steps = tuple(future_prediction_steps)
        self.future_prediction_enabled = bool(future_prediction_enabled)
        self.future_prediction_dim = int(future_prediction_dim)
        if self.actor_use_estimated_state and not self.state_estimator_enabled:
            raise ValueError("Estimated-state Actor input requires the state estimator.")
        if any(k <= 0 for k in self.future_prediction_steps) or not self.future_prediction_steps:
            raise ValueError("Future prediction offsets must be positive and nonempty.")
        if self.future_prediction_enabled and self.future_prediction_dim <= 0:
            raise ValueError("Future prediction requires an IMU+tendon dimension.")
        self.state_target_layout = state_layout(self.state_estimator_include_foot_position)
        head_names = {'root_linear_velocity': 'root_velocity', 'foot_contact': 'contact',
                      'foot_local_position': 'foot_position'}
        if self.state_estimator_enabled:
            for name, dim in self.state_target_layout.items():
                setattr(self, head_names.get(name, name) + '_head',
                        build_mlp(self.recurrent_hidden_dim, (64,), dim))
        self.future_heads = nn.ModuleList(
            build_mlp(self.recurrent_hidden_dim, (64,), self.future_prediction_dim)
            for _ in self.future_prediction_steps
        ) if self.future_prediction_enabled else nn.ModuleList()
        actor_input_dim = self.recurrent_hidden_dim + (
            sum(self.state_target_layout.values()) if self.actor_use_estimated_state else 0)
        self.recurrent = nn.GRUCell(self.recurrent_input_dim, self.recurrent_hidden_dim)
        # Keep the public ``actor`` name for checkpoint/export compatibility.
        self.actor = build_mlp(
            actor_input_dim,
            actor_hidden_dims,
            self.action_dim,
        )
        self.gate = (
            build_mlp(
                actor_input_dim,
                gate_hidden_dims,
                self.num_experts,
            )
            if self.use_moe
            else None
        )
        self.experts = nn.ModuleList(
            build_mlp(
                actor_input_dim,
                expert_hidden_dims,
                self.action_dim,
                output_activation=nn.Tanh(),
            )
            for _ in range(self.num_experts)
        ) if self.use_moe else nn.ModuleList()
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
        sensor_history = obs[:, -self.recurrent_input_dim :]
        return self.recurrent(sensor_history, hidden_state)

    def _actor_outputs(
        self,
        hidden_state: torch.Tensor,
        state_predictions: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_features = hidden_state
        if self.actor_use_estimated_state:
            estimates = self.predict_state(hidden_state) if state_predictions is None else state_predictions
            state = [estimates[k].sigmoid() if k == 'foot_contact' else estimates[k]
                     for k in self.state_target_layout]
            actor_features = torch.cat((hidden_state, *state), dim=-1)
        base_logits = self.actor(actor_features)
        if not self.use_moe:
            return (
                torch.tanh(base_logits),
                hidden_state.new_zeros((hidden_state.shape[0], 1)),
                hidden_state.new_zeros((hidden_state.shape[0], self.action_dim)),
            )
        assert self.gate is not None
        gate_weights = torch.softmax(self.gate(actor_features), dim=-1)
        expert_actions = torch.stack(
            tuple(expert(actor_features) for expert in self.experts),
            dim=1,
        )
        residual = torch.sum(gate_weights.unsqueeze(-1) * expert_actions, dim=1)
        mean = torch.tanh(base_logits + self.residual_scale * residual)
        return mean, gate_weights, residual

    def _distribution_from_hidden(
        self,
        hidden_state: torch.Tensor,
        state_predictions: dict[str, torch.Tensor] | None = None,
    ) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        mean, gate_weights, residual = self._actor_outputs(hidden_state, state_predictions)
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

    def predict_state(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.state_estimator_enabled:
            return {}
        names = {'root_linear_velocity': 'root_velocity', 'foot_contact': 'contact',
                 'foot_local_position': 'foot_position'}
        return {key: getattr(self, names.get(key, key) + '_head')(hidden_state)
                for key in self.state_target_layout}

    def auxiliary_sequence(self, obs, initial_hidden, episode_starts):
        hidden = initial_hidden
        states = {key: [] for key in self.state_target_layout} if self.state_estimator_enabled else {}
        futures = [[] for _ in self.future_heads]
        for t in range(obs.shape[0]):
            hidden = self._next_hidden(obs[t], hidden, episode_starts[t])
            for key, prediction in self.predict_state(hidden).items():
                states[key].append(prediction)
            for values, head in zip(futures, self.future_heads):
                values.append(head(hidden))
        return ({key: torch.stack(values) for key, values in states.items()},
                [torch.stack(values) for values in futures])

    def estimator_metadata(self) -> dict:
        return dict(state_estimator_enabled=self.state_estimator_enabled,
                    state_estimator_include_foot_position=self.state_estimator_include_foot_position,
                    actor_use_estimated_state=self.actor_use_estimated_state,
                    future_prediction_enabled=self.future_prediction_enabled,
                    future_prediction_steps=list(self.future_prediction_steps),
                    future_prediction_dim=self.future_prediction_dim)

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

    def rollout_action_and_value(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor | None = None,
        privileged_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return only the tensors needed while collecting a rollout.

        This intentionally omits entropy, the median critic and diagnostic
        MoE outputs.  PPO evaluates those quantities during the gradient
        update, while rollout needs only an action, its likelihood, the value
        baseline and the next recurrent state.  Keeping this hot path narrow
        also makes its one-step CUDA graph quick to compile and replay.
        """

        next_hidden = self._next_hidden(obs, hidden_state, episode_start)
        dist, _gate_weights, _residual = self._distribution_from_hidden(next_hidden)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        critic_features = self._critic_features(next_hidden, privileged_context)
        value = self.critic(critic_features).squeeze(-1)
        return action, log_prob, value, next_hidden

    def evaluate_sequence_shared(self, obs, initial_hidden, episode_starts, actions,
                                 privileged_context=None):
        """One GRU/head pass per tick for PPO, state and action supervision.

        All returned tensors share the same differentiable hidden trajectory.
        Contact estimates remain logits here; Actor alone applies sigmoid.
        """
        if obs.ndim != 3:
            raise ValueError(f"Expected time-major observations, got {tuple(obs.shape)}.")
        hidden = initial_hidden
        outputs = [[] for _ in range(10)]
        for step in range(obs.shape[0]):
            hidden = self._next_hidden(obs[step], hidden, episode_starts[step])
            estimates = self.predict_state(hidden)
            dist, weights, residual = self._distribution_from_hidden(hidden, estimates)
            critic = self._critic_features(hidden, None if privileged_context is None else privileged_context[step])
            state = (torch.cat(tuple(estimates.values()), dim=-1) if estimates
                     else hidden.new_empty((hidden.shape[0], 0)))
            future = (torch.cat(tuple(head(hidden) for head in self.future_heads), dim=-1)
                      if self.future_prediction_enabled else hidden.new_empty((hidden.shape[0], 0)))
            values = (dist.log_prob(actions[step]).sum(-1), dist.entropy().sum(-1),
                      self.critic(critic).squeeze(-1), self.median_critic(critic).squeeze(-1),
                      weights, residual, self.predict_context(hidden), dist.mean, state, future)
            for collection, value in zip(outputs, values):
                collection.append(value)
        stacked = tuple(torch.stack(values) for values in outputs)
        return (*stacked[:4], hidden, *stacked[4:])

    def evaluate_sequence(self, obs, initial_hidden, episode_starts, actions,
                          privileged_context=None):
        """Legacy eight-output API; trainer uses the shared eleven-output API."""
        return self.evaluate_sequence_shared(obs, initial_hidden, episode_starts,
                                             actions, privileged_context)[:8]

    def action_mean_sequence(
        self,
        obs: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Return deterministic Actor means for a time-major rollout.

        This is kept separate from likelihood evaluation so the distillation
        loss compares teacher actions with the deployable deterministic Actor,
        rather than with sampled Gaussian exploration actions.
        """

        if obs.ndim != 3:
            raise ValueError(f"Expected time-major observations, got {tuple(obs.shape)}.")
        hidden = initial_hidden
        means: list[torch.Tensor] = []
        for step in range(obs.shape[0]):
            hidden = self._next_hidden(obs[step], hidden, episode_starts[step])
            mean, _gate_weights, _residual = self._actor_outputs(hidden)
            means.append(mean)
        return torch.stack(means)

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
            observation[:, -self.policy.recurrent_input_dim :],
            recurrent_state,
        )
        action, _weights, _residual = self.policy._actor_outputs(next_hidden)
        return torch.clamp(action, -1.0, 1.0), next_hidden
