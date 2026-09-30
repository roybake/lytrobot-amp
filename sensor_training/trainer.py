"""BPO-AMP trainer extension that records sensor-observation metadata."""

from __future__ import annotations

from collections import deque
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lytrobot_amp.ppo_amp import PPOAMPTrainer
from lytrobot_amp.networks import AmpDiscriminator

from .networks import RecurrentSensorActorExport, SensorActorCritic
from .ppo_config import SensorPPOAMPConfig


def _sensor_metadata_compatible(saved: dict, current: dict) -> bool:
    """Compare actor layouts while allowing a stand-to-walk schedule change.

    Checkpoint metadata passes through JSON, which turns tuples into lists.
    The three schedule values alter command *values* but not actor dimensions
    or field order; fine-tuning must be able to change them.
    """
    saved_normalized = json.loads(json.dumps(saved))
    current_normalized = json.loads(json.dumps(current))
    for metadata in (saved_normalized, current_normalized):
        gait = metadata.get("gait_command")
        if isinstance(gait, dict):
            for key in (
                "motion_loop",
                "stand_start_prob",
                "stand_start_motion_phase",
                "stand_start_phase_delay_steps",
                "stand_start_transition_min_steps",
                "stand_start_transition_max_steps",
            ):
                gait.pop(key, None)
    return saved_normalized == current_normalized


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
        if env.obs_dim % env.sensor_frame_dim != 0:
            raise ValueError(
                "Sensor observation dimension must contain an integer number of frames."
            )
        self.actor_history_frames = env.obs_dim // env.sensor_frame_dim
        context_weights = env.privileged_context_metadata.get("loss_weights")
        if context_weights is None:
            context_weights = (1.0,) * env.privileged_context_dim
        if len(context_weights) != env.privileged_context_dim:
            raise ValueError(
                "Privileged-context loss weights must match the context dimension."
            )
        self.privileged_context_loss_weights = torch.as_tensor(
            context_weights,
            dtype=torch.float32,
            device=self.device,
        )

        self.actor_critic = SensorActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
            init_std=cfg.init_std,
            min_action_std=cfg.min_action_std,
            max_action_std=cfg.max_action_std,
            sensor_frame_dim=env.sensor_frame_dim,
            actor_history_frames=self.actor_history_frames,
            recurrent_hidden_dim=cfg.recurrent_hidden_dim,
            use_moe=cfg.use_moe,
            num_experts=cfg.num_experts,
            gate_hidden_dims=cfg.gate_hidden_dims,
            expert_hidden_dims=cfg.expert_hidden_dims,
            residual_scale=cfg.residual_scale,
            privileged_context_dim=env.privileged_context_dim,
        ).to(self.device)
        self.discriminator = AmpDiscriminator(
            input_dim=env.amp_obs_dim,
            hidden_dims=cfg.discriminator_hidden_dims,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=cfg.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=cfg.discriminator_learning_rate
        )
        self._compiled_evaluate_sequence = None
        self._compiled_rollout_action_and_value = None
        self._compiled_rollout_amp_reward = None
        self._policy_compile_failed = False
        self._rollout_compile_failed = False
        if self.device.type == "cuda":
            # RTX 30-series cards support TF32 tensor cores.  This only
            # affects float32 matmul precision and substantially improves the
            # MLP-heavy recurrent update; it does not alter tensor shapes or
            # the rollout/BPTT semantics.
            torch.set_float32_matmul_precision("high")
            enabled_compilers: list[str] = []
            if bool(cfg.compile_rollout):
                self._compiled_rollout_action_and_value = torch.compile(
                    self.actor_critic.rollout_action_and_value,
                    mode="reduce-overhead",
                    fullgraph=True,
                    dynamic=False,
                )
                if cfg.amp_enabled:
                    self._compiled_rollout_amp_reward = torch.compile(
                        self.discriminator.reward,
                        mode="reduce-overhead",
                        fullgraph=True,
                        dynamic=False,
                    )
                enabled_compilers.append("rollout")
            if bool(cfg.compile_policy_update):
                # Compile a bounded BPTT chunk, not the entire rollout
                # window.  This remains opt-in because recurrent backward
                # graph compilation is expensive on mid-range CUDA GPUs.
                self._compiled_evaluate_sequence = torch.compile(
                    self.actor_critic.evaluate_sequence,
                    mode="reduce-overhead",
                    fullgraph=True,
                    dynamic=False,
                )
                enabled_compilers.append(
                    f"policy-step({int(cfg.compile_policy_chunk_steps)})"
                )
            if enabled_compilers:
                print(
                    "[INFO] compilation enabled for "
                    f"{', '.join(enabled_compilers)}; "
                    "the first matching call will compile a bounded graph "
                    "(the policy backward graph can take a few minutes once)."
                )
        self.obs = env.reset()
        self.recurrent_state = self.actor_critic.initial_state(env.num_envs)
        self.episode_starts = torch.ones(
            env.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.global_step = 0
        self.episode_reward_sums = torch.zeros(env.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        window = max(1, int(cfg.episode_metric_window))
        self.completed_episode_rewards: deque[float] = deque(maxlen=window)
        self.completed_episode_lengths: deque[int] = deque(maxlen=window)
        self.completed_episode_successes: deque[float] = deque(maxlen=window)
        self.loaded_domain_randomization_scale: float | None = None

    def _evaluate_policy_sequence(
        self,
        obs: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        actions: torch.Tensor,
        privileged_contexts: torch.Tensor,
    ):
        """Evaluate a recurrent minibatch, falling back safely if compilation fails."""

        compiled = self._compiled_evaluate_sequence
        if compiled is None:
            return self.actor_critic.evaluate_sequence(
               obs,
               initial_hidden,
               episode_starts,
               actions,
               privileged_contexts,
           )
        try:
            chunk_steps = max(1, int(self.cfg.compile_policy_chunk_steps))
            if obs.shape[0] <= chunk_steps:
                return tuple(
                    value.clone()
                    for value in compiled(
                        obs,
                        initial_hidden,
                        episode_starts,
                        actions,
                        privileged_contexts,
                    )
                )

            hidden = initial_hidden
            chunks: list[tuple[torch.Tensor, ...]] = []
            for start in range(0, obs.shape[0], chunk_steps):
                end = min(start + chunk_steps, obs.shape[0])
                chunk = tuple(
                    value.clone()
                    for value in compiled(
                        obs[start:end],
                        hidden,
                        episode_starts[start:end],
                        actions[start:end],
                        privileged_contexts[start:end],
                    )
                )
                hidden = chunk[4]
                chunks.append(chunk)
            return (
                torch.cat([chunk[0] for chunk in chunks]),
                torch.cat([chunk[1] for chunk in chunks]),
                torch.cat([chunk[2] for chunk in chunks]),
                torch.cat([chunk[3] for chunk in chunks]),
                hidden,
                torch.cat([chunk[5] for chunk in chunks]),
                torch.cat([chunk[6] for chunk in chunks]),
                torch.cat([chunk[7] for chunk in chunks]),
            )
        except Exception as exc:
            if not self._policy_compile_failed:
                print(
                    "[WARN] recurrent policy compilation failed; "
                    f"falling back to eager updates: {type(exc).__name__}: {exc}"
                )
            self._policy_compile_failed = True
            self._compiled_evaluate_sequence = None
            return self.actor_critic.evaluate_sequence(
                obs,
                initial_hidden,
                episode_starts,
                actions,
                privileged_contexts,
            )

    def _rollout_action_and_value(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor,
        privileged_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the cached one-step rollout graph, with an eager fallback."""

        compiled = self._compiled_rollout_action_and_value
        if compiled is None:
            return self.actor_critic.rollout_action_and_value(
                obs, hidden_state, episode_start, privileged_context
            )
        try:
            # reduce-overhead replays CUDA graphs with reusable output
            # buffers.  These tensors survive an AMP graph invocation and an
            # environment step, so detach them from that reusable storage.
            return tuple(
                value.clone()
                for value in compiled(obs, hidden_state, episode_start, privileged_context)
            )
        except Exception as exc:
            if not self._rollout_compile_failed:
                print(
                    "[WARN] recurrent rollout compilation failed; "
                    f"falling back to eager rollout: {type(exc).__name__}: {exc}"
                )
            self._rollout_compile_failed = True
            self._compiled_rollout_action_and_value = None
            return self.actor_critic.rollout_action_and_value(
                obs, hidden_state, episode_start, privileged_context
            )

    def _rollout_amp_reward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Use the cached discriminator-reward graph, with an eager fallback."""

        compiled = self._compiled_rollout_amp_reward
        if compiled is None:
            return self.discriminator.reward(amp_obs)
        try:
            return compiled(amp_obs).clone()
        except Exception as exc:
            if not self._rollout_compile_failed:
                print(
                    "[WARN] AMP rollout compilation failed; "
                    f"falling back to eager rollout: {type(exc).__name__}: {exc}"
                )
            self._rollout_compile_failed = True
            self._compiled_rollout_amp_reward = None
            return self.discriminator.reward(amp_obs)

    def reset_training_env(self) -> None:
        super().reset_training_env()
        self.recurrent_state.zero_()
        self.episode_starts.fill_(1.0)

    def _select_behavior_action(self, action, log_prob, next_hidden, episode_starts):
        """Extension point for supervised roll-in; ordinary PPO stays on-policy."""
        return action, log_prob, torch.ones_like(log_prob)

    def collect_rollout(self) -> dict[str, torch.Tensor | np.ndarray]:
        """Collect recurrent trajectories while preserving episode boundaries."""

        num_steps = self.cfg.num_steps
        num_envs = self.env.num_envs
        obs = torch.empty((num_steps, num_envs, self.env.obs_dim), device=self.device)
        actions = torch.empty((num_steps, num_envs, self.env.action_dim), device=self.device)
        log_probs = torch.empty((num_steps, num_envs), device=self.device)
        values = torch.empty((num_steps, num_envs), device=self.device)
        rewards = torch.empty((num_steps, num_envs), device=self.device)
        task_rewards = torch.empty_like(rewards)
        amp_rewards = torch.empty_like(rewards)
        dones = torch.empty_like(rewards)
        time_outs = torch.empty_like(rewards)
        timeout_bootstrap_values = torch.zeros_like(rewards)
        successes = torch.empty_like(rewards)
        episode_starts = torch.empty_like(rewards)
        policy_sample_mask = torch.empty_like(rewards)
        amp_features = torch.empty(
            (num_steps, num_envs, self.env.amp_obs_dim),
            device=self.device,
        )
        privileged_contexts = torch.empty(
            (num_steps, num_envs, self.env.privileged_context_dim),
            device=self.device,
        )
        completed_reward_values = torch.zeros_like(rewards)
        completed_length_values = torch.zeros_like(rewards)
        episode_progress_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        termination_episode_step_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        termination_episode_step_count = torch.zeros((), dtype=torch.float32, device=self.device)
        termination_episode_step_min = torch.full((), torch.inf, dtype=torch.float32, device=self.device)
        termination_episode_step_max = torch.zeros((), dtype=torch.float32, device=self.device)
        nonfinite_task_reward_count = torch.zeros((), dtype=torch.float32, device=self.device)
        reward_component_keys: tuple[str, ...] = ()
        reward_component_sums: torch.Tensor | None = None
        initial_recurrent_state = self.recurrent_state.detach().clone()

        for step in range(num_steps):
            obs_tensor = self._as_device_tensor(self.obs)
            amp_tensor = self._as_device_tensor(self.env.amp_observe())
            privileged_context = self._as_device_tensor(
                self.env.privileged_observe()
            )
            current_episode_starts = self.episode_starts
            with torch.no_grad():
                (
                    action,
                    log_prob,
                    value,
                    next_recurrent_state,
                ) = self._rollout_action_and_value(
                    obs_tensor,
                    self.recurrent_state,
                    current_episode_starts,
                    privileged_context,
                )
                action, log_prob, on_policy = self._select_behavior_action(
                    action, log_prob, next_recurrent_state, current_episode_starts)
                amp_reward = (
                    self._rollout_amp_reward(amp_tensor)
                    if self.cfg.amp_enabled
                    else torch.zeros(num_envs, device=self.device)
                )

            env_action = action if getattr(self.env, "is_torch_env", False) else action.cpu().numpy()
            next_obs, task_reward_value, done_value, info = self.env.step(env_action)
            if "terminal_observation" in info:
                terminal_ids = self._as_device_tensor(
                    info["terminal_env_ids"]
                ).to(dtype=torch.long)
                terminal_obs = self._as_device_tensor(info["terminal_observation"])
                terminal_context = self._as_device_tensor(
                    info["terminal_privileged_context"]
                )
                with torch.no_grad():
                    terminal_value = self.actor_critic.recurrent_action_and_value(
                        terminal_obs,
                        next_recurrent_state.index_select(0, terminal_ids),
                        torch.zeros(len(terminal_ids), device=self.device),
                        privileged_context=terminal_context,
                    )[3]
                timeout_bootstrap_values[step, terminal_ids] = terminal_value
            episode_step = self._as_device_tensor(info["episode_step"])
            episode_progress_sum += episode_step.sum()
            component_items = [
                (key, value)
                for key, value in info.items()
                if key not in (
                    "episode_step", "time_outs", "successes",
                    "terminal_observation", "terminal_privileged_context",
                    "terminal_env_ids",
                )
            ]
            if not reward_component_keys:
                reward_component_keys = tuple(key for key, _value in component_items)
            component_matrix = torch.stack(
                tuple(
                    self._as_device_tensor(value).reshape(num_envs)
                    for _key, value in component_items
                ),
                dim=0,
            )
            component_matrix = torch.clamp(
                torch.nan_to_num(component_matrix, nan=0.0, posinf=0.0, neginf=0.0),
                -1.0e6,
                1.0e6,
            )
            step_component_sums = component_matrix.sum(dim=1)
            reward_component_sums = (
                step_component_sums
                if reward_component_sums is None
                else reward_component_sums + step_component_sums
            )

            task_reward = self._as_device_tensor(task_reward_value)
            done = self._as_device_tensor(done_value)
            invalid_task_reward = ~torch.isfinite(task_reward)
            nonfinite_task_reward_count += invalid_task_reward.sum()
            task_reward = torch.where(
                invalid_task_reward,
                torch.full_like(task_reward, -2.0),
                task_reward,
            )
            done = torch.maximum(done, invalid_task_reward.to(done.dtype))
            time_out = self._as_device_tensor(
                info.get("time_outs", np.zeros(num_envs, dtype=np.float32))
            )
            real_termination = (done > 0.0) & (time_out <= 0.0)
            termination_steps = episode_step.to(torch.float32)
            termination_episode_step_sum += torch.where(
                real_termination, termination_steps, torch.zeros_like(termination_steps)
            ).sum()
            termination_episode_step_count += real_termination.sum()
            termination_episode_step_min = torch.minimum(
                termination_episode_step_min,
                torch.where(
                    real_termination,
                    termination_steps,
                    torch.full_like(termination_steps, torch.inf),
                ).min(),
            )
            termination_episode_step_max = torch.maximum(
                termination_episode_step_max,
                torch.where(
                    real_termination, termination_steps, torch.zeros_like(termination_steps)
                ).max(),
            )
            success = self._as_device_tensor(info.get("successes", time_out))
            amp_reward = torch.nan_to_num(amp_reward, nan=0.0, posinf=0.0, neginf=0.0)
            if self.cfg.amp_enabled and self.cfg.amp_startup_gate:
                gait_progress = self._as_device_tensor(
                    info.get("gait_progress", np.ones(num_envs, dtype=np.float32))
                )
                amp_gate = torch.clamp(
                    torch.nan_to_num(
                        gait_progress,
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ),
                    0.0,
                    1.0,
                )
                amp_reward = amp_reward * amp_gate
            total_reward = self.cfg.task_reward_lerp * task_reward
            if self.cfg.amp_enabled:
                total_reward += (
                    (1.0 - self.cfg.task_reward_lerp)
                    * self.cfg.amp_reward_coef
                    * amp_reward
                )
            total_reward = float(max(self.cfg.reward_scale, 1.0e-6)) * total_reward
            total_reward = torch.nan_to_num(
                total_reward,
                nan=-0.5,
                posinf=-0.5,
                neginf=-0.5,
            )

            self.episode_reward_sums = torch.nan_to_num(
                self.episode_reward_sums,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            self.episode_reward_sums += total_reward
            self.episode_lengths += 1
            done_mask = done > 0.0
            completed_reward_values[step] = torch.where(
                done_mask,
                self.episode_reward_sums,
                torch.zeros_like(self.episode_reward_sums),
            )
            completed_length_values[step] = torch.where(
                done_mask,
                self.episode_lengths.to(torch.float32),
                torch.zeros_like(total_reward),
            )
            self.episode_reward_sums.masked_fill_(done_mask, 0.0)
            self.episode_lengths.masked_fill_(done_mask, 0)

            obs[step] = obs_tensor
            actions[step] = action
            log_probs[step] = log_prob
            values[step] = value
            rewards[step] = total_reward
            task_rewards[step] = task_reward
            amp_rewards[step] = amp_reward
            dones[step] = done
            time_outs[step] = time_out
            successes[step] = success
            episode_starts[step] = current_episode_starts
            policy_sample_mask[step] = on_policy
            amp_features[step] = amp_tensor
            privileged_contexts[step] = privileged_context
            self.obs = next_obs
            self.recurrent_state = next_recurrent_state.detach()
            self.episode_starts = done.detach()
            self.global_step += num_envs

        with torch.no_grad():
            next_obs_tensor = self._as_device_tensor(self.obs)
            next_privileged_context = self._as_device_tensor(
                self.env.privileged_observe()
            )
            next_value = self.actor_critic.recurrent_action_and_value(
                next_obs_tensor,
                self.recurrent_state,
                self.episode_starts,
                privileged_context=next_privileged_context,
            )[3]

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(num_steps)):
            terminal = torch.clamp(dones[step] - time_outs[step], min=0.0, max=1.0)
            next_values = next_value if step == num_steps - 1 else values[step + 1]
            next_values = torch.where(
                time_outs[step] > 0.0,
                timeout_bootstrap_values[step],
                next_values,
            )
            bootstrap_mask = 1.0 - terminal
            trace_mask = 1.0 - torch.clamp(dones[step], min=0.0, max=1.0)
            delta = (
                rewards[step]
                + self.cfg.gamma * next_values * bootstrap_mask
                - values[step]
            )
            last_gae = (
                delta
                + self.cfg.gamma
                * self.cfg.gae_lambda
                * trace_mask
                * last_gae
            )
            advantages[step] = last_gae
        returns = advantages + values

        completed_payload = torch.stack(
            (completed_reward_values, completed_length_values, dones, successes)
        ).detach().cpu().numpy()
        completed_mask = completed_payload[2] > 0.0
        completed_episode_rewards: list[np.ndarray] = []
        completed_episode_lengths: list[np.ndarray] = []
        if np.any(completed_mask):
            done_rewards = completed_payload[0][completed_mask].astype(np.float32, copy=False)
            done_lengths = completed_payload[1][completed_mask].astype(np.int32, copy=False)
            done_successes = completed_payload[3][completed_mask].astype(np.float32, copy=False)
            completed_episode_rewards.append(done_rewards)
            completed_episode_lengths.append(done_lengths)
            self.completed_episode_rewards.extend(float(value) for value in done_rewards)
            self.completed_episode_lengths.extend(int(value) for value in done_lengths)
            self.completed_episode_successes.extend(float(value) for value in done_successes)

        if reward_component_sums is None:
            reward_component_sums = torch.zeros(0, dtype=torch.float32, device=self.device)
        component_means = reward_component_sums / max(num_steps * num_envs, 1)
        has_termination = termination_episode_step_count > 0.0
        termination_episode_step_mean = torch.where(
            has_termination,
            termination_episode_step_sum / termination_episode_step_count.clamp_min(1.0),
            torch.zeros_like(termination_episode_step_sum),
        )
        termination_episode_step_min = torch.where(
            has_termination,
            termination_episode_step_min,
            torch.zeros_like(termination_episode_step_min),
        )
        summary_components = torch.cat(
            (
                component_means,
                (episode_progress_sum / max(num_steps * num_envs, 1)).reshape(1),
                torch.stack((
                    termination_episode_step_mean,
                    termination_episode_step_min,
                    termination_episode_step_max,
                )),
            )
        )
        component_values = summary_components.detach().cpu().tolist()
        reward_components = {
            key: float(component_values[index])
            for index, key in enumerate(reward_component_keys)
        }
        self._last_termination_episode_step_metrics = {
            "termination_episode_step_mean": float(component_values[-3]),
            "termination_episode_step_min": float(component_values[-2]),
            "termination_episode_step_max": float(component_values[-1]),
        }
        # This runs once per rollout after the existing summary transfer, so
        # force acceptance checks add no per-policy-tick CUDA synchronization.
        force_nonzero = reward_components.get("foot_force/nonzero_fraction")
        force_saturation = reward_components.get("foot_force/saturation_fraction")
        if self.env.is_torch_env and force_nonzero is not None and force_nonzero <= 0.0:
            raise RuntimeError("Warp foot force stayed zero for the full rollout")
        if self.env.is_torch_env and force_saturation is not None and force_saturation > 0.95:
            raise RuntimeError(
                f"Warp foot force saturation exceeded 0.95 ({force_saturation:.4f})"
            )

        return {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "returns": returns,
            "advantages": advantages,
            "task_rewards": task_rewards,
            "amp_rewards": amp_rewards,
            "dones": dones,
            "time_outs": time_outs,
            "timeout_bootstrap_values": timeout_bootstrap_values,
            "successes": successes,
            "episode_starts": episode_starts,
            "policy_sample_mask": policy_sample_mask,
            "initial_recurrent_state": initial_recurrent_state,
            "amp_features": amp_features,
            "privileged_contexts": privileged_contexts,
            "completed_episode_rewards": completed_episode_rewards,
            "completed_episode_lengths": completed_episode_lengths,
            "episode_progress": float(component_values[-4]),
            "reward_components": reward_components,
            "nonfinite_task_reward_count": nonfinite_task_reward_count,
        }

    def train_iteration(self) -> dict[str, float]:
        metrics = super().train_iteration()
        metrics.update(getattr(self, "_last_termination_episode_step_metrics", {}))
        return metrics

    def _actor_auxiliary_loss(
        self,
        obs: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        env_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        """Extension point for isolated trainers; direct training adds nothing."""

        del obs, initial_hidden, episode_starts, env_indices
        return None

    def update_policy(
        self,
        rollout: dict[str, torch.Tensor | np.ndarray],
    ) -> dict[str, float]:
        """Update whole environment sequences with truncated BPTT."""

        obs = rollout["obs"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        returns = rollout["returns"]
        old_values = rollout["values"]
        episode_starts = rollout["episode_starts"]
        initial_hidden = rollout["initial_recurrent_state"]
        privileged_contexts = rollout["privileged_contexts"]
        advantages = rollout["advantages"]
        policy_sample_mask = rollout.get("policy_sample_mask", torch.ones_like(advantages))
        assert all(torch.is_tensor(value) for value in (
            obs,
            actions,
            old_log_probs,
            returns,
            old_values,
            episode_starts,
            initial_hidden,
            privileged_contexts,
            advantages,
        ))
        finite_envs = (
            torch.isfinite(obs).all(dim=(0, 2))
            & torch.isfinite(actions).all(dim=(0, 2))
            & torch.isfinite(old_log_probs).all(dim=0)
            & torch.isfinite(returns).all(dim=0)
            & torch.isfinite(old_values).all(dim=0)
            & torch.isfinite(initial_hidden).all(dim=1)
            & torch.isfinite(privileged_contexts).all(dim=(0, 2))
            & torch.isfinite(advantages).all(dim=0)
        )
        dropped_sample_count = int(
            ((~finite_envs).sum() * obs.shape[0]).detach().cpu()
        )
        obs = obs[:, finite_envs]
        actions = actions[:, finite_envs]
        old_log_probs = old_log_probs[:, finite_envs]
        returns = returns[:, finite_envs]
        old_values = old_values[:, finite_envs]
        episode_starts = episode_starts[:, finite_envs]
        initial_hidden = initial_hidden[finite_envs]
        privileged_contexts = privileged_contexts[:, finite_envs]
        advantages = advantages[:, finite_envs]
        policy_sample_mask = policy_sample_mask[:, finite_envs]
        env_batch_size = obs.shape[1]
        minibatches = max(1, min(self.cfg.num_minibatches, env_batch_size))
        metric_names = (
            "policy_loss",
            "value_loss",
            "median_value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "bpo_ratio_error",
            "bpo_target_ratio",
            "bpo_policy_weight",
            "bpo_median_advantage_abs",
            "gate_change_loss",
            "expert_load_balance_loss",
            "residual_l2_loss",
            "privileged_context_loss",
        )
        if env_batch_size == 0:
            metrics = {key: 0.0 for key in metric_names}
            metrics["learning_rate"] = self._policy_learning_rate()
            metrics["policy_updates_skipped_nonfinite"] = 0.0
            metrics["policy_samples_dropped_nonfinite"] = float(dropped_sample_count)
            metrics["importance_ratio_log_clip_fraction"] = 0.0
            return metrics

        metric_sums = torch.zeros(len(metric_names), dtype=torch.float32, device=self.device)
        update_count = 0
        skipped_update_count = 0
        clipped_ratio_count = torch.zeros((), dtype=torch.long, device=self.device)
        ratio_sample_count = 0
        bpo_lambda = float(max(self.cfg.bpo_lambda, 1.0e-6))
        ratio_eps = float(max(self.cfg.clip_coef, 0.0))
        log_ratio_limit = float(max(self.cfg.importance_ratio_log_clip, 1.0))
        stop_policy_update = False
        kl_stop = (
            float(max(self.cfg.target_kl, 0.0))
            * float(max(self.cfg.target_kl_stop_multiplier, 1.0))
        )

        for _epoch in range(self.cfg.update_epochs):
            epoch_kl_sum = torch.zeros((), dtype=torch.float32, device=self.device)
            epoch_update_count = 0
            env_indices = torch.randperm(env_batch_size, device=self.device)
            for mb in torch.tensor_split(env_indices, minibatches):
                (
                    new_log_prob,
                    entropy,
                    new_value,
                    median_value,
                    _next_hidden,
                    gate_weights,
                    residual,
                    context_prediction,
                ) = self._evaluate_policy_sequence(
                    obs[:, mb],
                    initial_hidden[mb],
                    episode_starts[:, mb],
                    actions[:, mb],
                    privileged_contexts[:, mb],
                )
                mb_returns = returns[:, mb]
                mb_old_values = old_values[:, mb]
                raw_log_ratio = new_log_prob - old_log_probs[:, mb]
                finite_log_ratio = torch.isfinite(raw_log_ratio)
                clipped_ratio_count += (
                    (raw_log_ratio.abs() > log_ratio_limit) & finite_log_ratio
                ).sum()
                ratio_sample_count += int(raw_log_ratio.numel())
                log_ratio = torch.clamp(
                    raw_log_ratio,
                    -log_ratio_limit,
                    log_ratio_limit,
                )
                ratio = log_ratio.exp()
                sample_mask = policy_sample_mask[:, mb]
                def policy_mean(value):
                    return (value * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)

                if self.cfg.deepmimic_ppo:
                    # MimicKit PPO: min(A*r, A*clip(r,1-eps,1+eps)).
                    mb_advantage = advantages[:, mb]
                    advantage_mean = policy_mean(mb_advantage)
                    advantage_std = policy_mean((mb_advantage-advantage_mean).square()).sqrt()
                    normalized_advantage = (
                        (mb_advantage-advantage_mean) / advantage_std.clamp_min(1e-6)
                    ).clamp(-4.0, 4.0)
                    policy_loss = -policy_mean(torch.minimum(
                        normalized_advantage * ratio,
                        normalized_advantage * ratio.clamp(
                            1.0 - ratio_eps, 1.0 + ratio_eps
                        ),
                    ))
                    median_advantage = mb_returns - median_value
                    target_ratio = torch.ones_like(ratio)
                    policy_weight = torch.ones_like(ratio)
                    ratio_error = (ratio - 1.0).abs()
                else:
                    with torch.no_grad():
                        median_advantage = mb_returns - median_value
                        target_ratio = 1.0 + ratio_eps * torch.tanh(
                            median_advantage / (2.0 * bpo_lambda)
                        )
                        policy_weight = (
                            (mb_returns - new_value).abs()
                            + float(max(self.cfg.bpo_tv_coef, 0.0))
                        )
                    ratio_error = (target_ratio - ratio).abs()
                    policy_loss = policy_mean(ratio_error * policy_weight)

                value_clip = float(max(self.cfg.value_clip_coef, 0.0))
                value_pred_clipped = (
                    mb_old_values
                    + torch.clamp(
                        new_value - mb_old_values,
                        -value_clip,
                        value_clip,
                    )
                    if value_clip > 0.0
                    else new_value
                )
                value_losses = (new_value - mb_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(
                    value_losses,
                    value_losses_clipped,
                ).mean()
                median_value_loss = (
                    policy_loss.new_zeros(()) if self.cfg.deepmimic_ppo else
                    (
                        bpo_lambda
                        * self._bpo_soft_abs(
                            (mb_returns.detach() - median_value) / bpo_lambda
                        )
                    ).mean()
                )
                entropy_loss = policy_mean(entropy)

                if self.cfg.use_moe and gate_weights.shape[0] > 1:
                    valid_transition = 1.0 - episode_starts[1:, mb]
                    gate_delta = (gate_weights[1:] - gate_weights[:-1]).square().mean(dim=-1)
                    gate_change_loss = (
                        (gate_delta * valid_transition).sum()
                        / valid_transition.sum().clamp_min(1.0)
                    )
                else:
                    gate_change_loss = gate_weights.new_zeros(())
                if self.cfg.use_moe:
                    mean_expert_load = gate_weights.mean(dim=(0, 1))
                    expert_load_balance_loss = (
                        self.cfg.num_experts * mean_expert_load.square().sum() - 1.0
                    )
                    residual_l2_loss = residual.square().mean()
                else:
                    expert_load_balance_loss = policy_loss.new_zeros(())
                    residual_l2_loss = policy_loss.new_zeros(())
                context_loss_elements = F.smooth_l1_loss(
                    context_prediction,
                    privileged_contexts[:, mb],
                    reduction="none",
                )
                privileged_context_loss = (
                    context_loss_elements
                    * self.privileged_context_loss_weights.reshape(1, 1, -1)
                ).sum() / (
                    self.privileged_context_loss_weights.sum()
                    * context_loss_elements.shape[0]
                    * context_loss_elements.shape[1]
                )
                auxiliary_actor_loss = self._actor_auxiliary_loss(
                    obs[:, mb],
                    initial_hidden[mb],
                    episode_starts[:, mb],
                    mb,
                )

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    + self.cfg.median_value_coef * median_value_loss
                    - self.cfg.entropy_coef * entropy_loss
                    + self.cfg.gate_change_coef * gate_change_loss
                    + self.cfg.expert_load_balance_coef * expert_load_balance_loss
                    + self.cfg.residual_l2_coef * residual_l2_loss
                    + self.cfg.privileged_context_loss_coef
                    * privileged_context_loss
                )
                if auxiliary_actor_loss is not None:
                    loss = loss + auxiliary_actor_loss
                stepped = self._optimizer_step_if_finite(
                    loss,
                    self.actor_critic,
                    self.optimizer,
                    self.cfg.max_grad_norm,
                )
                if not stepped:
                    skipped_update_count += 1
                    continue
                self.actor_critic.clamp_log_std_()

                with torch.no_grad():
                    approx_kl = policy_mean((ratio - 1.0) - log_ratio)
                    clip_fraction = policy_mean(((ratio - 1.0).abs() > ratio_eps).float())
                    if (
                        kl_stop > 0.0
                        and torch.isfinite(approx_kl)
                        and float(approx_kl.detach().cpu()) > kl_stop
                    ):
                        stop_policy_update = True
                metric_sums += torch.stack(
                    (
                        policy_loss,
                        value_loss,
                        median_value_loss,
                        entropy_loss,
                        approx_kl,
                        clip_fraction,
                        ratio_error.mean(),
                        target_ratio.mean(),
                        policy_weight.mean(),
                        median_advantage.abs().mean(),
                        gate_change_loss,
                        expert_load_balance_loss,
                        residual_l2_loss,
                        privileged_context_loss,
                    )
                ).detach()
                update_count += 1
                epoch_kl_sum += approx_kl.detach()
                epoch_update_count += 1
                if stop_policy_update:
                    break

            mean_epoch_kl = (
                float((epoch_kl_sum / epoch_update_count).cpu())
                if epoch_update_count > 0
                else 0.0
            )
            self._adapt_policy_learning_rate(mean_epoch_kl)
            if (
                stop_policy_update
                or (
                    self.cfg.target_kl > 0.0
                    and epoch_update_count > 0
                    and mean_epoch_kl > self.cfg.target_kl
                )
            ):
                break

        metric_values = (metric_sums / max(update_count, 1)).cpu().tolist()
        metrics = {
            key: float(metric_values[index])
            for index, key in enumerate(metric_names)
        }
        metrics["learning_rate"] = self._policy_learning_rate()
        metrics["policy_updates_skipped_nonfinite"] = float(skipped_update_count)
        metrics["policy_samples_dropped_nonfinite"] = float(dropped_sample_count)
        metrics["importance_ratio_log_clip_fraction"] = float(
            clipped_ratio_count.detach().cpu() / max(ratio_sample_count, 1)
        )
        return metrics

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> int:
        checkpoint = torch.load(Path(path), map_location=self.device)
        architecture = checkpoint.get("sensor_policy_architecture")
        if not isinstance(architecture, dict) or not bool(architecture.get("recurrent", False)):
            raise ValueError(
                "Checkpoint uses the legacy feed-forward sensor policy; recurrent MoE "
                "training must start in a new run instead of resuming that checkpoint."
            )
        saved_context_dim = int(architecture.get("privileged_context_dim", 0))
        # Old recurrent sensor checkpoints predate this switch and use MoE.
        saved_use_moe = bool(architecture.get("use_moe", True))
        if saved_use_moe != self.cfg.use_moe:
            raise ValueError(
                "Checkpoint MoE architecture does not match this run: "
                f"saved use_moe={saved_use_moe}, current={self.cfg.use_moe}."
            )
        if saved_context_dim != self.env.privileged_context_dim:
            raise ValueError(
                "Checkpoint has no compatible implicit-terrain estimator: "
                f"saved privileged_context_dim={saved_context_dim}, "
                f"current={self.env.privileged_context_dim}. Start a new run."
            )
        saved_history_frames = int(architecture.get("actor_history_frames", 1))
        if saved_history_frames != self.actor_history_frames:
            raise ValueError(
                "Checkpoint Actor sensor-history input is incompatible: "
                f"saved actor_history_frames={saved_history_frames}, "
                f"current={self.actor_history_frames}. Older checkpoints read only "
                "the latest frame and cannot be resumed into the multi-frame GRU; "
                "start a new run."
            )
        saved_metadata = checkpoint.get("sensor_observation")
        if saved_metadata is None:
            raise ValueError(
                "Checkpoint has no sensor_observation metadata; use a checkpoint produced by sensor_training."
            )
        current_metadata = self.env.observation_metadata
        if not _sensor_metadata_compatible(saved_metadata, current_metadata):
            raise ValueError(
                "Checkpoint sensor observation layout does not match the current sensor_training layout."
            )
        next_iteration = super().load(path, load_optimizers=load_optimizers)
        self.recurrent_state.zero_()
        self.episode_starts.fill_(1.0)
        return next_iteration

    def _checkpoint_payload(self, iteration: int) -> dict:
        checkpoint = super()._checkpoint_payload(iteration)
        checkpoint.update({
            "sensor_observation": self.env.observation_metadata,
            "sensor_policy_architecture": {
                "name": (
                    "proprioceptive_history_gru_context_residual_moe"
                    if self.cfg.use_moe
                    else "proprioceptive_history_gru_context"
                ),
                "recurrent": True,
                "use_moe": self.cfg.use_moe,
                "sensor_frame_dim": self.env.sensor_frame_dim,
                "actor_history_frames": self.actor_history_frames,
                "recurrent_input_dim": self.actor_critic.recurrent_input_dim,
                "recurrent_hidden_dim": self.cfg.recurrent_hidden_dim,
                "num_experts": self.cfg.num_experts,
                "gate_hidden_dims": list(self.cfg.gate_hidden_dims),
                "expert_hidden_dims": list(self.cfg.expert_hidden_dims),
                "residual_scale": self.cfg.residual_scale,
                "privileged_context_dim": self.env.privileged_context_dim,
                "privileged_context_fields": list(
                    self.env.privileged_context_metadata["fields"]
                ),
                "privileged_context_actor_input": False,
                "asymmetric_critic": True,
                "recurrent_state_reset": "episode_start",
            },
        })
        return checkpoint

    def _export_onnx(self, path: Path, checkpoint: dict) -> None:
        """Export recurrent sensor policy with explicit hidden-state I/O."""

        import onnx

        # CUDA GRUCell is traced as aten::_thnn_fused_gru_cell, which the
        # legacy opset-17 exporter cannot lower.  Export an isolated CPU copy;
        # this leaves the live CUDA policy and optimizer untouched.
        export_policy = copy.deepcopy(self.actor_critic).to("cpu").eval()
        export_actor = RecurrentSensorActorExport(export_policy).eval()
        dummy_observation = torch.zeros(
            (1, self.env.obs_dim),
            dtype=torch.float32,
        )
        dummy_state = export_policy.initial_state(1)
        dummy_episode_start = torch.ones(
            1,
            dtype=torch.float32,
        )
        torch.onnx.export(
            export_actor,
            (dummy_observation, dummy_state, dummy_episode_start),
            path,
            input_names=("observation", "recurrent_state", "episode_start"),
            output_names=("action", "next_recurrent_state"),
            dynamic_axes={
                "observation": {0: "batch"},
                "recurrent_state": {0: "batch"},
                "episode_start": {0: "batch"},
                "action": {0: "batch"},
                "next_recurrent_state": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

        model = onnx.load(path)
        metadata = {
            "format_version": 2,
            "iteration": checkpoint["iteration"],
            "global_step": checkpoint["global_step"],
            "obs_dim": checkpoint["obs_dim"],
            "action_dim": checkpoint["action_dim"],
            "amp_obs_dim": checkpoint["amp_obs_dim"],
            "ppo_amp_config": checkpoint["ppo_amp_config"],
            "env_config": checkpoint["env_config"],
            "common_joint_names": checkpoint["common_joint_names"],
            "sensor_observation": checkpoint["sensor_observation"],
            "sensor_policy_architecture": checkpoint["sensor_policy_architecture"],
        }
        onnx.helper.set_model_props(
            model,
            {"lytrobot_amp.metadata": json.dumps(metadata, ensure_ascii=False)},
        )
        onnx.checker.check_model(model)
        onnx.save(model, path)
