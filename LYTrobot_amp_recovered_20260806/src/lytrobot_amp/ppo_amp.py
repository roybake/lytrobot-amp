"""Bounded Policy Optimization with an AMP discriminator for the LYT robot environment."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .env import LytRobotAmpEnv
from .networks import ActorCritic, AmpDiscriminator


@dataclass
class PPOAMPConfig:
    seed: int = 1
    learning_rate: float = 3.0e-4
    discriminator_learning_rate: float = 3.0e-5
    reward_scale: float = 0.25
    num_steps: int = 24
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    bpo_lambda: float = 0.001
    bpo_tv_coef: float = 0.0
    value_clip_coef: float = 1.0
    entropy_coef: float = 0.002
    value_coef: float = 0.5
    median_value_coef: float = 0.5
    max_grad_norm: float = 1.0
    importance_ratio_log_clip: float = 20.0
    target_kl: float = 0.0
    target_kl_stop_multiplier: float = 2.0
    adaptive_lr: bool = True
    min_learning_rate: float = 1.0e-5
    max_learning_rate: float = 3.0e-4
    kl_adapt_factor: float = 1.5
    amp_enabled: bool = True
    amp_reward_coef: float = 1.20
    task_reward_lerp: float = 0.50
    amp_discriminator_updates: int = 1
    amp_grad_penalty_coef: float = 2.0
    amp_minibatch_size: int = 1024
    amp_label_smoothing: float = 0.20
    amp_obs_noise_std: float = 0.05
    amp_mirror_ref_prob: float = 0.50
    actor_hidden_dims: tuple[int, ...] = (256, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    discriminator_hidden_dims: tuple[int, ...] = (256, 128)
    init_std: float = 0.35
    min_action_std: float = 0.05
    max_action_std: float = 0.60
    episode_metric_window: int = 4096


class PPOAMPTrainer:
    def __init__(
        self,
        env: LytRobotAmpEnv,
        cfg: PPOAMPConfig,
        device: str,
        log_dir: str | Path,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.rng = np.random.default_rng(cfg.seed + 1000)

        self.actor_critic = ActorCritic(
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
        self.completed_episode_successes: deque[float] = deque(maxlen=window)
        self.loaded_domain_randomization_scale: float | None = None

    def reset_training_env(self) -> None:
        """Synchronously reset every world and discard partial episode totals."""

        self.obs = self.env.reset()
        self.episode_reward_sums.zero_()
        self.episode_lengths.zero_()

    def _policy_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _set_policy_learning_rate(self, learning_rate: float) -> None:
        learning_rate = float(learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _adapt_policy_learning_rate(self, approx_kl: float) -> None:
        if not self.cfg.adaptive_lr or self.cfg.target_kl <= 0.0 or not np.isfinite(approx_kl):
            return
        factor = float(max(self.cfg.kl_adapt_factor, 1.01))
        current_lr = self._policy_learning_rate()
        min_lr = float(max(self.cfg.min_learning_rate, 0.0))
        max_lr = float(max(self.cfg.max_learning_rate, min_lr))
        target_kl = float(self.cfg.target_kl)
        if approx_kl > target_kl:
            self._set_policy_learning_rate(max(min_lr, current_lr / factor))
        elif 0.0 < approx_kl < 0.25 * target_kl:
            self._set_policy_learning_rate(min(max_lr, current_lr * factor))

    def _as_device_tensor(self, value: torch.Tensor | np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    @staticmethod
    def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _bpo_soft_abs(x: torch.Tensor) -> torch.Tensor:
        return -0.5 * x + F.softplus(x)

    @staticmethod
    def _optimizer_step_if_finite(
        loss: torch.Tensor,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
    ) -> bool:
        """Apply one update without allowing a non-finite gradient to poison parameters."""

        optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss.detach()).item()):
            return False

        loss.backward()
        try:
            nn.utils.clip_grad_norm_(
                module.parameters(),
                max_grad_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as exc:
            optimizer.zero_grad(set_to_none=True)
            if "non-finite" not in str(exc).lower():
                raise
            return False

        optimizer.step()
        return True

    @staticmethod
    def _nonfinite_tensor_names(value: object, prefix: str = "") -> list[str]:
        names: list[str] = []
        if torch.is_tensor(value):
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                names.append(prefix or "<tensor>")
        elif isinstance(value, dict):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                names.extend(PPOAMPTrainer._nonfinite_tensor_names(item, child))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                names.extend(PPOAMPTrainer._nonfinite_tensor_names(item, f"{prefix}[{index}]"))
        return names

    @classmethod
    def _assert_checkpoint_finite(cls, checkpoint: dict) -> None:
        bad_names: list[str] = []
        for key in (
            "actor_critic",
            "discriminator",
            "optimizer",
            "discriminator_optimizer",
        ):
            if key in checkpoint:
                bad_names.extend(cls._nonfinite_tensor_names(checkpoint[key], key))
        if bad_names:
            preview = ", ".join(bad_names[:6])
            if len(bad_names) > 6:
                preview += f", ... ({len(bad_names)} tensors total)"
            raise ValueError(f"Refusing non-finite training checkpoint: {preview}")

    def collect_rollout(self) -> dict[str, torch.Tensor | np.ndarray]:
        num_steps = self.cfg.num_steps
        num_envs = self.env.num_envs
        obs_dim = self.env.obs_dim
        action_dim = self.env.action_dim
        amp_dim = self.env.amp_obs_dim

        obs = torch.empty((num_steps, num_envs, obs_dim), device=self.device)
        actions = torch.empty((num_steps, num_envs, action_dim), device=self.device)
        log_probs = torch.empty((num_steps, num_envs), device=self.device)
        values = torch.empty((num_steps, num_envs), device=self.device)
        rewards = torch.empty((num_steps, num_envs), device=self.device)
        task_rewards = torch.empty((num_steps, num_envs), device=self.device)
        amp_rewards = torch.empty((num_steps, num_envs), device=self.device)
        dones = torch.empty((num_steps, num_envs), device=self.device)
        time_outs = torch.empty((num_steps, num_envs), device=self.device)
        successes = torch.empty((num_steps, num_envs), device=self.device)
        amp_features = torch.empty((num_steps, num_envs, amp_dim), device=self.device)
        completed_reward_values = torch.zeros((num_steps, num_envs), device=self.device)
        completed_length_values = torch.zeros((num_steps, num_envs), device=self.device)
        episode_progress_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        nonfinite_task_reward_count = torch.zeros((), dtype=torch.float32, device=self.device)
        reward_component_keys: tuple[str, ...] = ()
        reward_component_sums: torch.Tensor | None = None

        for step in range(num_steps):
            obs_tensor = self._as_device_tensor(self.obs)
            amp_tensor = self._as_device_tensor(self.env.amp_observe())
            with torch.no_grad():
                action, log_prob, _, value = self.actor_critic.get_action_and_value(obs_tensor)
                if self.cfg.amp_enabled:
                    amp_reward = self.discriminator.reward(amp_tensor)
                else:
                    amp_reward = torch.zeros(num_envs, device=self.device)

            env_action = action if getattr(self.env, "is_torch_env", False) else action.cpu().numpy()
            next_obs, task_reward_value, done_value, _info = self.env.step(env_action)
            episode_step = self._as_device_tensor(_info["episode_step"])
            episode_progress_sum = episode_progress_sum + episode_step.sum()
            component_items = [
                (key, info_value)
                for key, info_value in _info.items()
                if key not in ("episode_step", "time_outs", "successes")
            ]
            if not reward_component_keys:
                reward_component_keys = tuple(key for key, _value in component_items)
            component_matrix = torch.stack(
                tuple(self._as_device_tensor(value).reshape(num_envs) for _key, value in component_items),
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
            time_out = self._as_device_tensor(_info.get("time_outs", np.zeros(num_envs, dtype=np.float32)))
            success = self._as_device_tensor(_info.get("successes", time_out))
            amp_reward = torch.nan_to_num(amp_reward, nan=0.0, posinf=0.0, neginf=0.0)
            total_reward = self.cfg.task_reward_lerp * task_reward
            if self.cfg.amp_enabled:
                total_reward = total_reward + (1.0 - self.cfg.task_reward_lerp) * self.cfg.amp_reward_coef * amp_reward
            total_reward = float(max(self.cfg.reward_scale, 1.0e-6)) * total_reward
            total_reward = torch.nan_to_num(total_reward, nan=-0.5, posinf=-0.5, neginf=-0.5)
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
                done_mask, self.episode_reward_sums, torch.zeros_like(self.episode_reward_sums)
            )
            completed_length_values[step] = torch.where(
                done_mask, self.episode_lengths.to(torch.float32), torch.zeros_like(total_reward)
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
            amp_features[step] = amp_tensor
            self.obs = next_obs
            self.global_step += num_envs

        with torch.no_grad():
            next_obs_tensor = self._as_device_tensor(self.obs)
            next_value = self.actor_critic.get_action_and_value(next_obs_tensor)[3]

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(num_steps)):
            terminal = torch.clamp(dones[step] - time_outs[step], min=0.0, max=1.0)
            if step == num_steps - 1:
                next_non_terminal = 1.0 - terminal
                next_values = next_value
            else:
                next_non_terminal = 1.0 - terminal
                next_values = values[step + 1]
            delta = rewards[step] + self.cfg.gamma * next_values * next_non_terminal - values[step]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae
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
        summary_components = torch.cat(
            (
                component_means,
                (episode_progress_sum / max(num_steps * num_envs, 1)).reshape(1),
            )
        )
        component_values = summary_components.detach().cpu().tolist()
        episode_progress = float(component_values[-1])
        reward_components = {
            key: float(component_values[index]) for index, key in enumerate(reward_component_keys)
        }

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
            "successes": successes,
            "amp_features": amp_features,
            "completed_episode_rewards": completed_episode_rewards,
            "completed_episode_lengths": completed_episode_lengths,
            "episode_progress": episode_progress,
            "reward_components": reward_components,
            "nonfinite_task_reward_count": nonfinite_task_reward_count,
        }

    def update_policy(self, rollout: dict[str, torch.Tensor | np.ndarray]) -> dict[str, float]:
        obs = rollout["obs"].reshape((-1, self.env.obs_dim))
        actions = rollout["actions"].reshape((-1, self.env.action_dim))
        old_log_probs = rollout["log_probs"].reshape(-1)
        returns = rollout["returns"].reshape(-1)
        old_values = rollout["values"].reshape(-1)

        finite_samples = (
            torch.isfinite(obs).all(dim=1)
            & torch.isfinite(actions).all(dim=1)
            & torch.isfinite(old_log_probs)
            & torch.isfinite(returns)
            & torch.isfinite(old_values)
        )
        dropped_sample_count = int((~finite_samples).sum().detach().cpu())
        if dropped_sample_count > 0:
            obs = obs[finite_samples]
            actions = actions[finite_samples]
            old_log_probs = old_log_probs[finite_samples]
            returns = returns[finite_samples]
            old_values = old_values[finite_samples]

        batch_size = obs.shape[0]
        minibatches = max(1, min(self.cfg.num_minibatches, batch_size))
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
        )
        if batch_size == 0:
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
        kl_stop = float(max(self.cfg.target_kl, 0.0)) * float(max(self.cfg.target_kl_stop_multiplier, 1.0))
        for _epoch in range(self.cfg.update_epochs):
            epoch_kl_sum = torch.zeros((), dtype=torch.float32, device=self.device)
            epoch_update_count = 0
            indices = torch.randperm(batch_size, device=self.device)
            for mb in torch.tensor_split(indices, minibatches):
                _, new_log_prob, entropy, new_value = self.actor_critic.get_action_and_value(obs[mb], actions[mb])
                median_value = self.actor_critic.median_value(obs[mb])
                raw_log_ratio = new_log_prob - old_log_probs[mb]
                finite_log_ratio = torch.isfinite(raw_log_ratio)
                clipped_ratio_count += (
                    (raw_log_ratio.abs() > log_ratio_limit) & finite_log_ratio
                ).sum()
                ratio_sample_count += int(raw_log_ratio.numel())
                log_ratio = torch.clamp(raw_log_ratio, -log_ratio_limit, log_ratio_limit)
                ratio = log_ratio.exp()

                with torch.no_grad():
                    median_advantage = returns[mb] - median_value
                    target_ratio = 1.0 + ratio_eps * torch.tanh(median_advantage / (2.0 * bpo_lambda))
                    policy_weight = (returns[mb] - new_value).abs() + float(max(self.cfg.bpo_tv_coef, 0.0))
                ratio_error = (target_ratio - ratio).abs()
                policy_loss = (ratio_error * policy_weight).mean()

                value_clip = float(max(self.cfg.value_clip_coef, 0.0))
                if value_clip > 0.0:
                    value_pred_clipped = old_values[mb] + torch.clamp(
                        new_value - old_values[mb], -value_clip, value_clip
                    )
                else:
                    value_pred_clipped = new_value
                value_losses = (new_value - returns[mb]).pow(2)
                value_losses_clipped = (value_pred_clipped - returns[mb]).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                median_value_loss = (
                    bpo_lambda
                    * self._bpo_soft_abs((returns[mb].detach() - median_value) / bpo_lambda)
                ).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    + self.cfg.median_value_coef * median_value_loss
                    - self.cfg.entropy_coef * entropy_loss
                )
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
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > ratio_eps).float().mean()
                    if kl_stop > 0.0 and torch.isfinite(approx_kl) and float(approx_kl.detach().cpu()) > kl_stop:
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
                    )
                ).detach()
                update_count += 1
                epoch_kl_sum += approx_kl.detach()
                epoch_update_count += 1
                if stop_policy_update:
                    break

            mean_epoch_kl = (
                float((epoch_kl_sum / epoch_update_count).cpu()) if epoch_update_count > 0 else 0.0
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
        metrics = {key: float(metric_values[index]) for index, key in enumerate(metric_names)}
        metrics["learning_rate"] = self._policy_learning_rate()
        metrics["policy_updates_skipped_nonfinite"] = float(skipped_update_count)
        metrics["policy_samples_dropped_nonfinite"] = float(dropped_sample_count)
        metrics["importance_ratio_log_clip_fraction"] = float(
            clipped_ratio_count.detach().cpu() / max(ratio_sample_count, 1)
        )
        return metrics

    def update_discriminator(self, rollout: dict[str, torch.Tensor | np.ndarray]) -> dict[str, float]:
        if not self.cfg.amp_enabled:
            return {"amp_loss": 0.0, "amp_grad_penalty": 0.0, "amp_agent_prob": 0.0, "amp_ref_prob": 0.0}

        agent_features = rollout["amp_features"].reshape((-1, self.env.amp_obs_dim)).detach()
        batch_size = agent_features.shape[0]
        amp_batch_size = max(1, min(self.cfg.amp_minibatch_size, batch_size))
        metric_names = ("amp_loss", "amp_grad_penalty", "amp_agent_prob", "amp_ref_prob")
        metric_sums = torch.zeros(len(metric_names), dtype=torch.float32, device=self.device)
        updates = max(1, self.cfg.amp_discriminator_updates)
        completed_updates = 0
        skipped_updates = 0

        for _ in range(updates):
            idx = torch.as_tensor(
                self.rng.choice(batch_size, size=amp_batch_size, replace=batch_size < amp_batch_size),
                dtype=torch.long,
                device=self.device,
            )
            agent_batch = agent_features[idx]
            ref_np = self.env.motion.sample_features(
                amp_batch_size,
                self.rng,
                mirror_prob=self.cfg.amp_mirror_ref_prob,
            )
            ref_batch = torch.as_tensor(ref_np, dtype=torch.float32, device=self.device)

            noisy_agent_batch = self._amp_noisy_batch(agent_batch)
            noisy_ref_batch = self._amp_noisy_batch(ref_batch)
            logits_agent = self.discriminator(noisy_agent_batch)
            logits_ref = self.discriminator(noisy_ref_batch)
            smoothing = float(np.clip(self.cfg.amp_label_smoothing, 0.0, 0.45))
            agent_target = torch.full_like(logits_agent, smoothing)
            ref_target = torch.full_like(logits_ref, 1.0 - smoothing)
            loss_agent = F.binary_cross_entropy_with_logits(logits_agent, agent_target)
            loss_ref = F.binary_cross_entropy_with_logits(logits_ref, ref_target)
            amp_loss = 0.5 * (loss_agent + loss_ref)
            grad_penalty = self._gradient_penalty(agent_batch, ref_batch)
            loss = amp_loss + self.cfg.amp_grad_penalty_coef * grad_penalty

            stepped = self._optimizer_step_if_finite(
                loss,
                self.discriminator,
                self.discriminator_optimizer,
                self.cfg.max_grad_norm,
            )
            if not stepped:
                skipped_updates += 1
                continue

            with torch.no_grad():
                metric_sums += torch.stack(
                    (
                        amp_loss,
                        grad_penalty,
                        torch.sigmoid(logits_agent).mean(),
                        torch.sigmoid(logits_ref).mean(),
                    )
                ).detach()
            completed_updates += 1

        metric_values = (metric_sums / max(completed_updates, 1)).cpu().tolist()
        metrics = {key: float(metric_values[index]) for index, key in enumerate(metric_names)}
        metrics["amp_updates_skipped_nonfinite"] = float(skipped_updates)
        return metrics

    def _amp_noisy_batch(self, batch: torch.Tensor) -> torch.Tensor:
        noise_std = float(max(self.cfg.amp_obs_noise_std, 0.0))
        if noise_std <= 0.0:
            return batch
        return batch + noise_std * torch.randn_like(batch)

    def _gradient_penalty(self, agent_batch: torch.Tensor, ref_batch: torch.Tensor) -> torch.Tensor:
        if self.cfg.amp_grad_penalty_coef <= 0.0:
            return torch.zeros((), device=self.device)
        penalty_batch = torch.cat((agent_batch, ref_batch), dim=0).detach().requires_grad_(True)
        logits = self.discriminator(penalty_batch)
        gradients = torch.autograd.grad(
            logits.sum(),
            penalty_batch,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.pow(2).sum(dim=1).mean()

    def train_iteration(self) -> dict[str, float]:
        iteration_start = time.perf_counter()
        rollout = self.collect_rollout()
        rollout_end = time.perf_counter()
        policy_metrics = self.update_policy(rollout)
        policy_end = time.perf_counter()
        amp_metrics = self.update_discriminator(rollout)
        amp_end = time.perf_counter()

        reward = rollout["rewards"]
        task_reward = rollout["task_rewards"]
        amp_reward = rollout["amp_rewards"]
        dones = rollout["dones"]
        time_outs = rollout["time_outs"]
        successes = rollout["successes"]
        completed_episode_rewards = rollout["completed_episode_rewards"]
        completed_episode_lengths = rollout["completed_episode_lengths"]
        episode_progress = float(rollout.get("episode_progress", 0.0))
        nonfinite_task_reward_count = rollout.get("nonfinite_task_reward_count", 0.0)
        if torch.is_tensor(nonfinite_task_reward_count):
            nonfinite_task_reward_count = float(nonfinite_task_reward_count.detach().cpu())
        else:
            nonfinite_task_reward_count = float(nonfinite_task_reward_count)
        if len(completed_episode_rewards) > 0:
            rollout_episode_rewards = np.concatenate(completed_episode_rewards).astype(np.float32, copy=False)
            rollout_episode_reward = float(rollout_episode_rewards.mean())
        else:
            rollout_episode_reward = 0.0
        if len(completed_episode_lengths) > 0:
            rollout_episode_lengths = np.concatenate(completed_episode_lengths).astype(np.float32, copy=False)
            rollout_episode_length = float(rollout_episode_lengths.mean())
        else:
            rollout_episode_length = 0.0
        mean_episode_reward = (
            float(np.mean(self.completed_episode_rewards)) if len(self.completed_episode_rewards) > 0 else 0.0
        )
        mean_episode_length = (
            float(np.mean(self.completed_episode_lengths)) if len(self.completed_episode_lengths) > 0 else 0.0
        )
        mean_episode_success_rate = (
            float(np.mean(self.completed_episode_successes))
            if len(self.completed_episode_successes) > 0
            else 0.0
        )
        completed_count = dones.sum()
        successful_count = successes.sum()
        rollout_episode_success_rate = torch.where(
            completed_count > 0.0,
            successful_count / completed_count.clamp_min(1.0),
            torch.zeros_like(completed_count),
        )
        termination_fraction_tensor = torch.clamp(dones - time_outs, 0.0, 1.0).mean()
        summary_values = torch.stack(
            (
                reward.mean(),
                task_reward.mean(),
                amp_reward.mean(),
                dones.mean(),
                time_outs.mean(),
                termination_fraction_tensor,
                completed_count,
                successful_count,
                rollout_episode_success_rate,
            )
        ).detach().cpu().tolist()
        (
            rollout_reward,
            mean_task_reward,
            mean_amp_reward,
            done_fraction,
            timeout_fraction,
            termination_fraction,
            completed_episode_count,
            successful_episode_count,
            rollout_episode_success_rate_value,
        ) = map(float, summary_values)
        rollout_seconds = max(rollout_end - iteration_start, 1.0e-9)
        env_steps_per_second = float(self.cfg.num_steps * self.env.num_envs) / rollout_seconds
        metrics: dict[str, float] = {
            "global_step": float(self.global_step),
            "mean_reward": mean_episode_reward,
            "mean_length": mean_episode_length,
            "mean_episode_reward": mean_episode_reward,
            "mean_episode_length": mean_episode_length,
            "rollout_mean_reward": rollout_reward,
            "rollout_mean_task_reward": mean_task_reward,
            "rollout_mean_amp_reward": mean_amp_reward,
            "rollout_episode_reward": rollout_episode_reward,
            "rollout_episode_length": rollout_episode_length,
            "mean_task_reward": mean_task_reward,
            "mean_amp_reward": mean_amp_reward,
            "mean_episode_progress": episode_progress,
            "done_fraction": done_fraction,
            "timeout_fraction": timeout_fraction,
            "termination_fraction": termination_fraction,
            "completed_episode_count": completed_episode_count,
            "successful_episode_count": successful_episode_count,
            "mean_episode_success_rate": mean_episode_success_rate,
            "episode_success_rate": mean_episode_success_rate,
            "rollout_episode_success_rate": rollout_episode_success_rate_value,
            "timing_rollout_s": rollout_end - iteration_start,
            "timing_policy_update_s": policy_end - rollout_end,
            "timing_amp_update_s": amp_end - policy_end,
            "timing_iteration_s": amp_end - iteration_start,
            "throughput_env_steps_s": env_steps_per_second,
            "throughput_physics_steps_s": env_steps_per_second * int(self.env.cfg.decimation),
            "nonfinite_task_reward_count": nonfinite_task_reward_count,
            "nonfinite_task_reward_fraction": (
                nonfinite_task_reward_count / max(self.cfg.num_steps * self.env.num_envs, 1)
            ),
        }
        reward_components = rollout.get("reward_components", {})
        if isinstance(reward_components, dict):
            for key, value in reward_components.items():
                metrics[f"task_{key}"] = float(value)
        bottom_gate = metrics.get("task_bottom_gate", 0.0)
        if bottom_gate > 1.0e-8:
            metrics["bottom_depth_success_rate"] = float(
                np.clip(metrics.get("task_bottom_depth_success", 0.0) / bottom_gate, 0.0, 1.0)
            )
        else:
            metrics["bottom_depth_success_rate"] = 0.0
        metrics.update(policy_metrics)
        metrics.update(amp_metrics)
        return metrics

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> int:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self._assert_checkpoint_finite(checkpoint)
        expected = {
            "obs_dim": self.env.obs_dim,
            "action_dim": self.env.action_dim,
            "amp_obs_dim": self.env.amp_obs_dim,
        }
        for key, value in expected.items():
            if key in checkpoint and int(checkpoint[key]) != int(value):
                raise ValueError(f"Checkpoint {key}={checkpoint[key]} does not match current environment {value}.")

        actor_state = checkpoint["actor_critic"]
        missing, unexpected = self.actor_critic.load_state_dict(actor_state, strict=False)
        if any(key.startswith("median_critic.") for key in missing):
            self.actor_critic.median_critic.load_state_dict(self.actor_critic.critic.state_dict())
            print("[WARN] checkpoint has no BPO median_critic; initialized it from critic weights.")
        non_median_missing = [key for key in missing if not key.startswith("median_critic.")]
        if non_median_missing:
            print(f"[WARN] actor_critic checkpoint missing keys: {non_median_missing}")
        if unexpected:
            print(f"[WARN] ignored unexpected actor_critic checkpoint keys: {unexpected}")
        self.actor_critic.clamp_log_std_()
        if self.cfg.amp_enabled and "discriminator" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator"])

        if load_optimizers:
            if "optimizer" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                except ValueError as exc:
                    print(f"[WARN] skipped incompatible actor optimizer state: {exc}")
            if self.cfg.amp_enabled and "discriminator_optimizer" in checkpoint:
                self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        self._set_optimizer_lrs()

        env_config = checkpoint.get("env_config", {})
        if isinstance(env_config, dict) and "domain_randomization_scale" in env_config:
            self.loaded_domain_randomization_scale = float(env_config["domain_randomization_scale"])
        self.global_step = int(checkpoint.get("global_step", 0))
        return int(checkpoint.get("iteration", -1)) + 1

    def _set_optimizer_lrs(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.cfg.learning_rate)
        for group in self.discriminator_optimizer.param_groups:
            group["lr"] = float(self.cfg.discriminator_learning_rate)

    def _checkpoint_payload(self, iteration: int) -> dict:
        checkpoint = {
            "iteration": int(iteration),
            "global_step": int(self.global_step),
            "actor_critic": self.actor_critic.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "ppo_amp_config": asdict(self.cfg),
            "env_config": asdict(self.env.cfg),
            "obs_dim": self.env.obs_dim,
            "action_dim": self.env.action_dim,
            "amp_obs_dim": self.env.amp_obs_dim,
            "common_joint_names": self.env.motion.common_joint_names,
        }
        self._assert_checkpoint_finite(checkpoint)
        return checkpoint

    def _export_onnx(self, path: Path, checkpoint: dict) -> None:
        """Export the deterministic actor and embed everything needed by playback."""
        import onnx

        was_training = self.actor_critic.training
        self.actor_critic.eval()
        dummy_observation = torch.zeros(
            (1, self.env.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        try:
            torch.onnx.export(
                self.actor_critic.actor,
                dummy_observation,
                path,
                input_names=("observation",),
                output_names=("action",),
                dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
        finally:
            self.actor_critic.train(was_training)

        model = onnx.load(path)
        metadata = {
            "format_version": 1,
            "iteration": checkpoint["iteration"],
            "global_step": checkpoint["global_step"],
            "obs_dim": checkpoint["obs_dim"],
            "action_dim": checkpoint["action_dim"],
            "amp_obs_dim": checkpoint["amp_obs_dim"],
            "ppo_amp_config": checkpoint["ppo_amp_config"],
            "env_config": checkpoint["env_config"],
            "common_joint_names": checkpoint["common_joint_names"],
        }
        if "sensor_observation" in checkpoint:
            metadata["sensor_observation"] = checkpoint["sensor_observation"]
        onnx.helper.set_model_props(
            model,
            {"lytrobot_amp.metadata": json.dumps(metadata, ensure_ascii=False)},
        )
        onnx.checker.check_model(model)
        onnx.save(model, path)

    @staticmethod
    def _remove_stale_models(directory: Path, current_stem: str) -> None:
        """Keep only the newest successfully written PT/ONNX model pair."""

        for suffix in (".pt", ".onnx"):
            for candidate in directory.glob(f"model_*{suffix}"):
                if candidate.stem != current_stem:
                    candidate.unlink(missing_ok=True)

    def _save_checkpoint_pair(self, path: str | Path, checkpoint: dict) -> tuple[Path, Path]:
        self._assert_checkpoint_finite(checkpoint)
        pt_path = Path(path)
        if pt_path.suffix != ".pt":
            raise ValueError(f"Checkpoint path must end in .pt, got: {pt_path}")
        onnx_path = pt_path.with_suffix(".onnx")
        pt_path.parent.mkdir(parents=True, exist_ok=True)

        token = uuid.uuid4().hex
        temporary_pt = pt_path.parent / f".{pt_path.name}.{token}.tmp"
        temporary_onnx = onnx_path.parent / f".{onnx_path.name}.{token}.tmp"
        try:
            torch.save(checkpoint, temporary_pt)
            self._export_onnx(temporary_onnx, checkpoint)
            os.replace(temporary_pt, pt_path)
            os.replace(temporary_onnx, onnx_path)
        finally:
            temporary_pt.unlink(missing_ok=True)
            temporary_onnx.unlink(missing_ok=True)

        # Cleanup happens only after both replacements succeed. If validation,
        # serialization, or ONNX export fails, the previous healthy pair stays.
        self._remove_stale_models(pt_path.parent, pt_path.stem)
        return pt_path, onnx_path

    def save(self, path: str | Path, iteration: int) -> tuple[Path, Path]:
        checkpoint = self._checkpoint_payload(iteration)
        return self._save_checkpoint_pair(path, checkpoint)
