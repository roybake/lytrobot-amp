"""PPO with an AMP discriminator for the LYT robot environment."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

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
    value_clip_coef: float = 1.0
    entropy_coef: float = 0.002
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
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
        self.episode_reward_sums = np.zeros(env.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(env.num_envs, dtype=np.int32)
        window = max(1, int(cfg.episode_metric_window))
        self.completed_episode_rewards: deque[float] = deque(maxlen=window)
        self.completed_episode_lengths: deque[int] = deque(maxlen=window)

    def _as_device_tensor(self, value: torch.Tensor | np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    @staticmethod
    def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def collect_rollout(self) -> dict[str, torch.Tensor | np.ndarray]:
        num_steps = self.cfg.num_steps
        num_envs = self.env.num_envs
        obs_dim = self.env.obs_dim
        action_dim = self.env.action_dim
        amp_dim = self.env.amp_obs_dim

        obs = torch.zeros((num_steps, num_envs, obs_dim), device=self.device)
        actions = torch.zeros((num_steps, num_envs, action_dim), device=self.device)
        log_probs = torch.zeros((num_steps, num_envs), device=self.device)
        values = torch.zeros((num_steps, num_envs), device=self.device)
        rewards = torch.zeros((num_steps, num_envs), device=self.device)
        task_rewards = torch.zeros((num_steps, num_envs), device=self.device)
        amp_rewards = torch.zeros((num_steps, num_envs), device=self.device)
        dones = torch.zeros((num_steps, num_envs), device=self.device)
        time_outs = torch.zeros((num_steps, num_envs), device=self.device)
        amp_features = torch.zeros((num_steps, num_envs, amp_dim), device=self.device)
        completed_episode_rewards: list[np.ndarray] = []
        completed_episode_lengths: list[np.ndarray] = []
        episode_progress_values: list[float] = []
        reward_component_sums: dict[str, float] = {}
        reward_component_counts: dict[str, int] = {}

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
            done_np = self._as_numpy(done_value)
            episode_step_np = self._as_numpy(_info["episode_step"]).astype(np.int32)
            episode_progress_values.append(float(episode_step_np.mean()))
            for key, info_value in _info.items():
                if key in ("episode_step", "time_outs"):
                    continue
                if torch.is_tensor(info_value):
                    reward_component_sums[key] = reward_component_sums.get(key, 0.0) + float(
                        info_value.detach().sum().cpu()
                    )
                    reward_component_counts[key] = reward_component_counts.get(key, 0) + int(info_value.numel())
                else:
                    values_np = np.asarray(info_value, dtype=np.float32)
                    reward_component_sums[key] = reward_component_sums.get(key, 0.0) + float(values_np.sum())
                    reward_component_counts[key] = reward_component_counts.get(key, 0) + int(values_np.size)
            task_reward = self._as_device_tensor(task_reward_value)
            done = self._as_device_tensor(done_value)
            time_out = self._as_device_tensor(_info.get("time_outs", np.zeros(num_envs, dtype=np.float32)))
            total_reward = self.cfg.task_reward_lerp * task_reward
            if self.cfg.amp_enabled:
                total_reward = total_reward + (1.0 - self.cfg.task_reward_lerp) * self.cfg.amp_reward_coef * amp_reward
            total_reward = float(max(self.cfg.reward_scale, 1.0e-6)) * total_reward
            total_reward_np = self._as_numpy(total_reward).astype(np.float32, copy=False)
            self.episode_reward_sums += total_reward_np
            self.episode_lengths += 1

            obs[step] = obs_tensor
            actions[step] = action
            log_probs[step] = log_prob
            values[step] = value
            rewards[step] = total_reward
            task_rewards[step] = task_reward
            amp_rewards[step] = amp_reward
            dones[step] = done
            time_outs[step] = time_out
            amp_features[step] = amp_tensor
            if np.any(done_np > 0.0):
                done_mask = done_np.astype(bool)
                done_rewards = self.episode_reward_sums[done_mask].copy()
                done_lengths = self.episode_lengths[done_mask].copy()
                completed_episode_rewards.append(done_rewards)
                completed_episode_lengths.append(done_lengths)
                self.completed_episode_rewards.extend(float(value) for value in done_rewards)
                self.completed_episode_lengths.extend(int(value) for value in done_lengths)
                self.episode_reward_sums[done_mask] = 0.0
                self.episode_lengths[done_mask] = 0

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
            "amp_features": amp_features,
            "completed_episode_rewards": completed_episode_rewards,
            "completed_episode_lengths": completed_episode_lengths,
            "episode_progress": float(np.mean(episode_progress_values)) if episode_progress_values else 0.0,
            "reward_components": {
                key: reward_component_sums[key] / max(reward_component_counts.get(key, 0), 1)
                for key in reward_component_sums
            },
        }

    def update_policy(self, rollout: dict[str, torch.Tensor | np.ndarray]) -> dict[str, float]:
        obs = rollout["obs"].reshape((-1, self.env.obs_dim))
        actions = rollout["actions"].reshape((-1, self.env.action_dim))
        old_log_probs = rollout["log_probs"].reshape(-1)
        returns = rollout["returns"].reshape(-1)
        old_values = rollout["values"].reshape(-1)
        advantages = rollout["advantages"].reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

        batch_size = obs.shape[0]
        indices = np.arange(batch_size)
        minibatches = max(1, min(self.cfg.num_minibatches, batch_size))

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        update_count = 0

        for _epoch in range(self.cfg.update_epochs):
            self.rng.shuffle(indices)
            for minibatch_indices in np.array_split(indices, minibatches):
                mb = torch.as_tensor(minibatch_indices, dtype=torch.long, device=self.device)
                _, new_log_prob, entropy, new_value = self.actor_critic.get_action_and_value(obs[mb], actions[mb])
                log_ratio = new_log_prob - old_log_probs[mb]
                ratio = log_ratio.exp()

                pg_loss1 = -advantages[mb] * ratio
                pg_loss2 = -advantages[mb] * torch.clamp(
                    ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

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
                entropy_loss = entropy.mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                self.actor_critic.clamp_log_std_()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()
                metrics["policy_loss"] += float(policy_loss.detach().cpu())
                metrics["value_loss"] += float(value_loss.detach().cpu())
                metrics["entropy"] += float(entropy_loss.detach().cpu())
                metrics["approx_kl"] += float(approx_kl.detach().cpu())
                metrics["clip_fraction"] += float(clip_fraction.detach().cpu())
                update_count += 1

        for key in metrics:
            metrics[key] /= max(update_count, 1)
        return metrics

    def update_discriminator(self, rollout: dict[str, torch.Tensor | np.ndarray]) -> dict[str, float]:
        if not self.cfg.amp_enabled:
            return {"amp_loss": 0.0, "amp_grad_penalty": 0.0, "amp_agent_prob": 0.0, "amp_ref_prob": 0.0}

        agent_features = rollout["amp_features"].reshape((-1, self.env.amp_obs_dim)).detach()
        batch_size = agent_features.shape[0]
        amp_batch_size = max(1, min(self.cfg.amp_minibatch_size, batch_size))
        metrics = {"amp_loss": 0.0, "amp_grad_penalty": 0.0, "amp_agent_prob": 0.0, "amp_ref_prob": 0.0}
        updates = max(1, self.cfg.amp_discriminator_updates)

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

            self.discriminator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.max_grad_norm)
            self.discriminator_optimizer.step()

            with torch.no_grad():
                metrics["amp_loss"] += float(amp_loss.detach().cpu())
                metrics["amp_grad_penalty"] += float(grad_penalty.detach().cpu())
                metrics["amp_agent_prob"] += float(torch.sigmoid(logits_agent).mean().detach().cpu())
                metrics["amp_ref_prob"] += float(torch.sigmoid(logits_ref).mean().detach().cpu())

        for key in metrics:
            metrics[key] /= updates
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
        rollout = self.collect_rollout()
        policy_metrics = self.update_policy(rollout)
        amp_metrics = self.update_discriminator(rollout)

        reward = rollout["rewards"]
        task_reward = rollout["task_rewards"]
        amp_reward = rollout["amp_rewards"]
        dones = rollout["dones"]
        completed_episode_rewards = rollout["completed_episode_rewards"]
        completed_episode_lengths = rollout["completed_episode_lengths"]
        episode_progress = float(rollout.get("episode_progress", 0.0))
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
        rollout_reward = float(reward.mean().detach().cpu())
        metrics: dict[str, float] = {
            "global_step": float(self.global_step),
            "mean_reward": mean_episode_reward,
            "mean_length": mean_episode_length,
            "mean_episode_reward": mean_episode_reward,
            "mean_episode_length": mean_episode_length,
            "rollout_mean_reward": rollout_reward,
            "rollout_mean_task_reward": float(task_reward.mean().detach().cpu()),
            "rollout_mean_amp_reward": float(amp_reward.mean().detach().cpu()),
            "rollout_episode_reward": rollout_episode_reward,
            "rollout_episode_length": rollout_episode_length,
            "mean_task_reward": float(task_reward.mean().detach().cpu()),
            "mean_amp_reward": float(amp_reward.mean().detach().cpu()),
            "mean_episode_progress": episode_progress,
            "done_fraction": float(dones.mean().detach().cpu()),
        }
        reward_components = rollout.get("reward_components", {})
        if isinstance(reward_components, dict):
            for key, value in reward_components.items():
                metrics[f"task_{key}"] = float(value)
        metrics.update(policy_metrics)
        metrics.update(amp_metrics)
        return metrics

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> int:
        checkpoint = torch.load(Path(path), map_location=self.device)
        expected = {
            "obs_dim": self.env.obs_dim,
            "action_dim": self.env.action_dim,
            "amp_obs_dim": self.env.amp_obs_dim,
        }
        for key, value in expected.items():
            if key in checkpoint and int(checkpoint[key]) != int(value):
                raise ValueError(f"Checkpoint {key}={checkpoint[key]} does not match current environment {value}.")

        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.actor_critic.clamp_log_std_()
        if self.cfg.amp_enabled and "discriminator" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator"])

        if load_optimizers:
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.cfg.amp_enabled and "discriminator_optimizer" in checkpoint:
                self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        self._set_optimizer_lrs()

        self.global_step = int(checkpoint.get("global_step", 0))
        return int(checkpoint.get("iteration", -1)) + 1

    def _set_optimizer_lrs(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.cfg.learning_rate)
        for group in self.discriminator_optimizer.param_groups:
            group["lr"] = float(self.cfg.discriminator_learning_rate)

    def save(self, path: str | Path, iteration: int) -> None:
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
