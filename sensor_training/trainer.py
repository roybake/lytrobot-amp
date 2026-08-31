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
from lytrobot_amp.networks import ActorCritic, AmpDiscriminator
from lytrobot_amp.training_defaults import CONTROL_FREQUENCY_HZ, require_100hz_control

from .networks import RecurrentSensorActorExport, SensorActorCritic
from .ppo_config import SensorPPOAMPConfig


class SensorHistoryPPOAMPTrainer(PPOAMPTrainer):
    @staticmethod
    def _sensor_observation_layout_signature(metadata: object) -> dict:
        """Keep resume checks strict for actor inputs, not reward semantics.

        ``sensor_observation`` also records squat task diagnostics.  A reward
        coefficient or the policy for ending a successful episode changes the
        learning objective, but neither changes a saved actor's input layout.
        Treat only those explicitly listed fields as mutable across resumes.
        """

        if not isinstance(metadata, dict):
            return {}
        signature = copy.deepcopy(metadata)
        signature.pop("success_definition", None)
        phase_feedback = signature.get("phase_feedback")
        if isinstance(phase_feedback, dict):
            phase_feedback.pop("cycle_completion_bonus", None)
            phase_feedback.pop("terminate_on_cycle_completion", None)
        return signature

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

        policy_frame_dim = int(
            getattr(env, "policy_frame_dim", env.sensor_frame_dim)
        )
        self.actor_critic = SensorActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
            init_std=cfg.init_std,
            min_action_std=cfg.min_action_std,
            max_action_std=cfg.max_action_std,
            sensor_frame_dim=policy_frame_dim,
            recurrent_hidden_dim=cfg.recurrent_hidden_dim,
            privileged_context_dim=env.privileged_context_dim,
            estimator_context_dim=env.estimator_context_dim,
        ).to(self.device)
        self.discriminator = AmpDiscriminator(
            input_dim=env.amp_obs_dim,
            hidden_dims=cfg.discriminator_hidden_dims,
        ).to(self.device)
        self.teacher_actor_critic: ActorCritic | None = None
        teacher_path = cfg.teacher_checkpoint
        if teacher_path is not None:
            checkpoint = torch.load(Path(teacher_path), map_location=self.device)
            saved_obs_dim = int(checkpoint.get("obs_dim", -1))
            if saved_obs_dim != int(env.base_env.obs_dim):
                raise ValueError(
                    "Teacher observation dimension does not match the student base environment: "
                    f"teacher={saved_obs_dim}, base={env.base_env.obs_dim}."
                )
            if int(checkpoint.get("action_dim", -1)) != int(env.action_dim):
                raise ValueError("Teacher action dimension does not match the student.")
            saved_cfg = checkpoint.get("ppo_amp_config", {})
            if not isinstance(saved_cfg, dict):
                raise ValueError("Teacher checkpoint has no valid PPO-AMP configuration.")
            teacher = ActorCritic(
                obs_dim=saved_obs_dim,
                action_dim=env.action_dim,
                actor_hidden_dims=tuple(saved_cfg.get("actor_hidden_dims", (256, 128))),
                critic_hidden_dims=tuple(saved_cfg.get("critic_hidden_dims", (256, 128))),
                init_std=float(saved_cfg.get("init_std", 0.35)),
                min_action_std=float(saved_cfg.get("min_action_std", 0.05)),
                max_action_std=float(saved_cfg.get("max_action_std", 0.60)),
            ).to(self.device)
            teacher.load_state_dict(checkpoint["actor_critic"], strict=True)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self.teacher_actor_critic = teacher
        if cfg.distill_teacher_control_target:
            if self.teacher_actor_critic is None:
                raise ValueError(
                    "Teacher control-target distillation requires a frozen Teacher."
                )
            if abs(float(env.base_env.cfg.reference_ctrl_blend)) > 1.0e-12:
                raise ValueError(
                    "Teacher control-target distillation requires the Student "
                    "environment to use reference_ctrl_blend=0."
                )
        fused_optimizer = bool(cfg.fused_optimizer and self.device.type == "cuda")
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=cfg.learning_rate,
            fused=fused_optimizer,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.discriminator_learning_rate,
            fused=fused_optimizer,
        )
        self._compiled_policy_update = bool(
            cfg.compile_policy and self.device.type == "cuda"
        )
        self._evaluate_policy_sequence = self.actor_critic.evaluate_sequence
        if self._compiled_policy_update:
            self._evaluate_policy_sequence = torch.compile(
                self.actor_critic.evaluate_sequence,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )
            print(
                "[INFO] recurrent BPTT torch.compile enabled; the first policy "
                "update performs a one-time CUDA compilation."
            )

        if fused_optimizer:
            print("[INFO] fused CUDA Adam enabled for policy and AMP updates.")

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

    def _student_space_teacher_target(
        self, teacher_action: torch.Tensor
    ) -> torch.Tensor:
        """Map the Teacher's reference residual to the Student control space.

        The 08-06 Teacher adds its action around the retargeted tendon target,
        while the deployable Student adds its action around measured tendon
        lengths.  Distilling the raw Teacher residual therefore teaches a
        different command.  Convert the Teacher's intended actuator target
        into the equivalent Student action before applying the imitation loss.
        """

        if not self.cfg.distill_teacher_control_target:
            return teacher_action
        base_env = self.env.base_env
        control_scale = (
            base_env.ctrl_half_range
            * base_env.action_scale
            * float(base_env.cfg.action_residual_scale)
        ).clamp_min(1.0e-6)
        teacher_ctrl = torch.clamp(
            base_env.last_reference_ctrl + teacher_action * control_scale,
            min=base_env.ctrl_low,
            max=base_env.ctrl_high,
        )
        current_ctrl = base_env._current_tendon_ctrl()
        return torch.clamp(
            (teacher_ctrl - current_ctrl) / control_scale,
            -1.0,
            1.0,
        )

    def reset_training_env(self) -> None:
        super().reset_training_env()
        self.recurrent_state.zero_()
        self.episode_starts.fill_(1.0)

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
        successes = torch.empty_like(rewards)
        episode_starts = torch.empty_like(rewards)
        amp_features = torch.empty(
            (num_steps, num_envs, self.env.amp_obs_dim),
            device=self.device,
        )
        teacher_actions = torch.empty_like(actions) if self.teacher_actor_critic else None
        privileged_contexts = torch.empty(
            (num_steps, num_envs, self.env.privileged_context_dim),
            device=self.device,
        )
        estimator_contexts = torch.empty(
            (num_steps, num_envs, self.env.estimator_context_dim),
            device=self.device,
        )
        completed_reward_values = torch.zeros_like(rewards)
        completed_length_values = torch.zeros_like(rewards)
        episode_progress_sum = torch.zeros((), dtype=torch.float32, device=self.device)
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
            estimator_context = self._as_device_tensor(
                self.env.estimator_observe()
            )
            current_episode_starts = self.episode_starts
            with torch.no_grad():
                (
                    action,
                    log_prob,
                    _entropy,
                    value,
                    _median_value,
                    next_recurrent_state,
                ) = self.actor_critic.recurrent_action_and_value(
                    obs_tensor,
                    self.recurrent_state,
                    current_episode_starts,
                    privileged_context=privileged_context,
                )
                amp_reward = (
                    self.discriminator.reward(amp_tensor)
                    if self.cfg.amp_enabled
                    else torch.zeros(num_envs, device=self.device)
                )
                if teacher_actions is not None:
                    teacher_obs = self._as_device_tensor(self.env.base_env.observe())
                    teacher_action = self.teacher_actor_critic.act_inference(teacher_obs)
                    teacher_actions[step] = self._student_space_teacher_target(
                        teacher_action
                    )

            env_action = action if getattr(self.env, "is_torch_env", False) else action.cpu().numpy()
            next_obs, task_reward_value, done_value, info = self.env.step(env_action)
            episode_step = self._as_device_tensor(info["episode_step"])
            episode_progress_sum += episode_step.sum()
            component_items = [
                (key, value)
                for key, value in info.items()
                if key not in ("episode_step", "time_outs", "successes")
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
            success = self._as_device_tensor(info.get("successes", time_out))
            amp_reward = torch.nan_to_num(amp_reward, nan=0.0, posinf=0.0, neginf=0.0)
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
            amp_features[step] = amp_tensor
            privileged_contexts[step] = privileged_context
            estimator_contexts[step] = estimator_context
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
            next_non_terminal = 1.0 - terminal
            delta = (
                rewards[step]
                + self.cfg.gamma * next_values * next_non_terminal
                - values[step]
            )
            last_gae = (
                delta
                + self.cfg.gamma
                * self.cfg.gae_lambda
                * next_non_terminal
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
        summary_components = torch.cat(
            (
                component_means,
                (episode_progress_sum / max(num_steps * num_envs, 1)).reshape(1),
            )
        )
        component_values = summary_components.detach().cpu().tolist()
        reward_components = {
            key: float(component_values[index])
            for index, key in enumerate(reward_component_keys)
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
            "episode_starts": episode_starts,
            "initial_recurrent_state": initial_recurrent_state,
            "amp_features": amp_features,
            "teacher_actions": teacher_actions,
            "privileged_contexts": privileged_contexts,
            "estimator_contexts": estimator_contexts,
            "completed_episode_rewards": completed_episode_rewards,
            "completed_episode_lengths": completed_episode_lengths,
            "episode_progress": float(component_values[-1]),
            "reward_components": reward_components,
            "nonfinite_task_reward_count": nonfinite_task_reward_count,
        }

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
        estimator_contexts = rollout["estimator_contexts"]
        teacher_actions = rollout.get("teacher_actions")
        assert all(torch.is_tensor(value) for value in (
            obs,
            actions,
            old_log_probs,
            returns,
            old_values,
            episode_starts,
            initial_hidden,
            privileged_contexts,
            estimator_contexts,
        ))

        finite_envs = (
            torch.isfinite(obs).all(dim=(0, 2))
            & torch.isfinite(actions).all(dim=(0, 2))
            & torch.isfinite(old_log_probs).all(dim=0)
            & torch.isfinite(returns).all(dim=0)
            & torch.isfinite(old_values).all(dim=0)
            & torch.isfinite(initial_hidden).all(dim=1)
            & torch.isfinite(privileged_contexts).all(dim=(0, 2))
            & torch.isfinite(estimator_contexts).all(dim=(0, 2))
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
        estimator_contexts = estimator_contexts[:, finite_envs]
        if torch.is_tensor(teacher_actions):
            teacher_actions = teacher_actions[:, finite_envs]

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
            "privileged_context_loss",
            "distillation_loss",
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
                if self._compiled_policy_update:
                    # reduce-overhead mode reuses CUDA-graph output storage.
                    # Mark every optimizer update as a new step so outputs from
                    # the previous minibatch may be safely recycled.
                    torch.compiler.cudagraph_mark_step_begin()
                (
                    new_log_prob,
                    entropy,
                    new_value,
                    median_value,
                    _next_hidden,
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
                policy_loss = (ratio_error * policy_weight).mean()

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
                    bpo_lambda
                    * self._bpo_soft_abs(
                        (mb_returns.detach() - median_value) / bpo_lambda
                    )
                ).mean()
                entropy_loss = entropy.mean()

                privileged_context_loss = F.smooth_l1_loss(
                    context_prediction,
                    estimator_contexts[:, mb],
                )
                hidden = initial_hidden[mb]
                student_means: list[torch.Tensor] = []
                for step in range(obs.shape[0]):
                    hidden = self.actor_critic._next_hidden(
                        obs[step, mb], hidden, episode_starts[step, mb]
                    )
                    student_means.append(self.actor_critic._actor_mean(hidden))
                distillation_loss = (
                    F.smooth_l1_loss(torch.stack(student_means), teacher_actions[:, mb])
                    if torch.is_tensor(teacher_actions)
                    else torch.zeros((), dtype=obs.dtype, device=self.device)
                )

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    + self.cfg.median_value_coef * median_value_loss
                    - self.cfg.entropy_coef * entropy_loss
                    + self.cfg.privileged_context_loss_coef
                    * privileged_context_loss
                    + self.cfg.distillation_coef * distillation_loss
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
                        privileged_context_loss,
                        distillation_loss,
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
                "Checkpoint uses the legacy feed-forward sensor policy; recurrent GRU/MLP "
                "training must start in a new run instead of resuming that checkpoint."
            )
        if architecture.get("name") != "proprioceptive_gru_context_mlp":
            raise ValueError(
                "Checkpoint recurrent policy architecture is incompatible with "
                "the current single-GRU/MLP policy; start a new run instead of "
                "resuming it."
            )
        saved_env_cfg = checkpoint.get("env_config", {})
        if not isinstance(saved_env_cfg, dict):
            raise ValueError("Checkpoint has no valid env_config timing metadata.")
        require_100hz_control(
            float(saved_env_cfg.get("timestep", 0.0)),
            int(saved_env_cfg.get("decimation", 0)),
            source="checkpoint",
        )
        require_100hz_control(
            float(self.env.cfg.timestep),
            int(self.env.cfg.decimation),
            source="current training",
        )
        saved_collision_profile = str(
            saved_env_cfg.get("robot_collision_profile", "full_body")
        )
        current_collision_profile = str(
            getattr(self.env.cfg, "robot_collision_profile", "full_body")
        )
        if saved_collision_profile != current_collision_profile:
            if not bool(
                getattr(self.env.cfg, "allow_collision_profile_migration", False)
            ):
                raise ValueError(
                    "Checkpoint collision profile does not match current training: "
                    f"saved={saved_collision_profile!r}, "
                    f"current={current_collision_profile!r}. Use the saved profile "
                    "for exact continuation, or explicitly pass "
                    "--allow-collision-profile-migration in a new log directory."
                )
            print(
                "[WARN] explicitly migrating checkpoint collision profile: "
                f"{saved_collision_profile} -> {current_collision_profile}"
            )
        saved_context_dim = int(architecture.get("privileged_context_dim", 0))
        if saved_context_dim != self.env.privileged_context_dim:
            raise ValueError(
                "Checkpoint has no compatible asymmetric-Critic context: "
                f"saved privileged_context_dim={saved_context_dim}, "
                f"current={self.env.privileged_context_dim}. Start a new run."
            )
        saved_context_version = int(architecture.get("privileged_context_version", 0))
        current_context_version = int(
            self.env.privileged_context_metadata.get("version", 0)
        )
        if saved_context_version != current_context_version:
            raise ValueError(
                "Checkpoint privileged-context version does not match current "
                f"training: saved={saved_context_version}, "
                f"current={current_context_version}. Start a new run."
            )
        saved_estimator_dim = int(architecture.get("estimator_context_dim", 0))
        if saved_estimator_dim != self.env.estimator_context_dim:
            raise ValueError(
                "Checkpoint estimator target dimension does not match current "
                f"training: saved={saved_estimator_dim}, "
                f"current={self.env.estimator_context_dim}. Start a new run."
            )
        saved_metadata = checkpoint.get("sensor_observation")
        if saved_metadata is None:
            raise ValueError(
                "Checkpoint has no sensor_observation metadata; use a checkpoint produced by sensor_training."
            )
        current_metadata = self.env.observation_metadata
        if (
            self._sensor_observation_layout_signature(saved_metadata)
            != self._sensor_observation_layout_signature(current_metadata)
        ):
            raise ValueError(
                "Checkpoint sensor observation layout does not match the current "
                "sensor_training layout. Actor inputs, history length, IMU layout, "
                "or phase features changed."
            )
        saved_task = str(checkpoint.get("training_task", "")).strip()
        current_task = str(getattr(self.env.cfg, "training_task", "")).strip()
        if saved_task and current_task and saved_task != current_task:
            raise ValueError(
                "Checkpoint belongs to a different independent training task: "
                f"saved={saved_task!r}, current={current_task!r}."
            )
        next_iteration = super().load(path, load_optimizers=load_optimizers)
        self.recurrent_state.zero_()
        self.episode_starts.fill_(1.0)
        return next_iteration

    def _checkpoint_payload(self, iteration: int) -> dict:
        checkpoint = super()._checkpoint_payload(iteration)
        checkpoint.update({
            "training_task": str(getattr(self.env.cfg, "training_task", "")),
            "distillation": {
                "teacher_checkpoint": self.cfg.teacher_checkpoint,
                "teacher_frozen": self.teacher_actor_critic is not None,
                "coefficient": float(self.cfg.distillation_coef),
            },
            "sensor_observation": self.env.observation_metadata,
            "sensor_policy_architecture": {
                "name": "proprioceptive_gru_context_mlp",
                "recurrent": True,
                # Legacy key retained for playback/evaluation loaders. Every
                # task now uses the same physical IMU+tendon frame contract.
                "sensor_frame_dim": int(
                    getattr(
                        self.env,
                        "policy_frame_dim",
                        self.env.sensor_frame_dim,
                    )
                ),
                "physical_sensor_frame_dim": self.env.sensor_frame_dim,
                "recurrent_hidden_dim": self.cfg.recurrent_hidden_dim,
                "control_frequency_hz": CONTROL_FREQUENCY_HZ,
                "control_period_s": float(self.env.dt),
                "privileged_context_dim": self.env.privileged_context_dim,
                "privileged_context_version": int(
                    self.env.privileged_context_metadata["version"]
                ),
                "privileged_context_fields": list(
                    self.env.privileged_context_metadata["fields"]
                ),
                "privileged_context_groups": self.env.privileged_context_metadata[
                    "groups"
                ],
                "privileged_context_actor_input": False,
                "asymmetric_critic": True,
                "estimator_context_dim": self.env.estimator_context_dim,
                "estimator_context_fields": list(
                    self.env.estimator_context_metadata["fields"]
                ),
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
            "format_version": 3,
            "iteration": checkpoint["iteration"],
            "global_step": checkpoint["global_step"],
            "obs_dim": checkpoint["obs_dim"],
            "action_dim": checkpoint["action_dim"],
            "amp_obs_dim": checkpoint["amp_obs_dim"],
            "ppo_amp_config": checkpoint["ppo_amp_config"],
            "env_config": checkpoint["env_config"],
            "training_task": checkpoint.get("training_task", ""),
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
