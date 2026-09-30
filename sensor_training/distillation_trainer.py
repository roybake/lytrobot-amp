"""Trainer used only by privileged-Teacher sensor distillation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .distillation import PrivilegedTeacher, linear_distillation_coefficient
from .distillation_config import SensorDistillationPPOAMPConfig
from .distillation_env import SensorDistillationWarpAmpEnv
from .trainer import SensorHistoryPPOAMPTrainer


class SensorDistillationPPOAMPTrainer(SensorHistoryPPOAMPTrainer):
    """Add Teacher labels and imitation loss without changing direct training."""

    def __init__(
        self,
        env: SensorDistillationWarpAmpEnv,
        cfg: SensorDistillationPPOAMPConfig,
        device: str,
        log_dir: str | Path,
    ) -> None:
        if not cfg.teacher_checkpoint:
            raise ValueError("Distillation trainer requires a Teacher checkpoint.")
        super().__init__(env, cfg, device, log_dir)
        self.teacher = PrivilegedTeacher(
            cfg.teacher_checkpoint,
            expected_observation_dim=env.teacher_obs_dim,
            expected_action_dim=env.action_dim,
            student_control_config={
                "reference_ctrl_blend": float(env.cfg.reference_ctrl_blend),
                "action_residual_scale": float(env.cfg.action_residual_scale),
                "distal_action_scale": float(env.cfg.distal_action_scale),
                "decimation": float(env.cfg.decimation),
                "motion_file": str(env.cfg.motion_file),
            },
            device=self.device,
        )
        self.distillation_iteration = 0
        self._compiled_teacher_act = None
        if self.device.type == "cuda" and bool(cfg.compile_rollout):
            self._compiled_teacher_act = torch.compile(
                self.teacher.policy.act_inference,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            )
        self._active_teacher_actions: torch.Tensor | None = None
        self._teacher_loss_sum = torch.zeros((), device=self.device)
        self._teacher_mae_sum = torch.zeros((), device=self.device)
        self._teacher_update_count = 0
        self._latest_teacher_labels = None
        self._use_teacher_action = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

    def _teacher_action_loss_coef(self) -> float:
        return linear_distillation_coefficient(
            self.cfg.teacher_action_loss_coef,
            self.cfg.teacher_action_loss_final_coef,
            self.distillation_iteration,
            self.cfg.teacher_action_loss_decay_iterations,
        )

    def _teacher_act(self, observation: torch.Tensor) -> torch.Tensor:
        compiled = self._compiled_teacher_act
        if compiled is None:
            return self.teacher.act(observation)
        try:
            return compiled(observation).clone()
        except Exception as exc:
            print(
                "[WARN] Teacher rollout compilation failed; falling back to eager "
                f"inference: {type(exc).__name__}: {exc}"
            )
            self._compiled_teacher_act = None
            return self.teacher.act(observation)

    @torch.no_grad()
    def _teacher_label(self) -> torch.Tensor:
        raw_action = self._teacher_act(self.env.teacher_observe())
        raw_action = raw_action * self.env.teacher_action_ramp(
            self.teacher.metadata.startup_action_ramp_steps
        )
        applied_residual = self.env.remap_teacher_action(
            raw_action,
            teacher_reference_blend=self.teacher.metadata.reference_ctrl_blend,
            teacher_action_residual_scale=self.teacher.metadata.action_residual_scale,
            teacher_distal_action_scale=self.teacher.metadata.distal_action_scale,
        )
        # Labels describe the Actor output before the Student's own startup
        # ramp. Without this division equal ramps apply the fade twice.
        student_ramp = self.env.teacher_action_ramp(self.env.cfg.startup_action_ramp_steps)
        return torch.clamp(applied_residual / student_ramp.clamp_min(1e-6), -1., 1.)

    def _teacher_behavior_probability(self) -> float:
        return linear_distillation_coefficient(
            self.cfg.teacher_action_probability,
            self.cfg.teacher_action_probability_final,
            self.distillation_iteration,
            self.cfg.teacher_action_probability_decay_iterations)

    def _select_behavior_action(self, action, log_prob, next_hidden, episode_starts):
        self._latest_teacher_labels = self._teacher_label()
        draw = torch.rand(self.env.num_envs, device=self.device) < self._teacher_behavior_probability()
        self._use_teacher_action = torch.where(
            episode_starts > .5, draw, self._use_teacher_action)
        executed = torch.where(self._use_teacher_action[:, None],
                               self._latest_teacher_labels, action)
        distribution, _, _ = self.actor_critic._distribution_from_hidden(next_hidden)
        executed_log_prob = distribution.log_prob(executed).sum(dim=-1)
        return executed, executed_log_prob, (~self._use_teacher_action).to(torch.float32)

    def _rollout_teacher_label(self):
        if self._latest_teacher_labels is None:
            raise RuntimeError('Teacher labels must be generated before stepping physics.')
        return self._latest_teacher_labels

    def collect_rollout(self) -> dict[str, torch.Tensor | np.ndarray]:
        self.env.begin_teacher_rollout(self._rollout_teacher_label)
        labels: torch.Tensor | None = None
        try:
            rollout = super().collect_rollout()
        finally:
            labels = self.env.end_teacher_rollout()
        if labels.shape[0] != self.cfg.num_steps:
            raise RuntimeError(
                "Teacher label count does not match the Student rollout: "
                f"labels={labels.shape[0]}, rollout_steps={self.cfg.num_steps}."
            )
        rollout["teacher_actions"] = labels
        return rollout

    def _actor_auxiliary_loss(
        self,
        obs: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        env_indices: torch.Tensor,
        action_means: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._active_teacher_actions is None:
            raise RuntimeError("Teacher targets are unavailable during the policy update.")
        # PPO supplies deterministic means from the SAME recurrent forward.
        # The fallback preserves this helper's direct-call compatibility.
        action_mean = action_means if action_means is not None else self.actor_critic.action_mean_sequence(
            obs, initial_hidden, episode_starts)
        targets = self._active_teacher_actions[:, env_indices]
        scale = max(float(self.cfg.teacher_action_loss_scale), 1e-6)
        teacher_loss = F.smooth_l1_loss(action_mean / scale, targets / scale)
        with torch.no_grad():
            self._teacher_loss_sum += teacher_loss.detach()
            self._teacher_mae_sum += (action_mean - targets).abs().mean().detach()
            self._teacher_update_count += 1
        return self._teacher_action_loss_coef() * teacher_loss

    @staticmethod
    def _finite_rollout_envs(
        rollout: dict[str, torch.Tensor | np.ndarray],
    ) -> torch.Tensor:
        tensor_names = (
            "obs",
            "actions",
            "log_probs",
            "returns",
            "values",
            "episode_starts",
            "initial_recurrent_state",
            "privileged_contexts",
        )
        values = {name: rollout[name] for name in tensor_names}
        if not all(torch.is_tensor(value) for value in values.values()):
            raise TypeError("Distillation rollout fields must all be tensors.")
        return (
            torch.isfinite(values["obs"]).all(dim=(0, 2))
            & torch.isfinite(values["actions"]).all(dim=(0, 2))
            & torch.isfinite(values["log_probs"]).all(dim=0)
            & torch.isfinite(values["returns"]).all(dim=0)
            & torch.isfinite(values["values"]).all(dim=0)
            & torch.isfinite(values["initial_recurrent_state"]).all(dim=1)
            & torch.isfinite(values["privileged_contexts"]).all(dim=(0, 2))
        )

    def update_policy(
        self,
        rollout: dict[str, torch.Tensor | np.ndarray],
    ) -> dict[str, float]:
        teacher_actions = rollout.get("teacher_actions")
        if not torch.is_tensor(teacher_actions):
            raise TypeError("Distillation rollout has no Teacher action tensor.")
        if not torch.isfinite(teacher_actions).all():
            raise RuntimeError("Teacher produced a non-finite distillation target.")
        finite_envs = self._finite_rollout_envs(rollout)
        self._active_teacher_actions = teacher_actions[:, finite_envs]
        self._teacher_loss_sum.zero_()
        self._teacher_mae_sum.zero_()
        self._teacher_update_count = 0
        try:
            metrics = super().update_policy(rollout)
        finally:
            self._active_teacher_actions = None
        count = max(self._teacher_update_count, 1)
        metrics["teacher_action_loss"] = float((self._teacher_loss_sum / count).cpu())
        metrics["teacher_action_mae"] = float((self._teacher_mae_sum / count).cpu())
        metrics["teacher_action_loss_coef"] = self._teacher_action_loss_coef()
        metrics["teacher_behavior_probability"] = self._teacher_behavior_probability()
        metrics["student_on_policy_fraction"] = float(rollout['policy_sample_mask'].mean().cpu())
        self.distillation_iteration += 1
        return metrics

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> int:
        checkpoint = torch.load(Path(path), map_location=self.device)
        next_iteration = super().load(path, load_optimizers=load_optimizers)
        self.distillation_iteration = (
            0
            if self.cfg.distillation_reset_schedule
            else int(checkpoint.get("distillation_iteration", next_iteration))
        )
        return next_iteration

    def _checkpoint_payload(self, iteration: int) -> dict:
        checkpoint = super()._checkpoint_payload(iteration)
        checkpoint.update(
            {
                "distillation_iteration": int(self.distillation_iteration),
                "distillation": {
                    "enabled": True,
                    "teacher": {
                        "checkpoint": self.teacher.metadata.checkpoint,
                        "iteration": self.teacher.metadata.iteration,
                        "global_step": self.teacher.metadata.global_step,
                        "observation_dim": self.teacher.metadata.observation_dim,
                        "action_dim": self.teacher.metadata.action_dim,
                        "reference_ctrl_blend": self.teacher.metadata.reference_ctrl_blend,
                        "action_residual_scale": self.teacher.metadata.action_residual_scale,
                        "distal_action_scale": self.teacher.metadata.distal_action_scale,
                        "decimation": self.teacher.metadata.decimation,
                        "startup_action_ramp_steps": self.teacher.metadata.startup_action_ramp_steps,
                        "action_semantics_remapped": True,
                        "observes_executed_student_action_history": True,
                    },
                    "teacher_action_loss_coef": self._teacher_action_loss_coef(),
                    "teacher_action_loss_scale": self.cfg.teacher_action_loss_scale,
                    "teacher_behavior_probability": self._teacher_behavior_probability(),
                    "actor_labels_pre_startup_ramp": True,
                },
            }
        )
        return checkpoint
