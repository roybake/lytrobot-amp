"""Regression checks that direct walking and distillation stay separate."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from lytrobot_amp.networks import ActorCritic
from lytrobot_amp.ppo_amp import PPOAMPConfig
from sensor_training.distillation_config import SensorDistillationPPOAMPConfig
from sensor_training.distillation_trainer import SensorDistillationPPOAMPTrainer
from sensor_training.ppo_config import SensorPPOAMPConfig
from sensor_training.tests.test_recurrent_policy import _DummySensorEnv
from sensor_training.trainer import SensorHistoryPPOAMPTrainer


class _DummyDistillationEnv(_DummySensorEnv):
    teacher_obs_dim = 6

    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(
            decimation=2,
            startup_action_ramp_steps=20,
            reference_ctrl_blend=0.0,
            action_residual_scale=0.5,
            distal_action_scale=1.0,
            motion_file="walk1_mjcf_repaired.npz",
        )
        self._labeler = None
        self._labels: list[torch.Tensor] = []

    def teacher_observe(self) -> torch.Tensor:
        return torch.full((self.num_envs, self.teacher_obs_dim), float(self.step_count))

    def teacher_action_ramp(self, _ramp_steps: int) -> torch.Tensor:
        return torch.ones(self.num_envs, 1)

    def remap_teacher_action(self, action: torch.Tensor, **_kwargs) -> torch.Tensor:
        return action

    def begin_teacher_rollout(self, labeler) -> None:
        self._labeler = labeler
        self._labels = []

    def end_teacher_rollout(self) -> torch.Tensor:
        labels = torch.stack(self._labels)
        self._labeler = None
        self._labels = []
        return labels

    def step(self, action):
        assert self._labeler is not None
        self._labels.append(self._labeler().detach().clone())
        return super().step(action)


def _save_teacher(path: Path) -> None:
    policy = ActorCritic(
        obs_dim=6,
        action_dim=2,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
    )
    torch.save(
        {
            "iteration": 3,
            "global_step": 12,
            "obs_dim": 6,
            "action_dim": 2,
            "actor_critic": policy.state_dict(),
            "ppo_amp_config": {
                "actor_hidden_dims": (8,),
                "critic_hidden_dims": (8,),
            },
            "env_config": {
                "reference_ctrl_blend": 0.0,
                "action_residual_scale": 0.5,
                "distal_action_scale": 1.0,
                "decimation": 2,
                "startup_action_ramp_steps": 0,
                "motion_file": "walk1_mjcf_repaired.npz",
            },
        },
        path,
    )


def _small_config(config_type, **kwargs):
    return config_type(
        num_steps=2,
        update_epochs=1,
        num_minibatches=1,
        amp_enabled=False,
        compile_rollout=False,
        compile_policy_update=False,
        recurrent_hidden_dim=8,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        gate_hidden_dims=(8,),
        expert_hidden_dims=(8,),
        **kwargs,
    )


def test_direct_configs_contain_no_distillation_options() -> None:
    assert "teacher_checkpoint" not in {field.name for field in fields(PPOAMPConfig)}
    assert "teacher_checkpoint" not in {field.name for field in fields(SensorPPOAMPConfig)}
    assert "teacher_checkpoint" in {
        field.name for field in fields(SensorDistillationPPOAMPConfig)
    }


def test_direct_trainer_collects_and_updates_without_teacher_data() -> None:
    trainer = SensorHistoryPPOAMPTrainer(
        _DummySensorEnv(), _small_config(SensorPPOAMPConfig), "cpu", "."
    )
    rollout = trainer.collect_rollout()
    metrics = trainer.update_policy(rollout)

    assert not hasattr(trainer, "teacher")
    assert "teacher_actions" not in rollout
    assert not any(name.startswith("teacher_") for name in metrics)


def test_distillation_trainer_adds_teacher_data_only_on_its_path() -> None:
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "teacher.pt"
        _save_teacher(checkpoint)
        cfg = _small_config(
            SensorDistillationPPOAMPConfig,
            teacher_checkpoint=str(checkpoint),
            teacher_action_loss_coef=0.5,
            teacher_action_loss_final_coef=0.1,
            teacher_action_loss_decay_iterations=10,
        )
        trainer = SensorDistillationPPOAMPTrainer(
            _DummyDistillationEnv(), cfg, "cpu", directory
        )
        rollout = trainer.collect_rollout()
        metrics = trainer.update_policy(rollout)

    assert tuple(rollout["teacher_actions"].shape) == (2, 2, 2)
    assert metrics["teacher_action_loss"] >= 0.0
    assert metrics["teacher_action_mae"] >= 0.0
    assert metrics["teacher_action_loss_coef"] == 0.5
