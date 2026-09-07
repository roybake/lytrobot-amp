from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort
import torch

from lytrobot_amp.networks import ActorCritic
from lytrobot_amp.ppo_amp import PPOAMPTrainer
from scripts.train import (
    BEST_MODEL_FILENAME,
    BestCheckpointTracker,
    LATEST_MODEL_FILENAME,
    best_model_path,
    latest_model_path,
)


class ModelExportTests(unittest.TestCase):
    def _trainer(self) -> PPOAMPTrainer:
        trainer = PPOAMPTrainer.__new__(PPOAMPTrainer)
        trainer.device = torch.device("cpu")
        trainer.env = SimpleNamespace(obs_dim=7)
        trainer.actor_critic = ActorCritic(obs_dim=7, action_dim=3)
        return trainer

    @staticmethod
    def _checkpoint(iteration: int) -> dict:
        return {
            "iteration": iteration,
            "global_step": 100 * iteration,
            "actor_critic": {},
            "obs_dim": 7,
            "action_dim": 3,
            "amp_obs_dim": 5,
            "ppo_amp_config": {"actor_hidden_dims": [256, 128]},
            "env_config": {"target_velocity": 1.0},
            "common_joint_names": ["joint"],
            "conditional_amp": {
                "version": 1,
                "condition_fields": ["motion_id"],
                "condition_dim": 4,
                "actor_input": False,
            },
        }

    def test_save_exports_pair_and_removes_previous_pair(self) -> None:
        trainer = self._trainer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.csv"
            metrics_path.write_text("iteration,reward\n", encoding="utf-8")
            old_pt, old_onnx = trainer._save_checkpoint_pair(
                root / "model_1.pt", self._checkpoint(0)
            )
            current_pt, current_onnx = trainer._save_checkpoint_pair(
                root / "model_2.pt", self._checkpoint(1)
            )

            self.assertFalse(old_pt.exists())
            self.assertFalse(old_onnx.exists())
            self.assertTrue(current_pt.exists())
            self.assertTrue(current_onnx.exists())
            self.assertEqual(
                sorted(path.name for path in root.glob("model_*")),
                ["model_2.onnx", "model_2.pt"],
            )
            self.assertTrue(metrics_path.exists())

            model = onnx.load(current_onnx)
            properties = {item.key: item.value for item in model.metadata_props}
            metadata = json.loads(properties["lytrobot_amp.metadata"])
            self.assertEqual(metadata["iteration"], 1)
            self.assertEqual(metadata["obs_dim"], 7)
            self.assertEqual(metadata["action_dim"], 3)
            self.assertEqual(metadata["conditional_amp"]["condition_dim"], 4)
            self.assertFalse(metadata["conditional_amp"]["actor_input"])

            observation = np.random.default_rng(1).normal(size=(4, 7)).astype(np.float32)
            with torch.no_grad():
                expected = trainer.actor_critic.act_inference(torch.from_numpy(observation)).numpy()
            session = ort.InferenceSession(
                str(current_onnx), providers=("CPUExecutionProvider",)
            )
            actual = session.run(None, {session.get_inputs()[0].name: observation})[0]
            np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)

            final_pt, final_onnx = trainer._save_checkpoint_pair(
                root / "model_final.pt", self._checkpoint(2)
            )
            self.assertFalse(current_pt.exists())
            self.assertFalse(current_onnx.exists())
            self.assertTrue(final_pt.exists())
            self.assertTrue(final_onnx.exists())
            self.assertEqual(
                sorted(path.name for path in root.glob("model_*")),
                ["model_final.onnx", "model_final.pt"],
            )

    def test_all_training_saves_share_one_canonical_latest_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            self.assertEqual(LATEST_MODEL_FILENAME, "model_final.pt")
            self.assertEqual(latest_model_path(root), root / "model_final.pt")
            self.assertEqual(BEST_MODEL_FILENAME, "model_best.pt")
            self.assertEqual(best_model_path(root), root / "model_best.pt")

    def test_latest_and_best_model_pairs_coexist(self) -> None:
        trainer = self._trainer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_pt, final_onnx = trainer._save_checkpoint_pair(
                latest_model_path(root), self._checkpoint(1)
            )
            best_pt, best_onnx = trainer._save_checkpoint_pair(
                best_model_path(root), self._checkpoint(2)
            )
            trainer._save_checkpoint_pair(latest_model_path(root), self._checkpoint(3))

            for path in (final_pt, final_onnx, best_pt, best_onnx):
                self.assertTrue(path.exists())

    def test_serial_best_checkpoint_prioritizes_completion_over_reward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracker = BestCheckpointTracker(
                directory,
                serial_course=True,
                window_size=1,
            )
            tracker.observe(
                {
                    "rollout_mean_reward": 100.0,
                    "task_course_stage": 1.0,
                    "task_course_start_stage": 0.0,
                }
            )
            tracker.commit(10)
            tracker.observe(
                {
                    "rollout_mean_reward": -10.0,
                    "task_course_suffix_complete": 0.01,
                }
            )
            self.assertTrue(tracker.is_improvement())

    def test_serial_best_checkpoint_rejects_entrance_only_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracker = BestCheckpointTracker(
                directory,
                serial_course=True,
                window_size=1,
            )
            tracker.observe(
                {
                    "rollout_mean_reward": 100.0,
                    "task_course_stage": 0.0,
                    "task_course_stage_progress": 0.95,
                    "task_course_stage_completion": 0.0,
                }
            )
            self.assertFalse(tracker.is_improvement())

            tracker.observe(
                {
                    "rollout_mean_reward": 1.0,
                    "task_course_stage_completion": 0.001,
                }
            )
            self.assertTrue(tracker.is_improvement())

    def test_failed_new_export_preserves_previous_model_pair(self) -> None:
        trainer = self._trainer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_pt, old_onnx = trainer._save_checkpoint_pair(
                root / "model_1.pt", self._checkpoint(0)
            )

            with mock.patch.object(
                trainer,
                "_export_onnx",
                side_effect=RuntimeError("export failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "export failed"):
                    trainer._save_checkpoint_pair(
                        root / "model_2.pt",
                        self._checkpoint(1),
                    )

            self.assertTrue(old_pt.exists())
            self.assertTrue(old_onnx.exists())
            self.assertFalse((root / "model_2.pt").exists())
            self.assertFalse((root / "model_2.onnx").exists())


if __name__ == "__main__":
    unittest.main()
