from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch

from lytrobot_amp.networks import ActorCritic
from lytrobot_amp.ppo_amp import PPOAMPTrainer


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
        }

    def test_save_exports_pair_and_removes_previous_pair(self) -> None:
        trainer = self._trainer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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

            model = onnx.load(current_onnx)
            properties = {item.key: item.value for item in model.metadata_props}
            metadata = json.loads(properties["lytrobot_amp.metadata"])
            self.assertEqual(metadata["iteration"], 1)
            self.assertEqual(metadata["obs_dim"], 7)
            self.assertEqual(metadata["action_dim"], 3)

            observation = np.random.default_rng(1).normal(size=(4, 7)).astype(np.float32)
            with torch.no_grad():
                expected = trainer.actor_critic.act_inference(torch.from_numpy(observation)).numpy()
            session = ort.InferenceSession(
                str(current_onnx), providers=("CPUExecutionProvider",)
            )
            actual = session.run(None, {session.get_inputs()[0].name: observation})[0]
            np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
