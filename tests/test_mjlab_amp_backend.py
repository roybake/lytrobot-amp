"""The mjlab physics backend must preserve the LYT AMP task and tendons."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lytrobot_amp.env import LytRobotEnvCfg
from lytrobot_amp.env_mjlab import LytRobotMjlabAmpEnv
from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
from lytrobot_amp.ppo_amp import PPOAMPConfig, PPOAMPTrainer


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mjlab AMP integration")
class MjlabAmpBackendTests(unittest.TestCase):
    def test_matches_direct_warp_and_trains_amp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with np.load(ROOT / "robot/reference_data/walk1_mjcf.npz", allow_pickle=True) as archive:
                payload = {
                    key: archive[key][:80] if key in ("qpos", "qvel") else archive[key]
                    for key in archive.files
                }
            motion_file = Path(tmp) / "motion.npz"
            np.savez(motion_file, **payload)
            cfg = LytRobotEnvCfg(
                model_xml=str(ROOT / "robot/MJCFmodelposition.xml"),
                motion_file=str(motion_file),
                num_envs=2,
                seed=11,
                domain_randomization=False,
                warp_cuda_graph=False,
                episode_timeout_jitter_steps=0,
            )
            direct = LytRobotWarpAmpEnv(cfg, device="cuda")
            mjlab = LytRobotMjlabAmpEnv(cfg, device="cuda")
            try:
                self.assertEqual(mjlab.model.ntendon, 14)
                self.assertEqual(mjlab.model.nflex, 0)
                direct.reset()
                mjlab.reset()
                torch.testing.assert_close(direct.qpos, mjlab.qpos)
                action = torch.zeros((2, direct.action_dim), device="cuda")
                direct_obs, direct_reward, _, _ = direct.step(action)
                mjlab_obs, mjlab_reward, _, _ = mjlab.step(action)
                torch.testing.assert_close(direct.qpos, mjlab.qpos, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(direct_obs, mjlab_obs, atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(direct_reward, mjlab_reward, atol=1e-4, rtol=1e-4)

                trainer = PPOAMPTrainer(
                    mjlab,
                    PPOAMPConfig(
                        num_steps=2,
                        update_epochs=1,
                        num_minibatches=1,
                        amp_enabled=True,
                        amp_minibatch_size=4,
                        actor_hidden_dims=(16,),
                        critic_hidden_dims=(16,),
                        discriminator_hidden_dims=(16,),
                    ),
                    device="cuda",
                    log_dir=tmp,
                )
                metrics = trainer.train_iteration()
                self.assertTrue(np.isfinite(metrics["amp_loss"]))
                self.assertEqual(trainer.global_step, 4)
            finally:
                direct.close()
                mjlab.close()
