from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from lytrobot_amp.env import LytRobotEnvCfg
from lytrobot_amp.training_defaults import (
    MULTIMOTION_STAIR_STEP_DEPTH_M,
    MULTIMOTION_STAIR_STEP_HEIGHT_M,
    ROUGH_TERRAIN_HEIGHT_M,
    SLOPE_ANGLE_DEG,
    STAIR_STAGE2_STEP_HEIGHT_M,
    TRAINING_TARGET_SPEED_MPS,
)
from sensor_training.envs import (
    SensorHistoryCpuAmpEnv,
    _normalized_terrain_context,
)
from sensor_training.networks import SensorActorCritic
from sensor_training.train_multimotion import default_train_args


REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION_NAMES = (
    "walk",
    "rough_walk",
    "stair_up",
    "stair_down",
)


class SensorMultimotionEntrypointTests(unittest.TestCase):
    def test_defaults_keep_multimotion_group_without_skill_command(self) -> None:
        args = default_train_args()

        self.assertEqual(args[args.index("--log-group") + 1], "multimotion")
        self.assertEqual(args[args.index("--training-task") + 1], "multimotion")
        self.assertIn("--no-skill-conditioning", args)
        tags = args[args.index("--wandb-tags") + 1].split(",")
        self.assertIn("sensor-only", tags)
        self.assertIn("imu", tags)
        self.assertIn("tendon", tags)
        self.assertIn("--use-reference-velocity", args)
        self.assertNotIn("--no-use-reference-velocity", args)
        # The task definition is stage-neutral.  Stage 1/2 launchers append
        # the exact domain-randomization contract.
        self.assertNotIn("--domain-randomization", args)
        self.assertNotIn("--warp-global-domain-randomization", args)

        def value(option: str) -> float:
            return float(args[args.index(option) + 1])

        self.assertAlmostEqual(
            value("--target-velocity"),
            TRAINING_TARGET_SPEED_MPS,
            places=6,
        )
        self.assertEqual(value("--fall-penalty"), 10.0)
        self.assertEqual(value("--startup-action-ramp-steps"), 0.0)
        self.assertEqual(args.count("--target-velocity"), 1)
        self.assertEqual(value("--reference-ctrl-blend"), 0.0)
        self.assertEqual(value("--motion-transition-blend-steps"), 0.0)
        self.assertEqual(
            value("--rough-terrain-height"),
            ROUGH_TERRAIN_HEIGHT_M,
        )
        self.assertEqual(value("--slope-angle-deg"), SLOPE_ANGLE_DEG)
        self.assertEqual(value("--stair-step-height"), MULTIMOTION_STAIR_STEP_HEIGHT_M)
        self.assertEqual(value("--stair-step-depth"), MULTIMOTION_STAIR_STEP_DEPTH_M)
        self.assertNotIn("--specific-cot-limit", args)
        terrain_start = args.index("--terrain-types") + 1
        terrain_end = next(
            index
            for index in range(terrain_start, len(args))
            if args[index].startswith("--")
        )
        self.assertEqual(
            args[terrain_start:terrain_end],
            ["rough", "slope_up", "slope_down", "stairs"],
        )

    def test_real_multimotion_env_exposes_only_current_56_sensor_values(self) -> None:
        motion_dir = REPO_ROOT / "robot" / "reference_data" / "multimotion"
        env = SensorHistoryCpuAmpEnv(
            LytRobotEnvCfg(
                model_xml=str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"),
                motion_files=tuple(
                    str(motion_dir / f"{name}_omniretarget.npz")
                    for name in MOTION_NAMES
                ),
                motion_names=MOTION_NAMES,
                motion_loop=(True, False, False, False),
                skill_conditioning=False,
                use_reference_velocity=True,
                terrain_types=(
                    "rough",
                    "slope_up",
                    "slope_down",
                    "stairs",
                ),
                terrain_asset_dir=str(
                    REPO_ROOT / "robot" / "terrains" / "more_discolrl"
                ),
                serial_terrain_course=True,
                terrain_course_variants=1,
                rough_terrain_height=ROUGH_TERRAIN_HEIGHT_M,
                slope_angle_deg=SLOPE_ANGLE_DEG,
                stair_step_height=STAIR_STAGE2_STEP_HEIGHT_M,
                stair_step_depth=MULTIMOTION_STAIR_STEP_DEPTH_M,
                num_envs=1,
                stand_start_prob=0.0,
                motion_reset_prob=1.0,
                reset_noise_scale=0.0,
                domain_randomization=False,
                episode_timeout_jitter_steps=0,
                reference_ctrl_blend=0.0,
            )
        )
        try:
            observation = env.reset()
            self.assertEqual(env.obs_dim, 56)
            self.assertEqual(env.sensor_history_obs_dim, 336)
            self.assertEqual(observation.shape, (1, 56))
            self.assertTrue(np.isfinite(observation).all())
            privileged_context = env.privileged_observe()
            self.assertEqual(
                privileged_context.shape,
                (1, env.privileged_context_dim),
            )
            self.assertTrue(np.isfinite(privileged_context).all())
            self.assertFalse(
                env.observation_metadata["privileged_training_context"][
                    "actor_input"
                ]
            )
            current_ctrl = env.base_env._current_tendon_ctrl(env.base_env.datas[0])
            reference_ctrl = np.full_like(current_ctrl, 1.0e6)
            np.testing.assert_allclose(
                env.base_env._reference_base_ctrl(
                    reference_ctrl,
                    env.base_env.datas[0],
                ),
                current_ctrl,
            )
            for stage, scenario in enumerate(
                env.base_env.course_stage_scenario_names[0]
            ):
                env.base_env.course_stage_ids[0] = stage
                np.testing.assert_allclose(
                    env.privileged_observe()[0, 5:],
                    _normalized_terrain_context(scenario, env.base_env.cfg),
                )
            env.base_env.course_stage_ids[0] = 0

            reference = env.motion.sample_features(
                8,
                np.random.default_rng(7),
                mirror_prob=0.5,
            )
            self.assertEqual(reference.shape, (8, env.amp_obs_dim))
            self.assertTrue(np.isfinite(reference).all())

            policy = SensorActorCritic(
                env.obs_dim,
                env.action_dim,
                sensor_frame_dim=env.sensor_frame_dim,
                privileged_context_dim=env.privileged_context_dim,
            )
            self.assertEqual(policy.recurrent.input_size, env.sensor_frame_dim)
            self.assertEqual(policy.recurrent.hidden_size, 128)
            self.assertEqual(policy.actor[0].in_features, 128)
            self.assertEqual(
                policy.critic[0].in_features,
                128 + env.privileged_context_dim,
            )
            self.assertEqual(policy.actor[0].in_features, 128)
            self.assertEqual(
                policy.context_estimator[-2].out_features,
                env.privileged_context_dim,
            )
            with torch.no_grad():
                recurrent_state = policy.initial_state(1)
                action, recurrent_state = policy.act_inference_recurrent(
                    torch.from_numpy(observation),
                    recurrent_state,
                    torch.ones(1),
                )
                action = action.numpy()
            self.assertEqual(tuple(recurrent_state.shape), (1, 128))
            self.assertFalse(torch.equal(recurrent_state, torch.zeros_like(recurrent_state)))
            observation, reward, done, _info = env.step(action)
            self.assertEqual(observation.shape, (1, 56))
            self.assertTrue(np.isfinite(observation).all())
            self.assertTrue(np.isfinite(reward).all())
            self.assertTrue(np.isfinite(done).all())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
