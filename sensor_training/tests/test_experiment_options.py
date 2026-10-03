"""Sensor recipe CLI options must reach the environment and PPO config."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from unittest.mock import Mock, patch

from scripts import train as base_train
from sensor_training import distillation_entrypoints
from sensor_training.config import SensorObservationConfig
from sensor_training.distillation_config import SensorDistillationPPOAMPConfig
from sensor_training.distillation_recipes.run_improved import TRAINING_ARGS
from sensor_training.experiment_options import install_training_options, option_parser


class ExperimentOptionsTests(unittest.TestCase):
    def test_improved_recipe_options_reach_environment_and_ppo(self) -> None:
        environment = Mock()

        def configure_training(_path: str, *, run_name: str) -> None:
            self.assertEqual(run_name, '__main__')
            install_training_options(base_train)
            args = base_train.parse_args()
            cfg = base_train.PPOAMPConfig()
            base_train.LytRobotWarpAmpEnv(object(), device='cuda')
            observation = environment.call_args.kwargs['observation_cfg']

            self.assertEqual(observation.frame_count, 8)
            self.assertEqual(args.history_steps, 7)
            self.assertTrue(observation.include_tendon_velocity)
            self.assertTrue(observation.include_tendon_tracking_error)
            self.assertEqual(args.entropy_coef, 0.004)
            self.assertEqual(cfg.entropy_coef, 0.004)
            self.assertEqual(cfg.init_std, 0.18)
            self.assertEqual(cfg.recurrent_hidden_dim, 192)
            self.assertTrue(cfg.state_estimator_enabled)
            self.assertTrue(cfg.actor_use_estimated_state)
            self.assertEqual(args.num_envs, 4096)
            self.assertEqual(args.teacher_checkpoint, TRAINING_ARGS[2])

        with (
            patch.object(sys, 'argv', list(TRAINING_ARGS)),
            patch.multiple(
                base_train,
                parse_args=base_train.parse_args,
                PPOAMPConfig=SensorDistillationPPOAMPConfig,
                DISTILLATION_CLI_ENABLED=True,
                LytRobotAmpEnv=environment,
                LytRobotWarpAmpEnv=environment,
                LytRobotMjlabAmpEnv=environment,
            ),
            patch.object(distillation_entrypoints.runpy, 'run_path', configure_training),
        ):
            distillation_entrypoints.run_stage1()

    def test_history_options_include_the_current_frame(self) -> None:
        for arguments, frames in ((['--frame-count', '1'], 1),
                                  (['--frame-count=8'], 8), (['--history-steps', '7'], 8)):
            with self.subTest(arguments=arguments):
                options = option_parser().parse_args(arguments)
                observation = SensorObservationConfig(history_steps=options.history_steps)
                self.assertEqual(observation.frame_count, frames)

    def test_invalid_or_conflicting_history_options_are_rejected(self) -> None:
        for arguments in (['--frame-count', '0'], ['--frame-count', '-1'],
                          ['--frame-count', '8', '--history-steps', '5']):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    option_parser().parse_args(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_omitted_optional_parameters_preserve_distillation_defaults(self) -> None:
        environment = Mock()
        with (
            patch.object(sys, 'argv', ['train_distill.py']),
            patch.multiple(
                base_train,
                parse_args=base_train.parse_args,
                PPOAMPConfig=SensorDistillationPPOAMPConfig,
                LytRobotAmpEnv=environment,
                LytRobotWarpAmpEnv=Mock(),
                LytRobotMjlabAmpEnv=Mock(),
            ),
        ):
            install_training_options(base_train)
            cfg = base_train.PPOAMPConfig()
            base_train.LytRobotAmpEnv(object())
            self.assertEqual(environment.call_args.kwargs['observation_cfg'].frame_count, 6)
            self.assertEqual(cfg.entropy_coef, 0.003)
            self.assertEqual(cfg.init_std, 0.15)
            self.assertEqual(cfg.gamma, 0.995)
            self.assertEqual(cfg.gae_lambda, 0.97)


if __name__ == '__main__':
    unittest.main()
