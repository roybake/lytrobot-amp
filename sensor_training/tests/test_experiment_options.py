"""Sensor recipe CLI options must reach the environment and PPO config."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import train as base_train
from sensor_training import distillation_entrypoints
from sensor_training import train as sensor_train
from sensor_training import train_stairs_deepmimic as stair_train
from sensor_training.config import SensorObservationConfig
from sensor_training.distillation_config import SensorDistillationPPOAMPConfig
from sensor_training.distillation_recipes.run_improved import TRAINING_ARGS
from sensor_training.experiment_options import install_training_options, option_parser
from sensor_training.stair_deepmimic import SensorStairDeepMimicConfig
from lytrobot_amp.env import LytRobotEnvCfg


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

    def test_stair_entrypoint_records_curriculum_in_args_and_ppo_config(self) -> None:
        for arguments, expected in (
            (['--stair-rsi-warmup-iterations', '5000',
              '--stair-rsi-ramp-iterations', '20000'], (5000, 20000)),
            (['--stair-rsi-warmup-iterations=5000',
              '--stair-rsi-ramp-iterations=20000'], (5000, 20000)),
            (['--stair-rsi-warmup-iterations', '0',
              '--stair-rsi-ramp-iterations', '0'], (0, 0)),
            ([], (2000, 10000)),
        ):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                def configure_training() -> None:
                    args = base_train.parse_args()
                    cfg = base_train.PPOAMPConfig(num_steps=args.num_steps)
                    self.assertIsInstance(cfg, SensorStairDeepMimicConfig)
                    self.assertEqual(args.num_envs, 4096)
                    self.assertEqual(args.max_iterations, 40000)
                    self.assertEqual(cfg.privileged_context_loss_coef, 0.0)
                    base_train.save_config(Path(directory), args, LytRobotEnvCfg(), cfg)
                    saved = json.loads((Path(directory) / 'params' / 'config.json').read_text())
                    for name, value in zip(
                        ('stair_rsi_warmup_iterations', 'stair_rsi_ramp_iterations'), expected,
                    ):
                        self.assertEqual(saved['ppo_amp'][name], value)
                        if arguments:
                            self.assertEqual(saved['args'][name], value)

                with (
                    patch.object(sys, 'argv', ['train_stairs_deepmimic.py', *arguments,
                                              '--max-iterations', '40000', '--num-envs', '4096']),
                    patch.multiple(
                        sensor_train,
                        SensorPPOAMPConfig=sensor_train.SensorPPOAMPConfig,
                        SensorHistoryPPOAMPTrainer=sensor_train.SensorHistoryPPOAMPTrainer,
                    ),
                    patch.multiple(
                        base_train,
                        parse_args=base_train.parse_args,
                        PPOAMPConfig=base_train.PPOAMPConfig,
                        PPOAMPTrainer=base_train.PPOAMPTrainer,
                        LytRobotAmpEnv=base_train.LytRobotAmpEnv,
                        LytRobotWarpAmpEnv=base_train.LytRobotWarpAmpEnv,
                        LytRobotMjlabAmpEnv=base_train.LytRobotMjlabAmpEnv,
                        main=configure_training,
                    ),
                    patch('scripts.audit_stair_reference.audit_reference',
                          return_value={'alignment_pass': True, 'contact_geometry_pass': True}),
                ):
                    stair_train.main()

    def test_invalid_stair_curriculum_iterations_are_rejected(self) -> None:
        for name in ('--stair-rsi-warmup-iterations', '--stair-rsi-ramp-iterations'):
            for value in ('-1', '1.5', '5000l'):
                with self.subTest(name=name, value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        option_parser(SensorStairDeepMimicConfig).parse_args([name, value])
                    self.assertEqual(error.exception.code, 2)

    def test_stair_curriculum_options_remain_isolated_from_other_sensor_training(self) -> None:
        for config in (sensor_train.SensorPPOAMPConfig, SensorDistillationPPOAMPConfig):
            arguments = ['--stair-rsi-warmup-iterations', '5000']
            with self.subTest(config=config):
                options, remaining = option_parser(config).parse_known_args(arguments)
                self.assertEqual(remaining, arguments)
                self.assertFalse(hasattr(options, 'stair_rsi_warmup_iterations'))


if __name__ == '__main__':
    unittest.main()
