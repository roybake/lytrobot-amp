from __future__ import annotations

from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import train
from scripts.train_multimotion import default_train_args as multimotion_args
from scripts.train_multimotion_no_stairs import default_train_args as no_stairs_args
from scripts.train_multimotion_no_stairs_stage1 import main as no_stairs_stage1_main
from scripts.train_multimotion_no_stairs_stage2 import main as no_stairs_stage2_main
from scripts.train_stairs import default_train_args as stairs_args
from scripts.two_stage_task import build_task_stage_args
from sensor_training.train_multimotion import default_train_args as sensor_multimotion_args


ROOT = Path(__file__).resolve().parents[1]


class TwoStageTrainingTaskTests(unittest.TestCase):
    def test_task_launchers_use_the_requested_log_trees(self) -> None:
        expected_roots = {
            "scripts/train_multimotion_stage1.py": ROOT / "logs" / "multimotion",
            "scripts/train_multimotion_stage2.py": ROOT / "logs" / "multimotion",
            "sensor_training/train_multimotion_stage1.py": ROOT / "logs" / "multimotion",
            "sensor_training/train_multimotion_stage2.py": ROOT / "logs" / "multimotion",
            "scripts/train_stairs_stage1.py": ROOT / "logs" / "stairs",
            "scripts/train_stairs_stage2.py": ROOT / "logs" / "stairs",
        }
        for launcher, expected_root in expected_roots.items():
            with self.subTest(launcher=launcher), patch(
                "scripts.two_stage_task.run_task_stage"
            ) as mocked_run:
                runpy.run_path(str(ROOT / launcher), run_name="__main__")
                self.assertEqual(mocked_run.call_args.kwargs["log_root"], expected_root)

    def test_stair_free_launchers_call_the_canonical_walking_stages(self) -> None:
        cases = (
            (
                no_stairs_stage1_main,
                "scripts.train_multimotion_no_stairs_stage1.run_stage1",
            ),
            (
                no_stairs_stage2_main,
                "scripts.train_multimotion_no_stairs_stage2.run_stage2",
            ),
        )
        for launcher, target in cases:
            with self.subTest(target=target), patch(target) as canonical_stage, patch.object(
                sys, "argv", ["launcher.py", "--no-wandb"]
            ):
                launcher()
                canonical_stage.assert_called_once_with()
                self.assertEqual(
                    sys.argv[sys.argv.index("--training-task") + 1],
                    "multimotion_no_stairs",
                )
                self.assertEqual(
                    Path(sys.argv[sys.argv.index("--log-root") + 1]),
                    ROOT / "logs" / "multimotion" / "no_stairs",
                )
                self.assertIn("--no-wandb", sys.argv)

    def test_task_defaults_do_not_enable_domain_randomization(self) -> None:
        for args in (no_stairs_args(), stairs_args(), sensor_multimotion_args()):
            self.assertNotIn("--domain-randomization", args)
            self.assertNotIn("--warp-global-domain-randomization", args)

    def test_stage1_forces_all_domain_randomization_off(self) -> None:
        args = build_task_stage_args(
            [
                "--domain-randomization",
                "--warp-global-domain-randomization",
                "--global-dr-resample-interval",
                "99",
                "--no-wandb",
            ],
            stage="stage1",
            log_root=ROOT / "logs" / "sensor_policy" / "test_task",
        )
        self.assertIn("--no-domain-randomization", args)
        self.assertNotIn("--domain-randomization", args)
        self.assertIn("--no-warp-global-domain-randomization", args)
        self.assertNotIn("--warp-global-domain-randomization", args)
        self.assertEqual(args[args.index("--global-dr-resample-interval") + 1], "0")
        self.assertEqual(args[args.index("--stage") + 1], "stage1")
        self.assertEqual(args[args.index("--log-group") + 1], "stage1")
        self.assertIn("--no-wandb", args)

    def test_stage2_requires_a_stage1_or_stage2_checkpoint(self) -> None:
        with self.assertRaisesRegex(SystemExit, "requires --stage1-checkpoint"):
            build_task_stage_args(
                [],
                stage="stage2",
                log_root=ROOT / "logs" / "sensor_policy" / "test_task",
            )

    def test_stage2_maps_stage1_checkpoint_and_forces_randomization(self) -> None:
        args = build_task_stage_args(
            [
                "--stage1-checkpoint",
                "/tmp/stage1.pt",
                "--no-domain-randomization",
                "--no-warp-global-domain-randomization",
            ],
            stage="stage2",
            log_root=ROOT / "logs" / "sensor_policy" / "test_task",
        )
        self.assertEqual(args[args.index("--resume-checkpoint") + 1], "/tmp/stage1.pt")
        self.assertIn("--resume-reset-progress", args)
        self.assertIn("--resume-reset-optimizer", args)
        self.assertIn("--domain-randomization", args)
        self.assertNotIn("--no-domain-randomization", args)
        self.assertIn("--warp-global-domain-randomization", args)
        self.assertNotIn("--no-warp-global-domain-randomization", args)
        self.assertEqual(args[args.index("--dr-curriculum-mode") + 1], "adaptive")
        self.assertEqual(args[args.index("--dr-curriculum-initial-scale") + 1], "0.0")
        self.assertEqual(args[args.index("--global-dr-resample-interval") + 1], "250")
        self.assertEqual(args[args.index("--stage") + 1], "stage2")
        self.assertEqual(args[args.index("--log-group") + 1], "stage2")

    def test_every_missing_task_now_has_two_stage_launchers(self) -> None:
        launchers = (
            "train_multimotion_stage1.py",
            "train_multimotion_stage2.py",
            "train_multimotion_no_stairs_stage1.py",
            "train_multimotion_no_stairs_stage2.py",
            "train_stairs_stage1.py",
            "train_stairs_stage2.py",
            "train_multimotion_walk_stage1.py",
            "train_multimotion_walk_stage2.py",
        )
        for launcher in launchers:
            self.assertTrue((ROOT / "scripts" / launcher).is_file(), launcher)
        self.assertTrue(
            (ROOT / "sensor_training" / "train_multimotion_stage1.py").is_file()
        )
        self.assertTrue(
            (ROOT / "sensor_training" / "train_multimotion_stage2.py").is_file()
        )

    def test_effective_parser_contract_for_every_new_two_stage_task(self) -> None:
        tasks = (
            ("multimotion", multimotion_args()),
            ("stairs_up_down", stairs_args()),
        )
        for task_name, task_args in tasks:
            log_root = ROOT / "logs" / "sensor_policy" / task_name
            stage1 = build_task_stage_args([], stage="stage1", log_root=log_root)
            with patch.object(sys, "argv", ["train.py", *task_args, *stage1]):
                parsed_stage1 = train.parse_args()
            self.assertEqual(parsed_stage1.stage, "stage1")
            self.assertFalse(parsed_stage1.domain_randomization)
            self.assertFalse(parsed_stage1.warp_global_domain_randomization)
            self.assertEqual(parsed_stage1.log_group, "stage1")
            self.assertEqual(parsed_stage1.training_task, task_name)
            self.assertIsNone(parsed_stage1.seed)
            self.assertTrue(parsed_stage1.episode_random_seed)

            stage2 = build_task_stage_args(
                ["--stage1-checkpoint", "/tmp/stage1.pt"],
                stage="stage2",
                log_root=log_root,
            )
            with patch.object(sys, "argv", ["train.py", *task_args, *stage2]):
                parsed_stage2 = train.parse_args()
            self.assertEqual(parsed_stage2.stage, "stage2")
            self.assertTrue(parsed_stage2.domain_randomization)
            self.assertTrue(parsed_stage2.warp_global_domain_randomization)
            self.assertEqual(parsed_stage2.dr_curriculum_mode, "adaptive")
            self.assertEqual(parsed_stage2.log_group, "stage2")
            self.assertEqual(parsed_stage2.training_task, task_name)
            self.assertIsNone(parsed_stage2.seed)
            self.assertTrue(parsed_stage2.episode_random_seed)

    def test_training_engine_rejects_unrandomized_stage2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_final.pt"
            checkpoint.touch()
            argv = [
                "train.py",
                "--stage",
                "stage2",
                "--resume-checkpoint",
                str(checkpoint),
                "--no-domain-randomization",
            ]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(
                ValueError,
                "requires domain randomization",
            ):
                train.main()

    def test_training_engine_rejects_stage2_without_checkpoint(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["train.py", "--stage", "stage2", "--domain-randomization"],
        ), self.assertRaisesRegex(ValueError, "requires a Stage 1 checkpoint"):
            train.main()

    def test_training_engine_rejects_stage2_without_warp_shared_randomization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_final.pt"
            checkpoint.touch()
            argv = [
                "train.py",
                "--stage",
                "stage2",
                "--resume-checkpoint",
                str(checkpoint),
                "--domain-randomization",
                "--no-warp-global-domain-randomization",
            ]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(
                ValueError,
                "requires Warp shared-parameter domain randomization",
            ):
                train.main()


if __name__ == "__main__":
    unittest.main()
