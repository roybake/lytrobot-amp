from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

from scripts.train import (
    make_log_dir,
    terrain_course_random_start_prob_at_iteration,
)
from scripts.train_multimotion import (
    AMP_DISCRIMINATOR_UPDATES,
    AMP_MINIBATCH_SIZE,
    COURSE_RANDOM_START_RAMP_ITERATIONS,
    COURSE_RANDOM_START_WARMUP_ITERATIONS,
    PARALLEL_ENVS,
    PPO_MINIBATCHES,
    ROLLOUT_STEPS,
    default_train_args,
)
from lytrobot_amp.training_defaults import (
    MULTIMOTION_STAIR_STEP_DEPTH_M,
    MULTIMOTION_STAIR_STEP_HEIGHT_M,
    ROUGH_TERRAIN_HEIGHT_M,
    SLOPE_ANGLE_DEG,
    TRAINING_TARGET_SPEED_MPS,
)


class MultimotionLogPathTests(unittest.TestCase):
    def test_multimotion_defaults_use_group_and_no_fixed_run_name(self) -> None:
        args = default_train_args()

        group_index = args.index("--log-group")
        self.assertEqual(args[group_index + 1], "multimotion")
        self.assertNotIn("--run-name", args)

    def test_multimotion_defaults_reward_forward_course_progress(self) -> None:
        args = default_train_args()

        def value(option: str) -> float:
            return float(args[args.index(option) + 1])

        self.assertGreater(value("--forward-velocity-reward-coef"), 0.0)
        self.assertAlmostEqual(
            value("--target-velocity"),
            TRAINING_TARGET_SPEED_MPS,
            places=6,
        )
        self.assertGreater(value("--forward-progress-reward-coef"), 0.0)
        self.assertGreater(value("--forward-velocity-deficit-penalty-coef"), 0.0)
        self.assertLessEqual(value("--forward-velocity-tracking-sigma"), 0.20)
        self.assertGreater(
            value("--forward-velocity-excess-penalty-coef"),
            value("--forward-velocity-deficit-penalty-coef"),
        )
        self.assertGreater(value("--min-forward-progress-speed"), 0.0)
        self.assertGreater(value("--forward-progress-window-steps"), 0.0)
        self.assertGreaterEqual(value("--min-forward-progress-speed"), 0.10)
        self.assertEqual(value("--fall-penalty"), 10.0)
        self.assertEqual(value("--startup-action-ramp-steps"), 0.0)
        self.assertIn("--terminate-incomplete-motion-end", args)
        self.assertGreater(value("--side-foot-pos-coef"), 0.0)
        self.assertGreater(value("--step-gap-coef"), 0.0)
        self.assertGreater(value("--swing-advance-coef"), 0.0)
        self.assertGreater(value("--foot-slip-penalty-coef"), 0.0)
        self.assertIn("--terrain-course-phase-sync", args)
        self.assertGreater(value("--terrain-course-phase-lookahead"), 0.0)
        self.assertEqual(value("--motion-transition-blend-steps"), 15.0)
        self.assertGreater(value("--terrain-course-stage-reward"), 0.0)
        self.assertGreater(value("--terrain-course-random-start-prob"), 0.0)
        self.assertLess(value("--terrain-course-random-start-prob"), 1.0)
        self.assertEqual(
            value("--terrain-course-random-start-warmup-iterations"),
            COURSE_RANDOM_START_WARMUP_ITERATIONS,
        )
        self.assertEqual(
            value("--terrain-course-random-start-ramp-iterations"),
            COURSE_RANDOM_START_RAMP_ITERATIONS,
        )

        names = args[
            args.index("--motion-names") + 1 : args.index("--motion-weights")
        ]
        self.assertEqual(
            names,
            ["walk", "rough_walk", "stair_up", "stair_down"],
        )

    def test_random_stage_curriculum_preserves_full_route_warmup(self) -> None:
        target = 0.80
        warmup = COURSE_RANDOM_START_WARMUP_ITERATIONS
        ramp = COURSE_RANDOM_START_RAMP_ITERATIONS

        self.assertGreater(warmup, 0)
        self.assertGreater(ramp, 0)
        self.assertEqual(
            terrain_course_random_start_prob_at_iteration(
                warmup - 1, target, warmup, ramp
            ),
            0.0,
        )
        self.assertEqual(
            terrain_course_random_start_prob_at_iteration(
                warmup, target, warmup, ramp
            ),
            0.0,
        )
        self.assertAlmostEqual(
            terrain_course_random_start_prob_at_iteration(
                warmup + ramp // 2, target, warmup, ramp
            ),
            target / 2.0,
        )
        self.assertEqual(
            terrain_course_random_start_prob_at_iteration(
                warmup + ramp, target, warmup, ramp
            ),
            target,
        )

    def test_multimotion_defaults_are_balanced_for_4096_parallel_envs(self) -> None:
        args = default_train_args()

        def integer(option: str) -> int:
            return int(args[args.index(option) + 1])

        self.assertEqual(integer("--num-envs"), PARALLEL_ENVS)
        self.assertEqual(integer("--num-steps"), ROLLOUT_STEPS)
        self.assertEqual(integer("--minibatches"), PPO_MINIBATCHES)
        self.assertEqual(integer("--amp-minibatch-size"), AMP_MINIBATCH_SIZE)
        self.assertEqual(
            integer("--amp-disc-updates"), AMP_DISCRIMINATOR_UPDATES
        )
        self.assertEqual(integer("--episode-metric-window"), PARALLEL_ENVS)
        rollout_batch = PARALLEL_ENVS * ROLLOUT_STEPS
        self.assertEqual(rollout_batch, 98_304)
        self.assertEqual(rollout_batch % PPO_MINIBATCHES, 0)
        self.assertEqual(rollout_batch // PPO_MINIBATCHES, 12_288)
        self.assertEqual(integer("--warp-nconmax"), 32)
        self.assertEqual(integer("--warp-njmax"), 128)

    def test_multimotion_defaults_cover_training_terrain_without_energy_penalty(self) -> None:
        args = default_train_args()

        def value(option: str) -> float:
            return float(args[args.index(option) + 1])

        start = args.index("--terrain-types") + 1
        end = next(index for index in range(start, len(args)) if args[index].startswith("--"))
        terrains = set(args[start:end])
        self.assertEqual(terrains, {"rough", "slope_up", "slope_down", "stairs"})
        self.assertEqual(value("--rough-terrain-height"), ROUGH_TERRAIN_HEIGHT_M)
        self.assertEqual(value("--slope-angle-deg"), SLOPE_ANGLE_DEG)
        self.assertEqual(value("--stair-step-height"), MULTIMOTION_STAIR_STEP_HEIGHT_M)
        self.assertEqual(value("--stair-step-depth"), MULTIMOTION_STAIR_STEP_DEPTH_M)
        self.assertNotIn("--specific-cot-limit", args)
        self.assertEqual(value("--specific-cot-penalty-coef"), 0.0)

    def test_multimotion_group_creates_timestamp_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = make_log_dir(
                directory,
                stage="stage1",
                run_name=None,
                log_group="multimotion",
            )

            self.assertEqual(log_dir.parent, Path(directory).resolve() / "multimotion")
            self.assertRegex(log_dir.name, re.compile(r"\d{2}-\d{2}-\d{2}:\d{2}:\d{2}"))
            self.assertTrue(log_dir.is_dir())

    def test_default_training_still_groups_by_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = make_log_dir(directory, stage="stage2", run_name="named")

            self.assertEqual(log_dir, Path(directory).resolve() / "stage2" / "named")


if __name__ == "__main__":
    unittest.main()
