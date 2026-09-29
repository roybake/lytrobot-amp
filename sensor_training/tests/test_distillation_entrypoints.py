"""CLI defaults for the two clearly separated distillation log groups."""

from __future__ import annotations

import sys
import unittest

from sensor_training import distillation_entrypoints


class DistillationEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_argv = sys.argv[:]
        self.previous_run_path = distillation_entrypoints.runpy.run_path
        self.captured: list[str] | None = None
        self.captured_path: str | None = None
        distillation_entrypoints.runpy.run_path = self._capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        sys.argv = self.previous_argv
        distillation_entrypoints.runpy.run_path = self.previous_run_path  # type: ignore[assignment]

    def _capture(self, path: str, *, run_name: str) -> None:
        self.assertEqual(run_name, "__main__")
        self.captured_path = path
        self.captured = list(sys.argv[1:])

    def test_stage1_uses_a_dedicated_gait_log_group(self) -> None:
        sys.argv = ["train_distill_stage1.py", "--teacher-checkpoint", "/tmp/teacher.pt"]
        distillation_entrypoints.run_stage1()

        assert self.captured is not None
        assert self.captured_path is not None
        self.assertTrue(self.captured_path.endswith("train_distill.py"))
        self.assertIn("--log-group", self.captured)
        self.assertEqual(self.captured[self.captured.index("--log-group") + 1], "stage1_gait")
        self.assertEqual(self.captured[self.captured.index("--max-iterations") + 1], "30000")
        self.assertEqual(self.captured[self.captured.index("--decimation") + 1], "2")
        self.assertEqual(self.captured[self.captured.index("--num-envs") + 1], "2048")
        self.assertEqual(self.captured[self.captured.index("--num-steps") + 1], "48")
        self.assertEqual(
            self.captured[self.captured.index("--compile-policy-chunk-steps") + 1],
            "8",
        )
        self.assertEqual(
            self.captured[self.captured.index("--reference-ctrl-blend") + 1],
            "0.0",
        )
        self.assertEqual(
            self.captured[self.captured.index("--motion-file") + 1].split("/")[-1],
            "walk1_omniretarget.npz",
        )
        self.assertEqual(
            self.captured[self.captured.index("--sensor-knee-gait-reward-coef") + 1],
            "1.0",
        )
        self.assertIn("--no-domain-randomization", self.captured)

    def test_stage2_resumes_student_and_uses_a_robust_log_group(self) -> None:
        sys.argv = [
            "train_distill_stage2.py",
            "--student-checkpoint", "/tmp/student.pt",
            "--teacher-checkpoint", "/tmp/teacher.pt",
        ]
        distillation_entrypoints.run_stage2()

        assert self.captured is not None
        self.assertNotIn("--student-checkpoint", self.captured)
        self.assertEqual(
            self.captured[self.captured.index("--resume-checkpoint") + 1],
            "/tmp/student.pt",
        )
        self.assertIn("--resume-reset-progress", self.captured)
        self.assertIn("--distillation-reset-schedule", self.captured)
        self.assertIn("--domain-randomization", self.captured)
        self.assertEqual(self.captured[self.captured.index("--log-group") + 1], "stage2_robust")
        self.assertEqual(self.captured[self.captured.index("--max-iterations") + 1], "30000")
        self.assertEqual(self.captured[self.captured.index("--decimation") + 1], "2")
        self.assertEqual(self.captured[self.captured.index("--num-envs") + 1], "2048")
        self.assertEqual(self.captured[self.captured.index("--num-steps") + 1], "48")
        self.assertEqual(
            self.captured[self.captured.index("--compile-policy-chunk-steps") + 1],
            "8",
        )

    def test_stage1_rejects_non_100hz_control(self) -> None:
        sys.argv = [
            "train_distill_stage1.py",
            "--teacher-checkpoint", "/tmp/teacher.pt",
            "--decimation", "4",
        ]
        with self.assertRaisesRegex(ValueError, "requires 100 Hz"):
            distillation_entrypoints.run_stage1()

    def test_stage2_resume_preserves_its_distillation_schedule(self) -> None:
        sys.argv = [
            "train_distill_stage2.py",
            "--resume-distillation-checkpoint", "/tmp/stage2.pt",
            "--teacher-checkpoint", "/tmp/teacher.pt",
        ]
        distillation_entrypoints.run_stage2()

        assert self.captured is not None
        self.assertEqual(
            self.captured[self.captured.index("--resume-checkpoint") + 1],
            "/tmp/stage2.pt",
        )
        self.assertNotIn("--resume-reset-progress", self.captured)
        self.assertNotIn("--distillation-reset-schedule", self.captured)


if __name__ == "__main__":
    unittest.main()
