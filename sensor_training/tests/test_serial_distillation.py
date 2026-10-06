"""Plan construction for the one-command serial-distillation pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sensor_training.train_serial_distillation import _phase_command, _phase_paths


class SerialDistillationPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            no_wandb=True,
            device="cuda",
            sim_backend="warp",
            num_envs=64,
            decimation=2,
            stage1_iterations=30000,
            stage2_iterations=30000,
        )
        self.paths = _phase_paths("10-06-12:00:00")

    def test_all_phase_paths_are_role_and_stage_specific(self) -> None:
        self.assertIn("distillation/10-06-12:00:00/teacher", str(self.paths["teacher_stage1"]))
        self.assertIn("distillation/10-06-12:00:00/teacher/stage2", str(self.paths["teacher_stage2"]))
        self.assertIn("distillation/10-06-12:00:00/student", str(self.paths["student_stage1"]))
        self.assertIn("distillation/10-06-12:00:00/student", str(self.paths["student_stage2"]))

        self.assertNotEqual(self.paths["student_stage1"], self.paths["student_stage2"])

    def test_fresh_stage2_commands_chain_the_prior_phase_checkpoints(self) -> None:
        teacher = _phase_command("teacher_stage2", self.args, self.paths, resume_checkpoint=None)
        student = _phase_command("student_stage2", self.args, self.paths, resume_checkpoint=None)

        self.assertIn("--stage1-checkpoint", teacher)
        self.assertIn(str(self.paths["teacher_stage1"] / "model_final.pt"), teacher)
        self.assertIn("--student-checkpoint", student)
        self.assertIn(str(self.paths["student_stage1"] / "model_final.pt"), student)
        self.assertIn("--teacher-checkpoint", student)
        self.assertIn(str(self.paths["teacher_stage2"] / "model_final.pt"), student)
        self.assertIn("--no-wandb", teacher)
        self.assertIn("--decimation", student)
        self.assertEqual(student[student.index("--decimation") + 1], "2")
        self.assertEqual(student[student.index("--num-steps") + 1], "48")
        # 64 teacher envs * 24 steps == 32 student envs * 48 steps.
        self.assertEqual(student[student.index("--num-envs") + 1], "32")

    def test_student_rejects_non_100hz_control(self) -> None:
        self.args.decimation = 4
        with self.assertRaisesRegex(ValueError, "requires 100 Hz"):
            _phase_command(
                "student_stage1",
                self.args,
                self.paths,
                resume_checkpoint=None,
            )

    def test_interrupted_student_stage2_uses_its_own_resume_mode(self) -> None:
        checkpoint = self.paths["student_stage2"] / "model_final.pt"
        command = _phase_command(
            "student_stage2",
            self.args,
            self.paths,
            resume_checkpoint=checkpoint,
        )

        self.assertIn("--resume-distillation-checkpoint", command)
        self.assertIn(str(checkpoint), command)
        self.assertNotIn("--student-checkpoint", command)

    def test_downstream_phases_prefer_a_retained_best_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                name: root / name
                for name in ("teacher_stage1", "teacher_stage2", "student_stage1", "student_stage2")
            }
            for name in ("teacher_stage1", "teacher_stage2", "student_stage1"):
                paths[name].mkdir()
                (paths[name] / "model_best.pt").touch()

            teacher = _phase_command("teacher_stage2", self.args, paths, resume_checkpoint=None)
            student = _phase_command("student_stage2", self.args, paths, resume_checkpoint=None)

            self.assertIn(str(paths["teacher_stage1"] / "model_best.pt"), teacher)
            self.assertIn(str(paths["teacher_stage2"] / "model_best.pt"), student)
            self.assertIn(str(paths["student_stage1"] / "model_best.pt"), student)


if __name__ == "__main__":
    unittest.main()
