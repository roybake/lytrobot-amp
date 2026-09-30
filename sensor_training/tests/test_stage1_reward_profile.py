"""Regression checks for the sensor task-reward profiles."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_entrypoint_argv(
    entrypoint: Path,
    *extra_args: str,
) -> list[str]:
    """Execute a thin entrypoint until it dispatches to its train module."""

    captured: dict[str, list[str]] = {}
    fake_runpy = types.ModuleType("runpy")

    def capture(_path: str, *, run_name: str) -> None:
        captured["argv"] = list(sys.argv[1:])

    fake_runpy.run_path = capture  # type: ignore[attr-defined]
    previous_runpy = sys.modules.get("runpy")
    previous_argv = sys.argv[:]
    try:
        sys.modules["runpy"] = fake_runpy
        sys.argv = [str(entrypoint), *extra_args]
        namespace = {
            "__name__": "__main__",
            "__file__": str(entrypoint),
            "__package__": None,
        }
        exec(compile(entrypoint.read_bytes(), str(entrypoint), "exec"), namespace)
    finally:
        sys.argv = previous_argv
        if previous_runpy is None:
            del sys.modules["runpy"]
        else:
            sys.modules["runpy"] = previous_runpy
    return captured["argv"]


def _remove_option(args: list[str], name: str) -> str:
    """Remove a two-token CLI option and return its configured value."""

    index = args.index(name)
    value = args[index + 1]
    del args[index : index + 2]
    return value


class SensorStage1RewardProfileTests(unittest.TestCase):
    def test_sensor_resume_accepts_schedule_changes_but_not_layout_changes(self) -> None:
        from sensor_training.trainer import _sensor_metadata_compatible

        saved = {
            "actor_input_fields": ["imu", "gait_command"],
            "gait_command": {
                "stand_start_phase_delay_steps": 50,
                "stand_start_transition_min_steps": 150,
                "stand_start_transition_max_steps": 250,
                "lookahead_seconds": [0.0, 0.1],
            },
        }
        current = {
            "actor_input_fields": ("imu", "gait_command"),
            "gait_command": {
                "stand_start_phase_delay_steps": 20,
                "stand_start_transition_min_steps": 80,
                "stand_start_transition_max_steps": 100,
                "lookahead_seconds": (0.0, 0.1),
            },
        }
        self.assertTrue(_sensor_metadata_compatible(saved, current))
        current["gait_command"]["lookahead_seconds"] = (0.0, 0.2)
        self.assertFalse(_sensor_metadata_compatible(saved, current))

    def test_sensor_stage1_is_independent_by_default(self) -> None:
        sensor_args = _default_entrypoint_argv(
            REPO_ROOT / "sensor_training" / "train_stage1.py",
        )

        self.assertNotIn("--teacher-checkpoint", sensor_args)
        self.assertNotIn("--teacher-action-loss-coef", sensor_args)

    def test_sensor_stage1_uses_deployable_reference_tracking(self) -> None:
        sensor_args = _default_entrypoint_argv(REPO_ROOT / "sensor_training" / "train_stage1.py")

        self.assertIn("--no-sensor-stability-reward", sensor_args)
        # A fixed 1e-5 rate (the old default) is 30x below the recipe that
        # produced the accepted gait and must not come back.
        self.assertNotIn("--no-adaptive-lr", sensor_args)
        self.assertNotIn("--sensor-natural-gait-reward-coef", sensor_args)
        self.assertEqual(_remove_option(sensor_args, "--reference-ctrl-blend"), "1.0")
        self.assertEqual(
            Path(_remove_option(sensor_args, "--motion-file")).name,
            "walk1_omniretarget.npz",
        )
        self.assertEqual(_remove_option(sensor_args, "--target-velocity"), "0.87")
        self.assertEqual(_remove_option(sensor_args, "--symmetry-reward-coef"), "1.2")
        self.assertEqual(_remove_option(sensor_args, "--symmetry-foot-pos-sigma"), "0.08")
        self.assertEqual(_remove_option(sensor_args, "--side-foot-pos-coef"), "0.0")
        self.assertEqual(_remove_option(sensor_args, "--step-gap-coef"), "0.0")
        self.assertEqual(_remove_option(sensor_args, "--swing-advance-coef"), "0.0")
        self.assertEqual(_remove_option(sensor_args, "--decimation"), "2")
        self.assertEqual(_remove_option(sensor_args, "--num-envs"), "3072")
        self.assertEqual(_remove_option(sensor_args, "--num-steps"), "64")
        self.assertEqual(_remove_option(sensor_args, "--stand-start-prob"), "0.5")
        self.assertEqual(_remove_option(sensor_args, "--motion-reset-prob"), "0.5")
        self.assertEqual(_remove_option(sensor_args, "--action-residual-scale"), "0.5")
        self.assertEqual(_remove_option(sensor_args, "--startup-action-ramp-steps"), "50")
        self.assertEqual(_remove_option(sensor_args, "--learning-rate"), "3.0e-4")
        self.assertEqual(_remove_option(sensor_args, "--min-learning-rate"), "1.0e-5")
        self.assertEqual(_remove_option(sensor_args, "--max-learning-rate"), "3.0e-4")
        self.assertEqual(_remove_option(sensor_args, "--clip-coef"), "0.2")
        self.assertEqual(_remove_option(sensor_args, "--target-kl"), "0.025")
        self.assertEqual(_remove_option(sensor_args, "--target-kl-stop-multiplier"), "2.0")
        self.assertEqual(_remove_option(sensor_args, "--min-action-std"), "0.05")
        self.assertEqual(_remove_option(sensor_args, "--max-action-std"), "0.6")
        self.assertEqual(_remove_option(sensor_args, "--task-reward-lerp"), "0.5")
        self.assertEqual(_remove_option(sensor_args, "--amp-reward-coef"), "1.2")
        self.assertEqual(_remove_option(sensor_args, "--fall-penalty"), "2.0")
        self.assertEqual(_remove_option(sensor_args, "--episode-steps"), "1200")
        self.assertEqual(_remove_option(sensor_args, "--stand-start-phase-delay-steps"), "25")
        self.assertEqual(_remove_option(sensor_args, "--stand-start-transition-min-steps"), "100")
        self.assertEqual(_remove_option(sensor_args, "--stand-start-transition-max-steps"), "200")
        self.assertEqual(_remove_option(sensor_args, "--min-forward-progress-speed"), "0.1")
        self.assertEqual(_remove_option(sensor_args, "--forward-progress-grace-steps"), "400")
        self.assertEqual(_remove_option(sensor_args, "--forward-progress-window-steps"), "100")
        self.assertEqual(
            _remove_option(sensor_args, "--forward-progress-termination-penalty"),
            "2.0",
        )

    def test_sensor_stage2_keeps_deployable_tracking_while_adding_robustness(self) -> None:
        sensor_args = _default_entrypoint_argv(REPO_ROOT / "sensor_training" / "train_stage2.py")

        self.assertIn("--no-sensor-stability-reward", sensor_args)
        self.assertNotIn("--no-adaptive-lr", sensor_args)
        self.assertNotIn("--sensor-natural-gait-reward-coef", sensor_args)
        self.assertEqual(_remove_option(sensor_args, "--reference-ctrl-blend"), "1.0")
        self.assertEqual(
            Path(_remove_option(sensor_args, "--motion-file")).name,
            "walk1_omniretarget.npz",
        )
        self.assertEqual(_remove_option(sensor_args, "--target-velocity"), "0.87")
        self.assertEqual(_remove_option(sensor_args, "--learning-rate"), "3.0e-4")
        self.assertEqual(_remove_option(sensor_args, "--task-reward-lerp"), "0.5")
        self.assertEqual(_remove_option(sensor_args, "--amp-reward-coef"), "1.2")
        self.assertEqual(_remove_option(sensor_args, "--action-residual-scale"), "0.5")
        self.assertEqual(_remove_option(sensor_args, "--min-action-std"), "0.05")
        self.assertEqual(_remove_option(sensor_args, "--max-action-std"), "0.6")
        self.assertEqual(_remove_option(sensor_args, "--max-iterations"), "30000")
        self.assertEqual(_remove_option(sensor_args, "--num-envs"), "3072")
        self.assertEqual(_remove_option(sensor_args, "--num-steps"), "64")
        self.assertEqual(_remove_option(sensor_args, "--stand-start-prob"), "0.5")
        self.assertEqual(_remove_option(sensor_args, "--motion-reset-prob"), "0.5")
        self.assertEqual(_remove_option(sensor_args, "--stand-start-phase-delay-steps"), "25")
        self.assertEqual(_remove_option(sensor_args, "--stand-start-transition-min-steps"), "100")
        self.assertEqual(_remove_option(sensor_args, "--stand-start-transition-max-steps"), "200")
        self.assertIn("--domain-randomization", sensor_args)

    def test_privileged_teacher_entrypoint_defaults_stay_untouched(self) -> None:
        # Sensor and privileged Stage1 must share the same task profile so a
        # comparison measures observation/network differences rather than a
        # different motion prior, target speed or reward surface.
        teacher_args = _default_entrypoint_argv(
            REPO_ROOT / "scripts" / "train_stage1.py"
        )
        sensor_args = _default_entrypoint_argv(
            REPO_ROOT / "sensor_training" / "train_stage1.py"
        )

        shared_options = (
            "--motion-file",
            "--target-velocity",
            "--symmetry-reward-coef",
            "--symmetry-foot-pos-sigma",
            "--side-foot-pos-coef",
            "--step-gap-coef",
            "--swing-advance-coef",
        )
        for option in shared_options:
            with self.subTest(option=option):
                teacher_value = _remove_option(teacher_args, option)
                sensor_value = _remove_option(sensor_args, option)
                if option == "--motion-file":
                    teacher_value = Path(teacher_value).name
                    sensor_value = Path(sensor_value).name
                self.assertEqual(sensor_value, teacher_value)

    def test_sensor_entrypoints_require_100_hz(self) -> None:
        for entrypoint in ("train_stage1.py", "train_stage2.py"):
            with self.subTest(entrypoint=entrypoint, control_hz=100):
                args_100hz = _default_entrypoint_argv(
                    REPO_ROOT / "sensor_training" / entrypoint,
                )
                self.assertEqual(_remove_option(args_100hz, "--decimation"), "2")
                self.assertEqual(_remove_option(args_100hz, "--num-steps"), "64")
                self.assertEqual(_remove_option(args_100hz, "--num-envs"), "3072")

            with self.subTest(entrypoint=entrypoint, control_hz=50):
                with self.assertRaisesRegex(ValueError, "requires 100 Hz"):
                    _default_entrypoint_argv(
                        REPO_ROOT / "sensor_training" / entrypoint,
                        "--decimation", "4",
                    )


if __name__ == "__main__":
    unittest.main()
