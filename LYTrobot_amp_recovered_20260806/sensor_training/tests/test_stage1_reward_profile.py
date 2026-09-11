"""Regression checks for the sensor task-reward profiles."""

from __future__ import annotations

import sys
import types
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_entrypoint_argv(entrypoint: Path) -> list[str]:
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
        sys.argv = [str(entrypoint)]
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
    def test_sensor_stage1_uses_the_base_stage1_reward_profile(self) -> None:
        base_args = _default_entrypoint_argv(REPO_ROOT / "scripts" / "train_stage1.py")
        sensor_args = _default_entrypoint_argv(REPO_ROOT / "sensor_training" / "train_stage1.py")

        # Sensor-only deployment must not obtain the reference controller as
        # an unobserved gait-phase input.  Apart from this actuator-baseline
        # setting, every default—including the reward target—is inherited
        # verbatim from the ordinary stage1 entrypoint.
        _remove_option(sensor_args, "--reference-ctrl-blend")
        self.assertEqual(sensor_args, base_args)
        self.assertNotIn("--use-reference-velocity", sensor_args)
        self.assertNotIn("--use-reference-velocity", base_args)

    def test_sensor_stage2_keeps_the_stage1_reward_profile(self) -> None:
        base_args = _default_entrypoint_argv(REPO_ROOT / "scripts" / "train_stage2.py")
        sensor_args = _default_entrypoint_argv(REPO_ROOT / "sensor_training" / "train_stage2.py")

        _remove_option(sensor_args, "--reference-ctrl-blend")
        self.assertEqual(_remove_option(sensor_args, "--max-iterations"), "30000")
        _remove_option(base_args, "--max-iterations")
        self.assertEqual(Counter(sensor_args), Counter(base_args))
        for reward_override in (
            "--use-reference-velocity",
            "--forward-velocity-reward-coef",
            "--forward-progress-reward-coef",
            "--forward-velocity-deficit-penalty-coef",
            "--forward-velocity-excess-penalty-coef",
        ):
            self.assertNotIn(reward_override, sensor_args)


if __name__ == "__main__":
    unittest.main()
