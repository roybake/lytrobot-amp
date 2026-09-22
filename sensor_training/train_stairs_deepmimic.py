#!/usr/bin/env python3
"""Train stair up/down imitation with a deployable IMU+tendon actor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_stairs import (  # noqa: E402
    REFERENCE_DIR,
    STAIR_EPISODE_STEPS,
    STAIR_MAX_ITERATIONS,
    STAIR_UP_DOWN_EPISODE_STEPS,
    _validate_fixed_geometry,
    default_train_args,
)


EXACT_STAIR_REFERENCE = REFERENCE_DIR / "stairs.npz"
EXACT_STAIR_MODEL = REPO_ROOT / "robot" / "terrains" / "stairs.xml"
# 912 frames at 40 Hz need 2,280 control steps at 100 Hz.  Leave enough room
# for the fixed 30-step terminal hold used by the stair completion contract.
EXACT_STAIR_EPISODE_STEPS = 2_400
EXACT_STAIR_STEP_COUNT = 3
EXACT_STAIR_TOP_LENGTH_M = 2.19646


def _has_option(argv: list[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in argv)


def _validate_exact_contract(user_args: list[str]) -> None:
    """Reject overrides that could de-synchronise the solved clip and MJCF."""

    for option in (
        "--model-xml",
        "--motion-file",
        "--motion-files",
        "--motion-names",
        "--motion-weights",
        "--motion-loop",
        "--terrain-types",
        "--terrain-embedded-in-model",
        "--no-terrain-embedded-in-model",
        "--preserve-motion-world-xy",
        "--no-preserve-motion-world-xy",
        "--stair-terrain",
        "--no-stair-terrain",
        "--stair-fixed-start",
        "--no-stair-fixed-start",
        "--stair-step-height",
        "--stair-step-depth",
        "--stair-tread-width",
        "--stair-step-count",
        "--stair-width",
        "--stair-top-length",
        "--stair-distance-tolerance",
        "--stair-distance-failure-penalty",
        "--stair-lateral-tolerance",
        "--stair-lateral-failure-penalty",
        "--stair-terminal-hold-steps",
        "--episode-steps",
        "--decimation",
        "--motion-reset-prob",
        "--stand-start-prob",
        "--reset-noise-scale",
        "--action-noise-std",
        "--reference-ctrl-blend",
        "--action-residual-scale",
    ):
        if _has_option(user_args, option):
            raise SystemExit(f"精确楼梯 DeepMimic 入口不接受 {option} 覆盖。")


def exact_train_args() -> list[str]:
    """Return the fixed contract for stairs.npz + stairs.xml."""

    return [
        "--model-xml", str(EXACT_STAIR_MODEL),
        "--motion-files", str(EXACT_STAIR_REFERENCE),
        "--motion-names", "stairs",
        "--motion-weights", "1.0",
        "--motion-loop", "false",
        "--no-skill-conditioning",
        "--stair-terrain",
        "--terrain-embedded-in-model",
        "--preserve-motion-world-xy",
        "--stair-fixed-start",
        "--stair-step-height", "0.13",
        "--stair-step-depth", "0.32",
        "--stair-step-count", str(EXACT_STAIR_STEP_COUNT),
        "--stair-top-length", f"{EXACT_STAIR_TOP_LENGTH_M:g}",
        "--stair-width", "2.0",
        "--reference-ctrl-blend", "1.0",
        "--action-residual-scale", "0.5",
        "--use-reference-velocity",
        "--motion-reset-prob", "0.0",
        "--stand-start-prob", "0.0",
        "--reset-noise-scale", "0.0",
        "--action-noise-std", "0.0",
        "--no-episode-random-seed",
        "--episode-steps", str(EXACT_STAIR_EPISODE_STEPS),
        "--episode-timeout-jitter-steps", "0",
        "--stair-distance-tolerance", "0.05",
        "--stair-distance-failure-penalty", "5.0",
        "--stair-lateral-tolerance", "0.30",
        "--stair-lateral-failure-penalty", "5.0",
        "--stair-terminal-hold-steps", "30",
        "--decimation", "2",
        "--wandb-tags",
        "lytrobot,deepmimic,sensor-only,stairs,exact-xml,continuous-up-down",
    ]


def build_train_args(
    direction: str = "exact", user_args: list[str] | None = None
) -> list[str]:
    """Keep reward, sensor architecture and reference controls unambiguous."""
    user_args = list(user_args or ())
    if direction == "exact":
        _validate_exact_contract(user_args)
        args = exact_train_args()
    else:
        episode_steps = (
            STAIR_UP_DOWN_EPISODE_STEPS if direction == "up_down"
            else STAIR_EPISODE_STEPS
        )
        _validate_fixed_geometry(user_args, episode_steps)
        args = default_train_args(direction)
    for option in (
        "--amp", "--no-amp", "--deepmimic-reward", "--task-reward-lerp",
        "--sensor-stability-reward", "--no-sensor-stability-reward",
        "--lcp-weight", "--deepmimic-ppo", "--motion-file", "--motion-files", "--motion-names",
        "--use-reference-velocity", "--no-use-reference-velocity",
    ):
        if any(arg == option or arg.startswith(option + "=") for arg in user_args):
            raise SystemExit(f"纯传感器 DeepMimic 楼梯入口不接受 {option} 覆盖。")
    args += [
        "--deepmimic-reward", "--no-amp", "--task-reward-lerp", "1.0",
        # The onboard command generator stores one deployable velocity command,
        # not a privileged framewise root-velocity lookup.
        "--no-use-reference-velocity",
        "--lcp-weight", "0.002", "--deepmimic-ppo", "--stage", "stage1",
        "--max-iterations", str(STAIR_MAX_ITERATIONS),
        "--device", "cuda", "--sim-backend", "warp",
        "--no-domain-randomization", "--log-root",
        str(REPO_ROOT / "logs" / "sensor_policy"),
        "--log-group", f"stair_{direction}_deepmimic_sensor",
    ]
    return args + user_args


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--direction",
        choices=("exact", "up", "down", "up_down"),
        default="exact",
        help="exact 使用 stairs.npz + stairs.xml；其余值保留旧分片课程。",
    )
    parser.add_argument("--check-data", action="store_true")
    options, user_args = parser.parse_known_args()
    clip = (
        EXACT_STAIR_REFERENCE
        if options.direction == "exact"
        else REFERENCE_DIR / f"stair_{options.direction}_omniretarget.npz"
    )
    model_xml = (
        EXACT_STAIR_MODEL
        if options.direction == "exact"
        else REPO_ROOT / "robot" / "MJCFmodelposition.xml"
    )
    if not clip.is_file():
        raise SystemExit(
            f"缺少楼梯机器人参考：{clip}\n"
            "补齐重定向数据后先运行 --check-data，再启动纯传感器训练。"
        )
    if options.check_data:
        import mujoco
        from lytrobot_amp.motion import MotionReference

        model = mujoco.MjModel.from_xml_path(str(model_xml))
        reference = MotionReference(
            clip,
            model,
            zero_root_xy=options.direction != "exact",
            loop_override=False,
        )
        duration = (reference.frame_count - 1) / reference.frequency
        displacement = (
            reference.reference_qpos[-1, :2]
            - reference.reference_qpos[0, :2]
        )
        print(
            f"楼梯数据可加载：model={model_xml}, motion={clip}, "
            f"frames={reference.frame_count}, frequency={reference.frequency:g}Hz, "
            f"duration={duration:.3f}s, root_xy_delta={displacement.tolist()}"
        )
        return

    args = build_train_args(options.direction, user_args)
    import sensor_training.train as sensor_train
    from sensor_training.stair_deepmimic import (
        SensorStairDeepMimicConfig,
        SensorStairDeepMimicTrainer,
    )

    sensor_train.SensorPPOAMPConfig = SensorStairDeepMimicConfig
    sensor_train.SensorHistoryPPOAMPTrainer = SensorStairDeepMimicTrainer
    sys.argv = [str(REPO_ROOT / "sensor_training" / "train.py"), *args]
    sensor_train.main()


if __name__ == "__main__":
    main()
