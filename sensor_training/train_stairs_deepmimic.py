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


def build_train_args(direction: str, user_args: list[str]) -> list[str]:
    """Keep reward, sensor architecture and reference controls unambiguous."""
    episode_steps = (
        STAIR_UP_DOWN_EPISODE_STEPS if direction == "up_down"
        else STAIR_EPISODE_STEPS
    )
    _validate_fixed_geometry(user_args, episode_steps)
    for option in (
        "--amp", "--no-amp", "--deepmimic-reward", "--task-reward-lerp",
        "--sensor-stability-reward", "--no-sensor-stability-reward",
        "--lcp-weight", "--deepmimic-ppo", "--motion-file", "--motion-files", "--motion-names",
        "--use-reference-velocity", "--no-use-reference-velocity",
    ):
        if any(arg == option or arg.startswith(option + "=") for arg in user_args):
            raise SystemExit(f"纯传感器 DeepMimic 楼梯入口不接受 {option} 覆盖。")
    args = default_train_args(direction)
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
    parser.add_argument("--direction", choices=("up", "down", "up_down"), default="up_down")
    parser.add_argument("--check-data", action="store_true")
    options, user_args = parser.parse_known_args()
    clip = REFERENCE_DIR / f"stair_{options.direction}_omniretarget.npz"
    if not clip.is_file():
        raise SystemExit(
            f"缺少楼梯机器人参考：{clip}\n"
            "补齐重定向数据后先运行 --check-data，再启动纯传感器训练。"
        )
    if options.check_data:
        import mujoco
        from lytrobot_amp.motion import MotionReference

        model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"))
        MotionReference(clip, model, loop_override=False)
        print(f"楼梯参考可加载：{clip}")
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
