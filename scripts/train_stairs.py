#!/usr/bin/env python3
"""固定楼梯动作课程的公共入口。

支持分别训练上楼/下楼，也支持加载一个连续的上楼—平台—下楼参考。所有
专项课程都关闭随机动作起点，并把动作参考作为位置控制基准。

通常直接使用两个薄包装：

    uv run python scripts/train_stair_up.py
    uv run python scripts/train_stair_down.py
    uv run python scripts/train_stair_up_down.py

本模块也支持 ``--direction up|down|up_down``，便于脚本化调用。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lytrobot_amp.terrain import (  # noqa: E402
    STAIR_TRAINING_STEP_COUNT,
    STAIR_TRAINING_STEP_HEIGHT_M,
    STAIR_TRAINING_TOP_LENGTH_M,
    STAIR_TRAINING_TREAD_WIDTH_M,
)


REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
STAIR_LATERAL_WIDTH_M = 2.0
STAIR_START_X_M = -0.95
STAIR_DOWN_OFFSET_M = 0.25
# The references are 3.53 s / 3.93 s long at 40 Hz.  At 100 Hz control this
# leaves a small fixed tail for the final stable contact before timeout.
STAIR_EPISODE_STEPS = 500
STAIR_UP_DOWN_EPISODE_STEPS = 2200
STAIR_MAX_ITERATIONS = 30_000


def _has_option(argv: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def _option_value(argv: list[str], option: str) -> float | str | None:
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(f"{option}="):
            return value.partition("=")[2]
    return None


def _validate_fixed_geometry(
    user_args: list[str], episode_steps: int = STAIR_EPISODE_STEPS
) -> None:
    """Reject overrides that would silently invalidate the stair reference."""

    expected = {
        "--stair-step-height": STAIR_TRAINING_STEP_HEIGHT_M,
        "--stair-step-depth": STAIR_TRAINING_TREAD_WIDTH_M,
        "--stair-tread-width": STAIR_TRAINING_TREAD_WIDTH_M,
        "--stair-step-count": STAIR_TRAINING_STEP_COUNT,
        "--stair-width": STAIR_LATERAL_WIDTH_M,
        "--stair-top-length": STAIR_TRAINING_TOP_LENGTH_M,
        "--stair-start-x": STAIR_START_X_M,
        "--stair-down-offset": STAIR_DOWN_OFFSET_M,
    }
    for option, target in expected.items():
        value = _option_value(user_args, option)
        if value is None:
            continue
        try:
            actual = float(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{option} 必须保持固定值 {target}。") from exc
        if abs(actual - float(target)) > 1.0e-7:
            raise SystemExit(
                f"专项楼梯课程禁止修改 {option}={target}；收到 {value!r}。"
            )

    # The wrappers are intentionally specialist entry points, not aliases for
    # the general multi-motion trainer.  In particular, allowing a second
    # ``--motion-files`` occurrence would make argparse select another clip and
    # silently disable the fixed-distance contract.
    for option in (
        "--model-xml",
        "--motion-file",
        "--motion-files",
        "--motion-names",
        "--motion-weights",
        "--motion-loop",
        "--terrain-types",
        "--no-stair-terrain",
        "--no-stair-fixed-start",
        "--no-use-reference-velocity",
        "--episode-random-seed",
        "--domain-randomization",
    ):
        if _has_option(user_args, option):
            raise SystemExit(
                f"专项楼梯课程固定使用 OmniRetarget {option.replace('--', '')}，"
                "不接受该覆盖参数。"
            )

    fixed_numeric = {
        "--reference-ctrl-blend": 1.0,
        "--action-residual-scale": 0.5,
        "--motion-reset-prob": 0.0,
        "--stand-start-prob": 0.0,
        "--reset-noise-scale": 0.0,
        "--action-noise-std": 0.0,
        "--episode-steps": float(episode_steps),
        "--episode-timeout-jitter-steps": 0.0,
        "--stair-distance-tolerance": 0.05,
        "--stair-distance-failure-penalty": 5.0,
        "--stair-lateral-tolerance": 0.30,
        "--stair-lateral-failure-penalty": 5.0,
        "--stair-terminal-hold-steps": 30.0,
        "--decimation": 2.0,
    }
    for option, target in fixed_numeric.items():
        value = _option_value(user_args, option)
        if value is None:
            continue
        try:
            actual = float(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{option} 必须保持固定值 {target}。") from exc
        if abs(actual - target) > 1.0e-7:
            raise SystemExit(
                f"专项楼梯课程禁止修改 {option}={target}；收到 {value!r}。"
            )


def default_train_args(direction: str) -> list[str]:
    """Return deterministic defaults for one stair direction."""

    direction = str(direction).lower()
    if direction not in {"up", "down", "up_down"}:
        raise ValueError("direction must be 'up', 'down', or 'up_down'.")
    clip = f"stair_{direction}"
    episode_steps = (
        STAIR_UP_DOWN_EPISODE_STEPS if direction == "up_down" else STAIR_EPISODE_STEPS
    )
    return [
        "--motion-files",
        str(REFERENCE_DIR / f"{clip}_omniretarget.npz"),
        "--motion-names",
        clip,
        "--motion-weights",
        "1.0",
        "--motion-loop",
        "false",
        "--no-skill-conditioning",
        "--stair-terrain",
        "--stair-step-height",
        f"{STAIR_TRAINING_STEP_HEIGHT_M:g}",
        "--stair-step-depth",
        f"{STAIR_TRAINING_TREAD_WIDTH_M:g}",
        "--stair-step-count",
        str(STAIR_TRAINING_STEP_COUNT),
        "--stair-top-length",
        f"{STAIR_TRAINING_TOP_LENGTH_M:g}",
        "--stair-width",
        f"{STAIR_LATERAL_WIDTH_M:g}",
        "--stair-start-x",
        f"{STAIR_START_X_M:g}",
        "--stair-down-offset",
        f"{STAIR_DOWN_OFFSET_M:g}",
        "--stair-fixed-start",
        # Reference controls encode the fixed demonstrated action.  PPO learns
        # only a bounded residual around it, while phase and reset remain fixed.
        "--reference-ctrl-blend",
        "1.0",
        "--action-residual-scale",
        "0.5",
        "--use-reference-velocity",
        "--motion-reset-prob",
        "0.0",
        "--stand-start-prob",
        "0.0",
        "--reset-noise-scale",
        "0.0",
        "--action-noise-std",
        "0.0",
        "--no-episode-random-seed",
        "--episode-steps",
        str(episode_steps),
        "--episode-timeout-jitter-steps",
        "0",
        "--log-group",
        f"stair_{direction}",
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,stairs,stair-specialist,"
            f"stair-{direction},fixed-reference,cmu114,human-gait,mirror-symmetric,"
            "height-0.13m,tread-0.32m,steps-5"
        ),
        "--symmetry-reward-coef",
        "1.8",
        "--symmetry-foot-pos-sigma",
        "0.06",
        "--symmetry-foot-weight",
        "0.70",
        "--symmetry-contact-weight",
        "0.20",
        "--symmetry-action-weight",
        "0.10",
        "--amp-mirror-ref-prob",
        "0.5",
    ]


def main(direction: str | None = None) -> None:
    user_args = list(sys.argv[1:])
    if direction is None:
        # This wrapper-level selector is consumed before train.py sees argv.
        direction = "up"
        remaining: list[str] = []
        index = 0
        while index < len(user_args):
            value = user_args[index]
            if value == "--direction":
                if index + 1 >= len(user_args):
                    raise SystemExit("--direction requires up, down, or up_down")
                direction = user_args[index + 1]
                index += 2
                continue
            if value.startswith("--direction="):
                direction = value.partition("=")[2]
                index += 1
                continue
            remaining.append(value)
            index += 1
        user_args = remaining
    episode_steps = (
        STAIR_UP_DOWN_EPISODE_STEPS
        if str(direction).lower() == "up_down"
        else STAIR_EPISODE_STEPS
    )
    _validate_fixed_geometry(user_args, episode_steps)
    defaults = default_train_args(direction)
    if not _has_option(user_args, "--stage"):
        defaults.extend(("--stage", "stage1"))
    if not _has_option(user_args, "--max-iterations"):
        defaults.extend(("--max-iterations", str(STAIR_MAX_ITERATIONS)))
    if not _has_option(user_args, "--device"):
        defaults.extend(("--device", "cuda"))
    if not _has_option(user_args, "--sim-backend"):
        defaults.extend(("--sim-backend", "warp"))
    if not _has_option(user_args, "--domain-randomization") and not _has_option(
        user_args, "--no-domain-randomization"
    ):
        defaults.append("--no-domain-randomization")
    # Import lazily so importing this module for tests never starts training.
    sys.argv = [str(HERE / "train.py"), *defaults, *user_args]
    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
