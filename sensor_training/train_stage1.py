#!/usr/bin/env python3
"""Independent sensor-only stage1 training with an AMP gait prior."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensor_training.training_defaults import (
    recurrent_cli_defaults,
    stage_policy_rollout_envs,
)


# Use exactly the same natural-gait reference as the privileged Stage1
# entrypoint.  Keeping the motion, target speed and explicit reward overrides
# aligned makes comparisons isolate the deployable sensor observation and
# recurrent policy instead of silently changing the task.
DEFAULT_GAIT_REFERENCE = (
    REPO_ROOT / "robot" / "reference_data" / "walk1_mjcf_repaired.npz"
)


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


if __name__ == "__main__":
    default_decimation, default_num_steps, _default_envs = recurrent_cli_defaults(
        sys.argv[1:]
    )
    default_num_envs = stage_policy_rollout_envs(default_num_steps, default_decimation)
    if not _has_arg("--stage"):
        sys.argv.extend(["--stage", "stage1"])
    if not (_has_arg("--domain-randomization") or _has_arg("--no-domain-randomization")):
        sys.argv.append("--no-domain-randomization")
    if not (_has_arg("--wandb") or _has_arg("--no-wandb")):
        sys.argv.append("--wandb")
    if not _has_arg("--device"):
        sys.argv.extend(["--device", "cuda"])
    if not _has_arg("--sim-backend"):
        sys.argv.extend(["--sim-backend", "warp"])
    # Stage1 trains directly without a teacher.  The Actor receives
    # only deployable IMU/tendon/action history plus an onboard gait command;
    # reference phase and tendon targets are therefore known at deployment.
    #
    # Motion, target speed and explicit gait-reward overrides intentionally
    # match scripts/train_stage1.py.  Sensor history, recurrent optimization
    # and the deployable actor interface remain the experimental differences.
    stage1_defaults = {
        # Keep a 0.64 s recurrent gradient window at the required 100 Hz
        # control rate while matching the privileged recipe's physics budget
        # per PPO update (393,216 substeps => 3072 worlds here).
        "--num-envs": str(default_num_envs),
        "--num-steps": str(default_num_steps),
        # Match scripts/train_stage1.py so observation architecture is the
        # principal experimental variable in privileged-vs-sensor runs.
        "--motion-file": str(DEFAULT_GAIT_REFERENCE),
        "--motion-loop": "true",
        "--target-velocity": "0.87",
        "--symmetry-reward-coef": "1.2",
        "--symmetry-foot-pos-sigma": "0.08",
        "--side-foot-pos-coef": "0.0",
        "--step-gap-coef": "0.0",
        "--swing-advance-coef": "0.0",
        # The position servo is driven by onboard reference tendon targets
        # (the same table the Actor already receives in its gait command) and
        # the Actor adds a bounded feedback residual.
        "--reference-ctrl-blend": "1.0",
        "--action-residual-scale": "0.5",
        # Sensor-policy training and deployment use 100 Hz by default.
        "--decimation": str(default_decimation),
        "--startup-action-ramp-steps": "50",
        "--save-interval": "500",
        "--fall-penalty": "2.0",
        # Half of the episodes reproduce the deployment transition from a
        # standing start; the rest start inside the reference gait so that the
        # policy learns the whole cycle instead of only the startup ramp.
        "--stand-start-prob": "0.5",
        "--motion-reset-prob": "0.5",
        "--stand-start-phase-delay-steps": "25",
        "--stand-start-transition-min-steps": "100",
        "--stand-start-transition-max-steps": "200",
        # 12 s +/- 2.4 s episodes: ~8-15 gait cycles per episode, so a fall is
        # never cheaper than walking and the value function sees steady gait.
        "--episode-steps": "1200",
        # Same PPO budget per update as the privileged Stage1 policy.  The
        # previous 1e-5 fixed rate disabled the adaptive schedule and made
        # every update 30x smaller than the recipe that works.
        "--learning-rate": "3.0e-4",
        "--min-learning-rate": "1.0e-5",
        "--max-learning-rate": "3.0e-4",
        "--clip-coef": "0.2",
        "--target-kl": "0.025",
        "--target-kl-stop-multiplier": "2.0",
        # ``max_grad_norm``/``entropy_coef`` are left at the sensor config
        # defaults (0.5 / 0.003): a recurrent policy can leave the good state
        # distribution after one large update, and that is the pair the
        # 400-iteration screen that matched the reference metrics was run with.
        "--min-action-std": "0.05",
        "--max-action-std": "0.6",
        "--task-reward-lerp": "0.5",
        "--amp-reward-coef": "1.2",
        # Only stop an episode for standing still; the reference clock keeps
        # moving, so a stationary policy already scores far below a walking
        # one and needs no extra shaping term.
        "--min-forward-progress-speed": "0.1",
        "--forward-progress-grace-steps": "400",
        "--forward-progress-window-steps": "100",
        "--forward-progress-termination-penalty": "2.0",
    }
    for name, value in stage1_defaults.items():
        if not _has_arg(name):
            sys.argv.extend([name, value])
    if not (_has_arg("--sensor-stability-reward") or _has_arg("--no-sensor-stability-reward")):
        sys.argv.append("--no-sensor-stability-reward")
    runpy.run_path(str(Path(__file__).resolve().parent / "train.py"), run_name="__main__")
