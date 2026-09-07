#!/usr/bin/env python3
"""Train the 08-06 privileged Teacher at 100 Hz without stair terrain."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

REFERENCE_DIR = ROOT / "robot" / "reference_data" / "multimotion"
REFERENCE_MODEL_XML = ROOT / "robot" / "MJCFmodelposition_0806.xml"

from scripts.train_multimotion_no_stairs import (
    COURSE_RANDOM_START_RAMP_ITERATIONS,
    COURSE_RANDOM_START_WARMUP_ITERATIONS,
    COURSE_VARIANTS,
)


def _assert_reference_model_override_is_compatible(argv: list[str]) -> None:
    """Reject a model override that cannot consume the Teacher reference clips."""

    expected = REFERENCE_MODEL_XML.resolve()
    for index, arg in enumerate(argv):
        if arg == "--model-xml":
            if index + 1 >= len(argv):
                raise SystemExit("--model-xml requires a path.")
            supplied = Path(argv[index + 1]).resolve()
        elif arg.startswith("--model-xml="):
            supplied = Path(arg.split("=", 1)[1]).resolve()
        else:
            continue
        if supplied != expected:
            raise SystemExit(
                "multimotion_no_stairs_amp_teacher requires "
                f"{expected}; received incompatible --model-xml {supplied}."
            )


def default_train_args() -> list[str]:
    """Return the recorded 08-06 settings with only clock/terrain changes.

    The no-stairs Teacher uses terrain-aligned references: ``walk`` for flat,
    ``rough_walk`` for rough, and the two slope-adapted clips for the uphill
    and downhill stages.  Their equal weights, optimizer settings, and
    privileged Actor contract come from the saved 08-06 run and commit 6162570.
    Control-step durations are doubled for the 50->100 Hz migration introduced
    by commit a6f45cf.  Stair reference clips are excluded, so AMP expert
    samples cannot teach stair-only contacts on a no-stairs course. The
    explicit slip, reference-tracking, and 100 Hz action-rate corrections below
    address the gait defects observed in the recorded policy.
    """

    return [
        "--training-task",
        "multimotion_no_stairs_amp_teacher",
        "--model-xml",
        str(REFERENCE_MODEL_XML),
        "--terrain-asset-dir",
        str(ROOT / "robot" / "terrains" / "more_discolrl"),
        "--motion-files",
        str(REFERENCE_DIR / "walk_omniretarget.npz"),
        str(REFERENCE_DIR / "rough_walk_omniretarget.npz"),
        str(REFERENCE_DIR / "slope_up_omniretarget.npz"),
        str(REFERENCE_DIR / "slope_down_omniretarget.npz"),
        "--motion-names",
        "walk",
        "rough_walk",
        "slope_up",
        "slope_down",
        "--motion-weights",
        "0.25",
        "0.25",
        "0.25",
        "0.25",
        "--motion-loop",
        "true",
        "false",
        "false",
        "false",
        "--skill-conditioning",
        "--legacy-reference-conditioned-actor",
        "--seed",
        "42",
        "--decimation",
        "2",
        "--robot-collision-profile",
        "feet",
        "--num-envs",
        "4096",
        "--num-steps",
        "24",
        "--epochs",
        "4",
        "--minibatches",
        "8",
        "--amp-minibatch-size",
        "4096",
        "--amp-disc-updates",
        "2",
        "--episode-metric-window",
        "4096",
        "--warp-nconmax",
        "32",
        "--warp-njmax",
        "128",
        "--use-reference-velocity",
        "--target-velocity",
        "0.862295",
        "--forward-velocity-tracking-sigma",
        "0.20",
        "--forward-velocity-reward-coef",
        "2.0",
        "--forward-progress-reward-coef",
        "1.0",
        "--forward-velocity-deficit-penalty-coef",
        "1.5",
        "--forward-velocity-excess-penalty-coef",
        "3.0",
        "--min-forward-progress-speed",
        "0.10",
        "--forward-progress-grace-steps",
        "100",
        "--forward-progress-window-steps",
        "100",
        "--forward-progress-termination-penalty",
        "2.0",
        "--fall-penalty",
        "10.0",
        "--terminate-incomplete-motion-end",
        "--side-foot-pos-coef",
        "0.50",
        "--step-gap-coef",
        "0.50",
        "--swing-advance-coef",
        "0.75",
        "--symmetry-reward-coef",
        "0.50",
        "--terrain-types",
        "rough",
        "slope_up",
        "slope_down",
        "--no-stair-terrain",
        "--serial-terrain-course",
        "--no-serial-course-single-motion",
        "--legacy-0806-terrain-model",
        "--terrain-course-variants",
        str(COURSE_VARIANTS),
        "--terrain-course-lane-spacing",
        # "3.0",  # 原始赛道中心间距；必须大于 terrain-lane-width
        "4.0",
        "--terrain-course-run-in",
        "2.90",
        "--terrain-course-run-out",
        "0.30",
        "--terrain-course-base-height",
        "0.30",
        "--terrain-course-completion-reward",
        "25.0",
        "--terrain-course-random-start-prob",
        "0.70",
        "--terrain-course-random-start-warmup-iterations",
        str(COURSE_RANDOM_START_WARMUP_ITERATIONS),
        "--terrain-course-random-start-ramp-iterations",
        str(COURSE_RANDOM_START_RAMP_ITERATIONS),
        "--terrain-course-random-assignment",
        "--no-terrain-course-phase-sync",
        "--no-serial-reference-state-init",
        "--terrain-course-stage-reward",
        "0.50",
        "--terrain-course-transition-survival-steps",
        "75",
        "--terrain-course-transition-success-reward",
        "2.0",
        "--terrain-course-stage-tolerance",
        "0.35",
        "--rough-terrain-height",
        "0.05",
        "--flat-lane-y",
        "3.0",
        "--rough-lane-y",
        "6.0",
        "--pit-lane-y",
        "9.0",
        "--gap-lane-y",
        "12.0",
        "--terrain-lane-width",
        # "2.0",  # 原始赛道宽度
        "3.0",
        "--gap-terrain-width",
        "0.20",
        "--slope-angle-deg",
        "6",
        "--stair-step-height",
        "0.2",
        "--stair-step-depth",
        "0.22",
        "--stair-step-count",
        "3",
        "--stair-start-x",
        "-0.95",
        "--stair-down-offset",
        "0.25",
        "--reset-noise-scale",
        "0.01",
        "--action-residual-scale",
        "0.5",
        "--distal-action-scale",
        "1.0",
        "--reference-ctrl-blend",
        "1.0",
        "--motion-reset-prob",
        "1.0",
        "--motion-transition-blend-steps",
        "30",
        "--foot-vel-tracking-sigma",
        "1.0",
        "--stance-foot-vel-sigma",
        "0.15",
        "--foot-vel-reward-coef",
        "0.35",
        "--stance-foot-reward-coef",
        "1.0",
        "--foot-slip-velocity-threshold",
        "0.08",
        "--foot-slip-penalty-coef",
        "0.50",
        "--contact-reward-coef",
        "0.7",
        "--missed-contact-penalty-coef",
        "0.8",
        "--side-foot-pos-sigma",
        "0.08",
        "--step-gap-sigma",
        "0.12",
        "--swing-advance-sigma",
        "0.10",
        "--distal-joint-vel-sigma",
        "2.0",
        "--startup-action-ramp-steps",
        "0",
        "--stand-start-prob",
        "0.0",
        "--stand-start-phase-delay-steps",
        "50",
        "--stand-start-transition-min-steps",
        "160",
        "--stand-start-transition-max-steps",
        "320",
        "--stand-start-push-delay-steps",
        "200",
        "--episode-steps",
        "3600",
        "--episode-timeout-jitter-steps",
        "480",
        "--solver-iterations",
        "10",
        "--push-interval-min-steps",
        "50",
        "--push-interval-max-steps",
        "200",
        "--push-duration-min-steps",
        "4",
        "--push-duration-max-steps",
        "16",
        "--ctrl-penalty-coef",
        "0.04",
        "--action-rate-penalty-coef",
        "0.4",
        "--distal-action-penalty-coef",
        "0.03",
        "--distal-action-rate-penalty-coef",
        "0.48",
        "--learning-rate",
        "0.0003",
        "--reward-scale",
        "0.25",
        "--disc-learning-rate",
        "0.00003",
        "--amp-reward-coef",
        "1.2",
        "--task-reward-lerp",
        "0.65",
        "--min-action-std",
        "0.05",
        "--max-action-std",
        "0.6",
        "--log-root",
        str(ROOT / "logs" / "multimotion"),
        "--log-group",
        "multimotion_no_stairs_amp_teacher",
        "--wandb-tags",
        "lytrobot,bpo,amp,multimotion,no-stairs,amp-teacher,0806,100hz,"
        "privileged,reference-conditioned,rough,slope-up,slope-down",
        "--max-iterations",
        "30000",
    ]


def main() -> None:
    import scripts.train as runner
    from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
    from lytrobot_amp.ppo_amp import PPOAMPConfig, PPOAMPTrainer

    _assert_reference_model_override_is_compatible(sys.argv[1:])
    runner.LytRobotWarpAmpEnv = LytRobotWarpAmpEnv
    runner.PPOAMPConfig = PPOAMPConfig
    runner.PPOAMPTrainer = PPOAMPTrainer
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    runner.main()


if __name__ == "__main__":
    main()
