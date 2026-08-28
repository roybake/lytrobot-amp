#!/usr/bin/env python3
"""Train a single-reference, sensor-only AMP walker on a stair-free course."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
ORDINARY_REFERENCE_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
AMP_REFERENCE = (
    REPO_ROOT / "robot" / "reference_data" / "no_stairs_fast" / "walk_omniretarget.npz"
)
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lytrobot_amp.training_defaults import (
    ROUGH_TERRAIN_HEIGHT_M,
    SLOPE_ANGLE_DEG,
    NO_STAIRS_FAST_TARGET_SPEED_MPS,
    TRAINING_TARGET_SPEED_MPS,
)


# Match the stable multi-motion curriculum: keep 20% of 4096 worlds learning
# the deployable entrance-to-exit route after the ramp, while the remaining
# worlds directly rehearse rough and slope stages so later terrain is not
# starved of samples.  Completion/checkpoint scoring still requires a full run.
COURSE_RANDOM_START_PROBABILITY = 0.80
COURSE_RANDOM_START_WARMUP_ITERATIONS = 5000
COURSE_RANDOM_START_RAMP_ITERATIONS = 10000


def _stair_free_course_args(
    *,
    motion_files: tuple[Path, Path],
    motion_loop: tuple[bool, bool],
    target_velocity: float,
    use_reference_velocity: bool,
    forward_velocity_reward_coef: float,
    forward_progress_reward_coef: float,
    forward_velocity_deficit_penalty_coef: float,
    serial_reference_state_init: bool,
) -> list[str]:
    args = [
        "--motion-files",
        *(str(path) for path in motion_files),
        "--motion-names",
        "walk",
        "rough_walk",
        "--motion-weights",
        "0.50",
        "0.50",
        "--motion-loop",
        *("true" if value else "false" for value in motion_loop),
        "--no-skill-conditioning",
        "--reference-ctrl-blend",
        "0.0",
        "--target-velocity",
        f"{target_velocity:g}",
        "--forward-velocity-tracking-sigma",
        "0.20",
        "--forward-velocity-reward-coef",
        f"{forward_velocity_reward_coef:g}",
        "--forward-progress-reward-coef",
        f"{forward_progress_reward_coef:g}",
        "--forward-velocity-deficit-penalty-coef",
        f"{forward_velocity_deficit_penalty_coef:g}",
        "--forward-velocity-excess-penalty-coef",
        "3.0",
        "--min-forward-progress-speed",
        "0.10",
        "--fall-penalty",
        "10.0",
        "--terminate-incomplete-motion-end",
        "--terrain-types",
        "flat",
        "rough",
        "slope_up",
        "slope_down",
        "--serial-terrain-course",
        # flat、rough 与成对上下坡构成 3 个随机块，共 6 种唯一顺序。
        "--terrain-course-variants",
        "6",
        "--terrain-course-random-assignment",
        "--terrain-course-random-start-prob",
        f"{COURSE_RANDOM_START_PROBABILITY:g}",
        "--terrain-course-random-start-warmup-iterations",
        str(COURSE_RANDOM_START_WARMUP_ITERATIONS),
        "--terrain-course-random-start-ramp-iterations",
        str(COURSE_RANDOM_START_RAMP_ITERATIONS),
        "--terrain-course-phase-sync",
        "--terrain-course-phase-lookahead",
        "0.10",
        "--terrain-course-stage-reward",
        "5.0",
        "--terrain-course-run-out",
        "3.10",
        "--rough-terrain-height",
        f"{ROUGH_TERRAIN_HEIGHT_M:g}",
        "--slope-angle-deg",
        f"{SLOPE_ANGLE_DEG:g}",
        "--motion-transition-blend-steps",
        "30",
        "--episode-steps",
        # 36 seconds at 100 Hz: enough for the ~18 m course at the 0.86 m/s
        # reference speed plus recovery time on rough ground and both slopes.
        "3600",
    ]
    if use_reference_velocity:
        args.append("--use-reference-velocity")
    if serial_reference_state_init:
        # The old serial reset discarded motion_reset_prob and forced every
        # world to phase zero with zero velocity.  Half reference-state resets
        # now teach continuation of a real gait; the other half still start
        # from the exact phase-zero rest state used by deployment playback.
        args.extend(
            [
                "--serial-reference-state-init",
                "--motion-reset-prob",
                "0.50",
                "--stand-start-prob",
                "0.0",
                "--startup-action-ramp-steps",
                "0",
            ]
        )
    return args


def default_ordinary_stair_free_course_args() -> list[str]:
    """Keep ordinary Stage 1/2 on the original stable-speed references."""

    return _stair_free_course_args(
        motion_files=(
            ORDINARY_REFERENCE_DIR / "walk_omniretarget.npz",
            ORDINARY_REFERENCE_DIR / "rough_walk_omniretarget.npz",
        ),
        motion_loop=(True, False),
        target_velocity=TRAINING_TARGET_SPEED_MPS,
        use_reference_velocity=False,
        forward_velocity_reward_coef=3.0,
        forward_progress_reward_coef=2.0,
        forward_velocity_deficit_penalty_coef=2.5,
        serial_reference_state_init=False,
    )


def default_stair_free_course_args() -> list[str]:
    """Single OmniRetarget gait reused explicitly at every course stage.

    ``walk_omniretarget`` is the only expert clip.  Its data are consumed by
    AMP, while task rewards use physical velocity, contacts and stability only.
    The serial-course adapter deliberately maps that same looping gait to flat,
    rough and both slope stages; terrain adaptation must therefore come from
    IMU/tendon feedback rather than a terrain-labelled reference.
    """

    return [
        "--motion-files",
        str(AMP_REFERENCE),
        "--motion-names",
        "walk",
        "--motion-weights",
        "1.0",
        "--motion-loop",
        "true",
        "--serial-course-single-motion",
        "--no-skill-conditioning",
        "--reference-ctrl-blend",
        "0.0",
        "--amp",
        "--amp-task-only",
        "--target-velocity",
        f"{NO_STAIRS_FAST_TARGET_SPEED_MPS:g}",
        "--forward-velocity-tracking-sigma",
        "0.30",
        "--forward-velocity-reward-coef",
        "3.0",
        "--forward-progress-reward-coef",
        "1.0",
        "--forward-velocity-deficit-penalty-coef",
        "1.0",
        # A requested target is a tracking command, not a speed ceiling.  Do
        # not penalize a faster recovery step; callers can override the target.
        "--forward-velocity-excess-penalty-coef",
        "0.0",
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
        "--terrain-types",
        "flat",
        "rough",
        "slope_up",
        "slope_down",
        "--serial-terrain-course",
        "--terrain-course-variants",
        "6",
        "--terrain-course-random-assignment",
        "--terrain-course-random-start-prob",
        f"{COURSE_RANDOM_START_PROBABILITY:g}",
        "--terrain-course-random-start-warmup-iterations",
        str(COURSE_RANDOM_START_WARMUP_ITERATIONS),
        "--terrain-course-random-start-ramp-iterations",
        str(COURSE_RANDOM_START_RAMP_ITERATIONS),
        "--terrain-course-phase-sync",
        "--terrain-course-phase-lookahead",
        "0.10",
        "--terrain-course-stage-reward",
        "5.0",
        "--terrain-course-run-out",
        "3.10",
        "--rough-terrain-height",
        f"{ROUGH_TERRAIN_HEIGHT_M:g}",
        "--slope-angle-deg",
        f"{SLOPE_ANGLE_DEG:g}",
        "--motion-transition-blend-steps",
        "0",
        "--serial-reference-state-init",
        "--motion-reset-prob",
        "0.50",
        "--stand-start-prob",
        "0.25",
        "--startup-action-ramp-steps",
        "40",
        "--episode-steps",
        "3600",
    ]


def default_train_args() -> list[str]:
    """Add the stair-free task identity and rewards to the shared course."""

    return [
        "--training-task",
        "multimotion_no_stairs_amp_single",
        *default_stair_free_course_args(),
        # Keep slip suppression, but do not let static stance/contact shaping
        # outscore sustained forward locomotion.
        "--foot-slip-penalty-coef",
        "2.0",
        # During exploration tolerate only shallow foot/terrain solver
        # compliance, penalize every millimetre, and terminate before overlap
        # becomes visible.
        "--terrain-penetration-tolerance",
        "0.01",
        "--terrain-penetration-penalty-coef",
        "20.0",
        "--contact-reward-coef",
        "0.35",
        "--alternating-step-reward-coef",
        "1.50",
        "--specific-cot-penalty-coef",
        "0.0",
        # The reference trajectory affects learning only through AMP.  Keep a
        # small physical task component for course completion and robustness.
        "--amp-reward-coef",
        "2.0",
        "--task-reward-lerp",
        "0.25",
        "--min-action-std",
        "0.05",
        # Match the stable 08-06 multi-motion optimizer/AMP sampling regime.
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
        "128",
        "--warp-njmax",
        "512",
        "--log-root",
        str(REPO_ROOT / "logs" / "multimotion" / "no_stairs_amp_single"),
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,amp-task-only,single-omniretarget,walk,no-stairs,"
            "serial-course,random-order,flat,rough,slope-up,slope-down,"
            "alternating-feet,velocity-tracking,imu,tendon,recurrent,gru,sensor-only"
        ),
    ]


def main() -> None:
    sys.argv = [str(HERE / "train.py"), *default_train_args(), *sys.argv[1:]]
    from sensor_training.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
