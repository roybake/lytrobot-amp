#!/usr/bin/env python3
"""Generate and verify the isolated 5 km/h small-step no-stairs references."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from analyze_gait import analyze_reference


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TARGET_SPEED = 5.0 / 3.6
TARGET_STEP = 0.48


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-step-length", type=float, default=TARGET_STEP)
    parser.add_argument("--max-iterations", type=int, default=4)
    args = parser.parse_args()
    if not 0.46 <= args.target_step_length <= 0.50:
        raise ValueError("target step length must be in 0.46--0.50 m")
    prepared = HERE / "prepared_no_stairs_fast"
    results = HERE / "results_no_stairs_fast"
    output = REPO_ROOT / "robot" / "reference_data" / "no_stairs_fast"
    requested_step = args.target_step_length
    walk = output / "walk_omniretarget.npz"
    for attempt in range(args.max_iterations):
        run(
            str(HERE / "prepare_clips.py"), "--clips", "walk", "--output-dir", str(prepared),
            "--target-step-length", f"{requested_step:.9g}",
        )
        run(
            str(HERE / "run_omniretarget.py"), "--input", str(prepared / "walk_human.npz"),
            "--output", str(results / "walk_lyt_raw.npz"),
        )
        run(
            str(HERE / "build_references.py"), "--results-dir", str(results),
            "--output-dir", str(output), "--target-walk-speed", f"{TARGET_SPEED:.10g}",
            "--clips", "walk",
        )
        gait = analyze_reference(walk)
        step = float(gait["mean_step_length_m"])
        cadence = float(gait["cadence_steps_per_min"])
        print(f"attempt={attempt + 1}: step={step:.4f}m cadence={cadence:.2f}/min")
        if abs(step - args.target_step_length) <= 0.02 and 170.0 <= cadence <= 180.0:
            break
        # Closed-loop spatial correction uses the *retargeted robot* step
        # measurement; it never rescales final qpos joint trajectories.
        requested_step *= args.target_step_length / max(step, 1.0e-6)
    else:
        raise RuntimeError("small-step search did not satisfy step/cadence limits")
    run(
        str(HERE / "build_terrain_references.py"), "--source", str(walk),
        "--output-dir", str(output), "--tasks", "rough_walk",
    )
    run(
        str(HERE / "validate_references.py"), "--motion-dir", str(output),
        "--no-stairs-fast", "--output", str(REPO_ROOT / "recording" / "reference" / "no_stairs_fast_validation.json"),
    )


if __name__ == "__main__":
    main()
