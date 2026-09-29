#!/usr/bin/env python3
"""Build a terrain-aligned human prior from measured 20 cm stair phases.

CMU 143_17 supplies ascent style and 141_08 supplies descent style. Their
single-step phases are time-warped, mirrored for alternating sides, and placed
on the exact tread surfaces read from stairs.xml. The planned foot and pelvis
targets are explicit outputs; this file is an OmniRetarget input, not a claim
that the source actor performed the resulting continuous course.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "source_data/cmu_stairs_20cm"
sys.path.insert(0, str(HERE))

from stairs_xml_scene import read_stair_treads

RIGHT = np.asarray([1, 2, 3, 4])
LEFT = np.asarray([5, 6, 7, 8])
HIPS = 0


def quintic(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def resample(values: np.ndarray, count: int) -> np.ndarray:
    position = np.linspace(0.0, len(values) - 1.0, count)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(values) - 1)
    blend = (position - low).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[low] * (1.0 - blend) + values[high] * blend


def mirror_human(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    result[..., 1] *= -1.0
    result[:, RIGHT], result[:, LEFT] = result[:, LEFT].copy(), result[:, RIGHT].copy()
    return result


def load_template(path: Path, count: int, mirror: bool) -> tuple[np.ndarray, float, str]:
    with np.load(path, allow_pickle=False) as archive:
        motion = np.asarray(archive["global_joint_positions"], dtype=np.float64)
        height = float(archive["height"])
        source = str(archive["source"])
    motion = resample(motion, count)
    if mirror:
        motion = mirror_human(motion)
    return motion, height, source


def swing_curve(start: np.ndarray, target: np.ndarray, phase: np.ndarray,
                clearance: float) -> tuple[np.ndarray, np.ndarray]:
    """Lift, cross the riser, then lower, with exact zero endpoint speeds."""
    result = np.repeat(start[None, :], len(phase), axis=0)
    high = max(float(start[2]), float(target[2])) + clearance
    descending = target[2] < start[2] - .02
    if descending:
        # Cross the edge early, leaving 30% of the step to lower through a
        # full 20 cm riser without a knee-velocity spike.
        lift = quintic((phase - .20) / .20)
        travel = quintic((phase - .35) / .25)
        lower = quintic((phase - .60) / .30)
    else:
        lift = quintic((phase - .20) / .35)
        travel = quintic((phase - .45) / .30)
        lower = quintic((phase - .75) / .15)
    result[:, 2] = start[2] + (high - start[2]) * lift
    result[:, :2] = start[:2] + (target[:2] - start[:2]) * travel[:, None]
    result[:, 2] += (target[2] - high) * lower
    result[phase >= .90] = target
    contact = (phase <= .20) | (phase >= .90)
    return result, contact


def warp_human(template: np.ndarray, desired_feet: np.ndarray,
               root: np.ndarray, scale: float) -> np.ndarray:
    """Preserve measured leg curvature while matching planned root and toes."""
    local = (template - template[:, HIPS:HIPS+1]) * scale
    result = local + root[:, None, :]
    for chain in (RIGHT, LEFT):
        side = 0 if chain[0] == RIGHT[0] else 1
        correction = desired_feet[:, side] - result[:, chain[-1]]
        # Keep the hip near the captured pelvis and distribute endpoint warp
        # down the chain. This supplies style only; robot constraints follow.
        weights = np.asarray([.05, .35, .72, 1.0])
        result[:, chain] += correction[:, None, :] * weights[None, :, None]
    result[:, HIPS] = root
    return result


def landing_plan(terrain_xml: Path) -> list[tuple[float, float, str]]:
    treads = read_stair_treads(terrain_xml)
    plan = [((t["x_min"] + t["x_max"]) / 2.0, t["height"], t["name"])
            for t in treads]
    # Finish on level ground with one landing directly beyond the last riser
    # and a second 0.32 m run-out, so both feet clear the final box edge.
    plan.extend([(treads[-1]["x_max"] + .16, 0.0, "ground_exit_0"),
                 (treads[-1]["x_max"] + .48, 0.0, "ground_exit_1")])
    return plan


def build(terrain_xml: Path, output: Path, fps: float, step_duration: float,
          max_steps: int = 0) -> Path:
    frames_per_step = int(round(fps * step_duration))
    if frames_per_step < 8:
        raise ValueError("Each step needs at least 8 intervals for smooth contact transitions.")
    plan = landing_plan(terrain_xml)
    # Each level uses step-to gait: right advances, then left joins it.
    moves = [(side, x, z, name) for x, z, name in plan for side in (0, 1)]
    if max_steps:
        moves = moves[:max_steps]
    up_source = SOURCE / "143_17_up_single_step_aligned_120hz.npz"
    down_source = SOURCE / "141_08_down_single_step_aligned_120hz.npz"
    current = np.asarray([[-.16, -.075, .003], [-.16, .075, .003]], dtype=float)
    all_human, all_feet, all_contact, all_root, provenance = [], [], [], [], []
    # Begin with a measured anticipatory crouch and leftward weight transfer.
    # Without this phase, the first stair alone starts from the XML's fully
    # straight pose while every later stair starts from an already flexed gait.
    prep_intervals = max(8, int(round(.8 * fps)))
    prep_count = prep_intervals + 1
    prep_template, prep_height, prep_source = load_template(
        up_source, 2 * prep_count - 1, False
    )
    prep_template = prep_template[:prep_count]
    prep_phase = np.linspace(0., 1., prep_count)
    prep_feet = np.repeat(current[None, :, :], prep_count, axis=0)
    prep_root = np.column_stack((np.full(prep_count, -.22),
                                 .025 * quintic(prep_phase),
                                 .838 - .048 * quintic(prep_phase)))
    prep_human = warp_human(prep_template, prep_feet, prep_root,
                            .875 / max(prep_height, 1e-6))
    all_human.append(prep_human)
    all_feet.append(prep_feet)
    all_contact.append(np.ones((prep_count, 2), dtype=np.float32))
    all_root.append(prep_root)
    previous_root = prep_root[-1].copy()
    provenance.append(dict(step="preparation", moving_side="none",
                           surface="entry_ground", source=prep_source,
                           duration_s=.8, both_feet_fixed=True))
    for step, (side, x, z, surface) in enumerate(moves):
        phase = np.linspace(0.0, 1.0, frames_per_step + 1)
        target = np.asarray([x, (-.075, .075)[side], z + .003])
        desired = np.repeat(current[None, :, :], len(phase), axis=0)
        desired[:, side], swing_contact = swing_curve(current[side], target, phase, .045)
        contacts = np.ones((len(phase), 2), dtype=np.float32)
        contacts[:, side] = swing_contact
        rising = target[2] > current[side, 2] + .02
        falling = target[2] < current[side, 2] - .02
        source_path = up_source if rising else down_source if falling else up_source
        # Source phases both have left support and right swing. Mirror for left.
        template, human_height, source_name = load_template(source_path, len(phase), side == 1)
        foot_mean = desired.mean(axis=1)
        support_level = np.where(contacts[:, 1-side] > .5,
                                 desired[:, 1-side, 2], desired[:, side, 2])
        root_guide = np.column_stack((foot_mean[:, 0] - .06,
                                      .42 * desired[:, 1-side, 1] + .58 * foot_mean[:, 1],
                                      support_level + .835))
        # Changing stance side on a 20 cm riser must not teleport the pelvis
        # between support levels at a step seam. Transfer weight and height
        # over the first 65% of the next move.
        root_blend = quintic(phase / .65)[:, None]
        root = previous_root[None, :] * (1.0 - root_blend) + root_guide * root_blend
        # Retain the measured pelvis bob after removing source drift.
        bob = template[:, HIPS, 2] - np.linspace(template[0, HIPS, 2],
                                                template[-1, HIPS, 2], len(template))
        root[:, 2] += np.clip(bob * (.875 / max(human_height, 1e-6)), -.025, .025)
        human = warp_human(template, desired, root, .875 / max(human_height, 1e-6))
        # Drop duplicated seam frame; its state is preserved by the next step.
        start = 1
        all_human.append(human[start:])
        all_feet.append(desired[start:])
        all_contact.append(contacts[start:])
        all_root.append(root[start:])
        provenance.append(dict(step=int(step), moving_side=("right", "left")[side],
                               surface=surface, x_m=float(x), height_m=float(z),
                               source=source_name, mirrored=bool(side == 1),
                               ascent=bool(rising), descent=bool(falling)))
        current[side] = target
        previous_root = root[-1].copy()
    human = np.concatenate(all_human)
    feet = np.concatenate(all_feet)
    contacts = np.concatenate(all_contact)
    roots = np.concatenate(all_root)
    metadata = {
        "method": "measured CMU single-step phase reuse plus explicit XML footstep plan",
        "not_original_continuous_mocap": True,
        "source_ascent": "CMU 143_17 frames 53..100 (zero-based, inclusive)",
        "source_descent": "CMU 141_08 frames 215..245 (zero-based, inclusive)",
        "gait": "step-to; right advances then left joins each surface",
        "step_duration_s": step_duration, "swing_clearance_m": .045,
        "foot_target_clearance_m": .003, "landing_plan": provenance,
        "terrain_xml": str(terrain_xml), "full_plan_step_count": 2 * len(plan),
        "generated_step_count": len(moves),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, global_joint_positions=human.astype(np.float32),
                        foot_targets=feet.astype(np.float32), foot_contact=contacts,
                        root_target=roots.astype(np.float32), fps=np.float32(fps),
                        joint_names=np.asarray(["Hips", "RightUpLeg", "RightLeg", "RightFoot",
                                                "RightToeBase", "LeftUpLeg", "LeftLeg", "LeftFoot",
                                                "LeftToeBase", "Spine", "Spine1"]),
                        source=np.asarray("CMU 143_17 ascent + CMU 141_08 descent"),
                        task=np.asarray("stairs_xml_up_down"), loop=np.asarray(False),
                        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    print(f"saved {len(human)} frames ({(len(human)-1)/fps:.2f}s, {len(moves)} steps) -> {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-xml", type=Path,
                        default=ROOT / "robot/terrains/more_discolrl/stairs.xml")
    parser.add_argument("--output", type=Path, default=HERE / "prepared/stairs_xml_human.npz")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--step-duration", type=float, default=1.5)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Generate only this many individual foot moves; 0 builds the full course.")
    args = parser.parse_args()
    build(args.terrain_xml, args.output, args.fps, args.step_duration, args.max_steps)


if __name__ == "__main__":
    main()
