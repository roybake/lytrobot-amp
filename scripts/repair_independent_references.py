#!/usr/bin/env python3
"""Repair rough reference collision clearance and render all independent terrains.

Preserves source archives. The rough gait is synchronized to the 5 m spatial
heightfield period, then minimally lifted using the *runtime* collision model.
No policy actions or dynamic simulation are involved in these FK videos.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mujoco
import numpy as np

from lytrobot_amp.env import LytRobotAmpEnv
from lytrobot_amp.terrain import standalone_slope_datum, terrain_spawn_offset
from scripts.audit_independent_references import launcher_config
from scripts.view_multimotion_retarget import VideoWriter


def interpolate_pose(qpos, phase, period):
    """Interpolate a translating cycle, including its unwrapped seam."""
    count = len(qpos)
    lo = int(np.floor(phase))
    alpha = phase - lo
    a, b = qpos[lo % count].copy(), qpos[(lo + 1) % count].copy()
    a[:3] += (lo // count) * period
    b[:3] += ((lo + 1) // count) * period
    pose = (1 - alpha) * a + alpha * b
    if np.dot(a[3:7], b[3:7]) < 0:
        pose[3:7] = (1 - alpha) * a[3:7] - alpha * b[3:7]
    pose[3:7] /= np.linalg.norm(pose[3:7])
    return pose


def period_for(qpos, frequency, key, env):
    period = np.zeros(3)
    period[0] = float(qpos[-1, 0] - qpos[0, 0] + np.median(np.diff(qpos[:, 0])))
    if key in {"slope_up", "slope_down"}:
        period[2] = standalone_slope_datum(env.more_terrain_cfg)[0] * period[0]
    return period


def set_pose(env, pose, key):
    data = env.datas[0]
    data.qpos[:] = pose
    data.qpos[:3] += terrain_spawn_offset(key, env.more_terrain_cfg, env.stair_cfg)
    data.qvel[:] = 0
    mujoco.mj_forward(env.model, data)


def required_lift(env, pose, key, clearance=0.00025, align_support=False):
    """Find minimal vertical lift with nonpenetrating actual contacts.

    Never change geom margins: that changes the hfield narrow-phase result.
    """
    set_pose(env, pose, key)
    if not align_support and env._terrain_penetration_depth(env.datas[0]) <= 1e-8:
        return 0.0
    low, high = (-0.08 if align_support else 0.0), 0.12
    trial = pose.copy()
    trial[2] += low
    set_pose(env, trial, key)
    if align_support and env._terrain_penetration_depth(env.datas[0]) <= 1e-8:
        raise RuntimeError("No supporting surface within 8 cm below reference.")
    trial = pose.copy()
    trial[2] += high
    set_pose(env, trial, key)
    if env._terrain_penetration_depth(env.datas[0]) > 1e-8:
        raise RuntimeError("Collision needs more than 12 cm correction; refuse root-lift repair.")
    for _ in range(18):
        mid = (low + high) / 2
        trial = pose.copy()
        trial[2] += mid
        set_pose(env, trial, key)
        if env._terrain_penetration_depth(env.datas[0]) > 1e-8:
            low = mid
        else:
            high = mid
    return high + clearance


def repair_rough(env, source, output):
    with np.load(source) as a:
        payload = {k: a[k].copy() for k in a.files}
    # The existing terrain-adapted first gait cycle has the same 48-frame
    # contact cadence as walk. Repeat five steps-pairs over exactly one tile.
    walk = np.load(ROOT / "robot/reference_data/multimotion/walk_omniretarget.npz")
    cycle_frames = len(walk["qpos"])
    source_hz = float(payload["frequency"])
    base = np.asarray(env.motion.motions[0].reference_qpos[:cycle_frames], dtype=float)
    source_period = period_for(base, source_hz, "rough", env)
    frequency = 100.0
    count = round(5.0 / env.cfg.target_velocity * frequency)
    duration = count / frequency
    phases = np.arange(count) * (5 * cycle_frames / count)
    qpos = np.stack([interpolate_pose(base, phase, source_period) for phase in phases])
    qpos[:, 0] = (qpos[:, 0] - qpos[0, 0]) * 5.0 / (5 * source_period[0])
    indices = np.floor(phases).astype(int) % cycle_frames
    contact = payload["foot_contact"][indices]
    correction = np.zeros(count)
    for i in range(count):
        correction[i] = required_lift(env, qpos[i], "rough", align_support=True)
    # Smooth a conservative envelope, keeping every correction at or above
    # the collision bound. Short periodic windows preserve cycle continuity.
    envelope = np.maximum.reduce([np.roll(correction, shift) for shift in range(-1, 2)])
    weights = np.exp(-0.5 * (np.arange(-4, 5, dtype=float) / 0.5)**2)
    weights /= weights.sum()
    smooth = sum(weight * np.roll(envelope, shift) for weight, shift in zip(weights, range(-4, 5)))
    correction = np.maximum(correction, smooth)
    qpos[:, 2] += correction
    period = np.asarray([5.0, 0.0, 0.0])
    # Validate and fix between saved samples too, including the loop seam.
    for _ in range(3):
        extra = np.zeros(count)
        for i in range(count):
            lift = required_lift(env, interpolate_pose(qpos, i + 0.5, period), "rough")
            extra[i] = max(extra[i], lift)
            extra[(i + 1) % count] = max(extra[(i + 1) % count], lift)
        if extra.max() == 0:
            break
        qpos[:, 2] += extra
        correction += extra
    qvel = np.empty((count, env.model.nv))
    for i in range(count):
        mujoco.mj_differentiatePos(env.model, qvel[i], 2 / frequency,
                                 interpolate_pose(qpos, i - 1, period),
                                 interpolate_pose(qpos, i + 1, period))
    metadata = {"source_metadata": json.loads(str(payload["metadata"]))}
    metadata.update(repair="runtime_collision_root_clearance_and_5m_period",
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    loop_translation_m=period.tolist(), frequency=frequency, loop=True,
                    root_lift_max_m=float(correction.max()),
                    target_forward_speed_mps=5.0 / duration,
                    terrain_config=asdict(env.more_terrain_cfg))
    payload.update(qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32),
                   frequency=np.asarray(frequency), loop=np.asarray(True),
                   foot_contact=contact, metadata=np.asarray(json.dumps(metadata)))
    for name in ("foot_roll_phase", "foot_roll_mode"):
        payload[name] = payload[name][indices]
    with output.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    return float(correction.max())


def verify_and_render(key, source, output_dir, seconds, render):
    if render and any((output_dir / f"{key}.{ext}").exists() for ext in ("mp4", "png")):
        raise FileExistsError(f"Refusing to overwrite {key} videos/previews in {output_dir}")
    cfg = launcher_config(key)
    cfg.motion_files = (str(source),)
    env = LytRobotAmpEnv(cfg)
    try:
        with np.load(source) as a:
            qpos = np.asarray(env.motion.motions[0].reference_qpos, dtype=float)
            hz = float(a["frequency"])
            contact = a["foot_contact"].copy()
            metadata = json.loads(str(a["metadata"]))
        period = np.asarray(metadata.get("loop_translation_m", period_for(qpos, hz, key, env)))
        result = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "env_config": asdict(cfg), "loop_translation_m": period.tolist(),
                  "placements": []}
        from multimotion.retargeting.validate_references import BODY_PAIRS, geom_for_body
        pairs = [(geom_for_body(env.model, a), geom_for_body(env.model, b)) for a, b in BODY_PAIRS]
        minimum_cross_leg = 0.2
        qvel = np.empty((len(qpos), env.model.nv))
        for i, pose in enumerate(qpos):
            set_pose(env, pose, key)
            minimum_cross_leg = min(minimum_cross_leg, *(mujoco.mj_geomDistance(
                env.model, env.datas[0], a, b, .2, None) for a, b in pairs))
            mujoco.mj_differentiatePos(env.model, qvel[i], 2 / hz,
                                     interpolate_pose(qpos, i - 1, period),
                                     interpolate_pose(qpos, i + 1, period))
        result.update(minimum_cross_leg_distance_m=minimum_cross_leg,
                      max_root_vertical_speed_mps=float(np.abs(qvel[:, 2]).max()),
                      max_joint_speed_rad_s=float(np.abs(qvel[:, 6:]).max()),
                      mean_forward_speed_mps=float(period[0] * hz / len(qpos)),
                      finite=bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()))
        # Actual translations for the repaired rough cycle must be integer
        # tile periods, not arbitrary phase-independent offsets.
        travels = (0.0, 5.0, 10.0, 15.0) if key == "rough" else (0.0, 3.0, 9.0)
        for travel in travels:
            maximum, mismatch, airborne, nonfoot = 0.0, [], [], 0
            near_airborne, near_mismatch = [], []
            for i in range(2 * len(qpos)):
                pose = interpolate_pose(qpos, i / 2, period)
                pose[0] += travel
                if key in {"slope_up", "slope_down"}:
                    pose[2] += standalone_slope_datum(env.more_terrain_cfg)[0] * travel
                set_pose(env, pose, key)
                data = env.datas[0]
                maximum = max(maximum, env._terrain_penetration_depth(data))
                actual = env._foot_contacts(data)
                mismatch.append(np.mean(np.abs(actual - contact[(i // 2) % len(contact)])))
                airborne.append(actual.max() < .5)
                for c in data.contact[:data.ncon]:
                    other = int(c.geom2) if int(c.geom1) in env.terrain_geom_ids else int(c.geom1) if int(c.geom2) in env.terrain_geom_ids else None
                    nonfoot += int(other is not None and other not in env.contact_geom_id_to_index and c.dist < -1e-6)
                # Distinguish a deliberate sub-mm kinematic clearance from
                # floating. Probe only for measurement, then restore pose.
                probe = pose.copy()
                probe[2] -= 0.003
                set_pose(env, probe, key)
                near = env._foot_contacts(env.datas[0])
                near_airborne.append(near.max() < .5)
                near_mismatch.append(np.mean(np.abs(near - contact[(i // 2) % len(contact)])))
            result["placements"].append(dict(translation_m=travel, max_penetration_m=maximum,
                                              contact_mismatch_fraction=float(np.mean(mismatch)),
                                              airborne_fraction=float(np.mean(airborne)),
                                              no_support_within_3mm_fraction=float(np.mean(near_airborne)),
                                              contact_mismatch_within_3mm_fraction=float(np.mean(near_mismatch)),
                                              nonfoot_penetrating_contacts=nonfoot))
        if key == "rough" and any(p["max_penetration_m"] > 1e-6 for p in result["placements"]):
            raise RuntimeError("Repaired rough reference failed collision verification.")
        if render:
            camera = mujoco.MjvCamera()
            camera.azimuth, camera.elevation, camera.distance = 115, -15, 2.8
            option = mujoco.MjvOption()
            recorder = VideoWriter(env.model, output_dir / f"{key}.mp4", 960, 720, 40, 20, [])
            try:
                for frame in range(round(seconds * 40)):
                    set_pose(env, interpolate_pose(qpos, frame * hz / 40, period), key)
                    camera.lookat[:] = env.datas[0].qpos[:3]
                    camera.lookat[2] -= .2
                    recorder.write(env.datas[0], camera, option)
            finally:
                recorder.close()
            subprocess.run(["ffmpeg", "-n", "-hide_banner", "-loglevel", "error", "-ss", "1",
                            "-i", str(output_dir / f"{key}.mp4"), "-frames:v", "1",
                            str(output_dir / f"{key}.png")], check=True)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rough-reference", type=Path, help="Verify/render an already repaired archive.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    repaired = args.rough_reference
    if repaired is None:
        cfg = launcher_config("rough")
        # Always derive from the preserved source, never repeatedly repair
        # the launcher's already corrected archive.
        cfg.motion_files = (str(ROOT / "robot/reference_data/multimotion/rough_walk_omniretarget.npz"),)
        env = LytRobotAmpEnv(cfg)
        repaired = args.output_dir / "rough_walk_collision_repaired.npz"
        try:
            lift = repair_rough(env, Path(env.cfg.motion_files[0]), repaired)
            print(f"rough repaired: max lift {lift:.6f} m", flush=True)
        finally:
            env.close()
    report = {}
    for key in ("flat", "rough", "slope_up", "slope_down"):
        source = repaired if key == "rough" else Path(launcher_config(key).motion_files[0])
        report[key] = verify_and_render(key, source, args.output_dir, args.seconds, args.render)
        print(key, report[key]["placements"], flush=True)
        (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
