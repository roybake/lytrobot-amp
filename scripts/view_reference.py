#!/usr/bin/env python3
"""Replay an NPZ's original MuJoCo states and optionally export a video."""

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.view_stairs_reference import Playback, read_archive_states, joint_limit_violations, run_playback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-file', type=Path, default=ROOT / 'robot/reference_data/walk1_mjcf_repaired.npz')
    parser.add_argument('--model-xml', type=Path, default=ROOT / 'robot/MJCFmodelfinal.xml')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'recording/reference/walk1_mjcf_repaired.mp4')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--fps', type=float, default=40.0)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--forward', action='store_true', help='Integrate stored root XY velocity for in-place clips.')
    parser.add_argument('--cycles', type=int, default=1, help='Repeat the motion before finishing or looping.')
    args = parser.parse_args()
    if args.speed <= 0 or args.fps <= 0 or args.max_frames < 0 or args.cycles < 1:
        parser.error('speed/fps must be positive and max-frames nonnegative')
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    playback = read_archive_states(args.motion_file, model)
    if not 0 <= args.start_frame < playback.frame_count:
        parser.error('start-frame is outside the clip')
    if not np.isfinite(playback.qpos).all() or not np.isfinite(playback.qvel).all():
        raise ValueError('Motion contains nonfinite values')
    print(f'{playback.frame_count} frames @ {playback.frequency:g} Hz; {playback.duration:.2f} s; model layout matches', flush=True)
    print(f'Joint limit violations (radians): {joint_limit_violations(model, playback.qpos)}', flush=True)
    translating = np.max(np.abs(np.diff(playback.qpos[:, :2], axis=0))) > 1e-5
    if args.forward or translating:
        if args.forward and translating:
            parser.error('--forward requires an in-place reference with fixed root XY')
        # Include the next cycle's first frame, so the entire last interval is played.
        indices = np.arange(playback.frame_count * args.cycles + 1) % playback.frame_count
        qpos, qvel = playback.qpos[indices].copy(), playback.qvel[indices].copy()
        steps = 0.5 * (qvel[:-1, :2] + qvel[1:, :2]) / playback.frequency
        if translating:
            with np.load(args.motion_file, allow_pickle=True) as archive:
                displacement = np.asarray(archive['cycle_translation']) if 'cycle_translation' in archive.files else playback.qpos[-1, :2] - playback.qpos[0, :2] + playback.qvel[-1, :2] / playback.frequency
            qpos[:, :2] += (np.arange(len(indices)) // playback.frame_count)[:, None] * displacement
        else:
            qpos[1:, :2] = qpos[0, :2] + np.cumsum(steps, axis=0)
        args.loop_translation = qpos[-1, :2] - qpos[0, :2]
        playback = Playback(playback.source, qpos, qvel, playback.frequency)
        print(f'Forward playback: {playback.duration:.2f} s, displacement {args.loop_translation} m', flush=True)
    elif args.cycles != 1:
        parser.error('--cycles requires --forward or a translating reference')
    if args.check_only:
        return
    args.viewer = not args.record
    args.follow = True
    args.width, args.height, args.crf = 1280, 720, 18
    args.contact_points = True
    args.azimuth, args.elevation, args.distance, args.lookat = 90.0, -14.0, 2.5, None
    run_playback(args, model, mujoco.MjData(model), playback)


if __name__ == '__main__':
    main()
