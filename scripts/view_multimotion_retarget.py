#!/usr/bin/env python3
"""Visualize all OmniRetarget actions on MoRE complex terrain lanes."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mujoco
import numpy as np

from lytrobot_amp.motion import MotionReference
from lytrobot_amp.terrain import (
    MoreTerrainCfg,
    StairTerrainCfg,
    load_model,
    terrain_kind_from_motion,
    terrain_spawn_offset,
)


MODEL_XML = REPO_ROOT / "robot" / "MJCFmodelposition.xml"
MOTION_DIR = REPO_ROOT / "robot" / "reference_data" / "multimotion"
DEFAULT_OUTPUT = (
    REPO_ROOT / "recording" / "reference" / "multimotion_more_terrain_test.mp4"
)
CLIP_ORDER = (
    "walk",
    "turn_left",
    "turn_right",
    "rough_walk",
    "slope_up",
    "slope_down",
    "pit_walk",
    "gap_cross",
    "stair_up",
    "stair_down",
)


@dataclass(frozen=True)
class ClipSpec:
    key: str
    title: str
    subtitle: str
    filename: str
    azimuth: float
    distance: float


@dataclass(frozen=True)
class Segment:
    spec: ClipSpec
    motion: MotionReference
    start_frame: int
    intro_frames: int
    motion_frames: int
    outro_frames: int
    cycles: int

    @property
    def total_frames(self) -> int:
        return self.intro_frames + self.motion_frames + self.outro_frames

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.total_frames


CLIP_SPECS = {
    "walk": ClipSpec(
        key="walk",
        title="WALK",
        subtitle="CMU 69_01  |  cyclic gait",
        filename="walk_omniretarget.npz",
        azimuth=105.0,
        distance=2.35,
    ),
    "turn_left": ClipSpec(
        key="turn_left",
        title="TURN LEFT",
        subtitle="CMU 69_06  |  89.7 deg controlled",
        filename="turn_left_omniretarget.npz",
        azimuth=135.0,
        distance=2.45,
    ),
    "turn_right": ClipSpec(
        key="turn_right",
        title="TURN RIGHT",
        subtitle="CMU 69_06  |  88.0 deg controlled",
        filename="turn_right_omniretarget.npz",
        azimuth=45.0,
        distance=2.45,
    ),
    "rough_walk": ClipSpec(
        key="rough_walk",
        title="ROUGH WALK",
        subtitle="MoRE-discoRL heightfield  |  feet follow surface",
        filename="rough_walk_omniretarget.npz",
        azimuth=105.0,
        distance=2.55,
    ),
    "slope_up": ClipSpec(
        key="slope_up",
        title="SLOPE WALK UP",
        subtitle="derived from flat walk  |  6 deg full-foot terrain IK",
        filename="slope_up_omniretarget.npz",
        azimuth=105.0,
        distance=2.55,
    ),
    "slope_down": ClipSpec(
        key="slope_down",
        title="SLOPE WALK DOWN",
        subtitle="derived from flat walk  |  6 deg full-foot terrain IK",
        filename="slope_down_omniretarget.npz",
        azimuth=105.0,
        distance=2.55,
    ),
    "pit_walk": ClipSpec(
        key="pit_walk",
        title="PIT WALK",
        subtitle="MoRE-discoRL pit  |  feet on elevated terrain",
        filename="pit_walk_omniretarget.npz",
        azimuth=105.0,
        distance=2.65,
    ),
    "gap_cross": ClipSpec(
        key="gap_cross",
        title="GAP WALK",
        subtitle="MoRE curriculum gap 0.20 m  |  feet on platforms",
        filename="gap_cross_omniretarget.npz",
        azimuth=105.0,
        distance=2.75,
    ),
    "stair_up": ClipSpec(
        key="stair_up",
        title="STAIRS WALK UP",
        subtitle="CMU 114_07  |  feet on 3 calibrated steps",
        filename="stair_up_omniretarget.npz",
        azimuth=90.0,
        distance=2.65,
    ),
    "stair_down": ClipSpec(
        key="stair_down",
        title="STAIRS WALK DOWN",
        subtitle="CMU 114_07  |  feet on 3 calibrated steps",
        filename="stair_down_omniretarget.npz",
        azimuth=90.0,
        distance=2.65,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play or export all action and MoRE-terrain references on the LYT robot."
    )
    parser.add_argument("--clips", nargs="+", choices=CLIP_ORDER, default=list(CLIP_ORDER))
    parser.add_argument("--model-xml", default=str(MODEL_XML))
    parser.add_argument("--motion-dir", default=str(MOTION_DIR))
    parser.add_argument("--viewer", action="store_true", help="Also open the interactive MuJoCo viewer.")
    parser.add_argument(
        "--hold-viewer",
        action="store_true",
        help="Keep the interactive viewer open after playback until its window closes.",
    )
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--contact-sheet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--walk-cycles", type=int, default=3)
    parser.add_argument("--intro-seconds", type=float, default=0.55)
    parser.add_argument("--outro-seconds", type=float, default=0.35)
    return parser.parse_args()


def reference_stairs() -> StairTerrainCfg:
    """Return the stair geometry used by the multi-motion training configuration."""

    return StairTerrainCfg(
        enabled=True,
        step_height=0.13,
        step_depth=0.58,
        step_count=3,
        width=2.0,
        start_x=-0.95,
    )


def reference_more_terrains(clip_keys: list[str] | None = None) -> MoreTerrainCfg:
    """Return visualization terrain, including a serial slope course if needed."""

    requested = set(clip_keys or ())
    if requested & {"slope_up", "slope_down"}:
        terrain_types = ["gap", "pit", "rough", "slope_up", "slope_down", "stairs"]
        if requested <= {"slope_up", "slope_down"}:
            terrain_types = ["slope_up", "slope_down"]
        return MoreTerrainCfg(
            enabled=True,
            terrain_types=tuple(terrain_types),
            serial_course=True,
            course_orders=(tuple(terrain_types),),
            course_lane_spacing=3.0,
            course_base_height=0.30,
            lane_width=2.0,
        )

    return MoreTerrainCfg(
        enabled=True,
        terrain_types=("gap", "pit", "rough", "stairs"),
    )


def build_segments(
    model: mujoco.MjModel,
    clip_keys: list[str],
    motion_dir: Path,
    fps: float,
    speed: float,
    walk_cycles: int,
    intro_seconds: float,
    outro_seconds: float,
) -> list[Segment]:
    segments: list[Segment] = []
    next_frame = 0
    for key in clip_keys:
        spec = CLIP_SPECS[key]
        motion = MotionReference(motion_dir / spec.filename, model, zero_root_xy=False)
        cycles = walk_cycles if key == "walk" else 1
        duration = cycles * motion.frame_count / motion.frequency / speed
        segment = Segment(
            spec=spec,
            motion=motion,
            start_frame=next_frame,
            intro_frames=max(0, round(intro_seconds * fps)),
            motion_frames=max(1, math.ceil(duration * fps)),
            outro_frames=max(0, round(outro_seconds * fps)),
            cycles=cycles,
        )
        segments.append(segment)
        next_frame = segment.end_frame
    return segments


def sample_qpos(motion: MotionReference, source_position: float, loop: bool) -> tuple[np.ndarray, int]:
    low_unwrapped = int(math.floor(source_position))
    blend = float(source_position - low_unwrapped)
    if loop:
        cycle = low_unwrapped // motion.frame_count
        low = low_unwrapped % motion.frame_count
        high = (low + 1) % motion.frame_count
        low_qpos = np.asarray(motion.reference_qpos[low], dtype=np.float64).copy()
        high_qpos = np.asarray(motion.reference_qpos[high], dtype=np.float64).copy()
        stride = float(
            motion.reference_qpos[-1, 0]
            + np.median(np.diff(motion.reference_qpos[:, 0]))
        )
        low_qpos[0] += cycle * stride
        high_qpos[0] += (cycle + int(high == 0)) * stride
    else:
        low = min(low_unwrapped, motion.frame_count - 1)
        high = min(low + 1, motion.frame_count - 1)
        low_qpos = np.asarray(motion.reference_qpos[low], dtype=np.float64)
        high_qpos = np.asarray(motion.reference_qpos[high], dtype=np.float64)

    qpos = (
        (1.0 - blend) * low_qpos
        + blend * high_qpos
    ).astype(np.float64)
    quat0 = np.asarray(motion.reference_qpos[low, 3:7], dtype=np.float64)
    quat1 = np.asarray(motion.reference_qpos[high, 3:7], dtype=np.float64)
    if float(np.dot(quat0, quat1)) < 0.0:
        quat1 = -quat1
    quat = (1.0 - blend) * quat0 + blend * quat1
    qpos[3:7] = quat / max(float(np.linalg.norm(quat)), 1.0e-9)
    return qpos, low


def clip_pose(
    segment: Segment,
    local_frame: int,
    fps: float,
    speed: float,
    stairs: StairTerrainCfg,
    more: MoreTerrainCfg,
) -> tuple[np.ndarray, int, float]:
    motion = segment.motion
    if local_frame < segment.intro_frames:
        source_position = 0.0
        progress = 0.0
    elif local_frame >= segment.intro_frames + segment.motion_frames:
        source_position = float(
            motion.frame_count - 1 if segment.spec.key != "walk" else segment.cycles * motion.frame_count - 1
        )
        progress = 1.0
    else:
        playback_frame = local_frame - segment.intro_frames
        source_position = playback_frame / fps * speed * motion.frequency
        max_position = float(segment.cycles * motion.frame_count - 1)
        source_position = min(source_position, max_position)
        progress = playback_frame / max(segment.motion_frames - 1, 1)

    qpos, source_index = sample_qpos(
        motion,
        source_position,
        loop=segment.spec.key == "walk",
    )
    terrain_kind = terrain_kind_from_motion(segment.spec.key)
    qpos[:3] += terrain_spawn_offset(
        terrain_kind,
        more,
        stairs,
        stair_down=segment.spec.key == "stair_down",
        stair_down_offset=0.25,
        reference_contains_slope_elevation=terrain_kind in {"slope_up", "slope_down"},
    )
    return qpos, source_index, float(np.clip(progress, 0.0, 1.0))


def camera_for_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: ClipSpec,
) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = spec.distance
    camera.azimuth = spec.azimuth
    camera.elevation = -12.0
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id >= 0:
        camera.lookat[:] = data.xpos[pelvis_id]
        camera.lookat[2] += 0.10
    else:
        camera.lookat[:] = data.qpos[:3]
    return camera


def ffmpeg_overlay(segments: list[Segment], fps: float) -> str:
    font_candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    )
    font = next((path for path in font_candidates if path.exists()), None)
    if font is None:
        return ""

    filters: list[str] = []
    for segment in segments:
        start = segment.start_frame / fps
        end = segment.end_frame / fps
        enable = f"between(t,{start:.4f},{end:.4f})"
        filters.extend(
            (
                "drawtext="
                f"fontfile={font}:text='{segment.spec.title}':"
                "fontcolor=white:fontsize=44:x=38:y=34:"
                "box=1:boxcolor=black@0.58:boxborderw=14:"
                f"enable='{enable}'",
                "drawtext="
                f"fontfile={font}:text='{segment.spec.subtitle}':"
                "fontcolor=white:fontsize=22:x=40:y=105:"
                "box=1:boxcolor=black@0.45:boxborderw=8:"
                f"enable='{enable}'",
            )
        )
    filters.append(
        "drawtext="
        f"fontfile={font}:text='CONTACT TEST  |  heel > sole > toe':"
        "fontcolor=white@0.9:fontsize=19:x=24:y=h-th-20:"
        "box=1:boxcolor=black@0.35:boxborderw=7"
    )
    filters.append(
        "drawtext="
        f"fontfile={font}:text='OmniRetarget  |  LYT 14-DoF':"
        "fontcolor=white@0.9:fontsize=19:x=w-tw-24:y=h-th-20:"
        "box=1:boxcolor=black@0.35:boxborderw=7"
    )
    return ",".join(filters)


class VideoWriter:
    def __init__(
        self,
        model: mujoco.MjModel,
        output: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        segments: list[Segment],
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found; it is required for MP4 export.")
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        width = max(2, int(width) // 2 * 2)
        height = max(2, int(height) // 2 * 2)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.3f}",
            "-i",
            "-",
        ]
        overlay = ffmpeg_overlay(segments, fps)
        if overlay:
            command.extend(("-vf", overlay))
        command.extend(
            (
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            )
        )
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, data: mujoco.MjData, camera: mujoco.MjvCamera, option: mujoco.MjvOption) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input stream is unavailable.")
        self.renderer.update_scene(data, camera=camera, scene_option=option)
        self.process.stdin.write(self.renderer.render().tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        self.renderer.close()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}.")


def create_contact_sheet(
    video_path: Path,
    output_path: Path,
    segments: list[Segment],
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    sample_frames = [
        segment.start_frame + segment.intro_frames + segment.motion_frames // 2
        for segment in segments
    ]
    selection = "+".join(f"eq(n\\,{frame})" for frame in sample_frames)
    columns = 2 if len(sample_frames) > 1 else 1
    rows = math.ceil(len(sample_frames) / columns)
    video_filter = (
        f"select={selection},"
        f"scale=640:-2,"
        f"tile={columns}x{rows}:padding=8:margin=8:color=0x20242b"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            video_filter,
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=True,
    )


def _reference_forward_speed(motion: MotionReference) -> float:
    """Return the displayed root-X speed in native reference time."""

    qpos = np.asarray(motion.reference_qpos, dtype=np.float64)
    frequency = float(motion.frequency)
    if len(qpos) < 2 or frequency <= 0.0:
        return 0.0
    loop = bool(getattr(motion, "source_loop", False))
    duration = (len(qpos) - (0 if loop else 1)) / frequency
    distance = float(qpos[-1, 0] - qpos[0, 0])
    if loop:
        distance += float(np.median(np.diff(qpos[:, 0])))
    return distance / max(duration, 1.0e-9)


def main() -> None:
    args = parse_args()
    if not args.record and not args.viewer:
        raise ValueError("Enable at least one of --record or --viewer.")
    if args.fps <= 0.0 or args.speed <= 0.0:
        raise ValueError("--fps and --speed must be positive.")
    if not math.isclose(args.speed, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            "MuJoCo playback and recordings require --speed 1.0 so displayed "
            "world velocity matches the reference's physical robot velocity."
        )
    if args.walk_cycles < 1:
        raise ValueError("--walk-cycles must be at least one.")

    stairs = reference_stairs()
    more = reference_more_terrains(args.clips)
    model = load_model(args.model_xml, stairs, more)
    data = mujoco.MjData(model)
    segments = build_segments(
        model=model,
        clip_keys=args.clips,
        motion_dir=Path(args.motion_dir).expanduser().resolve(),
        fps=float(args.fps),
        speed=float(args.speed),
        walk_cycles=int(args.walk_cycles),
        intro_seconds=float(args.intro_seconds),
        outro_seconds=float(args.outro_seconds),
    )
    output = Path(args.output).expanduser().resolve()
    recorder = (
        VideoWriter(
            model,
            output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            crf=args.crf,
            segments=segments,
        )
        if args.record
        else None
    )

    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    if args.viewer:
        from mujoco import viewer as mj_viewer

        viewer_context = mj_viewer.launch_passive(model, data)
    else:
        viewer_context = nullcontext(None)

    try:
        with viewer_context as viewer:
            for segment in segments:
                print(
                    f"[{segment.spec.title}] {segment.motion.frame_count} source frames, "
                    f"{segment.motion.frequency:g} Hz, native_world_speed="
                    f"{_reference_forward_speed(segment.motion):.4f} m/s"
                )
                wall_start = time.perf_counter()
                for local_frame in range(segment.total_frames):
                    qpos, _, _ = clip_pose(
                        segment,
                        local_frame,
                        args.fps,
                        args.speed,
                        stairs,
                        more,
                    )
                    data.qpos[:] = qpos
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    camera = camera_for_pose(model, data, segment.spec)

                    if recorder is not None:
                        recorder.write(data, camera, option)
                    if viewer is not None:
                        viewer.cam.type = camera.type
                        viewer.cam.distance = camera.distance
                        viewer.cam.azimuth = camera.azimuth
                        viewer.cam.elevation = camera.elevation
                        viewer.cam.lookat[:] = camera.lookat
                        viewer.sync()
                        deadline = wall_start + (local_frame + 1) / args.fps
                        time.sleep(max(0.0, deadline - time.perf_counter()))
                        if not viewer.is_running():
                            break
                if viewer is not None and not viewer.is_running():
                    break
            if viewer is not None and args.hold_viewer and viewer.is_running():
                while viewer.is_running():
                    time.sleep(0.1)
    finally:
        if recorder is not None:
            recorder.close()

    if recorder is not None:
        print(f"Saved video: {output}")
        if args.contact_sheet:
            sheet = output.with_suffix(".png")
            create_contact_sheet(output, sheet, segments)
            print(f"Saved contact sheet: {sheet}")


if __name__ == "__main__":
    main()
