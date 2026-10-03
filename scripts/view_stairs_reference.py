#!/usr/bin/env python3
"""Replay the retargeted human stair clip on the exact stair terrain scene.

`robot/terrains/stairs.xml` already embeds both the tendon-driven LYT robot and
the three-up / platform / three-down treads, so one compiled model drives this
visualization and the DeepMimic stair entry point alike.  The clip itself is
`robot/reference_data/multimotion/stairs.npz`: 912 frames at 40 Hz.

`--source authored` replays the exported qpos/qvel exactly as the retargeter
wrote them; the patella slide joints then follow the knee through the model's
joint equality.  `--source training` replays `MotionReference` states, i.e. the
frames the DeepMimic resets use; that reference only tracks the fourteen
locomotion joints, so there the patella joints stay at qpos0.

The raw clip is offset from this stair scene by about one tread.  Playback
applies the accepted -0.39 m root-X alignment by default without editing the
NPZ; use `--root-x-offset 0` to inspect its original world coordinates.
`--check-only` prints the contact audit summary.
"""

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
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mujoco
import numpy as np

DEFAULT_MODEL = REPO_ROOT / "robot" / "terrains" / "stairs.xml"
DEFAULT_REFERENCE = (
    REPO_ROOT / "robot" / "reference_data" / "multimotion" / "stairs.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "recording" / "reference" / "stairs_xml_reference.mp4"
DEFAULT_ROOT_X_OFFSET = -0.39
ROOT_BODY_NAME = "pelvis"
REQUIRED_TREADS = ("stair_up_1", "stair_up_3", "top_platform", "stair_down_3")


@dataclass(frozen=True)
class Playback:
    """Reference states plus the clip metadata needed for playback."""

    source: str
    qpos: np.ndarray
    qvel: np.ndarray
    frequency: float

    @property
    def frame_count(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration(self) -> float:
        return (self.frame_count - 1) / max(self.frequency, 1.0e-9)

    @property
    def root_travel(self) -> float:
        return float(self.qpos[-1, 0] - self.qpos[0, 0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", default=str(DEFAULT_MODEL))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument(
        "--root-x-offset", type=float, default=DEFAULT_ROOT_X_OFFSET,
        help="仅回放时平移机器人根节点 X；原始 NPZ 保持不变。",
    )
    parser.add_argument(
        "--source",
        choices=("authored", "training"),
        default="authored",
        help="authored 直接播放 npz 姿态；training 播放 MotionReference 复位姿态。",
    )
    parser.add_argument("--viewer", action="store_true", help="打开 MuJoCo 交互窗口。")
    parser.add_argument("--record", action="store_true", help="导出 MP4。")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--crf", type=int, default=12)
    parser.add_argument("--fps", type=float, default=50.0, help="播放与录制帧率。")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率。")
    parser.add_argument("--start-frame", type=int, default=0, help="参考起始帧。")
    parser.add_argument("--max-frames", type=int, default=0, help="显示帧数上限，0 为整段。")
    parser.add_argument("--loop", action="store_true", help="viewer 内循环播放整段。")
    parser.add_argument("--follow", action="store_true", help="相机跟随机器人根节点。")
    parser.add_argument("--azimuth", type=float, default=90.0)
    parser.add_argument("--elevation", type=float, default=-14.0)
    parser.add_argument("--distance", type=float, default=-1.0, help="相机距离，<=0 自动。")
    parser.add_argument("--lookat", type=float, nargs=3, default=None)
    parser.add_argument(
        "--contact-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示接触点。",
    )
    parser.add_argument("--check-only", action="store_true", help="只打印数据与接触审计。")
    args = parser.parse_args()
    if args.fps <= 0.0:
        parser.error("--fps 必须为正数。")
    if args.speed <= 0.0:
        parser.error("--speed 必须为正数。")
    if args.max_frames < 0:
        parser.error("--max-frames 不能为负。")
    if args.loop and args.record and not args.viewer:
        parser.error("--loop 只在 --viewer 下有意义。")
    if not args.viewer and not args.record and not args.check_only:
        parser.error("请至少指定 --viewer、--record 或 --check-only 之一。")
    return args


def load_scene(model_xml: Path) -> mujoco.MjModel:
    """Compile the stair scene that already contains robot and treads."""

    if not model_xml.is_file():
        raise SystemExit(f"缺少楼梯场景：{model_xml}")
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    geom_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        for index in range(model.ngeom)
    }
    missing = [name for name in REQUIRED_TREADS if name not in geom_names]
    if missing:
        raise SystemExit(f"{model_xml} 缺少楼梯地形几何 {missing}，无法可视化楼梯行走。")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY_NAME) < 0:
        raise SystemExit(f"{model_xml} 缺少机器人根节点 {ROOT_BODY_NAME}。")
    return model


def read_archive_states(reference_file: Path, model: mujoco.MjModel) -> Playback:
    """Read the exported qpos/qvel and prove they match this model layout."""

    with np.load(reference_file, allow_pickle=True) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        joint_names = [str(name) for name in archive["joint_names"]]
        frequency = float(np.asarray(archive["frequency"]).item())
    if qpos.ndim != 2 or qvel.ndim != 2:
        raise SystemExit(f"参考的 qpos/qvel 必须为二维数组：{qpos.shape} / {qvel.shape}")
    model_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]
    if joint_names != model_joint_names:
        raise SystemExit(
            f"参考关节顺序与 {DEFAULT_MODEL.name} 不一致：{joint_names} != "
            f"{model_joint_names}；请改用 --source training。"
        )
    if qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
        raise SystemExit(
            f"参考维度 qpos={qpos.shape[1]}, qvel={qvel.shape[1]} 与模型 "
            f"nq={model.nq}, nv={model.nv} 不一致；请改用 --source training。"
        )
    return Playback(source="authored", qpos=qpos, qvel=qvel, frequency=frequency)


def load_playback(args: argparse.Namespace, model: mujoco.MjModel) -> Playback:
    """Build the frame table for the requested reference representation."""

    reference_file = Path(args.reference).expanduser().resolve()
    if not reference_file.is_file():
        raise SystemExit(f"缺少楼梯参考动作：{reference_file}")
    if args.source == "authored":
        playback = read_archive_states(reference_file, model)
    else:
        from lytrobot_amp.motion import MotionReference

        reference = MotionReference(
            reference_file, model, zero_root_xy=False, loop_override=False
        )
        playback = Playback(
            source="training",
            qpos=np.asarray(reference.reference_qpos, dtype=np.float64),
            qvel=np.asarray(reference.reference_qvel, dtype=np.float64),
            frequency=float(reference.frequency),
        )
    if args.root_x_offset == 0.0:
        return playback
    qpos = playback.qpos.copy()
    qpos[:, 0] += float(args.root_x_offset)
    return Playback(
        source=playback.source, qpos=qpos, qvel=playback.qvel,
        frequency=playback.frequency,
    )


def joint_limit_violations(model: mujoco.MjModel, qpos: np.ndarray) -> list[tuple[str, float]]:
    """Report joints whose reference samples leave the compiled model limits."""

    violations: list[tuple[str, float]] = []
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        address = int(model.jnt_qposadr[joint_id])
        low, high = (float(value) for value in model.jnt_range[joint_id])
        values = qpos[:, address]
        excess = float(np.max(np.maximum(0.0, np.maximum(low - values, values - high))))
        if excess > 1.0e-6:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            violations.append((str(joint_name), excess))
    return violations


def sample_state(playback: Playback, position: float) -> tuple[np.ndarray, np.ndarray]:
    """Blend neighbouring reference frames with a normalized root quaternion."""

    limit = playback.frame_count - 1
    position = min(max(float(position), 0.0), float(limit))
    low = int(math.floor(position))
    high = min(low + 1, limit)
    blend = position - low
    qpos = (1.0 - blend) * playback.qpos[low] + blend * playback.qpos[high]
    qvel = (1.0 - blend) * playback.qvel[low] + blend * playback.qvel[high]
    quat0 = playback.qpos[low, 3:7]
    quat1 = playback.qpos[high, 3:7]
    if float(np.dot(quat0, quat1)) < 0.0:
        quat1 = -quat1
    quat = (1.0 - blend) * quat0 + blend * quat1
    qpos[3:7] = quat / max(float(np.linalg.norm(quat)), 1.0e-9)
    return qpos, qvel


def course_bounds(playback: Playback) -> np.ndarray:
    """Return the lower/upper corner of the root travel used for framing."""

    root = playback.qpos[:, :3]
    return np.stack((root.min(axis=0), root.max(axis=0)))


def auto_lookat(bounds: np.ndarray) -> np.ndarray:
    return np.array(
        (
            0.5 * float(bounds[0, 0] + bounds[1, 0]),
            0.5 * float(bounds[0, 1] + bounds[1, 1]),
            0.45,
        )
    )


def auto_distance(bounds: np.ndarray, model: mujoco.MjModel, aspect: float) -> float:
    """Frame the whole course with the vertical field of view of the scene."""

    span = max(float(bounds[1, 0] - bounds[0, 0]), 1.0)
    fovy = math.radians(float(model.vis.global_.fovy))
    half_width = math.tan(fovy * 0.5) * max(aspect, 0.2)
    return max(3.0, 1.2 * 0.5 * span / max(half_width, 1.0e-3))


def place_camera(
    cam: mujoco.MjvCamera,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    args: argparse.Namespace,
    bounds: np.ndarray,
    aspect: float,
) -> None:
    """Point a free camera either at the whole course or at the root body."""

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = float(args.azimuth)
    cam.elevation = float(args.elevation)
    if args.follow:
        follow_camera_lookat(cam, model, data)
        cam.distance = float(args.distance) if args.distance > 0.0 else 3.4
        return
    cam.lookat[:] = (
        np.asarray(args.lookat, dtype=np.float64)
        if args.lookat is not None
        else auto_lookat(bounds)
    )
    cam.distance = (
        float(args.distance)
        if args.distance > 0.0
        else auto_distance(bounds, model, aspect)
    )


def follow_camera_lookat(
    cam: mujoco.MjvCamera, model: mujoco.MjModel, data: mujoco.MjData
) -> None:
    """Follow the pelvis while preserving mouse-controlled orbit and zoom."""

    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY_NAME)
    cam.lookat[:] = data.xpos[root_body]
    cam.lookat[2] += 0.05


class VideoWriter:
    """Pipe offscreen renders into ffmpeg with the repository codec settings."""

    def __init__(
        self,
        model: mujoco.MjModel,
        output: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.width = max(2, int(width) // 2 * 2)
        self.height = max(2, int(height) // 2 * 2)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.height)
        self.renderer = mujoco.Renderer(model, width=self.width, height=self.height)
        self.process = subprocess.Popen(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s:v", f"{self.width}x{self.height}", "-r", f"{fps:.3f}", "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
            ],
            stdin=subprocess.PIPE,
        )

    def write(
        self, data: mujoco.MjData, camera: mujoco.MjvCamera, option: mujoco.MjvOption
    ) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg 输入管道不可用。")
        self.renderer.update_scene(data, camera=camera, scene_option=option)
        self.process.stdin.write(self.renderer.render().tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        self.renderer.close()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}.")


def display_frame_count(args: argparse.Namespace, playback: Playback) -> int:
    """Number of display frames needed to play the clip once at --speed."""

    remaining = max(playback.frame_count - 1 - int(args.start_frame), 0)
    frames = int(round(remaining / playback.frequency / args.speed * args.fps)) + 1
    if args.max_frames > 0:
        frames = min(frames, int(args.max_frames))
    return max(frames, 1)


def run_playback(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    playback: Playback,
) -> None:
    """Drive the model through the clip, recording and/or viewing it."""

    bounds = course_bounds(playback)
    frames = display_frame_count(args, playback)
    aspect = args.width / max(args.height, 1)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = bool(args.contact_points)
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    render_camera = mujoco.MjvCamera()
    place_camera(render_camera, model, data, args, bounds, aspect)
    recorder = (
        VideoWriter(
            model,
            Path(args.output).expanduser().resolve(),
            args.width,
            args.height,
            args.fps,
            args.crf,
        )
        if args.record
        else None
    )
    viewer_context = nullcontext(None)
    if args.viewer:
        from mujoco import viewer as mj_viewer

        viewer_context = mj_viewer.launch_passive(model, data)
    try:
        with viewer_context as viewer:
            if viewer is not None:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = bool(
                    args.contact_points
                )
                place_camera(viewer.cam, model, data, args, bounds, aspect)
            loop_offset = np.zeros(2, dtype=np.float64)
            while True:
                wall_start = time.perf_counter()
                for index in range(frames):
                    source = (
                        int(args.start_frame)
                        + index / args.fps * playback.frequency * args.speed
                    )
                    pose, velocity = sample_state(playback, source)
                    pose = pose.copy()
                    pose[:2] += loop_offset
                    data.qpos[:] = pose
                    data.qvel[:] = velocity
                    mujoco.mj_forward(model, data)
                    if recorder is not None:
                        if args.follow:
                            follow_camera_lookat(render_camera, model, data)
                        recorder.write(data, render_camera, option)
                    if viewer is None:
                        continue
                    if args.follow:
                        with viewer.lock():
                            follow_camera_lookat(viewer.cam, model, data)
                    viewer.sync()
                    deadline = wall_start + (index + 1) / args.fps
                    time.sleep(max(0.0, deadline - time.perf_counter()))
                    if not viewer.is_running():
                        return
                if viewer is None or not args.loop:
                    break
                loop_offset += getattr(args, 'loop_translation', np.zeros(2))
            if viewer is not None:
                print("播放结束，窗口停在最后一帧，关闭窗口即退出。")
                while viewer.is_running():
                    time.sleep(0.1)
    finally:
        if recorder is not None:
            recorder.close()
    if recorder is not None:
        print(f"Saved video: {Path(args.output).expanduser().resolve()}")


def main() -> None:
    args = parse_args()
    model = load_scene(Path(args.model_xml).expanduser().resolve())
    playback = load_playback(args, model)
    print(
        f"场景 {Path(args.model_xml).name}：nq={model.nq}, nv={model.nv}；"
        f"参考 {Path(args.reference).name}（{playback.source}）："
        f"{playback.frame_count} 帧 @ {playback.frequency:g} Hz，"
        f"{playback.duration:.2f} s，根节点前进 {playback.root_travel:.3f} m，"
        f"根节点 X 平移 {args.root_x_offset:+.3f} m"
    )
    violations = joint_limit_violations(model, playback.qpos)
    print(
        "关节限位："
        + (
            ", ".join(f"{name} 越界 {excess:.4f}" for name, excess in violations)
            if violations
            else "全部在模型范围内"
        )
    )
    if args.check_only:
        from scripts.audit_stair_reference import audit_reference

        report = audit_reference(
            args.model_xml, args.reference, root_x_offset=args.root_x_offset
        )
        print(
            f"接触审计：最大足部穿透 {report['maximum_foot_penetration_m'] * 1000:.1f} mm"
            f"（第 {report['worst_penetration_frame']} 帧 {report['worst_penetration_pair']}），"
            f"视觉脚网格最大穿透 {report['maximum_visual_foot_mesh_penetration_m'] * 1000:.1f} mm，"
            f">2 cm 帧数 {report['frames_over_2cm_penetration']}，"
            f"台面外伸帧数 {report['tread_overhang_frames']} / "
            f"{report['tread_top_contact_frames']}，"
            f"最小台面余量 {report['minimum_visual_tread_edge_margin_m']:.3f} m，"
            f"最长无支撑 {report['longest_unsupported_run_s']:.2f} s，"
            f"通过={report['contact_geometry_pass']}"
        )
        if not report["contact_geometry_pass"]:
            print("提示：该参考与楼梯碰撞几何不兼容，画面中足部会与踏步重叠。")
        return

    run_playback(args, model, mujoco.MjData(model), playback)


if __name__ == "__main__":
    main()
