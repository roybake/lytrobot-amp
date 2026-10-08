#!/usr/bin/env python3
"""Run a trained policy checkpoint in the LYT MuJoCo environment."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import mujoco
import numpy as np

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.networks import ActorCritic
from lytrobot_amp.terrain import load_multi_robot_model


class OnnxPolicy:
    """Small NumPy adapter around an exported deterministic actor."""

    def __init__(self, path: str | Path) -> None:
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        # Small recurrent playback batches benefit from one CPU worker and
        # leave CPU resources available to the simulator and training jobs.
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(Path(path).expanduser()),
            sess_options=session_options,
            providers=("CPUExecutionProvider",),
        )
        self.inputs = {value.name: value for value in self.session.get_inputs()}
        self.outputs = {value.name: value for value in self.session.get_outputs()}
        self.input = self.inputs["observation"]
        self.output = self.outputs["action"]
        self.recurrent = "recurrent_state" in self.inputs
        self.recurrent_state: np.ndarray | None = None

    def validate_dimensions(self, obs_dim: int, action_dim: int) -> None:
        input_dim = self.input.shape[-1]
        output_dim = self.output.shape[-1]
        if isinstance(input_dim, int) and input_dim != obs_dim:
            raise ValueError(f"ONNX obs_dim={input_dim} does not match environment obs_dim={obs_dim}.")
        if isinstance(output_dim, int) and output_dim != action_dim:
            raise ValueError(f"ONNX action_dim={output_dim} does not match environment action_dim={action_dim}.")

    def act_inference(
        self,
        observation: np.ndarray,
        episode_start: np.ndarray | None = None,
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        feeds = {self.input.name: observation}
        output_names = (self.output.name,)
        if self.recurrent:
            hidden_dim = self.inputs["recurrent_state"].shape[-1]
            if not isinstance(hidden_dim, int):
                raise ValueError("ONNX recurrent_state has no fixed hidden dimension.")
            if self.recurrent_state is None or self.recurrent_state.shape[0] != observation.shape[0]:
                self.recurrent_state = np.zeros(
                    (observation.shape[0], hidden_dim),
                    dtype=np.float32,
                )
                episode_start = np.ones(observation.shape[0], dtype=np.float32)
            if episode_start is None:
                episode_start = np.zeros(observation.shape[0], dtype=np.float32)
            feeds["recurrent_state"] = self.recurrent_state
            feeds["episode_start"] = np.asarray(episode_start, dtype=np.float32)
            output_names = (self.output.name, "next_recurrent_state")
        outputs = self.session.run(output_names, feeds)
        if self.recurrent:
            self.recurrent_state = outputs[1]
        return outputs[0]


def load_onnx_metadata(path: str | Path) -> dict:
    import onnx

    model = onnx.load(Path(path).expanduser(), load_external_data=False)
    properties = {item.key: item.value for item in model.metadata_props}
    serialized = properties.get("lytrobot_amp.metadata")
    if serialized is None:
        raise ValueError(
            "ONNX model has no lytrobot_amp.metadata; use an ONNX model exported by this training code."
        )
    metadata = json.loads(serialized)
    if not isinstance(metadata, dict):
        raise ValueError("Invalid lytrobot_amp.metadata in ONNX model.")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play/evaluate a LYT BPO-AMP checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-xml", default=None)
    parser.add_argument(
        "--motion-file", default=None,
        help="Playback reference; defaults to the one saved in the checkpoint.",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of policy steps to run. Use 0 for no limit.")
    parser.add_argument("--target-velocity", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Playback RNG seed. Omit it to randomize each run; set it to reproduce one play test.",
    )
    parser.add_argument("--episode-random-seed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--training-aligned",
        action="store_true",
        help=(
            "Preserve reset noise and per-episode RNG behavior saved in the checkpoint. "
            "Use for distributional evaluation; the default viewer mode is deterministic."
        ),
    )
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True,
                        help="Open the local MuJoCo viewer by default; use --no-viewer for headless playback.")
    parser.add_argument(
        "--num-robots",
        type=int,
        default=None,
        help="Robots shown simultaneously; multimotion playback defaults to 3.",
    )
    parser.add_argument(
        "--serial-terrain-course",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use randomized all-terrain serial courses (enabled by default for multimotion).",
    )
    parser.add_argument(
        "--terrain-test",
        choices=("rough", "slope-up", "slope-down"),
        default=None,
        help=(
            "Run one independent sensor-policy route: flat→rough, flat→uphill, "
            "or downhill→flat. Each route has continuous terrain seams."
        ),
    )
    parser.add_argument(
        "--stage",
        default="auto",
        choices=("auto", "stage1", "stage2", "multimotion"),
        help="Checkpoint group used for recording output; auto infers it from the path and metadata.",
    )
    parser.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stand-start-prob", type=float, default=None)
    parser.add_argument("--stand-start-motion-phase", type=float, default=None)
    parser.add_argument("--stand-start-phase-delay-steps", type=int, default=None)
    parser.add_argument("--stand-start-transition-min-steps", type=int, default=None)
    parser.add_argument("--stand-start-transition-max-steps", type=int, default=None)
    parser.add_argument("--stand-start-push-delay-steps", type=int, default=None)
    parser.add_argument(
        "--camera-mode",
        default="follow",
        choices=("follow", "tracking", "free"),
        help="Viewer/recording camera mode. follow manually locks lookat to the pelvis every frame.",
    )
    parser.add_argument("--record-dir", default=str(REPO_ROOT / "recording"))
    parser.add_argument("--record-width", type=int, default=1920)
    parser.add_argument("--record-height", type=int, default=1080)
    parser.add_argument("--record-crf", type=int, default=12)
    parser.add_argument("--record-fps", type=float, default=30.0)
    return parser.parse_args()


def make_env_cfg(
    args: argparse.Namespace,
    checkpoint: dict,
    stage: str | None = None,
) -> LytRobotEnvCfg:
    checkpoint_env_cfg = checkpoint.get("env_config", {})
    model_xml = args.model_xml or checkpoint_env_cfg.get("model_xml") or str(
        REPO_ROOT / "robot" / "MJCFmodelposition.xml"
    )
    saved_motion_file = (
        checkpoint_env_cfg.get("motion_file")
        if isinstance(checkpoint_env_cfg, dict) else None
    )
    motion_file = args.motion_file or saved_motion_file or str(
        REPO_ROOT / "robot" / "reference_data" / "walk1_mjcf_repaired.npz"
    )
    cfg = LytRobotEnvCfg(
        model_xml=model_xml,
        motion_file=motion_file,
        num_envs=1,
        target_velocity=(
            float(args.target_velocity)
            if args.target_velocity is not None
            else float(checkpoint.get("env_config", {}).get("target_velocity", 0.87))
        ),
        reset_noise_scale=0.0,
        stand_start_prob=0.0,
    )
    if isinstance(checkpoint_env_cfg, dict):
        valid_fields = {field.name for field in fields(LytRobotEnvCfg)}
        for key, value in checkpoint_env_cfg.items():
            if key in valid_fields and key not in {"model_xml", "motion_file", "num_envs", "seed", "target_velocity"}:
                setattr(cfg, key, value)
    cfg.model_xml = model_xml
    cfg.motion_file = motion_file
    if not args.training_aligned:
        cfg.stair_phase_reset_prob = 0.0
    resolved_stage = stage or infer_stage(args.checkpoint, checkpoint, args.stage)
    robot_count = args.num_robots
    if robot_count is None:
        robot_count = 3 if resolved_stage == "multimotion" else 1
    if robot_count < 1:
        raise ValueError("--num-robots must be positive.")
    cfg.num_envs = int(robot_count)
    cfg.seed = int(args.seed)
    if args.target_velocity is not None:
        cfg.target_velocity = float(args.target_velocity)
    if not getattr(args, "training_aligned", False):
        cfg.reset_noise_scale = 0.0
    use_serial_course = (
        resolved_stage == "multimotion"
        if args.serial_terrain_course is None
        else bool(args.serial_terrain_course)
    )
    # Honor an explicit --no-serial-terrain-course even when the checkpoint
    # was trained with a serial course.  Previously the copied checkpoint
    # value stayed True because only the enabling branch assigned this field.
    cfg.serial_terrain_course = use_serial_course
    terrain_test = getattr(args, "terrain_test", None)
    if terrain_test is not None:
        terrain = terrain_test.replace("-", "_")
        cfg.terrain_types = (terrain,)
        cfg.serial_terrain_course = True
        # These are deployment tests, so the flat portions are the MJCF's
        # default MuJoCo floor (z=0), not the raised training-course datum.
        cfg.terrain_course_base_height = 0.0
        # Leave enough native floor for a stable gait to settle, while keeping
        # a short recording able to show the requested terrain transition.
        cfg.terrain_course_run_in = 0.75
        cfg.terrain_course_variants = max(int(robot_count), 1)
        cfg.terrain_course_random_assignment = False
        cfg.terrain_course_random_start_prob = 0.0
        cfg.stand_start_prob = 0.0
    if use_serial_course:
        cfg.terrain_course_variants = max(int(robot_count), 1)
        cfg.terrain_course_random_assignment = False
        # Playback is a strict full-route evaluation, never a curriculum reset.
        cfg.terrain_course_random_start_prob = 0.0
    if use_serial_course:
        cfg.stand_start_prob = 0.0
    if args.stand_start_prob is not None:
        cfg.stand_start_prob = float(args.stand_start_prob)
    stand_start_motion_phase = getattr(args, "stand_start_motion_phase", None)
    if stand_start_motion_phase is not None:
        cfg.stand_start_motion_phase = float(stand_start_motion_phase)
    if args.stand_start_phase_delay_steps is not None:
        cfg.stand_start_phase_delay_steps = int(args.stand_start_phase_delay_steps)
    if args.stand_start_transition_min_steps is not None:
        cfg.stand_start_transition_min_steps = int(args.stand_start_transition_min_steps)
    if args.stand_start_transition_max_steps is not None:
        cfg.stand_start_transition_max_steps = int(args.stand_start_transition_max_steps)
    if args.stand_start_push_delay_steps is not None:
        cfg.stand_start_push_delay_steps = int(args.stand_start_push_delay_steps)
    if args.domain_randomization is not None:
        cfg.domain_randomization = bool(args.domain_randomization)
    if args.episode_random_seed is not None:
        cfg.randomize_episode_seed = bool(args.episode_random_seed)
    elif getattr(args, "seed_was_provided", False) and not getattr(args, "training_aligned", False):
        cfg.randomize_episode_seed = False
    return cfg


def infer_stage(checkpoint_path: str | Path, checkpoint: dict, stage_arg: str) -> str:
    if stage_arg in {"stage1", "stage2", "multimotion"}:
        return stage_arg
    parts = set(Path(checkpoint_path).expanduser().parts)
    if "multimotion" in parts:
        return "multimotion"
    if "stage2" in parts:
        return "stage2"
    if "stage1" in parts:
        return "stage1"
    env_cfg = checkpoint.get("env_config", {})
    if isinstance(env_cfg, dict):
        motion_files = env_cfg.get("motion_files", ())
        if isinstance(motion_files, (list, tuple)) and len(motion_files) > 1:
            return "multimotion"
        if bool(env_cfg.get("domain_randomization", False)):
            return "stage2"
    return "stage1"


def checkpoint_recording_path(
    checkpoint_path: str | Path,
    checkpoint: dict,
    record_dir: str | Path,
    stage: str,
) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser()
    # Keep recording names portable across Linux, Windows, FAT, and exFAT.
    run_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", checkpoint_path.parent.name).rstrip(" .")
    if not run_name:
        run_name = "run"
    match = re.fullmatch(r"(?:model|modle)_(\d+)", checkpoint_path.stem)
    if match is not None:
        iteration = match.group(1)
    elif checkpoint_path.stem == "model_final" and "iteration" in checkpoint:
        iteration = str(int(checkpoint["iteration"]))
    else:
        iteration = checkpoint_path.stem

    output_dir = Path(record_dir).expanduser().resolve()
    role = checkpoint_path.parent.name
    if role in ("teacher", "student") and "distillation" in checkpoint_path.parts:
        run_dir = next(parent for parent in checkpoint_path.parents if parent.parent.name == "distillation")
        output_dir = output_dir / "distillation" / stage / f"{run_dir.name}-{role}"
    else:
        output_dir = output_dir / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / f"{run_name}-{iteration}.mp4"
    if not base_path.exists():
        return base_path

    suffix = 2
    while True:
        candidate = output_dir / f"{run_name}-{iteration}-{suffix}.mp4"
        if not candidate.exists():
            return candidate
        suffix += 1


def configure_tracking_camera(
    model: mujoco.MjModel,
    camera: mujoco.MjvCamera,
    data: mujoco.MjData | None = None,
    *,
    mode: str = "follow",
) -> None:
    pelvis_ids = playback_pelvis_body_ids(model)
    pelvis_body_id = pelvis_ids[0] if len(pelvis_ids) == 1 else -1
    if mode == "tracking" and pelvis_body_id >= 0:
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = pelvis_body_id
    else:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if len(pelvis_ids) > 1:
        camera.distance = max(8.0, 4.0 * len(pelvis_ids))
        camera.azimuth = 125.0
        camera.elevation = -24.0
    else:
        camera.distance = 2.8
        camera.azimuth = 90.0
        camera.elevation = -12.0
    if data is not None and mode != "free":
        update_follow_camera(model, camera, data, mode=mode)


def playback_pelvis_body_ids(model: mujoco.MjModel) -> list[int]:
    direct = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if direct >= 0:
        return [int(direct)]
    result: list[int] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if re.fullmatch(r"robot_\d+_pelvis", name):
            result.append(body_id)
    return result


def pelvis_camera_target(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    pelvis_ids = playback_pelvis_body_ids(model)
    if pelvis_ids:
        target = np.mean(np.asarray(data.xpos[pelvis_ids]), axis=0)
        target[2] += 0.12
        return target
    if data.qpos.size >= 3:
        return data.qpos[:3].copy()
    return None


def update_follow_camera(
    model: mujoco.MjModel,
    camera: mujoco.MjvCamera,
    data: mujoco.MjData,
    *,
    mode: str,
) -> None:
    if mode == "free":
        return
    if mode == "tracking" and camera.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
        return
    target = pelvis_camera_target(model, data)
    if target is not None:
        camera.lookat[:] = target


def sync_multi_robot_state(
    display_model: mujoco.MjModel,
    display_data: mujoco.MjData,
    env: LytRobotAmpEnv,
) -> None:
    """Mirror independent environment states into the display-only scene."""

    block = int(env.model.nq)
    expected = block * env.num_envs
    if display_model.nq != expected:
        raise ValueError(
            f"Display model nq={display_model.nq} does not match {env.num_envs}x{block}."
        )
    for env_id, source in enumerate(env.datas):
        start = env_id * block
        display_data.qpos[start : start + block] = source.qpos
    mujoco.mj_forward(display_model, display_data)


class MujocoVideoRecorder:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int,
        camera_mode: str,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        render_width = max(2, int(width) // 2 * 2)
        render_height = max(2, int(height) // 2 * 2)
        self.data = data
        self.model = model
        self.camera_mode = camera_mode
        self.output_path = output_path
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), render_width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), render_height)
        self.renderer = mujoco.Renderer(model, height=render_height, width=render_width)
        self.camera = mujoco.MjvCamera()
        configure_tracking_camera(model, self.camera, data, mode=camera_mode)

        fps_value = max(float(fps), 1.0)
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
            f"{self.renderer.width}x{self.renderer.height}",
            "-r",
            f"{fps_value:.3f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(int(crf)),
            "-profile:v",
            "high",
            "-level",
            "4.2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.fps = fps_value
        self.frame_count = 0

    def write_frame(self, camera: mujoco.MjvCamera | None = None, repeat: int = 1) -> None:
        if self.process.stdin is None:
            return
        if camera is None:
            camera = self.camera
        update_follow_camera(self.model, camera, self.data, mode=self.camera_mode)
        self.renderer.update_scene(self.data, camera=camera)
        frame = self.renderer.render()
        frame_bytes = frame.tobytes()
        for _ in range(max(1, int(repeat))):
            self.process.stdin.write(frame_bytes)
            self.frame_count += 1

    def close(self) -> None:
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            return_code = self.process.wait()
        finally:
            self.renderer.close()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self.output_path}")


def main() -> None:
    args = parse_args()
    args.seed_was_provided = args.seed is not None
    if args.seed is None:
        args.seed = secrets.randbits(32)
    torch.manual_seed(int(args.seed))
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    checkpoint_path = Path(args.checkpoint).expanduser()
    is_onnx = checkpoint_path.suffix.lower() == ".onnx"
    if is_onnx:
        checkpoint = load_onnx_metadata(checkpoint_path)
    elif checkpoint_path.suffix.lower() == ".pt":
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
    else:
        raise ValueError("--checkpoint must be a .pt or .onnx model file.")
    ppo_cfg = checkpoint.get("ppo_amp_config", {})
    stage = infer_stage(args.checkpoint, checkpoint, args.stage)

    env_cfg = make_env_cfg(args, checkpoint, stage)
    env = LytRobotAmpEnv(env_cfg)
    sensor_architecture = checkpoint.get("sensor_policy_architecture", {})
    recurrent_sensor_policy = bool(sensor_architecture.get("recurrent", False))
    if is_onnx:
        policy = OnnxPolicy(checkpoint_path)
        policy.validate_dimensions(env.obs_dim, env.action_dim)
        recurrent_sensor_policy = policy.recurrent
        recurrent_state = None
    else:
        if recurrent_sensor_policy:
            from sensor_training.networks import SensorActorCritic
            from sensor_training.experiment_options import estimator_kwargs

            policy = SensorActorCritic(
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                actor_hidden_dims=tuple(ppo_cfg.get("actor_hidden_dims", (256, 128))),
                critic_hidden_dims=tuple(ppo_cfg.get("critic_hidden_dims", (256, 128))),
                init_std=float(ppo_cfg.get("init_std", 0.8)),
                min_action_std=float(ppo_cfg.get("min_action_std", 0.05)),
                max_action_std=float(ppo_cfg.get("max_action_std", 0.80)),
                sensor_frame_dim=int(sensor_architecture["sensor_frame_dim"]),
                actor_history_frames=int(sensor_architecture.get("actor_history_frames", 1)),
                recurrent_hidden_dim=int(sensor_architecture["recurrent_hidden_dim"]),
                use_moe=bool(sensor_architecture.get("use_moe", True)),
                num_experts=int(sensor_architecture["num_experts"]),
                gate_hidden_dims=tuple(sensor_architecture["gate_hidden_dims"]),
                expert_hidden_dims=tuple(sensor_architecture["expert_hidden_dims"]),
                residual_scale=float(sensor_architecture["residual_scale"]),
                privileged_context_dim=int(sensor_architecture.get("privileged_context_dim", 0)),
                **estimator_kwargs(sensor_architecture),
            ).to(args.device)
            recurrent_state = policy.initial_state(env.num_envs)
        else:
            policy = ActorCritic(
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                actor_hidden_dims=tuple(ppo_cfg.get("actor_hidden_dims", (256, 128))),
                critic_hidden_dims=tuple(ppo_cfg.get("critic_hidden_dims", (256, 128))),
                init_std=float(ppo_cfg.get("init_std", 0.8)),
                min_action_std=float(ppo_cfg.get("min_action_std", 0.05)),
                max_action_std=float(ppo_cfg.get("max_action_std", 0.80)),
            ).to(args.device)
            recurrent_state = None
        missing, unexpected = policy.load_state_dict(checkpoint["actor_critic"], strict=False)
        if unexpected:
            print(f"[WARN] ignored unexpected actor_critic checkpoint keys: {unexpected}")
        non_median_missing = [key for key in missing if not key.startswith("median_critic.")]
        if non_median_missing:
            print(f"[WARN] actor_critic checkpoint missing keys: {non_median_missing}")
        policy.eval()
    print(
        f"[PLAY] seed={int(args.seed)} domain_randomization={env.domain_randomization} "
        f"episode_random_seed={env.cfg.randomize_episode_seed} camera_mode={args.camera_mode} "
        f"robots={env.num_envs} serial_course={env.cfg.serial_terrain_course}"
    )
    if env.cfg.serial_terrain_course:
        for index, order in enumerate(env.more_terrain_cfg.normalized_course_orders):
            print(f"[COURSE {index + 1}] {' -> '.join(order)}")

    obs = env.reset()
    episode_start = np.ones(env.num_envs, dtype=np.float32)
    viewer = None
    recorder = None
    display_model = env.model
    display_data = env.datas[0]
    if env.num_envs > 1 and args.viewer:
        display_model = load_multi_robot_model(
            env_cfg.model_xml,
            env.num_envs,
            env.stair_cfg,
            env.more_terrain_cfg,
        )
        display_data = mujoco.MjData(display_model)
        sync_multi_robot_state(display_model, display_data, env)
    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(display_model, display_data)
        record_path = checkpoint_recording_path(args.checkpoint, checkpoint, args.record_dir, stage)
        record_fps = args.record_fps if args.record_fps > 0.0 else 1.0 / max(env.dt, 1.0e-6)
        recorder = MujocoVideoRecorder(
            display_model,
            display_data,
            record_path,
            width=args.record_width,
            height=args.record_height,
            fps=record_fps,
            crf=args.record_crf,
            camera_mode=args.camera_mode,
        )
        print(f"[RECORD] saving {record_path}")
        configure_tracking_camera(display_model, viewer.cam, display_data, mode=args.camera_mode)
        viewer.sync()
        recorder.write_frame(getattr(viewer, "cam", None))

    rewards: list[float] = []
    forward_velocities: list[float] = []
    first_stair_episode: dict[str, float | str] | None = None
    try:
        step_count = 0
        while args.steps <= 0 or step_count < args.steps:
            if is_onnx:
                action = policy.act_inference(obs, episode_start=episode_start)
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
                if recurrent_sensor_policy:
                    start_tensor = torch.as_tensor(
                        episode_start,
                        dtype=torch.float32,
                        device=args.device,
                    )
                    action_tensor, recurrent_state = policy.act_inference_recurrent(
                        obs_tensor,
                        recurrent_state,
                        start_tensor,
                    )
                    action = action_tensor.cpu().numpy()
                else:
                    action = policy.act_inference(obs_tensor).cpu().numpy()
            obs, reward, done, _info = env.step(action)
            if env.cfg.stair_fixed_start and first_stair_episode is None and bool(done[0]):
                def stair_value(name: str, fallback: float = 0.0) -> float:
                    return float(np.asarray(_info.get(name, [fallback]))[0])

                route_success = stair_value("successes")
                if route_success:
                    reason = "route_complete"
                elif stair_value("stair_distance_failure"):
                    reason = "distance_failure"
                elif stair_value("stair_lateral_failure"):
                    reason = "lateral_failure"
                elif stair_value("deepmimic_route_failure"):
                    reason = "route_tracking_failure"
                elif stair_value("termination"):
                    reason = "fall_or_invalid_state"
                elif stair_value("incomplete_motion_end"):
                    reason = "motion_end_shortfall"
                elif stair_value("time_outs"):
                    reason = "timeout"
                else:
                    reason = "other"
                first_stair_episode = {
                    "steps": float(step_count + 1),
                    "success": route_success,
                    "reason": reason,
                    "distance_error": (
                        stair_value("stair_distance_error")
                        if reason in {"route_complete", "distance_failure"}
                        else float("nan")
                    ),
                    "xy_error": stair_value("deepmimic_root_xy_error", float("nan")),
                }
            episode_start = np.asarray(done, dtype=np.float32)
            rewards.extend(float(value) for value in reward)
            if "forward_velocity" in _info:
                forward_velocities.extend(
                    float(value) for value in _info["forward_velocity"]
                )
            step_count += 1
            if viewer is not None:
                if not viewer.is_running():
                    break
                if env.num_envs > 1:
                    sync_multi_robot_state(display_model, display_data, env)
                update_follow_camera(display_model, viewer.cam, display_data, mode=args.camera_mode)
                viewer.sync()
                if recorder is not None:
                    recorder.write_frame(getattr(viewer, "cam", None))
                time.sleep(env.dt)
    finally:
        if recorder is not None:
            recorder.close()
            print(f"[RECORD] saved {recorder.output_path} frames={recorder.frame_count}")
        if viewer is not None:
            viewer.close()
        env.close()

    mean_reward = sum(rewards) / max(len(rewards), 1)
    mean_forward_velocity = sum(forward_velocities) / max(
        len(forward_velocities), 1
    )
    print(
        f"[PLAY] policy_steps={step_count} robot_samples={len(rewards)} "
        f"mean_reward={mean_reward:.4f} "
        f"mean_forward_velocity={mean_forward_velocity:.4f}m/s"
    )
    if env.cfg.stair_fixed_start:
        if first_stair_episode is None:
            print(f"[STAIRS] first_episode=incomplete after {step_count} steps")
        else:
            distance_error = first_stair_episode["distance_error"]
            xy_error = first_stair_episode["xy_error"]
            print(
                f"[STAIRS] first_episode_steps={int(first_stair_episode['steps'])} "
                f"route_success={bool(first_stair_episode['success'])} "
                f"reason={first_stair_episode['reason']} "
                f"terminal_distance_error={f'{distance_error:.3f}m' if np.isfinite(distance_error) else 'n/a'} "
                f"terminal_xy_error={f'{xy_error:.3f}m' if np.isfinite(xy_error) else 'n/a'}"
            )


if __name__ == "__main__":
    main()
