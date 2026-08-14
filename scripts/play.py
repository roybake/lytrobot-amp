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

        self.session = ort.InferenceSession(
            str(Path(path).expanduser()),
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
    parser.add_argument("--model-xml", default=str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"))
    parser.add_argument("--motion-file", default=str(REPO_ROOT / "robot" / "reference_data" / "walk1_mjcf.npz"))
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of policy steps to run. Viewer playback defaults to no limit; "
            "headless playback defaults to 1000. Use 0 explicitly for no limit."
        ),
    )
    parser.add_argument(
        "--target-velocity",
        type=float,
        default=None,
        help="Reward diagnostic target; defaults to the checkpoint training target.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Playback RNG seed. Omit it to randomize each run; set it to reproduce one play test.",
    )
    parser.add_argument(
        "--episode-random-seed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Resample RNG state after every test reset. This is enabled by "
            "default unless a fixed --seed is supplied."
        ),
    )
    parser.add_argument("--viewer", action="store_true")
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
        "--stage",
        default="auto",
        choices=("auto", "stage1", "stage2", "multimotion"),
        help="Checkpoint group used for recording output; auto infers it from the path and metadata.",
    )
    parser.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--push-force-min", type=float, default=None)
    parser.add_argument("--push-force-max", type=float, default=None)
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=2_147_483_647,
        help="Per-episode timeout used during playback; defaults to effectively unlimited.",
    )
    parser.add_argument(
        "--episode-timeout-jitter-steps",
        type=int,
        default=0,
        help="Playback timeout jitter; disabled by default for repeatable action tests.",
    )
    parser.add_argument(
        "--motion-reset-prob",
        type=float,
        default=0.0,
        help="Probability of resetting at a random reference phase; playback starts at phase zero by default.",
    )
    parser.add_argument(
        "--squat-terminate-on-cycle-completion",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--stand-start-prob", type=float, default=None)
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
    parser.add_argument(
        "--record-dir",
        default=None,
        help=(
            "Recording root override. By default videos are stored beside the "
            "checkpoint under logs/<task>/<stage>/recordings/."
        ),
    )
    parser.add_argument("--record-width", type=int, default=1920)
    parser.add_argument("--record-height", type=int, default=1080)
    parser.add_argument("--record-crf", type=int, default=12)
    parser.add_argument("--record-fps", type=float, default=30.0)
    return parser.parse_args()


def resolve_step_limit(steps: int | None, *, viewer: bool) -> int:
    """Keep interactive tests alive until closed, while bounding headless runs."""

    if steps is None:
        return 0 if viewer else 1000
    return int(steps)


def make_env_cfg(
    args: argparse.Namespace,
    checkpoint: dict,
    stage: str | None = None,
) -> LytRobotEnvCfg:
    checkpoint_env_cfg = checkpoint.get("env_config", {})
    saved_target_velocity = (
        float(checkpoint_env_cfg.get("target_velocity", 1.0))
        if isinstance(checkpoint_env_cfg, dict)
        else 1.0
    )
    target_velocity = (
        saved_target_velocity
        if args.target_velocity is None
        else float(args.target_velocity)
    )
    cfg = LytRobotEnvCfg(
        model_xml=args.model_xml,
        motion_file=args.motion_file,
        num_envs=1,
        target_velocity=target_velocity,
        reset_noise_scale=0.0,
        stand_start_prob=0.0,
    )
    if isinstance(checkpoint_env_cfg, dict):
        valid_fields = {field.name for field in fields(LytRobotEnvCfg)}
        for key, value in checkpoint_env_cfg.items():
            if key in valid_fields and key not in {"model_xml", "motion_file", "num_envs", "seed", "target_velocity"}:
                setattr(cfg, key, value)
    cfg.model_xml = args.model_xml
    cfg.motion_file = args.motion_file
    resolved_stage = stage or infer_stage(args.checkpoint, checkpoint, args.stage)
    robot_count = args.num_robots
    if robot_count is None:
        robot_count = 3 if resolved_stage == "multimotion" else 1
    if robot_count < 1:
        raise ValueError("--num-robots must be positive.")
    cfg.num_envs = int(robot_count)
    cfg.seed = int(args.seed)
    cfg.target_velocity = target_velocity
    cfg.reset_noise_scale = 0.0
    # Playback is an endurance/repetition test, not a training-state sampler.
    # Start every reset at the action's phase-zero state and leave the episode
    # alive until a real termination or a task-defined successful completion.
    cfg.motion_reset_prob = float(getattr(args, "motion_reset_prob", 0.0))
    cfg.episode_steps = int(getattr(args, "episode_steps", 2_147_483_647))
    cfg.episode_timeout_jitter_steps = int(
        getattr(args, "episode_timeout_jitter_steps", 0)
    )
    use_serial_course = (
        bool(cfg.serial_terrain_course)
        if args.serial_terrain_course is None
        else bool(args.serial_terrain_course)
    )
    # Honor an explicit --no-serial-terrain-course even when the checkpoint
    # was trained with a serial course.  Previously the copied checkpoint
    # value stayed True because only the enabling branch assigned this field.
    cfg.serial_terrain_course = use_serial_course
    if use_serial_course:
        cfg.terrain_course_variants = max(int(robot_count), 1)
        cfg.terrain_course_random_assignment = False
        # Playback is a strict full-route evaluation, never a curriculum reset.
        cfg.terrain_course_random_start_prob = 0.0
        cfg.stand_start_prob = 0.0
    elif getattr(args, "stand_start_prob", None) is None:
        cfg.stand_start_prob = 1.0
    if getattr(args, "stand_start_prob", None) is not None:
        cfg.stand_start_prob = float(args.stand_start_prob)
    if getattr(args, "stand_start_phase_delay_steps", None) is not None:
        cfg.stand_start_phase_delay_steps = int(args.stand_start_phase_delay_steps)
    if getattr(args, "stand_start_transition_min_steps", None) is not None:
        cfg.stand_start_transition_min_steps = int(args.stand_start_transition_min_steps)
    if getattr(args, "stand_start_transition_max_steps", None) is not None:
        cfg.stand_start_transition_max_steps = int(args.stand_start_transition_max_steps)
    if getattr(args, "stand_start_push_delay_steps", None) is not None:
        cfg.stand_start_push_delay_steps = int(args.stand_start_push_delay_steps)
    if getattr(args, "domain_randomization", None) is not None:
        cfg.domain_randomization = bool(args.domain_randomization)
    if getattr(args, "push_force_min", None) is not None:
        cfg.push_force_min = float(args.push_force_min)
    if getattr(args, "push_force_max", None) is not None:
        cfg.push_force_max = float(args.push_force_max)
    if getattr(args, "squat_terminate_on_cycle_completion", None) is not None:
        cfg.squat_terminate_on_cycle_completion = bool(
            args.squat_terminate_on_cycle_completion
        )
    if getattr(args, "episode_random_seed", None) is not None:
        cfg.randomize_episode_seed = bool(args.episode_random_seed)
    elif getattr(args, "seed_was_provided", False):
        cfg.randomize_episode_seed = False
    else:
        # Playback randomness is a runtime contract, not checkpoint state.
        # This also keeps old deterministic checkpoints random by default.
        cfg.randomize_episode_seed = True
    return cfg


def infer_stage(checkpoint_path: str | Path, checkpoint: dict, stage_arg: str) -> str:
    if stage_arg in {"stage1", "stage2", "multimotion"}:
        return stage_arg
    parts = set(Path(checkpoint_path).expanduser().parts)
    training_task = str(checkpoint.get("training_task", "")).strip()
    env_cfg = checkpoint.get("env_config", {})
    if not training_task and isinstance(env_cfg, dict):
        training_task = str(env_cfg.get("training_task", "")).strip()
    if training_task == "multimotion":
        return "multimotion"
    if training_task in {"multimotion_no_stairs", "stairs_up_down"}:
        if "stage2" in parts or (
            isinstance(env_cfg, dict) and bool(env_cfg.get("domain_randomization", False))
        ):
            return "stage2"
        return "stage1"
    if "multimotion" in parts:
        return "multimotion"
    if "stage2" in parts:
        return "stage2"
    if "stage1" in parts:
        return "stage1"
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
    record_dir: str | Path | None,
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

    path_parts = set(checkpoint_path.parts)
    recording_stage = (
        "stage2" if "stage2" in path_parts else "stage1" if "stage1" in path_parts else stage
    )
    checkpoint_stage_dir = checkpoint_path.parent.parent
    if record_dir is None and checkpoint_stage_dir.name in {"stage1", "stage2"}:
        output_dir = checkpoint_stage_dir.resolve() / "recordings"
    elif record_dir is None:
        output_dir = checkpoint_stage_dir.resolve() / recording_stage / "recordings"
    else:
        output_dir = (
            Path(record_dir).expanduser().resolve()
            / recording_stage
            / "recordings"
        )
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
            raise RuntimeError("ffmpeg not found; install ffmpeg to save MuJoCo MP4 recordings.")

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
    args.steps = resolve_step_limit(args.steps, viewer=bool(args.viewer))
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
    if env_cfg.serial_terrain_course and not (
        env_cfg.forward_velocity_reward_coef > 0.0
        or env_cfg.forward_progress_reward_coef > 0.0
        or env_cfg.forward_velocity_deficit_penalty_coef > 0.0
    ):
        print(
            "[WARN] This checkpoint was trained without an explicit forward-course "
            "objective and may stand in place. It must be fine-tuned or retrained "
            "with the corrected multimotion defaults."
        )
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

            policy = SensorActorCritic(
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                actor_hidden_dims=tuple(ppo_cfg.get("actor_hidden_dims", (256, 128))),
                critic_hidden_dims=tuple(ppo_cfg.get("critic_hidden_dims", (256, 128))),
                init_std=float(ppo_cfg.get("init_std", 0.8)),
                min_action_std=float(ppo_cfg.get("min_action_std", 0.05)),
                max_action_std=float(ppo_cfg.get("max_action_std", 0.80)),
                sensor_frame_dim=int(sensor_architecture["sensor_frame_dim"]),
                recurrent_hidden_dim=int(sensor_architecture["recurrent_hidden_dim"]),
                num_experts=int(sensor_architecture["num_experts"]),
                gate_hidden_dims=tuple(sensor_architecture["gate_hidden_dims"]),
                expert_hidden_dims=tuple(sensor_architecture["expert_hidden_dims"]),
                residual_scale=float(sensor_architecture["residual_scale"]),
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
    stance_foot_speeds: list[float] = []
    completed_episodes = 0
    termination_count = 0
    timeout_count = 0
    success_count = 0
    squat_cycle_count = 0
    motion_repetition_count = 0
    try:
        step_count = 0
        while args.steps <= 0 or step_count < args.steps:
            previous_motion_phase = np.asarray(env.motion_phase).copy()
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
            episode_start = np.asarray(done, dtype=np.float32)
            completed = episode_start > 0.5
            completed_episodes += int(np.count_nonzero(completed))
            if "termination" in _info:
                termination_count += int(
                    np.count_nonzero(
                        completed
                        & (np.asarray(_info["termination"], dtype=np.float32) > 0.5)
                    )
                )
            if "time_outs" in _info:
                timeout_count += int(
                    np.count_nonzero(
                        completed
                        & (np.asarray(_info["time_outs"], dtype=np.float32) > 0.5)
                    )
                )
            if "successes" in _info:
                success_count += int(
                    np.count_nonzero(
                        completed
                        & (np.asarray(_info["successes"], dtype=np.float32) > 0.5)
                    )
                )
            if "squat_cycle_completed" in _info:
                squat_cycle_count += int(
                    np.count_nonzero(
                        np.asarray(
                            _info["squat_cycle_completed"], dtype=np.float32
                        )
                        > 0.5
                    )
                )
            current_motion_phase = np.asarray(env.motion_phase)
            motion_repetition_count += int(
                np.count_nonzero(
                    (~completed)
                    & (current_motion_phase + 1.0e-6 < previous_motion_phase)
                )
            )
            rewards.extend(float(value) for value in reward)
            if "forward_velocity" in _info:
                forward_velocities.extend(
                    float(value) for value in _info["forward_velocity"]
                )
            if "stance_foot_speed" in _info:
                stance_foot_speeds.extend(
                    float(value) for value in _info["stance_foot_speed"]
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
    mean_stance_foot_speed = sum(stance_foot_speeds) / max(
        len(stance_foot_speeds), 1
    )
    print(
        f"[PLAY] policy_steps={step_count} robot_samples={len(rewards)} "
        f"mean_reward={mean_reward:.4f} "
        f"mean_forward_velocity={mean_forward_velocity:.4f}m/s "
        f"mean_stance_foot_speed={mean_stance_foot_speed:.4f}m/s "
        f"episodes={completed_episodes} terminations={termination_count} "
        f"timeouts={timeout_count} successes={success_count} "
        f"repetitions={motion_repetition_count + timeout_count} "
        f"squat_cycles={squat_cycle_count}"
    )


if __name__ == "__main__":
    main()
