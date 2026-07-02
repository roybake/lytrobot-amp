#!/usr/bin/env python3
"""Run a trained policy checkpoint in the LYT MuJoCo environment."""

from __future__ import annotations

import argparse
from dataclasses import fields
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import mujoco
import numpy as np

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.networks import ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play/evaluate a LYT PPO-AMP checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-xml", default=str(REPO_ROOT / "robot" / "MJCFmodelposition.xml"))
    parser.add_argument("--motion-file", default=str(REPO_ROOT / "robot" / "reference_data" / "walk1.npz"))
    parser.add_argument("--steps", type=int, default=1000, help="Number of policy steps to run. Use 0 for no limit.")
    parser.add_argument("--target-velocity", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Playback RNG seed. Omit it to randomize each run; set it to reproduce one play test.",
    )
    parser.add_argument("--episode-random-seed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--stage", default="auto", choices=("auto", "stage1", "stage2"))
    parser.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--control-mode", default="auto", choices=("auto", "position", "hybrid"))
    parser.add_argument("--hybrid-force-scale", type=float, default=None)
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
    parser.add_argument("--record-dir", default=str(REPO_ROOT / "recording"))
    parser.add_argument("--record-width", type=int, default=1920)
    parser.add_argument("--record-height", type=int, default=1080)
    parser.add_argument("--record-crf", type=int, default=12)
    parser.add_argument("--record-fps", type=float, default=30.0)
    return parser.parse_args()


def make_env_cfg(args: argparse.Namespace, checkpoint: dict) -> LytRobotEnvCfg:
    cfg = LytRobotEnvCfg(
        model_xml=args.model_xml,
        motion_file=args.motion_file,
        num_envs=1,
        target_velocity=args.target_velocity,
        reset_noise_scale=0.0,
        stand_start_prob=0.0,
    )
    checkpoint_env_cfg = checkpoint.get("env_config", {})
    if isinstance(checkpoint_env_cfg, dict):
        valid_fields = {field.name for field in fields(LytRobotEnvCfg)}
        for key, value in checkpoint_env_cfg.items():
            if key in valid_fields and key not in {"model_xml", "motion_file", "num_envs", "seed", "target_velocity"}:
                setattr(cfg, key, value)
    cfg.model_xml = args.model_xml
    cfg.motion_file = args.motion_file
    cfg.num_envs = 1
    cfg.seed = int(args.seed)
    cfg.target_velocity = args.target_velocity
    cfg.reset_noise_scale = 0.0
    if args.control_mode != "auto":
        cfg.control_mode = args.control_mode
    if args.hybrid_force_scale is not None:
        cfg.hybrid_force_scale = float(args.hybrid_force_scale)
    if args.stand_start_prob is not None:
        cfg.stand_start_prob = float(args.stand_start_prob)
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
    elif getattr(args, "seed_was_provided", False):
        cfg.randomize_episode_seed = False
    return cfg


def infer_stage(checkpoint_path: str | Path, checkpoint: dict, stage_arg: str) -> str:
    if stage_arg in {"stage1", "stage2"}:
        return stage_arg
    parts = set(Path(checkpoint_path).expanduser().parts)
    if "stage2" in parts:
        return "stage2"
    if "stage1" in parts:
        return "stage1"
    env_cfg = checkpoint.get("env_config", {})
    if isinstance(env_cfg, dict) and bool(env_cfg.get("domain_randomization", False)):
        return "stage2"
    return "stage1"


def checkpoint_recording_path(
    checkpoint_path: str | Path,
    checkpoint: dict,
    record_dir: str | Path,
    stage: str,
) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser()
    run_name = checkpoint_path.parent.name
    match = re.fullmatch(r"(?:model|modle)_(\d+)", checkpoint_path.stem)
    if match is not None:
        iteration = match.group(1)
    elif checkpoint_path.stem == "model_final" and "iteration" in checkpoint:
        iteration = str(int(checkpoint["iteration"]))
    else:
        iteration = checkpoint_path.stem

    output_dir = Path(record_dir).expanduser().resolve() / stage
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
    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if mode == "tracking" and pelvis_body_id >= 0:
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = pelvis_body_id
    else:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.8
    camera.azimuth = 90.0
    camera.elevation = -12.0
    if data is not None and mode != "free":
        update_follow_camera(model, camera, data, mode=mode)


def pelvis_camera_target(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_body_id >= 0:
        target = data.xpos[pelvis_body_id].copy()
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
    args.seed_was_provided = args.seed is not None
    if args.seed is None:
        args.seed = secrets.randbits(32)
    torch.manual_seed(int(args.seed))
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    ppo_cfg = checkpoint.get("ppo_amp_config", {})

    env = LytRobotAmpEnv(make_env_cfg(args, checkpoint))
    policy = ActorCritic(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        actor_hidden_dims=tuple(ppo_cfg.get("actor_hidden_dims", (256, 128))),
        critic_hidden_dims=tuple(ppo_cfg.get("critic_hidden_dims", (256, 128))),
        init_std=float(ppo_cfg.get("init_std", 0.8)),
        min_action_std=float(ppo_cfg.get("min_action_std", 0.05)),
        max_action_std=float(ppo_cfg.get("max_action_std", 0.80)),
    ).to(args.device)
    policy.load_state_dict(checkpoint["actor_critic"])
    policy.eval()
    print(
        f"[PLAY] seed={int(args.seed)} domain_randomization={env.domain_randomization} "
        f"episode_random_seed={env.cfg.randomize_episode_seed} camera_mode={args.camera_mode}"
    )

    obs = env.reset()
    viewer = None
    recorder = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.datas[0])
        stage = infer_stage(args.checkpoint, checkpoint, args.stage)
        record_path = checkpoint_recording_path(args.checkpoint, checkpoint, args.record_dir, stage)
        record_fps = args.record_fps if args.record_fps > 0.0 else 1.0 / max(env.dt, 1.0e-6)
        recorder = MujocoVideoRecorder(
            env.model,
            env.datas[0],
            record_path,
            width=args.record_width,
            height=args.record_height,
            fps=record_fps,
            crf=args.record_crf,
            camera_mode=args.camera_mode,
        )
        print(f"[RECORD] saving {record_path}")
        configure_tracking_camera(env.model, viewer.cam, env.datas[0], mode=args.camera_mode)
        viewer.sync()
        recorder.write_frame(getattr(viewer, "cam", None))

    rewards: list[float] = []
    try:
        step_count = 0
        while args.steps <= 0 or step_count < args.steps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
            action = policy.act_inference(obs_tensor).cpu().numpy()
            obs, reward, done, _info = env.step(action)
            rewards.append(float(reward[0]))
            step_count += 1
            if viewer is not None:
                if not viewer.is_running():
                    break
                update_follow_camera(env.model, viewer.cam, env.datas[0], mode=args.camera_mode)
                viewer.sync()
                if recorder is not None:
                    recorder.write_frame(getattr(viewer, "cam", None))
                time.sleep(env.dt)
            if bool(done[0]):
                obs = env.reset()
    finally:
        if recorder is not None:
            recorder.close()
            print(f"[RECORD] saved {recorder.output_path} frames={recorder.frame_count}")
        if viewer is not None:
            viewer.close()
        env.close()

    mean_reward = sum(rewards) / max(len(rewards), 1)
    print(f"[PLAY] steps={len(rewards)} mean_reward={mean_reward:.4f}")


if __name__ == "__main__":
    main()
