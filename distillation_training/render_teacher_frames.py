#!/usr/bin/env python3
"""Render numbered PNG frames from a privileged distillation Teacher rollout."""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import mujoco
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distillation_training.networks import SharedActor
from distillation_training.runtime import env_step, make_env, tensor, validate_checkpoint


def write_rgb_png(path: Path, pixels) -> None:
    """Write an uint8 HxWx3 RGB array without an optional image dependency."""
    height, width, channels = pixels.shape
    if channels != 3 or pixels.dtype.name != "uint8":
        raise ValueError("renderer output must be an uint8 HxWx3 RGB array")
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=3))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    checkpoint = validate_checkpoint(args.checkpoint, "teacher")
    args.sensor_variant = checkpoint["sensor_variant"]
    args.num_envs = 1
    args.num_steps = 1
    args.device = "cpu"
    args.sim_backend = "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env, _ = make_env(args)
    actor = SharedActor(checkpoint["sensor_dim"])
    actor.load_state_dict(checkpoint["shared_actor"])
    actor.eval()
    observation = tensor(env.reset(), "cpu")
    base = env.base_env
    renderer = mujoco.Renderer(base.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.azimuth = 135.0
    camera.elevation = -18.0
    rewards: list[float] = []
    terminations = 0
    try:
        for frame in range(args.frames):
            with torch.no_grad():
                privileged = tensor(env.privileged_vector(), "cpu")
                action = actor.mean(observation, privileged)
            observation, reward, done, _ = env_step(env, action)
            rewards.append(float(reward.mean()))
            terminations += int(done.item() > 0)
            camera.lookat[:] = base.datas[0].qpos[:3]
            renderer.update_scene(base.datas[0], camera=camera)
            write_rgb_png(output_dir / f"frame_{frame:04d}.png", renderer.render())
    finally:
        renderer.close()
        env.close()
    mean_reward = sum(rewards) / len(rewards)
    print(
        f"[TEACHER FRAMES] frames={args.frames} mean_reward={mean_reward:.5f} "
        f"terminations={terminations} output={output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
