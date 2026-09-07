#!/usr/bin/env python3
"""Read-only FK/contact audit against the independent launcher's real terrain.

This checks kinematic references, not policy performance or dynamic feasibility.
JSON goes to stdout so existing motion assets/reports are never overwritten.
"""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mujoco
import numpy as np

from lytrobot_amp.env import LytRobotAmpEnv, LytRobotEnvCfg
from lytrobot_amp.terrain import standalone_slope_datum, terrain_spawn_offset
from multimotion.retargeting.analyze_gait import analyze_reference
from scripts.independent_terrain_task import INDEPENDENT_TERRAIN_SPECS, default_train_args
from scripts.train import parse_args


def launcher_config(key: str) -> LytRobotEnvCfg:
    with patch.object(sys, "argv", ["train.py", *default_train_args(key)]):
        args = parse_args()
    values = {f.name: getattr(args, f.name) for f in fields(LytRobotEnvCfg) if hasattr(args, f.name)}
    for name in ("motion_files", "motion_names", "motion_weights", "terrain_types"):
        values[name] = tuple(values[name] or ())
    values["motion_loop"] = tuple(v == "true" for v in args.motion_loop)
    values.update(num_envs=1, domain_randomization=False, reset_noise_scale=0.0, seed=42)
    return LytRobotEnvCfg(**values)


def audit_reference(key: str, travels=None) -> dict:
    cfg = launcher_config(key)
    if travels is None:
        with np.load(cfg.motion_files[0]) as archive:
            metadata = json.loads(str(archive["metadata"]))
        period = metadata.get("loop_translation_m")
        travels = tuple(i * period[0] for i in range(4)) if period else (0.0, 3.0, 9.0)
    env = LytRobotAmpEnv(cfg)
    try:
        motion = env.motion.motions[0]
        result = {"task": cfg.training_task, "gait": analyze_reference(Path(cfg.motion_files[0])),
                  "source_loop": motion.source_loop, "training_loop": cfg.motion_loop[0],
                  "joint_seam_max_rad": float(np.max(np.abs(motion.reference_qpos[0, 7:] - motion.reference_qpos[-1, 7:]))),
                  "placements": []}
        offset = terrain_spawn_offset(key, env.more_terrain_cfg, env.stair_cfg)
        data = env.datas[0]
        for travel in travels:
            contacts, penetrations = [], []
            for pose in motion.reference_qpos:
                data.qpos[:] = pose
                data.qpos[:3] += offset
                data.qpos[0] += travel
                if key in {"slope_up", "slope_down"}:
                    data.qpos[2] += standalone_slope_datum(env.more_terrain_cfg)[0] * travel
                mujoco.mj_forward(env.model, data)
                contacts.append(env._foot_contacts(data))
                penetrations.append(env._terrain_penetration_depth(data))
            contacts = np.asarray(contacts)
            result["placements"].append({
                "translation_m": travel,
                "contact_mismatch_fraction": float(np.mean(np.abs(contacts - motion.reference_foot_contact))),
                "airborne_fraction": float(np.mean(np.max(contacts, axis=1) < 0.5)),
                "max_penetration_m": max(penetrations),
            })
        return result
    finally:
        env.close()


if __name__ == "__main__":
    print(json.dumps({k: audit_reference(k) for k in INDEPENDENT_TERRAIN_SPECS}, indent=2))
