"""Shared defaults for one-terrain, sensor-only AMP training tasks."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "robot" / "reference_data" / "multimotion"
MODEL_XML = ROOT / "robot" / "MJCFmodelposition.xml"
TERRAIN_ASSET_DIR = ROOT / "robot" / "terrains" / "more_discolrl"


@dataclass(frozen=True)
class IndependentTerrainSpec:
    """Immutable launcher contract for one independent terrain experiment."""

    key: str
    task_name: str
    motion_name: str
    motion_filename: str
    terrain_type: str
    motion_loop: bool
    episode_steps: int


INDEPENDENT_TERRAIN_SPECS = {
    "flat": IndependentTerrainSpec(
        key="flat",
        task_name="independent_flat",
        motion_name="walk",
        motion_filename="walk_omniretarget.npz",
        terrain_type="flat",
        motion_loop=True,
        episode_steps=2000,
    ),
    "rough": IndependentTerrainSpec(
        key="rough",
        task_name="independent_rough",
        motion_name="rough_walk",
        motion_filename="rough_walk_omniretarget.npz",
        terrain_type="rough",
        motion_loop=True,
        episode_steps=2000,
    ),
    "slope_up": IndependentTerrainSpec(
        key="slope_up",
        task_name="independent_slope_up",
        motion_name="slope_up",
        motion_filename="slope_up_omniretarget.npz",
        terrain_type="slope_up",
        motion_loop=True,
        episode_steps=2000,
    ),
    "slope_down": IndependentTerrainSpec(
        key="slope_down",
        task_name="independent_slope_down",
        motion_name="slope_down",
        motion_filename="slope_down_omniretarget.npz",
        terrain_type="slope_down",
        motion_loop=True,
        episode_steps=2000,
    ),
}


def get_spec(key: str) -> IndependentTerrainSpec:
    try:
        return INDEPENDENT_TERRAIN_SPECS[str(key).lower()]
    except KeyError as exc:
        choices = ", ".join(INDEPENDENT_TERRAIN_SPECS)
        raise ValueError(f"Unknown independent terrain {key!r}; expected {choices}.") from exc


def default_train_args(key: str) -> list[str]:
    """Return a complete, non-distilled sensor-only AMP configuration.

    Each task has exactly one reference clip and one terrain type.  There is
    no serial course, skill label, reference-control blend, or Teacher action
    target, so its checkpoint cannot accidentally become a four-terrain model.
    """

    spec = get_spec(key)
    return [
        "--training-task",
        spec.task_name,
        "--model-xml",
        str(MODEL_XML),
        "--terrain-asset-dir",
        str(TERRAIN_ASSET_DIR),
        "--motion-files",
        str(REFERENCE_DIR / spec.motion_filename),
        "--motion-names",
        spec.motion_name,
        "--motion-weights",
        "1.0",
        "--motion-loop",
        "true" if spec.motion_loop else "false",
        "--no-skill-conditioning",
        "--reference-ctrl-blend",
        "0.0",
        "--terrain-types",
        spec.terrain_type,
        "--no-serial-terrain-course",
        "--no-serial-course-single-motion",
        "--standalone-terrain-field",
        "--amp",
        "--amp-task-only",
        # Keep the physical objective visible while AMP supplies the gait
        # style.  This prevents an in-place leg swing from being optimal.
        "--target-velocity",
        "0.862295",
        "--forward-velocity-tracking-sigma",
        "0.20",
        "--forward-velocity-reward-coef",
        "2.0",
        "--forward-progress-reward-coef",
        "1.0",
        "--forward-velocity-deficit-penalty-coef",
        "1.5",
        "--forward-velocity-excess-penalty-coef",
        "3.0",
        "--min-forward-progress-speed",
        "0.10",
        "--forward-progress-grace-steps",
        "100",
        "--forward-progress-window-steps",
        "100",
        "--forward-progress-termination-penalty",
        "2.0",
        "--fall-penalty",
        "10.0",
        "--terrain-penetration-tolerance",
        "0.01",
        "--terrain-penetration-penalty-coef",
        "20.0",
        "--foot-slip-velocity-threshold",
        "0.04",
        "--foot-slip-penalty-coef",
        "2.0",
        "--contact-reward-coef",
        "0.70",
        "--missed-contact-penalty-coef",
        "0.80",
        "--task-reward-lerp",
        "0.50",
        "--amp-reward-coef",
        "3.0",
        "--min-action-std",
        "0.05",
        "--motion-reset-prob",
        "1.0",
        # The downhill reference begins at the high end of its ramp; avoid a
        # flat-ground stand reset that would start below the reference.
        "--stand-start-prob",
        "0.0",
        "--startup-action-ramp-steps",
        "0",
        "--reset-noise-scale",
        "0.01",
        "--num-envs",
        "4096",
        "--num-steps",
        "24",
        "--epochs",
        "4",
        "--minibatches",
        "8",
        "--amp-minibatch-size",
        "4096",
        "--amp-disc-updates",
        "2",
        "--episode-metric-window",
        "4096",
        "--warp-nconmax",
        "128",
        "--warp-njmax",
        "512",
        "--episode-steps",
        str(spec.episode_steps),
        "--episode-timeout-jitter-steps",
        "0",
        "--save-interval",
        "500",
        "--log-root",
        str(ROOT / "logs" / "independent_terrain" / spec.key),
        "--log-group",
        spec.task_name,
        "--wandb-tags",
        (
            "lytrobot,bpo,amp,independent-terrain,sensor-only,imu,tendon,"
            f"{spec.key},direct-no-distillation,asymmetric-critic,recurrent,gru"
        ),
    ]


def launch(key: str) -> None:
    """Prepend the selected task defaults and enter the sensor trainer."""

    spec = get_spec(key)
    sys.argv = [
        str(ROOT / "scripts" / f"train_{spec.key}.py"),
        *default_train_args(spec.key),
        *sys.argv[1:],
    ]
    from sensor_training.train import main as train_main

    train_main()
