"""Control-mode helpers for tendon actuator grouping."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

CONTROL_MODE_POSITION = "position"
CONTROL_MODE_HYBRID = "hybrid"
CONTROL_MODES = (CONTROL_MODE_POSITION, CONTROL_MODE_HYBRID)
FORCE_ACTUATOR_SUFFIX = "_force"


def normalize_control_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in CONTROL_MODES:
        raise ValueError(f"Unsupported control_mode={mode!r}; expected one of {CONTROL_MODES}.")
    return value


def actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(actuator_id)) or ""


def position_actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = [
        actuator_id
        for actuator_id in range(model.nu)
        if not actuator_name(model, actuator_id).endswith(FORCE_ACTUATOR_SUFFIX)
    ]
    if not ids:
        raise RuntimeError("No position actuators found in MuJoCo model.")
    return np.asarray(ids, dtype=np.int32)


def force_actuator_ids_for_positions(
    model: mujoco.MjModel,
    position_ids: Sequence[int] | np.ndarray,
) -> np.ndarray:
    by_name = {actuator_name(model, actuator_id): actuator_id for actuator_id in range(model.nu)}
    ids: list[int] = []
    missing: list[str] = []
    for position_id in position_ids:
        base_name = actuator_name(model, int(position_id))
        force_name = f"{base_name}{FORCE_ACTUATOR_SUFFIX}"
        force_id = by_name.get(force_name)
        if force_id is None:
            missing.append(force_name)
        else:
            ids.append(int(force_id))
    if missing:
        raise RuntimeError(
            "Hybrid control requires one force actuator per position actuator; "
            f"missing {missing}."
        )
    return np.asarray(ids, dtype=np.int32)
