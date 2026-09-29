#!/usr/bin/env python3
"""Convert CMU ASF/AMC squat frames to OmniRetarget world keypoints.

The compact ASF/AMC FK parser is adapted from CalciferZh/AMCParser under the
MIT license; see ``third_party/AMCParser_LICENSE``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from transforms3d.euler import euler2mat

CMU_LENGTH_TO_METERS = (1.0 / 0.45) * 2.54 / 100.0
JOINT_NAMES = (
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "Spine",
    "Spine1",
)
ASF_TO_OUTPUT = {
    "Hips": "root",
    # ASF joint coordinates are the distal end of each bone.  Standard motion
    # keypoints name the proximal joint, so the leg chain intentionally shifts
    # by one bone (hipjoint->hip, femur->knee, tibia->ankle, foot->toe base).
    "RightUpLeg": "rhipjoint",
    "RightLeg": "rfemur",
    "RightFoot": "rtibia",
    "RightToeBase": "rfoot",
    "LeftUpLeg": "lhipjoint",
    "LeftLeg": "lfemur",
    "LeftFoot": "ltibia",
    "LeftToeBase": "lfoot",
    "Spine": "lowerback",
    "Spine1": "upperback",
}


@dataclass
class Joint:
    name: str
    direction: np.ndarray
    length: float
    axis_degrees: np.ndarray
    dof: tuple[str, ...]
    limits: tuple[tuple[float, float], ...]
    parent: Joint | None = None
    children: list[Joint] = field(default_factory=list)
    coordinate: np.ndarray = field(default_factory=lambda: np.zeros(3))
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        self.direction = np.asarray(self.direction, dtype=np.float64)
        self._axis_matrix = euler2mat(*np.deg2rad(self.axis_degrees))
        self._axis_matrix_inv = self._axis_matrix.T

    def set_motion(self, motion: dict[str, list[float]]) -> None:
        if self.name == "root":
            values = motion["root"]
            self.coordinate = np.asarray(values[:3], dtype=np.float64)
            rotation = euler2mat(*np.deg2rad(values[3:6]))
            self.matrix = self._axis_matrix @ rotation @ self._axis_matrix_inv
        else:
            rotation_degrees = np.zeros(3, dtype=np.float64)
            # Zero-DOF hierarchy helper joints (for example ``lhipjoint``)
            # are intentionally absent from AMC frames.
            values = motion.get(self.name, [])
            for value, channel in zip(values, self.dof, strict=True):
                rotation_degrees[{"rx": 0, "ry": 1, "rz": 2}[channel]] = value
            local_rotation = euler2mat(*np.deg2rad(rotation_degrees))
            assert self.parent is not None
            self.matrix = self.parent.matrix @ self._axis_matrix @ local_rotation @ self._axis_matrix_inv
            self.coordinate = self.parent.coordinate + self.length * (self.matrix @ self.direction)
        for child in self.children:
            child.set_motion(motion)


def _read_tokens(lines: list[str], index: int) -> tuple[list[str], int]:
    return lines[index].strip().split(), index + 1


def parse_asf(path: Path) -> dict[str, Joint]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(":bonedata") + 1
    lines = lines[start:]
    joints = {"root": Joint("root", np.zeros(3), 0.0, np.zeros(3), (), ())}
    index = 0
    while True:
        tokens, index = _read_tokens(lines, index)
        if tokens[0] == ":hierarchy":
            break
        if tokens[0] != "begin":
            raise ValueError(f"Unexpected ASF token: {tokens}")
        _, index = _read_tokens(lines, index)  # id
        name_tokens, index = _read_tokens(lines, index)
        direction_tokens, index = _read_tokens(lines, index)
        length_tokens, index = _read_tokens(lines, index)
        axis_tokens, index = _read_tokens(lines, index)
        name = name_tokens[1]
        direction = np.asarray(direction_tokens[1:], dtype=np.float64)
        length = float(length_tokens[1])
        axis = np.asarray(axis_tokens[1:4], dtype=np.float64)
        dof: tuple[str, ...] = ()
        limits: list[tuple[float, float]] = []
        tokens, index = _read_tokens(lines, index)
        if tokens[0] == "dof":
            dof = tuple(tokens[1:])
            for channel_index in range(len(dof)):
                limit_tokens, index = _read_tokens(lines, index)
                if channel_index == 0:
                    limit_tokens = limit_tokens[1:]
                limits.append((float(limit_tokens[0].lstrip("(")), float(limit_tokens[1].rstrip(")"))))
            tokens, index = _read_tokens(lines, index)
        if tokens[0] != "end":
            raise ValueError(f"Malformed ASF joint {name}: {tokens}")
        joints[name] = Joint(name, direction, length, axis, dof, tuple(limits))

    tokens, index = _read_tokens(lines, index)
    if tokens[0] != "begin":
        raise ValueError("Malformed ASF hierarchy.")
    while True:
        tokens, index = _read_tokens(lines, index)
        if tokens[0] == "end":
            break
        parent = joints[tokens[0]]
        for child_name in tokens[1:]:
            child = joints[child_name]
            parent.children.append(child)
            child.parent = parent
    return joints


def parse_amc(path: Path) -> list[dict[str, list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(":DEGREES") + 1
    frames: list[dict[str, list[float]]] = []
    current: dict[str, list[float]] | None = None
    for raw_line in lines[start:]:
        tokens = raw_line.strip().split()
        if not tokens:
            continue
        if len(tokens) == 1 and tokens[0].isdigit():
            if current is not None:
                frames.append(current)
            current = {}
            continue
        if current is None:
            raise ValueError("AMC joint data appeared before the first frame.")
        current[tokens[0]] = [float(value) for value in tokens[1:]]
    if current is not None:
        frames.append(current)
    return frames


def convert(asf_path: Path, amc_path: Path, start: int, end: int, stride: int) -> tuple[np.ndarray, float]:
    joints = parse_asf(asf_path)
    frames = parse_amc(amc_path)
    if not (0 <= start < end < len(frames)):
        raise ValueError(f"Requested [{start}, {end}] from {len(frames)} AMC frames.")
    selected = frames[start : end + 1 : stride]
    positions = np.zeros((len(selected), len(JOINT_NAMES), 3), dtype=np.float64)
    for frame_index, motion in enumerate(selected):
        joints["root"].set_motion(motion)
        for joint_index, output_name in enumerate(JOINT_NAMES):
            positions[frame_index, joint_index] = joints[ASF_TO_OUTPUT[output_name]].coordinate

    positions *= CMU_LENGTH_TO_METERS
    # CMU is right-handed Y-up with X forward and Z pointing to the subject's
    # right.  OmniRetarget and MuJoCo are right-handed Z-up with +Y to the
    # subject's left, hence [x, y, z] -> [x, -z, y].  Omitting the minus sign
    # mirrors the skeleton and makes a semantic right leg target the robot's
    # left side, which in turn produces severe leg crossing after retargeting.
    positions = positions[..., [0, 2, 1]]
    positions[..., 1] *= -1.0
    root_xy0 = positions[0, JOINT_NAMES.index("Hips"), :2].copy()
    positions[..., :2] -= root_xy0
    toe_indices = [JOINT_NAMES.index("RightToeBase"), JOINT_NAMES.index("LeftToeBase")]
    ground_z = float(np.min(positions[:, toe_indices, 2]))
    standing_frames = np.r_[0 : min(8, len(positions)), max(0, len(positions) - 8) : len(positions)]
    hip_height = float(np.median(positions[standing_frames, JOINT_NAMES.index("Hips"), 2]) - ground_z)
    return positions.astype(np.float32), hip_height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asf", type=Path, required=True)
    parser.add_argument("--amc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=340)
    parser.add_argument("--end", type=int, default=620)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    positions, hip_height = convert(args.asf, args.amc, args.start, args.end, args.stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        global_joint_positions=positions,
        height=np.float32(hip_height),
        fps=np.float32(120.0 / args.stride),
        joint_names=np.asarray(JOINT_NAMES),
        source=np.asarray("CMU MoCap subject 22 trial 14"),
        source_frames=np.asarray([args.start, args.end, args.stride], dtype=np.int32),
    )
    print(f"saved {args.output}: frames={len(positions)}, lower_body_height={hip_height:.4f}m")


if __name__ == "__main__":
    main()
