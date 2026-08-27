"""MoRE-discoRL complex terrains adapted to the LYT MuJoCo model.

The copied upstream files live in ``robot/terrains/more_discolrl``.  This
module turns their gap, pit, rough-heightfield, and stair parameters into one
multi-lane MuJoCo scene so a single batched policy can sample every terrain.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


MORE_TERRAIN_TYPES = (
    "flat",
    "gap",
    "pit",
    "rough",
    "slope_up",
    "slope_down",
    "stairs",
)
# The serial course is a straight, narrow terrain lane.  Ninety-degree
# turning clips cannot be concatenated in that lane without composing their
# world-frame poses (and the resulting S turn would leave the 2 m corridor).
# Keep turns in the motion library for dedicated turn training, but do not put
# discontinuous turn references in the all-terrain route.
COURSE_PREFIX_MOTIONS = ("walk",)
COURSE_SUFFIX_MOTIONS = ("walk",)
COURSE_TERRAIN_LENGTHS = {
    # Match the forward travel of the corresponding retargeted clips so
    # reference-stage changes remain spatially aligned with terrain seams.
    "flat": 3.10,
    "rough": 3.10,
    "slope_up": 3.10,
    "slope_down": 3.10,
    "pit": 3.10,
    "gap": 4.12,
    "stairs": 3.88,
}
# MoRE's evaluation XML leaves obstacle and heightfield ``rgba`` unspecified.
# MuJoCo compiles that default appearance to neutral gray.
MORE_XML_TERRAIN_RGBA = (0.5, 0.5, 0.5, 1.0)
MORE_MATTE_TERRAIN_MATERIAL = "more_matte_terrain"
MORE_TERRAIN_CHECKER_TEXTURE = "more_terrain_checker"
STRICT_CONTACT_MARGIN = 0.005
STRICT_CONTACT_SOLREF = (0.005, 1.0)
STRICT_CONTACT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
STRICT_RESET_CLEARANCE = 0.001


@dataclass(frozen=True)
class StairTerrainCfg:
    """A straight run-in, ascent, top platform, descent, and run-out track."""

    enabled: bool = False
    step_height: float = 0.13
    step_depth: float = 0.30
    step_count: int = 5
    width: float = 2.0
    run_in: float = 1.5
    top_length: float = 1.2
    run_out: float = 1.5
    start_x: float = 1.0
    center_y: float = 0.0

    @property
    def ascent_start(self) -> float:
        return self.start_x + self.run_in

    @property
    def top_start(self) -> float:
        return self.ascent_start + self.step_count * self.step_depth

    @property
    def descent_start(self) -> float:
        return self.top_start + self.top_length

    @property
    def end_x(self) -> float:
        return self.descent_start + self.step_count * self.step_depth + self.run_out


@dataclass(frozen=True)
class MoreTerrainCfg:
    """Parallel training lanes derived from the MoRE-discoRL scene files.

    ``gap`` and ``pit`` retain the upstream 0.30 m platform top.  The copied
    128x128 rough heightmap is spatially scaled to a robot-sized lane while
    preserving its normalized height samples.
    """

    enabled: bool = False
    terrain_types: tuple[str, ...] = MORE_TERRAIN_TYPES
    asset_dir: str = ""

    # In serial-course mode every lane contains every selected terrain.  Lanes
    # differ only in terrain order, which lets batched worlds share one MuJoCo
    # model while each episode still samples a random ordering.
    serial_course: bool = False
    course_orders: tuple[tuple[str, ...], ...] = ()
    course_lane_spacing: float = 3.0
    course_run_in: float = 2.90
    course_run_out: float = 3.10
    course_base_height: float = 0.30
    course_completion_tolerance: float = 1.0

    flat_lane_y: float = 3.0
    rough_lane_y: float = 6.0
    pit_lane_y: float = 9.0
    gap_lane_y: float = 12.0
    lane_width: float = 2.0

    rough_center_x: float = 2.0
    rough_half_length: float = 2.5
    rough_height: float = 0.05
    rough_base_depth: float = 0.05
    rough_spawn_x: float = 0.30
    slope_angle_deg: float = 6.0

    # The upstream training curriculum varies gaps continuously; 0.20 m is a
    # stable LYT starting level.  The copied evaluation XML uses 0.40 m.
    gap_first_center_x: float = 2.0
    gap_count: int = 3
    gap_width: float = 0.20
    gap_platform_half_length: float = 1.0
    gap_platform_height: float = 0.30
    gap_spawn_x: float = 1.515

    # Upstream pit.xml: center x=6, half length=5, half height=0.3.
    pit_center_x: float = 6.0
    pit_half_length: float = 5.0
    pit_platform_height: float = 0.30
    pit_spawn_x: float = 1.25

    @property
    def gap_platform_centers(self) -> tuple[float, ...]:
        pitch = 2.0 * self.gap_platform_half_length + self.gap_width
        return tuple(
            self.gap_first_center_x + index * pitch for index in range(self.gap_count)
        )

    @property
    def normalized_course_orders(self) -> tuple[tuple[str, ...], ...]:
        if self.course_orders:
            return self.course_orders
        return (self.terrain_types,)

    @property
    def course_lane_ys(self) -> tuple[float, ...]:
        count = len(self.normalized_course_orders)
        center = 0.5 * (count - 1)
        return tuple((index - center) * self.course_lane_spacing for index in range(count))


@dataclass(frozen=True)
class TerrainCourseSegment:
    """One terrain section in one serial-course lane."""

    kind: str
    start_x: float
    end_x: float

    @property
    def length(self) -> float:
        return self.end_x - self.start_x

    @property
    def center_x(self) -> float:
        return 0.5 * (self.start_x + self.end_x)


def randomized_course_orders(
    terrain_types: Sequence[str],
    count: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic, shuffled full-terrain permutations.

    The geometry must be compiled once for MuJoCo-Warp, so randomness is
    represented by a bank of lanes.  Episodes randomly select one of these
    lanes; if the requested count exceeds the number of unique permutations,
    shuffled permutations are reused.
    """

    values = validate_terrain_types(terrain_types)
    if not values:
        return ()
    if count < 1:
        raise ValueError("course variant count must be positive.")
    if "slope_up" in values and "slope_down" in values:
        blocks = [
            (value,)
            for value in values
            if value not in {"slope_up", "slope_down"}
        ]
        # An up/down pair returns to the common course datum, so it remains a
        # single block when terrain order is randomized.
        blocks.append(("slope_up", "slope_down"))
        permutations = [
            tuple(value for block in order for value in block)
            for order in itertools.permutations(blocks)
        ]
    else:
        permutations = list(itertools.permutations(values))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(permutations)
    return tuple(permutations[index % len(permutations)] for index in range(count))


def course_segments(
    cfg: MoreTerrainCfg,
    variant: int,
) -> tuple[TerrainCourseSegment, ...]:
    """Return ordered X intervals for one serial-course variant."""

    orders = cfg.normalized_course_orders
    if not 0 <= int(variant) < len(orders):
        raise IndexError(f"course variant {variant} outside [0, {len(orders)}).")
    cursor = float(cfg.course_run_in)
    result: list[TerrainCourseSegment] = []
    for kind in orders[int(variant)]:
        length = COURSE_TERRAIN_LENGTHS[kind]
        result.append(TerrainCourseSegment(kind, cursor, cursor + length))
        cursor += length
    return tuple(result)


def course_end_x(cfg: MoreTerrainCfg, variant: int) -> float:
    segments = course_segments(cfg, variant)
    last_x = segments[-1].end_x if segments else cfg.course_run_in
    return float(last_x + cfg.course_run_out)


def course_motion_sequence(order: Sequence[str]) -> tuple[str, ...]:
    """Expand a terrain order into the multi-action reference sequence."""

    result = list(COURSE_PREFIX_MOTIONS)
    for kind in order:
        if kind == "flat":
            result.append("walk")
        elif kind == "stairs":
            result.extend(("stair_up", "stair_down"))
        elif kind == "gap":
            result.append("gap_cross")
        elif kind == "rough":
            result.append("rough_walk")
        elif kind in {"slope_up", "slope_down"}:
            # The assessment policy has no terrain/skill command.  Reusing the
            # unrolled walking reference leaves adaptation to IMU+tendon input.
            result.append("rough_walk")
        elif kind == "pit":
            result.append("pit_walk")
        else:
            raise ValueError(f"Unknown course terrain kind: {kind!r}.")
    result.extend(COURSE_SUFFIX_MOTIONS)
    return tuple(result)


def course_stage_end_positions(
    cfg: MoreTerrainCfg,
    variant: int,
) -> tuple[float, ...]:
    """Return physical X boundaries aligned with ``course_motion_sequence``.

    Stage boundaries used to be reconstructed from raw clip displacement.  In
    particular that made the route depend on two mutually discontinuous 90°
    turn clips.  Geometry is the source of truth for a serial terrain course:
    the looping walk reaches the first terrain, every straight terrain ends at
    its compiled segment boundary, and stairs split at their midpoint into an
    ascent and descent stage.
    """

    result = [float(cfg.course_run_in)]
    for segment in course_segments(cfg, variant):
        if segment.kind == "stairs":
            result.extend((float(segment.center_x), float(segment.end_x)))
        else:
            result.append(float(segment.end_x))
    result.append(course_end_x(cfg, variant))
    return tuple(result)


def course_stage_scenarios(order: Sequence[str]) -> tuple[str, ...]:
    """Return assessment scenario labels aligned with course motion stages."""

    result = ["flat"] * len(COURSE_PREFIX_MOTIONS)
    for kind in order:
        if kind == "stairs":
            result.extend(("stair_up", "stair_down"))
        else:
            result.append(str(kind))
    result.extend(("flat",) * len(COURSE_SUFFIX_MOTIONS))
    return tuple(result)


def sample_course_start_stages(
    rng: np.random.Generator,
    batch_size: int,
    stage_count: int,
    random_start_prob: float,
) -> np.ndarray:
    """Sample stage-zero starts plus optional nonzero curriculum starts.

    Full-route episodes must continue to start at stage zero so their success
    rate stays meaningful.  Random nonzero starts expose the policy to later
    terrains before it has mastered every preceding stage, avoiding the
    stage-starvation observed when all resets started on the initial walk.
    """

    if batch_size < 0:
        raise ValueError("batch_size must be non-negative.")
    if stage_count < 1:
        raise ValueError("stage_count must be positive.")
    probability = float(np.clip(random_start_prob, 0.0, 1.0))
    stages = np.zeros(int(batch_size), dtype=np.int32)
    if batch_size == 0 or stage_count == 1 or probability <= 0.0:
        return stages
    curriculum = rng.random(int(batch_size)) < probability
    stages[curriculum] = rng.integers(
        1,
        int(stage_count),
        size=int(np.count_nonzero(curriculum)),
        dtype=np.int32,
    )
    return stages


def _validate_stairs(cfg: StairTerrainCfg) -> None:
    positive = {
        "step_height": cfg.step_height,
        "step_depth": cfg.step_depth,
        "width": cfg.width,
        "run_in": cfg.run_in,
        "top_length": cfg.top_length,
        "run_out": cfg.run_out,
    }
    invalid = [name for name, value in positive.items() if value <= 0.0]
    if invalid or cfg.step_count < 1:
        raise ValueError(
            f"Invalid staircase configuration: positive fields={invalid}, "
            f"step_count={cfg.step_count}."
        )


def _validate_more(cfg: MoreTerrainCfg) -> None:
    unknown = sorted(set(cfg.terrain_types) - set(MORE_TERRAIN_TYPES))
    if unknown:
        raise ValueError(
            f"Unknown MoRE terrain types {unknown}; expected a subset of "
            f"{MORE_TERRAIN_TYPES}."
        )
    positive = {
        "lane_width": cfg.lane_width,
        "rough_half_length": cfg.rough_half_length,
        "rough_height": cfg.rough_height,
        "gap_platform_half_length": cfg.gap_platform_half_length,
        "gap_platform_height": cfg.gap_platform_height,
        "pit_half_length": cfg.pit_half_length,
        "pit_platform_height": cfg.pit_platform_height,
    }
    invalid = [name for name, value in positive.items() if value <= 0.0]
    if invalid or cfg.gap_width <= 0.0 or cfg.gap_count < 2:
        raise ValueError(f"Invalid MoRE terrain configuration: {invalid} must be positive.")
    if cfg.serial_course:
        if ("slope_up" in cfg.terrain_types) != ("slope_down" in cfg.terrain_types):
            raise ValueError("serial courses must include slope_up and slope_down together.")
        if not 0.0 < float(cfg.slope_angle_deg) <= 6.0:
            raise ValueError("assessment slope_angle_deg must be in (0, 6].")
        if cfg.course_lane_spacing <= cfg.lane_width:
            raise ValueError("course_lane_spacing must be larger than lane_width.")
        if cfg.course_run_in <= 0.0 or cfg.course_run_out <= 0.0:
            raise ValueError("serial-course run-in and run-out must be positive.")
        if cfg.course_completion_tolerance < 0.0:
            raise ValueError("course_completion_tolerance cannot be negative.")
        expected = tuple(cfg.terrain_types)
        for order in cfg.normalized_course_orders:
            if len(order) != len(expected) or set(order) != set(expected):
                raise ValueError(
                    "Every serial-course order must contain each enabled terrain exactly once."
                )
            if "slope_up" in order:
                up_index = order.index("slope_up")
                if up_index + 1 >= len(order) or order[up_index + 1] != "slope_down":
                    raise ValueError("slope_up must be immediately followed by slope_down.")


def stair_boxes(cfg: StairTerrainCfg) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return ``(name, center, half_size)`` boxes for the MoRE-style staircase."""

    _validate_stairs(cfg)
    boxes: list[tuple[str, np.ndarray, np.ndarray]] = []
    half_width = 0.5 * cfg.width
    x = cfg.ascent_start
    for index in range(cfg.step_count):
        height = (index + 1) * cfg.step_height
        boxes.append(
            (
                f"stair_up_{index:02d}",
                np.asarray(
                    [x + 0.5 * cfg.step_depth, cfg.center_y, 0.5 * height]
                ),
                np.asarray([0.5 * cfg.step_depth, half_width, 0.5 * height]),
            )
        )
        x += cfg.step_depth

    top_height = cfg.step_count * cfg.step_height
    boxes.append(
        (
            "stair_top",
            np.asarray([x + 0.5 * cfg.top_length, cfg.center_y, 0.5 * top_height]),
            np.asarray([0.5 * cfg.top_length, half_width, 0.5 * top_height]),
        )
    )
    x += cfg.top_length

    for index in range(cfg.step_count):
        height = (cfg.step_count - index) * cfg.step_height
        boxes.append(
            (
                f"stair_down_{index:02d}",
                np.asarray(
                    [x + 0.5 * cfg.step_depth, cfg.center_y, 0.5 * height]
                ),
                np.asarray([0.5 * cfg.step_depth, half_width, 0.5 * height]),
            )
        )
        x += cfg.step_depth
    return boxes


def serial_course_stair_cfg(
    cfg: MoreTerrainCfg,
    stairs: StairTerrainCfg,
    variant: int,
) -> StairTerrainCfg:
    """Return the world-frame staircase compiled into one course lane.

    Reward and failure checks must use exactly the same relocated staircase as
    the collision geometry.  Keeping this calculation in one place prevents a
    serial-course policy from being rewarded against the old standalone stair
    coordinates while its feet collide with a different surface.
    """

    _validate_more(cfg)
    _validate_stairs(stairs)
    segment = next(
        (item for item in course_segments(cfg, variant) if item.kind == "stairs"),
        None,
    )
    if segment is None:
        raise ValueError(f"serial-course variant {variant} has no stairs segment.")
    flight_length = stairs.step_count * stairs.step_depth
    half_length = segment.center_x - segment.start_x
    if flight_length >= half_length:
        raise ValueError("serial stair half-segment is shorter than one stair flight.")
    requested_run_in = max(float(stairs.ascent_start), 1.0e-3)
    ascent_run_in = min(
        requested_run_in,
        max(half_length - flight_length - 1.0e-3, 1.0e-3),
    )
    top_length = segment.center_x - (
        segment.start_x + ascent_run_in + flight_length
    )
    return StairTerrainCfg(
        enabled=True,
        step_height=stairs.step_height,
        step_depth=stairs.step_depth,
        step_count=stairs.step_count,
        width=cfg.lane_width,
        run_in=ascent_run_in,
        top_length=top_length,
        run_out=1.0e-3,
        start_x=segment.start_x,
        center_y=cfg.course_lane_ys[int(variant)],
    )


def more_boxes(cfg: MoreTerrainCfg) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return the box fixtures copied from MoRE-discoRL's gap and pit scenes."""

    _validate_more(cfg)
    boxes: list[tuple[str, np.ndarray, np.ndarray]] = []
    half_width = 0.5 * cfg.lane_width
    if "gap" in cfg.terrain_types:
        for index, center_x in enumerate(cfg.gap_platform_centers):
            boxes.append(
                (
                    f"more_gap_platform_{index:02d}",
                    np.asarray(
                        [center_x, cfg.gap_lane_y, 0.5 * cfg.gap_platform_height]
                    ),
                    np.asarray(
                        [
                            cfg.gap_platform_half_length,
                            half_width,
                            0.5 * cfg.gap_platform_height,
                        ]
                    ),
                )
            )
    if "pit" in cfg.terrain_types:
        boxes.append(
            (
                "more_pit_platform",
                np.asarray(
                    [
                        cfg.pit_center_x,
                        cfg.pit_lane_y,
                        0.5 * cfg.pit_platform_height,
                    ]
                ),
                np.asarray(
                    [
                        cfg.pit_half_length,
                        half_width,
                        0.5 * cfg.pit_platform_height,
                    ]
                ),
            )
        )
    return boxes


def serial_course_boxes(
    cfg: MoreTerrainCfg,
    stairs: StairTerrainCfg,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return box fixtures for all randomized serial-course lanes."""

    _validate_more(cfg)
    _validate_stairs(stairs)
    boxes: list[tuple[str, np.ndarray, np.ndarray]] = []
    half_width = 0.5 * cfg.lane_width
    base_half_height = 0.5 * cfg.course_base_height

    def add_platform(name: str, x0: float, x1: float, y: float) -> None:
        if x1 <= x0:
            return
        boxes.append(
            (
                name,
                np.asarray([0.5 * (x0 + x1), y, base_half_height]),
                np.asarray([0.5 * (x1 - x0), half_width, base_half_height]),
            )
        )

    for variant, lane_y in enumerate(cfg.course_lane_ys):
        prefix = f"course_v{variant:02d}"
        add_platform(
            f"{prefix}_run_in",
            0.0,
            cfg.course_run_in,
            lane_y,
        )
        segments = course_segments(cfg, variant)
        for segment_index, segment in enumerate(segments):
            name = f"{prefix}_{segment_index:02d}_{segment.kind}"
            if segment.kind == "gap":
                platform_length = (
                    segment.length - (cfg.gap_count - 1) * cfg.gap_width
                ) / cfg.gap_count
                if platform_length <= 0.0:
                    raise ValueError("gap terrain is too short for its configured gaps.")
                cursor = segment.start_x
                for platform_index in range(cfg.gap_count):
                    add_platform(
                        f"{name}_platform_{platform_index:02d}",
                        cursor,
                        cursor + platform_length,
                        lane_y,
                    )
                    cursor += platform_length + cfg.gap_width
            elif segment.kind == "stairs":
                # Keep ascent in the first half and start descent at the stage
                # split.  The old compact layout put descent near the segment
                # end, about 1.3 m after the stair-down reference had already
                # begun lowering the pelvis.
                add_platform(name + "_base", segment.start_x, segment.end_x, lane_y)
                local_stairs = serial_course_stair_cfg(cfg, stairs, variant)
                for stair_name, center, half_size in stair_boxes(local_stairs):
                    shifted = center.copy()
                    shifted[2] += cfg.course_base_height
                    boxes.append((f"{name}_{stair_name}", shifted, half_size))
            elif segment.kind in {"flat", "pit"}:
                # Flat and MoRE's narrow pit safe platform both stay at the
                # common course datum, permitting arbitrary terrain ordering.
                add_platform(name, segment.start_x, segment.end_x, lane_y)
            elif segment.kind in {"rough", "slope_up", "slope_down"}:
                # Added as a heightfield or rotated ramp below.
                continue
            else:  # pragma: no cover - guarded by configuration validation.
                raise ValueError(f"Unknown serial terrain kind: {segment.kind!r}.")
        end = course_end_x(cfg, variant)
        last = segments[-1].end_x if segments else cfg.course_run_in
        add_platform(f"{prefix}_run_out", last, end, lane_y)
    return boxes


def serial_course_ramps(
    cfg: MoreTerrainCfg,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Return exact 6-degree up/down ramp geoms for serial-course lanes."""

    _validate_more(cfg)
    angle = np.deg2rad(float(cfg.slope_angle_deg))
    half_width = 0.5 * cfg.lane_width
    thickness = max(float(cfg.course_base_height), 0.05)
    result: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for variant, lane_y in enumerate(cfg.course_lane_ys):
        for index, segment in enumerate(course_segments(cfg, variant)):
            if segment.kind not in {"slope_up", "slope_down"}:
                continue
            signed_angle = -angle if segment.kind == "slope_up" else angle
            slope_length = segment.length / np.cos(angle)
            rise = segment.length * np.tan(angle)
            top_center_z = cfg.course_base_height + 0.5 * rise
            center = np.asarray(
                [
                    segment.center_x,
                    lane_y,
                    top_center_z - 0.5 * thickness * np.cos(angle),
                ],
                dtype=np.float64,
            )
            half_size = np.asarray(
                [0.5 * slope_length, half_width, 0.5 * thickness],
                dtype=np.float64,
            )
            quat = np.asarray(
                [
                    np.cos(0.5 * signed_angle),
                    0.0,
                    np.sin(0.5 * signed_angle),
                    0.0,
                ],
                dtype=np.float64,
            )
            result.append(
                (
                    f"course_v{variant:02d}_{index:02d}_{segment.kind}",
                    center,
                    half_size,
                    quat,
                )
            )
    return result


def _default_asset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "robot" / "terrains" / "more_discolrl"


def _add_matte_terrain_material(spec: mujoco.MjSpec) -> str:
    """Create MoRE's blue-gray checker as a glare-free terrain material."""

    texture = spec.add_texture()
    texture.name = MORE_TERRAIN_CHECKER_TEXTURE
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    texture.mark = mujoco.mjtMark.mjMARK_EDGE
    texture.rgb1 = (0.2, 0.3, 0.4)
    texture.rgb2 = (0.1, 0.2, 0.3)
    texture.markrgb = (0.8, 0.8, 0.8)
    texture.width = 300
    texture.height = 300

    material = spec.add_material()
    material.name = MORE_MATTE_TERRAIN_MATERIAL
    material.textures = [texture.name] + [""] * 9
    material.texuniform = 1
    material.texrepeat = (5.0, 5.0)
    material.rgba = MORE_XML_TERRAIN_RGBA
    material.specular = 0.0
    material.shininess = 0.0
    material.reflectance = 0.0
    material.metallic = 0.0
    material.roughness = 1.0
    return material.name


def _add_box(
    spec: mujoco.MjSpec,
    name: str,
    center: np.ndarray,
    half_size: np.ndarray,
    rgba: tuple[float, float, float, float],
    material: str,
) -> None:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos = center
    geom.size = half_size
    geom.rgba = rgba
    geom.material = material
    geom.friction = (1.0, 0.005, 0.0001)
    geom.group = 2
    geom.contype = 1
    geom.conaffinity = 0
    geom.condim = 3
    geom.margin = STRICT_CONTACT_MARGIN
    geom.solref = STRICT_CONTACT_SOLREF
    geom.solimp = STRICT_CONTACT_SOLIMP


def _add_ramp(
    spec: mujoco.MjSpec,
    name: str,
    center: np.ndarray,
    half_size: np.ndarray,
    quat: np.ndarray,
    material: str,
) -> None:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos = center
    geom.size = half_size
    geom.quat = quat
    geom.rgba = MORE_XML_TERRAIN_RGBA
    geom.material = material
    geom.friction = (1.0, 0.005, 0.0001)
    geom.group = 2
    geom.contype = 1
    geom.conaffinity = 0
    geom.condim = 3
    geom.margin = STRICT_CONTACT_MARGIN
    geom.solref = STRICT_CONTACT_SOLREF
    geom.solimp = STRICT_CONTACT_SOLIMP


def _attach_terrain(
    spec: mujoco.MjSpec,
    stairs: StairTerrainCfg | None,
    more: MoreTerrainCfg | None,
) -> None:
    """Attach configured terrain geometry to an existing model spec."""

    stair_enabled = stairs is not None and stairs.enabled
    more_enabled = more is not None and more.enabled and bool(more.terrain_types)
    if not stair_enabled and not more_enabled:
        return
    serial_course = bool(more_enabled and more is not None and more.serial_course)
    terrain_material = _add_matte_terrain_material(spec)
    if stair_enabled and not serial_course:
        assert stairs is not None
        for name, center, half_size in stair_boxes(stairs):
            _add_box(
                spec,
                name,
                center,
                half_size,
                MORE_XML_TERRAIN_RGBA,
                terrain_material,
            )

    if more_enabled:
        assert more is not None
        boxes = (
            serial_course_boxes(more, stairs or StairTerrainCfg(enabled=True))
            if serial_course
            else more_boxes(more)
        )
        for name, center, half_size in boxes:
            _add_box(
                spec,
                name,
                center,
                half_size,
                MORE_XML_TERRAIN_RGBA,
                terrain_material,
            )

        if serial_course and ({"slope_up", "slope_down"} & set(more.terrain_types)):
            for name, center, half_size, quat in serial_course_ramps(more):
                _add_ramp(spec, name, center, half_size, quat, terrain_material)

        if "rough" in more.terrain_types:
            asset_dir = (
                Path(more.asset_dir).expanduser().resolve()
                if more.asset_dir
                else _default_asset_dir()
            )
            heightmap = asset_dir / "height_field_lyt.png"
            if not heightmap.exists():
                raise FileNotFoundError(
                    f"Adapted MoRE-discoRL rough heightmap not found: {heightmap}"
                )
            hfield = spec.add_hfield()
            hfield.name = (
                "course_rough_hfield" if serial_course else "more_rough_hfield"
            )
            hfield.file = str(heightmap)
            hfield.size = (
                (
                    0.5 * COURSE_TERRAIN_LENGTHS["rough"]
                    if serial_course
                    else more.rough_half_length
                ),
                0.5 * more.lane_width,
                more.rough_height,
                more.rough_base_depth,
            )
            placements = (
                [
                    (
                        f"course_v{variant:02d}_{index:02d}_rough",
                        segment.center_x,
                        more.course_lane_ys[variant],
                        more.course_base_height,
                    )
                    for variant in range(len(more.normalized_course_orders))
                    for index, segment in enumerate(course_segments(more, variant))
                    if segment.kind == "rough"
                ]
                if serial_course
                else [
                    (
                        "more_rough",
                        more.rough_center_x,
                        more.rough_lane_y,
                        0.0,
                    )
                ]
            )
            for name, center_x, center_y, center_z in placements:
                geom = spec.worldbody.add_geom()
                geom.name = name
                geom.type = mujoco.mjtGeom.mjGEOM_HFIELD
                geom.hfieldname = hfield.name
                geom.pos = (center_x, center_y, center_z)
                geom.rgba = MORE_XML_TERRAIN_RGBA
                geom.material = terrain_material
                geom.friction = (1.0, 0.005, 0.0001)
                geom.group = 2
                geom.contype = 1
                geom.conaffinity = 0
                geom.condim = 3
                geom.margin = STRICT_CONTACT_MARGIN
                geom.solref = STRICT_CONTACT_SOLREF
                geom.solimp = STRICT_CONTACT_SOLIMP


def load_model(
    model_xml: str | Path,
    stairs: StairTerrainCfg | None = None,
    more: MoreTerrainCfg | None = None,
) -> mujoco.MjModel:
    """Load the robot MJCF and attach the selected MoRE terrain lanes."""

    model_path = Path(model_xml).expanduser().resolve()
    stair_enabled = stairs is not None and stairs.enabled
    more_enabled = more is not None and more.enabled and bool(more.terrain_types)
    if not stair_enabled and not more_enabled:
        return mujoco.MjModel.from_xml_path(str(model_path))

    spec = mujoco.MjSpec.from_file(str(model_path))
    _attach_terrain(spec, stairs, more)
    return spec.compile()


def load_multi_robot_model(
    model_xml: str | Path,
    robot_count: int,
    stairs: StairTerrainCfg | None = None,
    more: MoreTerrainCfg | None = None,
) -> mujoco.MjModel:
    """Build a display-only scene containing independent robot copies.

    Terrain is attached once while each robot subtree receives a unique prefix.
    The returned qpos layout contains one contiguous base-model block per robot,
    which allows playback to mirror states from independently simulated worlds.
    """

    if robot_count < 1:
        raise ValueError("robot_count must be positive.")
    model_path = Path(model_xml).expanduser().resolve()
    scene = mujoco.MjSpec()
    for index in range(robot_count):
        robot = mujoco.MjSpec.from_file(str(model_path))
        frame = scene.worldbody.add_frame(name=f"robot_{index}_frame")
        scene.attach(robot, prefix=f"robot_{index}_", frame=frame)
    _attach_terrain(scene, stairs, more)
    model = scene.compile()
    # Attaching full robot specs also attaches one copy of their scene lights.
    # Keep a single MoRE-strength key light so multi-robot terrain playback is
    # not overexposed by N identical lights.
    active_lights = np.flatnonzero(model.light_active)
    if len(active_lights) > 1:
        model.light_active[active_lights[1:]] = False
    return model


def terrain_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    """Resolve all static world geoms that may support a foot."""

    return np.flatnonzero(np.asarray(model.geom_bodyid) == 0).astype(np.int32)


def terrain_height_at(
    x: np.ndarray | float,
    cfg: StairTerrainCfg,
) -> np.ndarray:
    """Analytic top height of the straight staircase at world X."""

    values = np.asarray(x, dtype=np.float32)
    result = np.zeros_like(values)
    for _, center, half_size in stair_boxes(cfg):
        inside = np.abs(values - center[0]) <= half_size[0] + 1.0e-6
        result = np.where(inside, center[2] + half_size[2], result)
    return result


def terrain_height_from_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: np.ndarray | float,
    y: np.ndarray | float,
    *,
    ray_start_z: float = 4.0,
) -> np.ndarray:
    """Ray-cast the exact static terrain height, including the rough heightfield."""

    x_values, y_values = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )
    result = np.zeros(x_values.shape, dtype=np.float64)
    geom_id = np.full(1, -1, dtype=np.int32)
    terrain_group = np.zeros(6, dtype=np.uint8)
    terrain_group[2] = 1
    direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    for index in np.ndindex(x_values.shape):
        point = np.asarray(
            [x_values[index], y_values[index], ray_start_z], dtype=np.float64
        )
        distance = mujoco.mj_ray(
            model,
            data,
            point,
            direction,
            terrain_group,
            True,
            -1,
            geom_id,
        )
        result[index] = 0.0 if distance < 0.0 else ray_start_z - distance
    return result


def terrain_kind_from_motion(name: str) -> str:
    """Map a motion name to its matching terrain lane."""

    lowered = name.lower()
    for kind in ("slope_up", "slope_down", "gap", "pit", "rough", "stair"):
        if kind in lowered:
            return "stairs" if kind == "stair" else kind
    return "flat"


def terrain_spawn_offset(
    kind: str,
    more: MoreTerrainCfg,
    stairs: StairTerrainCfg,
    *,
    stair_down: bool = False,
    stair_down_offset: float = 0.0,
    course_variant: int = 0,
    course_stage: int | None = None,
) -> np.ndarray:
    """Return the world translation applied to a local retargeted trajectory."""

    if more.serial_course:
        lane_y = more.course_lane_ys[int(course_variant)]
        if course_stage is not None:
            stage_x: list[float] = [0.30]
            stage_z: list[float] = [more.course_base_height]
            for segment in course_segments(more, int(course_variant)):
                if segment.kind == "stairs":
                    stage_x.extend(
                        [segment.start_x, segment.center_x + stair_down_offset]
                    )
                    stage_z.extend(
                        [more.course_base_height, more.course_base_height]
                    )
                else:
                    local_x = {
                        "flat": 0.30,
                        "rough": 0.30,
                        "pit": 0.25,
                        "gap": 0.515,
                        "slope_up": 0.0,
                        "slope_down": 0.0,
                    }[segment.kind]
                    stage_x.append(segment.start_x + local_x)
                    if segment.kind in {"gap", "pit"}:
                        spawn_z = 0.0
                    elif segment.kind == "slope_down":
                        # A downhill segment starts at the crest shared with
                        # the preceding uphill ramp.  Spawning at the common
                        # course datum embedded the whole robot about 0.19 m
                        # inside this ramp.
                        spawn_z = more.course_base_height + (
                            segment.length
                            * np.tan(np.deg2rad(float(more.slope_angle_deg)))
                        )
                    else:
                        spawn_z = more.course_base_height
                    stage_z.append(spawn_z)
            last_x = course_segments(more, int(course_variant))[-1].end_x
            stage_x.append(last_x + 0.30)
            stage_z.append(more.course_base_height)
            index = int(course_stage)
            if not 0 <= index < len(stage_x):
                raise IndexError(
                    f"course stage {index} outside [0, {len(stage_x)})."
                )
            return np.asarray([stage_x[index], lane_y, stage_z[index]], dtype=np.float32)
        if kind == "flat":
            return np.asarray(
                [0.30, lane_y, more.course_base_height], dtype=np.float32
            )
        segment = next(
            item
            for item in course_segments(more, int(course_variant))
            if item.kind == kind
        )
        if kind == "rough":
            return np.asarray(
                [segment.start_x + 0.30, lane_y, more.course_base_height],
                dtype=np.float32,
            )
        if kind == "pit":
            return np.asarray([segment.start_x + 0.25, lane_y, 0.0], dtype=np.float32)
        if kind == "gap":
            return np.asarray([segment.start_x + 0.515, lane_y, 0.0], dtype=np.float32)
        if kind == "stairs":
            x = segment.start_x
            if stair_down:
                x += 0.5 * segment.length + stair_down_offset
            return np.asarray(
                [x, lane_y, more.course_base_height], dtype=np.float32
            )
        if kind in {"slope_up", "slope_down"}:
            spawn_z = more.course_base_height
            if kind == "slope_down":
                spawn_z += segment.length * np.tan(
                    np.deg2rad(float(more.slope_angle_deg))
                )
            return np.asarray(
                [segment.start_x, lane_y, spawn_z],
                dtype=np.float32,
            )

    if kind == "gap":
        return np.asarray([more.gap_spawn_x, more.gap_lane_y, 0.0], dtype=np.float32)
    if kind == "pit":
        return np.asarray([more.pit_spawn_x, more.pit_lane_y, 0.0], dtype=np.float32)
    if kind == "rough":
        return np.asarray(
            [more.rough_spawn_x, more.rough_lane_y, 0.0], dtype=np.float32
        )
    if kind == "stairs":
        x = stairs.descent_start + stair_down_offset if stair_down else 0.0
        return np.asarray([x, stairs.center_y, 0.0], dtype=np.float32)
    if kind == "flat":
        return np.asarray([0.0, more.flat_lane_y, 0.0], dtype=np.float32)
    raise ValueError(f"Unknown terrain kind: {kind!r}.")


def validate_terrain_types(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize CLI terrain names while preserving their order."""

    result = tuple(dict.fromkeys(str(value).lower() for value in values))
    unknown = sorted(set(result) - set(MORE_TERRAIN_TYPES))
    if unknown:
        raise ValueError(
            f"Unknown terrain types {unknown}; expected a subset of {MORE_TERRAIN_TYPES}."
        )
    return result
