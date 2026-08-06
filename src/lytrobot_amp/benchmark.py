"""Acceptance criteria shared by training, evaluation, and reports."""

from __future__ import annotations

from dataclasses import dataclass
import math


MIN_WALKING_SPEED_KMH = 5.0
MIN_WALKING_SPEED_MPS = MIN_WALKING_SPEED_KMH / 3.6
# Preserve the native timing of CMU subject 69 trial 01 after scaling and
# retargeting it to the LYT model.  The previous 1.4 m/s target compressed this
# source gait by about 1.62x and produced a much more aggressive 160-step/min
# reference.  Keep the external 5 km/h acceptance threshold independent: a
# native-speed policy is expected to fail that speed criterion until a later
# speed curriculum is introduced.
ORIGINAL_RETARGETED_WALK_SPEED_MPS = 0.8622949262039437
TRAINING_TARGET_SPEED_MPS = ORIGINAL_RETARGETED_WALK_SPEED_MPS
MAX_SPECIFIC_COT_J_PER_KG_M = 0.8
MAX_ROUGH_HEIGHT_DIFFERENCE_M = 0.05
MAX_SLOPE_DEG = 6.0
STAIR_HEIGHT_RANGE_M = (0.13, 0.20)
STAIR_TREAD_RANGE_M = (0.22, 0.32)
STAIR_ASSESSMENT_CASES = tuple(
    (height, tread)
    for height in STAIR_HEIGHT_RANGE_M
    for tread in STAIR_TREAD_RANGE_M
)
REQUIRED_SCENARIOS = (
    "flat",
    "rough",
    "slope_up",
    "slope_down",
    "stair_up",
    "stair_down",
)


def specific_cost_of_transport(
    energy_j: float,
    mass_kg: float,
    distance_m: float,
) -> float:
    """Return absolute mechanical energy per mass and forward distance."""

    if not all(math.isfinite(value) for value in (energy_j, mass_kg, distance_m)):
        return float("inf")
    if energy_j < 0.0 or mass_kg <= 0.0 or distance_m <= 0.0:
        return float("inf")
    return float(energy_j / (mass_kg * distance_m))


@dataclass(frozen=True)
class AcceptanceResult:
    mean_speed_mps: float
    flat_specific_cot_j_per_kg_m: float
    stable_scenarios: tuple[str, ...]
    course_successes: int

    @property
    def failures(self) -> tuple[str, ...]:
        result: list[str] = []
        if (
            not math.isfinite(self.mean_speed_mps)
            or self.mean_speed_mps < MIN_WALKING_SPEED_MPS
        ):
            result.append(
                f"speed {self.mean_speed_mps:.4f} m/s < {MIN_WALKING_SPEED_MPS:.4f} m/s"
            )
        if (
            not math.isfinite(self.flat_specific_cot_j_per_kg_m)
            or self.flat_specific_cot_j_per_kg_m > MAX_SPECIFIC_COT_J_PER_KG_M
        ):
            result.append(
                "flat CoT "
                f"{self.flat_specific_cot_j_per_kg_m:.4f} J/kg/m > "
                f"{MAX_SPECIFIC_COT_J_PER_KG_M:.4f} J/kg/m"
            )
        missing = sorted(set(REQUIRED_SCENARIOS) - set(self.stable_scenarios))
        if missing:
            result.append(f"unstable or unvisited scenarios: {missing}")
        if self.course_successes < 1:
            result.append("serial six-scenario course was never completed")
        return tuple(result)

    @property
    def passed(self) -> bool:
        return not self.failures
