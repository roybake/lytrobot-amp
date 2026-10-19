#!/usr/bin/env python3
"""Create the continuous 6 m, 3-degree ascent/descent MuJoCo scene."""

from math import radians, tan
from pathlib import Path


TERRAIN_DIR = Path(__file__).resolve().parents[1] / "robot" / "terrains"
SLOPE_DEG = 3.0
RUN_IN = 1.0
FIELD_LENGTH = 6.0
FIELD_WIDTH = 6.0
ROWS = 9
COLUMNS = 401


def height(x: float) -> float:
    """Integrate the smooth slope profile, with x measured from the toe."""
    gradient = tan(radians(SLOPE_DEG))
    if x <= 1.0:
        # A 1 m transition from the flat floor to the +3-degree straight.
        return gradient * (x**3 - 0.5 * x**4)
    if x <= 2.0:
        return gradient * (0.5 + x - 1.0)
    if x <= 4.0:
        # A 2 m crest smoothly changes the gradient from +3 to -3 degrees.
        t = (x - 2.0) / 2.0
        return gradient * (1.5 + 2.0 * (t - 2.0 * t**3 + t**4))
    if x <= 5.0:
        return gradient * (1.5 - (x - 4.0))
    # A 1 m transition from the -3-degree straight back to the flat floor.
    t = x - 5.0
    return gradient * (0.5 - t + t**3 - 0.5 * t**4)


def main() -> None:
    heights = [height(FIELD_LENGTH * column / (COLUMNS - 1)) for column in range(COLUMNS)]
    peak_height = max(heights)
    profile = " ".join(f"{value / peak_height:.17g}" for value in heights)
    elevation = "\n".join("              " + profile for _ in range(ROWS))
    center_x = RUN_IN + 0.5 * FIELD_LENGTH
    scene = f'''<mujoco model="LYT slope up and down scene">
  <compiler angle="radian" meshdir=".." inertiafromgeom="true" balanceinertia="true">
    <lengthrange inttotal="50" useexisting="true" />
  </compiler>
  <statistic center="4.5 0 0.5" extent="5.0" />
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.871 0.616 0.4 0.5" />
    <global azimuth="-130" elevation="-20" />
  </visual>
  <asset>
    <model name="lyt" file="../MJCFmodelfinal.xml" />
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <texture type="2d" name="terrain_groundplane" builtin="checker" rgb1="0.2 0.19 0.2" rgb2="0.1 0.11 0.11" width="50" height="50" />
    <material name="terrain_groundplane" texture="terrain_groundplane" texuniform="true" texrepeat="4 4" reflectance="0.5" />
    <material name="slope_material" rgba="0.4 0.4 0.4 1" reflectance="0" specular="0" />
    <!-- Flat run-in: 0..1 m. Entry: 1..2 m. +{SLOPE_DEG:g} degrees: 2..3 m.
         Smooth crest: 3..5 m. -{SLOPE_DEG:g} degrees: 5..6 m. Exit: 6..7 m. -->
    <hfield name="slope_hfield" size="{0.5 * FIELD_LENGTH:g} {0.5 * FIELD_WIDTH:g} {peak_height:.17g} 0.05" nrow="{ROWS}" ncol="{COLUMNS}"
            elevation="
{elevation}
            " />  <!-- size 第三个数控制坡面高度；6 米坡道 3° 对应 0.111366530976，2° 对应 0.074206635170。 -->
  </asset>
  <worldbody>
    <attach model="lyt" body="pelvis" prefix="" />
    <light pos="0 0 10.3" dir="0 0 -1.3" directional="true" diffuse="1.5 1.5 1.5" specular="0.1 0.1 0.1" castshadow="false" />
    <geom name="floor" size="0 0 0.05" type="plane" material="terrain_groundplane" rgba="0.8 0.9 0.8 1" friction="1 .1 .1" contype="1" conaffinity="1" condim="3" margin="0" group="2" />
    <geom name="slope" type="hfield" hfield="slope_hfield" pos="{center_x:g} 0 0" material="slope_material" rgba="0.4 0.4 0.4 1" friction="1 .1 .1" contype="1" conaffinity="1" condim="3" margin="0" group="2" />
  </worldbody>
</mujoco>
'''
    scene_path = TERRAIN_DIR / "slope.xml"
    scene_path.write_text(scene)
    print(f"Created {scene_path}; peak height {peak_height:.6f} m, straight slopes +/-{SLOPE_DEG:g} degrees.")


if __name__ == "__main__":
    main()
