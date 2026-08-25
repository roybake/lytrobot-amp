"""Shared terrain and motion defaults used by training entrypoints."""

from __future__ import annotations


# Preserve the native timing of CMU subject 69 trial 01 after scaling and
# retargeting it to the LYT model.
ORIGINAL_RETARGETED_WALK_SPEED_MPS = 0.8622949262039437
TRAINING_TARGET_SPEED_MPS = ORIGINAL_RETARGETED_WALK_SPEED_MPS

ROUGH_TERRAIN_HEIGHT_M = 0.05
SLOPE_ANGLE_DEG = 6.0
MULTIMOTION_STAIR_STEP_HEIGHT_M = 0.13
MULTIMOTION_STAIR_STEP_DEPTH_M = 0.22

# The dedicated stair curriculum starts with the easier geometry and switches
# to a harder geometry when domain randomization is enabled in Stage 2.
STAIR_STAGE1_STEP_HEIGHT_M = 0.13
STAIR_STAGE1_STEP_DEPTH_M = 0.32
STAIR_STAGE2_STEP_HEIGHT_M = 0.20
STAIR_STAGE2_STEP_DEPTH_M = 0.22
