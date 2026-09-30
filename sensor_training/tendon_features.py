"""Causal tendon features shared by simulation and the onboard controller.

Pass measured lengths in metres and reference lengths from the onboard table.
Call update exactly once per control tick. reset uses the current measurement
and clears filter memory, so an episode boundary cannot create a derivative spike.
"""
from __future__ import annotations
import numpy as np
import torch


class CausalTendonFeatures:
    def __init__(self, lengths, cfg):
        self.cfg = cfg
        self.previous = lengths.clone() if torch.is_tensor(lengths) else lengths.copy()
        self.velocity = torch.zeros_like(lengths) if torch.is_tensor(lengths) else np.zeros_like(lengths)

    def reset(self, lengths, ids):
        self.previous[ids] = lengths[ids]
        self.velocity[ids] = 0.

    def update(self, lengths, dt):
        if dt <= 0:
            raise ValueError('Control dt must be positive.')
        derivative = (lengths - self.previous) / dt
        clip = torch.clamp if torch.is_tensor(lengths) else np.clip
        # Clip before filtering to prevent a bad sample poisoning future ticks.
        limit = self.cfg.observation_clip * self.cfg.tendon_velocity_scale
        derivative = clip(derivative, -limit, limit)
        alpha = self.cfg.tendon_velocity_filter_alpha
        self.velocity = alpha * derivative + (1 - alpha) * self.velocity
        self.previous = lengths.clone() if torch.is_tensor(lengths) else lengths.copy()

    def frame(self, lengths, reference):
        values = []
        if self.cfg.include_tendon_velocity:
            values.append(self.velocity / self.cfg.tendon_velocity_scale)
        if self.cfg.include_tendon_tracking_error:
            values.append((lengths - reference) / self.cfg.tendon_tracking_error_scale)
        cat = torch.cat if torch.is_tensor(lengths) else np.concatenate
        result = cat(values, axis=1) if values else lengths[:, :0]
        clip = torch.clamp if torch.is_tensor(lengths) else np.clip
        return clip(result, -self.cfg.observation_clip, self.cfg.observation_clip)
