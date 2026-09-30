"""Causal simulated encoder/IMU output, shared by CPU, Warp and mjlab.

Only this layer sees simulator truth. Derived Actor features use its cached
output. Random samples are local to this layer and never use future samples.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class MeasurementSpec:
    noise_std: float = 0.
    dr_noise_std: float = 0.
    constant_bias: float = 0.
    reset_bias_range: float = 0.
    scale_error: float = 0.
    quantization: float = 0.
    max_delay_ticks: int = 0


class CausalMeasurement:
    def __init__(self, initial, spec: MeasurementSpec, *, center, lower, upper, seed=42):
        self.spec = spec
        self.torch = torch.is_tensor(initial)
        self.center, self.lower, self.upper = center, lower, upper
        self.rng = (torch.Generator(device=initial.device).manual_seed(seed)
                    if self.torch else np.random.default_rng(seed))
        self.bias = self._zeros_like(initial)
        self.gain = self.bias + 1.
        self.delay = (torch.zeros(initial.shape[0], dtype=torch.long, device=initial.device)
                      if self.torch else np.zeros(initial.shape[0], dtype=np.int64))
        self.output = self._copy(initial)
        self.history = (torch.stack([initial.clone() for _ in range(spec.max_delay_ticks + 1)])
                        if self.torch else np.stack([initial.copy() for _ in range(spec.max_delay_ticks + 1)]))

    def _zeros_like(self, value):
        return torch.zeros_like(value) if self.torch else np.zeros_like(value)

    def _copy(self, value):
        return value.clone() if self.torch else value.copy()

    def _random(self, shape, normal=False):
        if self.torch:
            return (torch.randn if normal else torch.rand)(shape, device=self.output.device,
                        dtype=self.output.dtype, generator=self.rng)
        return np.asarray(self.rng.standard_normal(shape) if normal else self.rng.random(shape), dtype=np.float32)

    def _encode(self, truth, scale, extra_noise_std=0., noise_sample=None):
        value = self.center + (truth - self.center) * self.gain + self.bias
        deviation = self.spec.noise_std + scale * self.spec.dr_noise_std + extra_noise_std
        if np.ndim(deviation) > 0 or float(deviation) > 0:
            value = value + deviation * (self._random(truth.shape, True) if noise_sample is None else noise_sample)
        if self.spec.quantization > 0:
            rounding = torch.round if self.torch else np.round
            value = rounding(value / self.spec.quantization) * self.spec.quantization
        if self.torch:
            value = torch.nan_to_num(value)
            return torch.maximum(torch.minimum(value, self.upper), self.lower)
        return np.clip(np.nan_to_num(value), self.lower, self.upper).astype(np.float32)

    def reset(self, truth, ids, *, scale=0., extra_noise_std=0.):
        # Generate samples in full vector shape so boolean/index selectors work
        # identically on both backends; only selected worlds are mutated.
        if not self.torch and (np.asarray(ids).size == 0 or
                               (np.asarray(ids).dtype == bool and not np.asarray(ids).any())):
            return
        spec = self.spec
        if spec.reset_bias_range > 0:
            sample = (2*self._random(truth.shape)-1) * spec.reset_bias_range * scale
            self.bias[ids] = sample[ids] + spec.constant_bias
        else:
            self.bias[ids] = spec.constant_bias
        if spec.scale_error > 0:
            sample = 1 + (2*self._random(truth.shape)-1) * spec.scale_error * scale
            self.gain[ids] = sample[ids]
        else:
            self.gain[ids] = 1.
        maximum = min(spec.max_delay_ticks, int(np.floor(spec.max_delay_ticks * scale + 1e-6)))
        if maximum:
            sample = (self._random((truth.shape[0],)) * (maximum+1))
            self.delay[ids] = sample[ids].to(torch.long) if self.torch else sample[ids].astype(np.int64)
        else:
            self.delay[ids] = 0
        value = self._encode(truth, scale, extra_noise_std)
        self.output[ids] = value[ids]
        # No pre-reset values remain in a delayed encoder stream.
        self.history[:, ids] = value[ids][None]

    def advance(self, truth, *, scale=0., extra_noise_std=0., noise_sample=None):
        value = self._encode(truth, scale, extra_noise_std, noise_sample)
        roll = torch.roll if self.torch else np.roll
        self.history = roll(self.history, -1, 0)
        self.history[-1] = value
        ids = (torch.arange(truth.shape[0], device=truth.device) if self.torch else np.arange(truth.shape[0]))
        self.output = self.history[self.spec.max_delay_ticks-self.delay, ids]
        return self.output
