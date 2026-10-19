"""LYT AMP environment using mjlab's public Simulation API for physics."""

from __future__ import annotations

import torch
import warp as wp

from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from .env_warp import (
    LytRobotWarpAmpEnv,
    _WARP_DYNAMIC_SOURCE_ERROR,
    _reload_mujoco_warp_reset_data,
)


class _ConfiguredModelCfg(MujocoCfg):
    """Keep the compiled LYT MJCF options configured by the base environment."""

    def apply(self, model) -> None:
        # The LYT environment has already applied timestep, solver and CCD
        # settings to this compiled MjModel before Simulation receives it.
        del model


class LytRobotMjlabAmpEnv(LytRobotWarpAmpEnv):
    """Run the existing LYT task and AMP features on mjlab.Simulation."""

    def _init_warp_data(self) -> None:
        wp.init()
        self.sim = Simulation(
            num_envs=self.num_envs,
            cfg=SimulationCfg(
                nconmax=max(1, int(self.cfg.warp_nconmax)),
                njmax=max(1, int(self.cfg.warp_njmax)),
                mujoco=_ConfiguredModelCfg(),
            ),
            model=self.model,
            device=str(self.device),
        )
        self.sim.use_cuda_graph = bool(self.cfg.warp_cuda_graph)
        self.wp_device = self.sim.wp_device
        self.wp_model = self.sim.wp_model
        self.wp_data = self.sim.wp_data
        self.wp_model.opt.timestep = wp.full(
            self.num_envs, float(self.cfg.timestep), dtype=wp.float32, device=self.wp_device,
        )
        self.sim.create_graph()
        self.sim.forward()
        self._init_warp_views()

    def _capture_warp_graphs(self) -> None:
        # Simulation captures its own step/forward/reset graphs at creation.
        pass

    def _warp_step(self) -> None:
        self.sim.step()

    def _warp_forward(self) -> None:
        self.sim.forward()

    def _reset_warp_worlds(self, ids: torch.Tensor) -> None:
        try:
            self.sim.reset(ids)
        except RuntimeError as exc:
            if _WARP_DYNAMIC_SOURCE_ERROR not in str(exc):
                raise
            _reload_mujoco_warp_reset_data()
            self.sim.reset(ids)
