"""Sensor-only environment; actor never receives reference or simulator truth."""
from __future__ import annotations
import numpy as np
import torch
from sensor_training.envs import SensorHistoryEnv
from sensor_training.config import SensorObservationConfig
from lytrobot_amp.foot_force import CpuFootForceSensor, FootForceSensorConfig, FOOT_FORCE_NAMES


class FootForceEnv(SensorHistoryEnv):
    def __init__(self, base_env, sensor_config=FootForceSensorConfig()):
        self.foot_force_config = sensor_config
        self.force_sensor = CpuFootForceSensor(base_env.model, base_env.contact_geom_ids,
                                               base_env.terrain_geom_ids, sensor_config)
        # SensorHistoryEnv fills the initial history inside super().__init__,
        # so all Warp-only tensors must exist before calling it.
        if bool(getattr(base_env, "is_torch_env", False)):
            self._force_anchor=torch.as_tensor(np.asarray(self.force_sensor.anchors),device=base_env.device)
            self._force_body=torch.as_tensor([self.force_sensor.geom_body[g[0]] for g in self.force_sensor.side_geoms],device=base_env.device)
        super().__init__(base_env, SensorObservationConfig(history_steps=0, gait_command_enabled=False))

    @property
    def sensor_frame_dim(self): return 62
    @property
    def gait_command_dim(self): return 0
    @property
    def obs_dim(self): return 62
    @property
    def observation_metadata(self):
        return {"sensor_version": "foot-force-stage1-v1", "actor_frame_dim": 62,
                "actor_input_fields": ("imu", "tendon_length", "foot_force"),
                "privileged_actor_input": False, "foot_force": self.foot_force_config.metadata()}

    def _allocate_histories(self):
        # This experiment intentionally has no reference/gait-command history.
        shape=(self.num_envs,1,62); amp=(self.num_envs,1,self.base_motion.feature_dim)
        if self.is_torch_env:
            self._actor_history=torch.zeros(shape,device=self.base_env.device); self._amp_history=torch.zeros(amp,device=self.base_env.device)
        else:
            self._actor_history=np.zeros(shape,np.float32); self._amp_history=np.zeros(amp,np.float32)

    def _normalized_sensor_frame(self):
        if self.is_torch_env:
            imu=self._sensordata[:,self._imu_addresses]/self._imu_divisors
            tendon=(self.base_env.ten_length[:,self.base_env.actuator_tendon_ids]-self.base_env.ctrl_mid)/self.base_env.ctrl_half_range
            force=self._warp_force()
            return torch.nan_to_num(torch.cat((imu,tendon,force),1)).clamp(-20,20)
        imu=np.stack([d.sensordata[self.imu_layout.addresses] for d in self.base_env.datas]).astype(np.float32)/self.imu_layout.divisors
        tendon=np.stack([d.ten_length[self.base_env.actuator_tendon_ids] for d in self.base_env.datas]).astype(np.float32)
        tendon=(tendon-self.base_env.ctrl_mid)/self.base_env.ctrl_half_range
        force=np.stack([self.force_sensor.normalized(d) for d in self.base_env.datas])
        return np.nan_to_num(np.concatenate((imu,tendon,force),1)).astype(np.float32)

    def _warp_force(self):
        """Continuous Warp solver-force adapter; this is intentionally not a
        contact flag.  The normal is decoded from the solver's EFC force."""
        b=self.base_env; out=torch.zeros((b.num_envs,6),device=b.device)
        valid=b._valid_contact_slots(); g1,g2=b.contact_geom[:,0],b.contact_geom[:,1]
        terrain1=torch.isin(g1,b.terrain_geom_ids); terrain2=torch.isin(g2,b.terrain_geom_ids)
        world=b.contact_worldid.clamp(0,b.num_envs-1).long()
        adr=__import__('warp').to_torch(b.wp_data.contact.efc_address)[:,0].long()
        efc=__import__('warp').to_torch(b.wp_data.efc.force)
        good=valid & (adr>=0)
        normal=torch.zeros_like(adr,dtype=torch.float32); normal[good]=efc[world[good],adr[good]].clamp_min(0)
        # Transform contact positions to the foot body frame and use precisely
        # the same geometry-derived Gaussian anchors as the CPU adapter.
        for side, geoms in enumerate(((int(b.contact_geom_id_values[2]),int(b.contact_geom_id_values[3])),(int(b.contact_geom_id_values[0]),int(b.contact_geom_id_values[1])))):
            matched=good & ((terrain1 & torch.isin(g2,torch.tensor(geoms,device=b.device))) | (terrain2 & torch.isin(g1,torch.tensor(geoms,device=b.device))))
            body=int(self._force_body[side]); rel=b.contact_pos-b.xpos[world,body]
            local=torch.einsum('nij,nj->ni',b.xmat[world,body].transpose(1,2),rel)[:,:2]
            d2=((local[:,None,:]-self._force_anchor[side][None])**2).sum(-1)
            weights=torch.softmax(-d2/(2*max(self.foot_force_config.distribution_sigma,1e-5)**2),dim=1)
            for zone in range(3): out[:,side*3+zone].scatter_add_(0,world,normal*matched*weights[:,zone])
        ref=max(float(np.sum(b.model.body_mass))*abs(float(b.model.opt.gravity[2]))/2,1e-6)
        return torch.nan_to_num((out/ref).clamp(0,self.foot_force_config.force_clip)/self.foot_force_config.force_clip)


class FootForceCpuAmpEnv(FootForceEnv):
    def __init__(self,cfg,device='cpu'): from lytrobot_amp.env import LytRobotAmpEnv; super().__init__(LytRobotAmpEnv(cfg,device=device))
class FootForceWarpAmpEnv(FootForceEnv):
    def __init__(self,cfg,device='cuda'): from lytrobot_amp.env_warp import LytRobotWarpAmpEnv; super().__init__(LytRobotWarpAmpEnv(cfg,device=device))
