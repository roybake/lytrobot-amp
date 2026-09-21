"""Sensor-only environment; actor never receives reference or simulator truth."""
from __future__ import annotations
import numpy as np
import torch
from sensor_training.envs import SensorHistoryEnv
from sensor_training.config import SensorObservationConfig
from lytrobot_amp.foot_force import (CpuFootForceSensor, FootForceSensorConfig,
                                     FootForceSensorDynamics, FOOT_FORCE_NAMES)


class FootForceEnv(SensorHistoryEnv):
    def __init__(self, base_env, sensor_config=FootForceSensorConfig()):
        self.foot_force_config = sensor_config
        self.force_sensor = CpuFootForceSensor(base_env.model, base_env.contact_geom_ids,
                                               base_env.terrain_geom_ids, sensor_config)
        # SensorHistoryEnv fills the initial history inside super().__init__,
        # so all Warp-only tensors must exist before calling it.
        if bool(getattr(base_env, "is_torch_env", False)):
            import warp as wp
            self._force_anchor=torch.as_tensor(np.asarray(self.force_sensor.anchors),device=base_env.device)
            self._force_body=torch.as_tensor([self.force_sensor.geom_body[g[0]] for g in self.force_sensor.side_geoms],device=base_env.device)
            count=int(base_env.contact_geom.shape[0]);wp_device=base_env.wp_data.contact.geom.device
            self._force_contact_ids=wp.array(np.arange(count,dtype=np.int32),dtype=wp.int32,device=wp_device)
            self._force_decoded=wp.zeros(count,dtype=wp.spatial_vector,device=wp_device)
        self.force_dynamics=FootForceSensorDynamics(base_env.num_envs,sensor_config,
            seed=int(base_env.cfg.seed)+2718,
            torch_device=base_env.device if bool(getattr(base_env,"is_torch_env",False)) else None)
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
                "privileged_actor_input": False, "command_dim": 0,
                "gait_command_enabled": False, "foot_force": self.foot_force_config.metadata()}

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
            raw=self._warp_force(); sensed=self.force_dynamics.apply(raw)
            force=torch.nan_to_num((sensed/self.force_sensor.force_ref).clamp(0,self.foot_force_config.force_clip)/self.foot_force_config.force_clip)
            self._last_force_normalized=force;self._last_force_raw=raw
            return torch.nan_to_num(torch.cat((imu,tendon,force),1)).clamp(-20,20)
        imu=np.stack([d.sensordata[self.imu_layout.addresses] for d in self.base_env.datas]).astype(np.float32)/self.imu_layout.divisors
        tendon=np.stack([d.ten_length[self.base_env.actuator_tendon_ids] for d in self.base_env.datas]).astype(np.float32)
        tendon=(tendon-self.base_env.ctrl_mid)/self.base_env.ctrl_half_range
        raw=np.stack([self.force_sensor.raw(d) for d in self.base_env.datas]);sensed=self.force_dynamics.apply(raw)
        force=np.nan_to_num(np.clip(sensed/self.force_sensor.force_ref,0,self.foot_force_config.force_clip)/self.foot_force_config.force_clip).astype(np.float32);self._last_force_raw=raw;self._last_force_normalized=force
        return np.nan_to_num(np.concatenate((imu,tendon,force),1)).astype(np.float32)

    def _warp_force(self):
        """Continuous Warp solver-force adapter; this is intentionally not a
        contact flag.  The normal is decoded from the solver's EFC force."""
        import warp as wp
        import mujoco_warp as mjw
        b=self.base_env; out=torch.zeros((b.num_envs,6),device=b.device)
        valid=b._valid_contact_slots(); g1,g2=b.contact_geom[:,0],b.contact_geom[:,1]
        terrain1=torch.isin(g1,b.terrain_geom_ids); terrain2=torch.isin(g2,b.terrain_geom_ids)
        world=b.contact_worldid.clamp(0,b.num_envs-1).long()
        # Use mujoco-warp's official decoder.  It handles pyramidal/elliptic
        # cones, per-world EFC addressing and friction rows; component zero in
        # the contact frame is the normal force, matching mj_contactForce.
        mjw.contact_force(b.wp_model,b.wp_data,self._force_contact_ids,False,self._force_decoded)
        normal=wp.to_torch(self._force_decoded)[:,0].clamp_min(0)
        good=valid
        # Transform contact positions to the foot body frame and use precisely
        # the same geometry-derived Gaussian anchors as the CPU adapter.
        for side, geoms in enumerate(((int(b.contact_geom_id_values[2]),int(b.contact_geom_id_values[3])),(int(b.contact_geom_id_values[0]),int(b.contact_geom_id_values[1])))):
            matched=good & ((terrain1 & torch.isin(g2,torch.tensor(geoms,device=b.device))) | (terrain2 & torch.isin(g1,torch.tensor(geoms,device=b.device))))
            body=int(self._force_body[side]); rel=b.contact_pos-b.xpos[world,body]
            local=torch.einsum('nij,nj->ni',b.xmat[world,body].transpose(1,2),rel)[:,:2]
            d2=((local[:,None,:]-self._force_anchor[side][None])**2).sum(-1)
            weights=torch.softmax(-d2/(2*max(self.foot_force_config.distribution_sigma,1e-5)**2),dim=1)
            for zone in range(3): out[:,side*3+zone].scatter_add_(0,world,normal*matched*weights[:,zone])
        return torch.nan_to_num(out).clamp_min(0)

    def reset(self,env_ids=None):
        self.force_dynamics.reset(env_ids)
        return super().reset(env_ids)

    def step(self,action):
        observation,reward,done,info=super().step(action);force=self._last_force_normalized
        left=force[:,:3].sum(1);right=force[:,3:].sum(1);eps=1e-5
        for index,name in enumerate(FOOT_FORCE_NAMES):info[f'foot_force/{name.removesuffix("_force")}']=force[:,index]
        def scalar(value):return float(value.detach().float().mean().cpu()) if torch.is_tensor(value) else float(np.mean(value))
        cast=lambda value:value.float() if torch.is_tensor(value) else value.astype(np.float32)
        info.update({'foot_force/left_total':left,'foot_force/right_total':right,
          'foot_force/left_contact_fraction':cast(left>eps),'foot_force/right_contact_fraction':cast(right>eps),
          'foot_force/double_support_fraction':cast((left>eps)&(right>eps)),'foot_force/flight_fraction':cast((left<=eps)&(right<=eps)),
          'foot_force/saturation_fraction':cast(force>=1.-1e-6).mean(1)})
        finite=bool(torch.isfinite(force).all()) if torch.is_tensor(force) else bool(np.isfinite(force).all())
        if not finite: print('[WARN] foot force contains NaN/Inf')
        elif scalar((force.abs() if torch.is_tensor(force) else np.abs(force)).sum(1)<eps)>.99: print('[WARN] foot force is zero for nearly every environment')
        elif scalar(info['foot_force/saturation_fraction'])>.99: print('[WARN] foot force is saturated for nearly every channel')
        return observation,reward,done,info


class FootForceCpuAmpEnv(FootForceEnv):
    def __init__(self,cfg,device='cpu'):
        if device != 'cpu': raise ValueError('FootForceCpuAmpEnv only supports device=cpu')
        from lytrobot_amp.env import LytRobotAmpEnv
        super().__init__(LytRobotAmpEnv(cfg))
class FootForceWarpAmpEnv(FootForceEnv):
    def __init__(self,cfg,device='cuda'): from lytrobot_amp.env_warp import LytRobotWarpAmpEnv; super().__init__(LytRobotWarpAmpEnv(cfg,device=device))


class NoForceEnv(SensorHistoryEnv):
    """Strict 56D control for the force ablation (all else is identical)."""
    def __init__(self,base_env):
        super().__init__(base_env,SensorObservationConfig(history_steps=0,gait_command_enabled=False))
    @property
    def sensor_frame_dim(self): return 56
    @property
    def gait_command_dim(self): return 0
    @property
    def obs_dim(self): return 56
    @property
    def observation_metadata(self):
        return {"sensor_version":"no-force-stage1-v1","actor_frame_dim":56,
                "actor_input_fields":("imu","tendon_length"),"command_dim":0,
                "gait_command_enabled":False,"privileged_actor_input":False}
    def _allocate_histories(self):
        shape=(self.num_envs,1,56); amp=(self.num_envs,1,self.base_motion.feature_dim)
        if self.is_torch_env:
            self._actor_history=torch.zeros(shape,device=self.base_env.device); self._amp_history=torch.zeros(amp,device=self.base_env.device)
        else:
            self._actor_history=np.zeros(shape,np.float32); self._amp_history=np.zeros(amp,np.float32)
    def _normalized_sensor_frame(self):
        if self.is_torch_env:
            imu=self._sensordata[:,self._imu_addresses]/self._imu_divisors
            tendon=(self.base_env.ten_length[:,self.base_env.actuator_tendon_ids]-self.base_env.ctrl_mid)/self.base_env.ctrl_half_range
            return torch.nan_to_num(torch.cat((imu,tendon),1)).clamp(-20,20)
        imu=np.stack([d.sensordata[self.imu_layout.addresses] for d in self.base_env.datas]).astype(np.float32)/self.imu_layout.divisors
        tendon=np.stack([d.ten_length[self.base_env.actuator_tendon_ids] for d in self.base_env.datas]).astype(np.float32)
        return np.nan_to_num(np.concatenate((imu,(tendon-self.base_env.ctrl_mid)/self.base_env.ctrl_half_range),1)).astype(np.float32)

class NoForceCpuAmpEnv(NoForceEnv):
    def __init__(self,cfg,device='cpu'):
        if device!='cpu': raise ValueError('NoForceCpuAmpEnv only supports device=cpu')
        from lytrobot_amp.env import LytRobotAmpEnv
        super().__init__(LytRobotAmpEnv(cfg))
class NoForceWarpAmpEnv(NoForceEnv):
    def __init__(self,cfg,device='cuda'):
        from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
        super().__init__(LytRobotWarpAmpEnv(cfg,device=device))
