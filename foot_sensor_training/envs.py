"""Sensor-only environment; actor never receives reference or simulator truth."""
from __future__ import annotations
import numpy as np
import torch
from sensor_training.envs import SensorHistoryEnv, sensor_stability_task_reward
from sensor_training.config import SensorObservationConfig
from lytrobot_amp.foot_force import (CpuFootForceSensor, FootForceSensorConfig,
                                     FootForceSensorDynamics, FOOT_FORCE_NAMES)
from .phase_clock import DeployablePhaseClock


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
        if bool(getattr(base_env,"is_torch_env",False)):
            self._force_sensor_output=torch.zeros((base_env.num_envs,6),device=base_env.device)
            self._last_force_raw=torch.zeros_like(self._force_sensor_output)
            self._last_force_normalized=torch.zeros_like(self._force_sensor_output)
        else:
            self._force_sensor_output=np.zeros((base_env.num_envs,6),np.float32)
            self._last_force_raw=np.zeros_like(self._force_sensor_output)
            self._last_force_normalized=np.zeros_like(self._force_sensor_output)
        self._force_updated_this_tick=False
        super().__init__(base_env, SensorObservationConfig(history_steps=0, gait_command_enabled=False))
        self._update_foot_force_sensor()
        self._fill_history(self._all_env_ids())

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
            force=self._read_foot_force_sensor()
            return torch.nan_to_num(torch.cat((imu,tendon,force),1)).clamp(-20,20)
        imu=np.stack([d.sensordata[self.imu_layout.addresses] for d in self.base_env.datas]).astype(np.float32)/self.imu_layout.divisors
        tendon=np.stack([d.ten_length[self.base_env.actuator_tendon_ids] for d in self.base_env.datas]).astype(np.float32)
        tendon=(tendon-self.base_env.ctrl_mid)/self.base_env.ctrl_half_range
        force=self._read_foot_force_sensor()
        return np.nan_to_num(np.concatenate((imu,tendon,force),1)).astype(np.float32)

    def _raw_foot_force(self):
        if self.is_torch_env:
            return self._warp_force()
        return np.stack([self.force_sensor.raw(data) for data in self.base_env.datas]).astype(np.float32)

    def _update_foot_force_sensor(self, env_ids=None):
        """Advance noise/filter/delay exactly once for the selected policy tick."""
        ids = self._all_env_ids() if env_ids is None else env_ids
        raw_all = self._raw_foot_force()
        raw = raw_all[ids]
        sensed = self.force_dynamics.apply(raw, ids)
        if self.is_torch_env:
            force=torch.nan_to_num(
                (sensed/self.force_sensor.force_ref).clamp(0,self.foot_force_config.force_clip)
                / self.foot_force_config.force_clip
            )
        else:
            force=np.nan_to_num(
                np.clip(sensed/self.force_sensor.force_ref,0,self.foot_force_config.force_clip)
                / self.foot_force_config.force_clip
            ).astype(np.float32)
        self._last_force_raw[ids]=raw
        self._last_force_normalized[ids]=force
        self._force_sensor_output[ids]=force
        self._force_updated_this_tick=True

    def _read_foot_force_sensor(self):
        """Pure cached read: observation/logging never advances sensor state."""
        return self._force_sensor_output

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
        self.base_env.reset(env_ids)
        ids=self._all_env_ids() if env_ids is None else env_ids
        self.force_dynamics.reset(ids)
        self._update_foot_force_sensor(ids)
        self._fill_history(ids)
        return self.observe()

    def _terminal_observation(self,ids):
        # Called by the base environment after physics and before autoreset.
        if not self._force_updated_this_tick:
            self._update_foot_force_sensor()
        return super()._terminal_observation(ids)

    def step(self,action):
        self._force_updated_this_tick=False
        _base_obs,reward,done,info=self.base_env.step(action)
        if bool(getattr(self.base_env.cfg,"sensor_stability_reward",False)):
            reward=sensor_stability_task_reward(
                info,fall_penalty=float(self.base_env.cfg.sensor_fall_penalty),
                stall_penalty=float(self.base_env.cfg.forward_progress_termination_penalty),
                knee_gait_reward_coef=float(self.base_env.cfg.sensor_knee_gait_reward_coef),
                natural_gait_reward_coef=float(self.base_env.cfg.sensor_natural_gait_reward_coef),
                mirror_reward_coef=float(self.base_env.cfg.sensor_mirror_reward_coef),
                stride_reward_coef=float(self.base_env.cfg.sensor_stride_reward_coef),
                foot_roll_reward_coef=float(self.base_env.cfg.sensor_foot_roll_reward_coef),
                alternation_reward_coef=float(self.base_env.cfg.sensor_alternation_reward_coef),
                hop_penalty_coef=float(self.base_env.cfg.sensor_hop_penalty_coef),
                tiptoe_penalty_coef=float(self.base_env.cfg.sensor_tiptoe_penalty_coef),
                stance_slip_penalty_coef=float(self.base_env.cfg.sensor_stance_slip_penalty_coef),
                sync_forward_penalty_coef=float(self.base_env.cfg.sensor_sync_forward_penalty_coef),
                velocity_excess_penalty_coef=float(self.base_env.cfg.sensor_velocity_excess_penalty_coef))
            info["sensor_stability_reward"]=reward
        if self.is_torch_env:
            done_ids=torch.nonzero(done>0,as_tuple=False).flatten()
        else:
            done_ids=np.nonzero(np.asarray(done)>0)[0].astype(np.int64)
        if not self._force_updated_this_tick:
            self._update_foot_force_sensor()
        if len(done_ids)>0:
            # The base environment has already autoreset these worlds.  Clear
            # every stateful sensor component before reading their reset pose.
            self.force_dynamics.reset(done_ids)
            self._update_foot_force_sensor(done_ids)
        self._append_history()
        if len(done_ids)>0:self._fill_history(done_ids)
        force=self._last_force_normalized
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
        return self.observe(),reward,done,info


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


class _PhaseObservationMixin:
    """Append only sin/cos from a controller-reconstructible fixed clock."""
    _phase_base_dim: int

    def _initialize_phase_clock(self,base_env):
        cfg=base_env.cfg
        if float(cfg.stand_start_prob)!=1.0 or float(cfg.motion_reset_prob)!=0.0:
            raise ValueError("phase experiments require stand_start_prob=1 and motion_reset_prob=0")
        if int(cfg.stand_start_transition_min_steps)!=int(cfg.stand_start_transition_max_steps):
            raise ValueError("deployable phase requires a fixed stand-start transition length")
        motion=base_env.motion
        if hasattr(motion,"motions"):
            if len(motion.motions)!=1:raise ValueError("deployable phase v1 supports one fixed motion")
            motion=motion.motions[0]
        self.phase_clock=DeployablePhaseClock(
            frame_count=float(motion.frame_count),frequency=float(motion.frequency),
            control_dt=float(base_env.dt),start_phase=float(cfg.stand_start_motion_phase),
            num_envs=base_env.num_envs,
            stand_start_phase_delay_steps=int(cfg.stand_start_phase_delay_steps),
            stand_start_transition_steps=int(cfg.stand_start_transition_min_steps),
            torch_device=base_env.device if bool(getattr(base_env,"is_torch_env",False)) else None)

    @property
    def sensor_frame_dim(self):return self._phase_base_dim+2
    @property
    def obs_dim(self):return self._phase_base_dim+2
    @property
    def observation_metadata(self):
        base=dict(super().observation_metadata)
        fields=tuple(base["actor_input_fields"])+("gait_phase_sin","gait_phase_cos")
        base.update({"sensor_version":f"{base['sensor_version']}-phase-v1",
          "actor_frame_dim":self.obs_dim,"actor_obs_dim":self.obs_dim,
          "actor_input_fields":fields,"phase_dim":2,"phase_fields":["gait_phase_sin","gait_phase_cos"],
          "phase_clock":self.phase_clock.metadata,"reference_tendon_actor_input":False})
        return base
    def _allocate_histories(self):
        shape=(self.num_envs,1,self.obs_dim);amp=(self.num_envs,1,self.base_motion.feature_dim)
        if self.is_torch_env:
            self._actor_history=torch.zeros(shape,device=self.base_env.device);self._amp_history=torch.zeros(amp,device=self.base_env.device)
        else:
            self._actor_history=np.zeros(shape,np.float32);self._amp_history=np.zeros(amp,np.float32)
    def _normalized_sensor_frame(self):
        sensor=super()._normalized_sensor_frame();phase=self.phase_clock.observe()
        return torch.cat((sensor,phase),1) if self.is_torch_env else np.concatenate((sensor,phase),1).astype(np.float32)
    def reset(self,env_ids=None):
        observation=super().reset(env_ids);ids=self._all_env_ids() if env_ids is None else env_ids
        self.phase_clock.reset(ids);self._fill_history(ids);return self.observe()
    def step(self,action):
        self.phase_clock.advance()
        observation,reward,done,info=super().step(action)
        ids=torch.nonzero(done>0,as_tuple=False).flatten() if self.is_torch_env else np.nonzero(np.asarray(done)>0)[0].astype(np.int64)
        if len(ids)>0:
            self.phase_clock.reset(ids);self._fill_history(ids);observation=self.observe()
        # Under the fixed reset/startup schedule the simulator reference and
        # controller clock must remain the same state, even though the default
        # sensor-stability task reward is phase independent.
        if self.is_torch_env:
            error=torch.remainder(self.phase_clock.phase-self.base_env.motion_phase+self.phase_clock.config.frame_count/2,self.phase_clock.config.frame_count)-self.phase_clock.config.frame_count/2
            aligned=bool((error.abs()<2e-4).all())
        else:
            error=np.mod(self.phase_clock.phase-self.base_env.motion_phase+self.phase_clock.config.frame_count/2,self.phase_clock.config.frame_count)-self.phase_clock.config.frame_count/2
            aligned=bool(np.all(np.abs(error)<2e-4))
        if not aligned:raise RuntimeError("deployable phase clock diverged from simulator reward phase")
        return observation,reward,done,info


class PhaseNoForceEnv(_PhaseObservationMixin,NoForceEnv):
    _phase_base_dim=56
    def __init__(self,base_env):self._initialize_phase_clock(base_env);super().__init__(base_env)
class PhaseFootForceEnv(_PhaseObservationMixin,FootForceEnv):
    _phase_base_dim=62
    def __init__(self,base_env,sensor_config=FootForceSensorConfig()):self._initialize_phase_clock(base_env);super().__init__(base_env,sensor_config)
class PhaseNoForceCpuAmpEnv(PhaseNoForceEnv):
    def __init__(self,cfg,device='cpu'):
        if device!='cpu':raise ValueError('PhaseNoForceCpuAmpEnv only supports device=cpu')
        from lytrobot_amp.env import LytRobotAmpEnv
        super().__init__(LytRobotAmpEnv(cfg))
class PhaseNoForceWarpAmpEnv(PhaseNoForceEnv):
    def __init__(self,cfg,device='cuda'):
        from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
        super().__init__(LytRobotWarpAmpEnv(cfg,device=device))
class PhaseFootForceCpuAmpEnv(PhaseFootForceEnv):
    def __init__(self,cfg,device='cpu'):
        if device!='cpu':raise ValueError('PhaseFootForceCpuAmpEnv only supports device=cpu')
        from lytrobot_amp.env import LytRobotAmpEnv
        super().__init__(LytRobotAmpEnv(cfg))
class PhaseFootForceWarpAmpEnv(PhaseFootForceEnv):
    def __init__(self,cfg,device='cuda'):
        from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
        super().__init__(LytRobotWarpAmpEnv(cfg,device=device))
