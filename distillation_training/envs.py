"""64D deployable frame plus training-only, interpretable 20D target."""
import torch, numpy as np
from foot_sensor_training.envs import PhaseFootForceEnv
from .config import PRIVILEGED_FIELDS
class DistillationEnv(PhaseFootForceEnv):
    @property
    def privileged_schema(self): return {"fields":PRIVILEGED_FIELDS,"dim":20,"actor_input":False,"reference_fields":False}
    def privileged_vector(self):
        b=self.base_env
        if self.is_torch_env:
            rotation=b.xmat[:,b.pelvis_body_id]
            vel=torch.bmm(rotation.transpose(1,2),b.qvel[:,:3,None]).squeeze(-1).clamp(-3,3)/3
            world_down=torch.tensor([0.,0.,-1.],device=b.device).expand(b.num_envs,3)
            gravity=torch.bmm(rotation.transpose(1,2),world_down[:,:,None]).squeeze(-1)
            # real tendon rate is simulator-only teacher target, never actor input
            rate=b.actuator_velocity[:,b.position_actuator_ids].clamp(-10,10)/10
            return torch.cat((vel,gravity,rate),1)
        vel=np.stack([d.xmat[b.pelvis_body_id].reshape(3,3).T@d.qvel[:3] for d in b.datas]).clip(-3,3)/3
        gravity=np.stack([d.xmat[b.pelvis_body_id].reshape(3,3).T@np.array([0,0,-1],np.float32) for d in b.datas])
        rate=np.stack([d.actuator_velocity[b.position_actuator_ids] for d in b.datas]).clip(-10,10)/10
        return np.concatenate((vel,gravity,rate),1).astype(np.float32)

class DistillationCpuEnv(DistillationEnv):
    def __init__(self,cfg):
        from lytrobot_amp.env import LytRobotAmpEnv
        super().__init__(LytRobotAmpEnv(cfg))

class DistillationWarpEnv(DistillationEnv):
    def __init__(self,cfg,device='cuda'):
        from lytrobot_amp.env_warp import LytRobotWarpAmpEnv
        super().__init__(LytRobotWarpAmpEnv(cfg,device=device))
