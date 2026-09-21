"""62D physical frame plus training-only, interpretable 20D privileged target."""
import torch, numpy as np
from foot_sensor_training.envs import FootForceEnv,FootForceCpuAmpEnv,FootForceWarpAmpEnv
from .config import PRIVILEGED_FIELDS
class DistillationEnv(FootForceEnv):
    @property
    def privileged_schema(self): return {"fields":PRIVILEGED_FIELDS,"dim":20,"actor_input":False,"reference_fields":False}
    def privileged_vector(self):
        b=self.base_env
        if self.is_torch_env:
            vel=b.qvel[:,:3].clamp(-3,3)/3; # local gravity from pelvis orientation
            gravity=torch.matmul(b.xmat[:,b.pelvis_body_id].transpose(1,2),torch.tensor([0.,0.,-1.],device=b.device).expand(b.num_envs,3))
            # real tendon rate is simulator-only teacher target, never actor input
            rate=b.actuator_velocity[:,b.position_actuator_ids].clamp(-10,10)/10
            return torch.cat((vel,gravity,rate),1)
        vel=np.stack([d.qvel[:3] for d in b.datas]).clip(-3,3)/3
        gravity=np.stack([d.xmat[b.pelvis_body_id].reshape(3,3).T@np.array([0,0,-1],np.float32) for d in b.datas])
        rate=np.stack([d.actuator_velocity[b.position_actuator_ids] for d in b.datas]).clip(-10,10)/10
        return np.concatenate((vel,gravity,rate),1).astype(np.float32)
