#!/usr/bin/env python3
"""Launch independent foot-force ablation with current-tendon action semantics."""
from pathlib import Path
import runpy,sys
root=Path(__file__).resolve().parents[1]
for p in (root,root/'src'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from scripts import train as t
from foot_sensor_training.envs import FootForceCpuAmpEnv,FootForceWarpAmpEnv
from sensor_training.ppo_config import SensorPPOAMPConfig
from sensor_training.trainer import SensorHistoryPPOAMPTrainer
def has(x): return any(a==x or a.startswith(x+'=') for a in sys.argv)
if __name__=='__main__':
    defaults={'--stage':'stage1','--sim-backend':'warp','--device':'cuda','--motion-file':str(root/'robot/reference_data/walk1_mjcf.npz'),'--reference-ctrl-blend':'0.0','--no-domain-randomization':'','--log-root':str(root/'logs/foot_sensor')}
    for k,v in defaults.items():
        if not has(k): sys.argv.extend([k] if not v else [k,v])
    t.LytRobotAmpEnv=FootForceCpuAmpEnv;t.LytRobotWarpAmpEnv=FootForceWarpAmpEnv;t.PPOAMPConfig=SensorPPOAMPConfig;t.PPOAMPTrainer=SensorHistoryPPOAMPTrainer;t.main()
