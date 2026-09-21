#!/usr/bin/env python3
"""58D IMU+tendon+deployable-phase ablation."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
for path in (root,root/'src'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from scripts import train as t
from foot_sensor_training.envs import PhaseNoForceCpuAmpEnv,PhaseNoForceWarpAmpEnv
from sensor_training.ppo_config import SensorPPOAMPConfig
from sensor_training.trainer import SensorHistoryPPOAMPTrainer
def has(x):return any(a==x or a.startswith(x+'=') for a in sys.argv)
if __name__=='__main__':
    defaults={'--stage':'stage1','--sim-backend':'warp','--device':'cuda','--motion-file':str(root/'robot/reference_data/walk1_mjcf.npz'),'--reference-ctrl-blend':'0.0','--no-domain-randomization':'','--sensor-stability-reward':'','--stand-start-prob':'1.0','--motion-reset-prob':'0.0','--stand-start-transition-min-steps':'120','--stand-start-transition-max-steps':'120','--log-root':str(root/'logs/foot_sensor_phase')}
    for k,v in defaults.items():
        if not has(k):sys.argv.extend([k] if not v else [k,v])
    t.LytRobotAmpEnv=PhaseNoForceCpuAmpEnv;t.LytRobotWarpAmpEnv=PhaseNoForceWarpAmpEnv
    t.PPOAMPConfig=SensorPPOAMPConfig;t.PPOAMPTrainer=SensorHistoryPPOAMPTrainer;t.main()
