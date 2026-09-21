"""Playback a 62D foot-force policy (PyTorch or exported ONNX)."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
for path in (root,root/'src'):
    if str(path) not in sys.path: sys.path.insert(0,str(path))
from scripts import play as base_play
from foot_sensor_training.envs import FootForceCpuAmpEnv

def main():
    if not any(x=='--stand-start-prob' or x.startswith('--stand-start-prob=') for x in sys.argv[1:]):
        sys.argv.extend(['--stand-start-prob','1.0'])
    base_play.LytRobotAmpEnv=FootForceCpuAmpEnv
    base_play.main()
if __name__=='__main__': main()
