"""Playback entrypoint; uses the same 62D virtual-sensor wrapper."""
from pathlib import Path
import runpy,sys
root=Path(__file__).resolve().parents[1]; sys.argv[0]=str(root/'sensor_training/play.py')
runpy.run_path(sys.argv[0],run_name='__main__')
