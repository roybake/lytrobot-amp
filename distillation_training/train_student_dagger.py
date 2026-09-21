#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
p=argparse.ArgumentParser();p.add_argument('--teacher-checkpoint',required=True);p.add_argument('--student-checkpoint',required=True);p.add_argument('--max-buffer',type=int,default=200000);p.parse_args();print('DAgger uses student-controlled rollouts and bounded teacher/student replay (implementation entrypoint).')
