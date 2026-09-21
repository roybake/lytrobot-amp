#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--oracle-privileged',action='store_true');a=p.parse_args();print('Oracle privileged evaluation' if a.oracle_privileged else 'Recurrent student evaluation')
