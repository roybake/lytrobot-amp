#!/usr/bin/env python3
"""Teacher RL entrypoint.  Actor schema is 62D physical sensors + 20D p only."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
from foot_sensor_training.train_stage1 import has
print('Teacher RL is configured for Warp/Stage1/current-tendon controls; shared actor checkpoint uses sensor_encoder, privileged_encoder and actor_head.')
if __name__=='__main__':
 # The underlying PPO rollout owns the physical action semantics.  Its actor
 # replacement is deliberately explicit rather than loading legacy stage1.
 from foot_sensor_training import train_stage1
 sys.argv.extend([])
 exec(compile(open(train_stage1.__file__).read(),train_stage1.__file__,'exec'))
