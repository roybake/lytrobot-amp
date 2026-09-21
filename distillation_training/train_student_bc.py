#!/usr/bin/env python3
import argparse,torch,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
from distillation_training.networks import SharedActor,HistoryEstimator
p=argparse.ArgumentParser();p.add_argument('--teacher-checkpoint',required=True);p.add_argument('--output',default='student_bc.pt');a=p.parse_args()
ck=torch.load(a.teacher_checkpoint,map_location='cpu'); actor=SharedActor();actor.load_state_dict(ck['shared_actor']);
for x in actor.parameters():x.requires_grad=False
torch.save({'history_estimator':HistoryEstimator().state_dict(),'teacher_checkpoint':a.teacher_checkpoint,'validation_required':True},a.output)
print('Initialized frozen-teacher BC checkpoint; run dataset/validation loop before DAgger.')
