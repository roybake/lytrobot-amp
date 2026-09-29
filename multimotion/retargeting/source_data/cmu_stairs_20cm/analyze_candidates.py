#!/usr/bin/env python3
"""Extract reproducible FK keypoints and stance candidates from downloaded CMU files."""
from pathlib import Path
import sys, json
import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_closing
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'squat_training/retargeting'))
from prepare_cmu_squat import convert, parse_amc, JOINT_NAMES
OUT=Path(__file__).resolve().parent
summaries={}
for name in ['143_17','113_19','113_20','141_07','141_08']:
 n=len(parse_amc(OUT/f'{name}.amc'))
 p,h=convert(OUT/f'{name.split("_")[0]}.asf',OUT/f'{name}.amc',0,n-1,1)
 np.savez_compressed(OUT/f'{name}_keypoints_120hz.npz',global_joint_positions=p,height=np.float32(h),fps=np.float32(120.),joint_names=np.asarray(JOINT_NAMES),source=np.asarray(f'CMU MoCap {name}'),source_frames=np.array([0,n-1,1]))
 # Orient progress by first principal component, preserving height and handedness.
 roots=p[:,0,:2]; delta=roots[-1]-roots[0]; _,_,v=np.linalg.svd(roots-roots.mean(0),full_matrices=False);forward=v[0]
 if forward@delta<0:forward=-forward
 lateral=np.array([-forward[1],forward[0]])
 p2=p.copy();p2[...,0]=p[...,:2]@forward;p2[...,1]=p[...,:2]@lateral
 ground=float(np.quantile(p[:,[4,8],2],.02)); p2[...,2]-=ground
 report={'frames':n,'duration_s':(n-1)/120,'hip_height_m':h,'ground_toe_estimate_m':ground,'forward_xy':forward.tolist(),'root_progress_m':[float(np.min(p2[:,0,0])),float(np.max(p2[:,0,0]))],'root_height_m':[float(np.min(p2[:,0,2])),float(np.max(p2[:,0,2]))],'contacts':[]}
 for side,idx in [('R',4),('L',8)]:
  sm=gaussian_filter1d(p2[:,idx,:],1.5,axis=0);vel=np.gradient(sm,1/120,axis=0)
  stable=(np.linalg.norm(vel,axis=-1)<.32)
  stable=binary_closing(stable,structure=np.ones(5))
  edges=np.diff(np.r_[False,stable,False].astype(int)); starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
  for a,b in zip(starts,ends):
   if b-a<10:continue
   report['contacts'].append({'side':side,'start_frame':int(a),'end_frame_exclusive':int(b),'duration_s':float((b-a)/120),'xyz_m':np.median(p2[a:b,idx],axis=0).round(4).tolist(),'root_x_m':float(np.median(p2[a:b,0,0]))})
 report['contacts'].sort(key=lambda q:q['start_frame'])
 summaries[name]=report
 print(name,json.dumps(report),flush=True)
(OUT/'candidate_analysis.json').write_text(json.dumps(summaries,indent=2)+'\n')
