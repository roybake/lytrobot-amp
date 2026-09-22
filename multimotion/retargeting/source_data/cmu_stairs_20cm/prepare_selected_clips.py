#!/usr/bin/env python3
"""Make +X-facing source clips; no synthetic motion or terrain edits are applied."""
from pathlib import Path
import json,numpy as np
OUT=Path(__file__).resolve().parent
meta=json.loads((OUT/'candidate_analysis.json').read_text())
clips=[('143_17','up_two_steps',0,155),('143_17','up_single_step',53,100),('141_08','down_two_steps',169,251),('141_08','down_single_step',215,245)]
rows=[]
for source,label,a,b in clips:
 data=np.load(OUT/f'{source}_keypoints_120hz.npz');p=data['global_joint_positions'][a:b+1].copy();f=np.asarray(meta[source]['forward_xy']);left=np.array([-f[1],f[0]])
 p0=p.copy();p[...,0]=p0[...,:2]@f;p[...,1]=p0[...,:2]@left;p[...,2]-=meta[source]['ground_toe_estimate_m'];offset=p[0,0,:2].copy();p[...,:2]-=offset
 path=OUT/f'{source}_{label}_aligned_120hz.npz'
 np.savez_compressed(path,global_joint_positions=p,joint_names=data['joint_names'],fps=np.float32(120.),height=np.float32(meta[source]['hip_height_m']),source=np.asarray(f'CMU MoCap {source}'),source_frames=np.array([a,b,1]),original_forward_xy=f,ground_toe_estimate_m=np.float32(meta[source]['ground_toe_estimate_m']),rotation_note=np.asarray('Rigid XY rotation only; Z shifted by estimated ground toe height. Toe keypoint is not robot sole.'))
 rows.append({'file':path.name,'source':source,'clip':label,'start_frame_zero_based':a,'end_frame_zero_based_inclusive':b,'source_fps':120,'original_duration_s':(b-a)/120,'original_forward_xy':f.tolist(),'root_progress_m':float(p[-1,0,0]-p[0,0,0]),'root_height_change_m':float(p[-1,0,2]-p[0,0,2]),'recommended_step_duration_s':0.7 if 'single' in label else None,'notes':'Use as measured human phase template, adapt stance anchors to exact target terrain with collision-constrained retargeting; do not assume source bones equal robot soles.'})
(OUT/'selected_clips.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
