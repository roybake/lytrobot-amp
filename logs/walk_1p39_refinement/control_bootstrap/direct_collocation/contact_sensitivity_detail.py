"""Locate the native geometry sources of finite-difference discontinuity."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;source=HERE/'solve_force_scaled.py'
ns={'__file__':str(source),'__name__':'geometry_analysis'}
sys.argv=[str(source),'--nodes','80','--name','geometry_detail','--initial-solution',str(HERE/'force80/solution.npz')]
exec(compile(source.read_text().split('initial=evaluate')[0],str(source),'exec'),ns)
np=ns['np'];mj=ns['mujoco'];m=ns['m'];cache=ns['cache'];evaluate=ns['evaluate'];decode=ns['decode'];x=ns['x0'].ravel()
direction=np.random.default_rng(42).normal(size=x.size);direction/=np.linalg.norm(direction);eps=1e-6
cache.clear();fp=evaluate(x+eps*direction,False)['fun'].copy();cache.clear();fm=evaluate(x-eps*direction,False)['fun'].copy();delta=(fp-fm)[:80*44].reshape(80,44)
frames=np.argsort(abs(delta).max(axis=1))[-5:]
qp,vp,cp=decode(x+eps*direction);qm,vm,cm=decode(x-eps*direction)
for frame in frames:
 info=[]
 for sign,qs,vs,cs in [('plus',qp,vp,cp),('minus',qm,vm,cm)]:
  d=mj.MjData(m);d.qpos[:]=qs[frame];d.qvel[:]=vs[frame];d.ctrl[:]=cs[frame];mj.mj_forward(m,d)
  info.append({'sign':sign,'lengths':d.actuator_length.tolist(),'actuator_forces':d.actuator_force.tolist(),'qacc':d.qacc.tolist(),'contacts':[(int(c.geom1),int(c.geom2),float(c.dist),c.pos.tolist(),c.frame[:3].tolist()) for c in d.contact]})
 print(json.dumps({'frame':int(frame),'residual_difference_max':float(abs(delta[frame]).max()),'pose_difference_max':float(abs(qp[frame]-qm[frame]).max()),'native':info}),flush=True)
