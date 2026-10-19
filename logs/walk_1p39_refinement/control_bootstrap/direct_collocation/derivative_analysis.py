"""Numerical analysis of native dynamics sensitivities at contact transitions."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
source=HERE/'solve_centered.py'
namespace={'__file__':str(source),'__name__':'sensitivity_analysis'}
sys.argv=[str(source),'--nodes','80','--name','sensitivity','--initial-solution',str(HERE/'centered80/solution.npz')]
exec(compile(source.read_text().split('initial=evaluate')[0],str(source),'exec'),namespace)
np=namespace['np'];mj=namespace['mujoco'];m=namespace['m'];evaluate=namespace['evaluate'];cache=namespace['cache'];x=namespace['x0'].ravel()
def value(xx):cache.clear();return evaluate(xx,False)['fun'].copy()
for disable in (False,True):
 if disable:m.opt.disableflags|=int(mj.mjtDisableBit.mjDSBL_WARMSTART)
 same1=value(x);xx=x.copy();xx[0]+=1e-5;value(xx);same2=value(x)
 direction=np.random.default_rng(42).normal(size=x.size);direction/=np.linalg.norm(direction)
 cache.clear();jac=evaluate(x,True)['jac'].copy();analytic=jac@direction
 rows=[]
 for eps in (1e-3,1e-4,1e-5,1e-6,1e-7):
  fd=(value(x+eps*direction)-value(x-eps*direction))/(2*eps)
  rows.append({'eps':eps,'relative_error':float(np.linalg.norm(fd-analytic)/np.linalg.norm(fd)),'fd_norm':float(np.linalg.norm(fd)),'analytic_norm':float(np.linalg.norm(analytic))})
 print(json.dumps({'warmstart_disabled':disable,'identical_input_difference':float(abs(same2-same1).max()),'jacobian':rows}),flush=True)
