from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;source=HERE/'solve_force_scaled.py';ns={'__file__':str(source),'__name__':'derivative_scale'}
sys.argv=[str(source),'--nodes','80','--name','derivative_scale','--initial-solution',str(HERE/'force80/solution.npz')]
text=source.read_text().split('initial=evaluate')[0]
# Define a tunable physical finite-difference size for the native tangent map.
text=text.replace('mjd_transitionFD(m,d,1e-6,True','mjd_transitionFD(m,d,native_eps,True')
ns['native_eps']=1e-6
exec(compile(text,str(source),'exec'),ns)
np=ns['np'];cache=ns['cache'];evaluate=ns['evaluate'];x=ns['x0'].ravel()
direction=np.random.default_rng(42).normal(size=x.size);direction/=np.linalg.norm(direction)
cache.clear();fp=evaluate(x+1e-6*direction,False)['fun'].copy();cache.clear();fm=evaluate(x-1e-6*direction,False)['fun'].copy();fd=(fp-fm)/2e-6
rows=[]
for eps in (1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9):
 ns['native_eps']=eps;cache.clear();jac=evaluate(x,True)['jac'].copy();pred=jac@direction
 row={'eps':eps,'relative_error':float(np.linalg.norm(fd-pred)/np.linalg.norm(fd)),'predicted_norm':float(np.linalg.norm(pred)),'fd_norm':float(np.linalg.norm(fd)),'max_J_entry':float(abs(jac.data).max())};rows.append(row);print(json.dumps(row),flush=True)
(HERE/'derivative_scale.json').write_text(json.dumps(rows,indent=2)+'\n')
