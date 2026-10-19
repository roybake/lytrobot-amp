from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;source=HERE/'solve_native.py';ns={'__file__':str(source),'__name__':'cache_analysis'}
sys.argv=[str(source),'--nodes','80','--name','cache_analysis','--initial-solution',str(HERE/'force80/solution.npz')]
exec(compile(source.read_text().split('initial=evaluate')[0],str(source),'exec'),ns)
np=ns['np'];mj=ns['mujoco'];m=ns['m'];cache=ns['cache'];evaluate=ns['evaluate'];x=ns['x0'].ravel()
cache.clear();f0=evaluate(x,False)['fun'].copy();x_before=x.copy();qbase=ns['qbase'].copy();
cache.clear();jac=evaluate(x,True)['jac'].copy();cache.clear();f1=evaluate(x,False)['fun'].copy()
print(json.dumps({'fun_after_jac_change':float(abs(f1-f0).max()),'parameters_after_jac_change':float(abs(x-x_before).max()),'base_after_jac_change':float(abs(qbase-ns['qbase']).max()),'model_timestep':m.opt.timestep,'ccd_tolerance':m.opt.ccd_tolerance}),flush=True)
direction=np.random.default_rng(42).normal(size=x.size);direction/=np.linalg.norm(direction);eps=1e-6
cache.clear();plus=evaluate(x+eps*direction,False)['fun'].copy();cache.clear();minus=evaluate(x-eps*direction,False)['fun'].copy();fd=(plus-minus)/(2*eps)
print(json.dumps({'fd_norm':float(np.linalg.norm(fd)),'J_norm':float(np.linalg.norm(jac@direction)),'relative_error':float(np.linalg.norm(fd-jac@direction)/np.linalg.norm(fd))}),flush=True)
