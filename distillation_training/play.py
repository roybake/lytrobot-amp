#!/usr/bin/env python3
import argparse,sys,torch,copy
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if str(root) not in sys.path: sys.path.insert(0,str(root))
from distillation_training.networks import SharedActor,HistoryEstimator,StudentExport
from distillation_training.runtime import (inherit_checkpoint_variant,make_env,
 tensor,env_step,validate_checkpoint)
from distillation_training.config import ACTION_DIM,HIDDEN_DIM,PHASE_FORCE,PHASE_NO_FORCE,RECURRENT_STATE_DIM,sensor_variant_spec
def export_onnx(actor,estimator,path,phase_clock_metadata=None,sensor_variant=None):
 inferred={PHASE_NO_FORCE.sensor_dim:PHASE_NO_FORCE.name,PHASE_FORCE.sensor_dim:PHASE_FORCE.name}
 if sensor_variant is None and actor.sensor_dim not in inferred:raise ValueError(f'unsupported actor sensor dimension: {actor.sensor_dim}')
 spec=sensor_variant_spec(sensor_variant or inferred[actor.sensor_dim])
 model=StudentExport(copy.deepcopy(actor),copy.deepcopy(estimator),spec.sensor_dim).cpu().eval();torch.onnx.export(model,(torch.zeros(1,spec.sensor_dim),torch.zeros(1,ACTION_DIM),torch.zeros(1,RECURRENT_STATE_DIM),torch.ones(1)),path,input_names=('observation','previous_action','recurrent_state','episode_start'),output_names=('action','next_recurrent_state'),dynamic_axes={name:{0:'batch'} for name in ('observation','previous_action','recurrent_state','episode_start','action','next_recurrent_state')},opset_version=17,dynamo=False)
 import onnx,json
 metadata={'sensor_variant':spec.name,'sensor_dim':spec.sensor_dim,'imu_dim':42,'tendon_dim':14,'foot_force_dim':6 if spec.use_foot_force else 0,'phase_dim':2,'phase_fields':['gait_phase_sin','gait_phase_cos'],'action_dim':ACTION_DIM,'estimator_input_dim':spec.estimator_input_dim,'history_estimator_input_dim':spec.estimator_input_dim,'hidden_dim':HIDDEN_DIM,'recurrent_hidden_dim':HIDDEN_DIM,'recurrent_state_dim':RECURRENT_STATE_DIM,'privileged_actor_input':False,'amp_actor_input':False,'reference_tendon_actor_input':False,'foot_force_sensor_version':'virtual-plantar-force-v1' if spec.use_foot_force else None,'phase_clock':phase_clock_metadata or {}}
 graph=onnx.load(path);onnx.helper.set_model_props(graph,{'lytrobot_amp.metadata':json.dumps(metadata)});onnx.checker.check_model(graph);onnx.save(graph,path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--teacher-checkpoint');p.add_argument('--oracle-privileged',action='store_true');p.add_argument('--steps',type=int,default=1000);p.add_argument('--export-onnx');p.add_argument('--num-envs',type=int,default=1);p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--sim-backend',choices=('cpu','warp'),default='cpu');p.add_argument('--seed',type=int,default=42);a=p.parse_args();device=torch.device(a.device)
 ck=validate_checkpoint(a.checkpoint,'student');spec=inherit_checkpoint_variant(a,('student',ck));actor=SharedActor(spec.sensor_dim).to(device);actor.load_state_dict(ck['shared_actor']);estimator=HistoryEstimator(spec.sensor_dim).to(device);estimator.load_state_dict(ck['history_estimator']);actor.eval();estimator.eval()
 if a.export_onnx:export_onnx(actor,estimator,a.export_onnx,ck['sensor_metadata']['phase_clock'],spec)
 env,_=make_env(a);obs=tensor(env.reset(),device);pa=torch.zeros(a.num_envs,14,device=device);pt=obs[:,42:56].clone();hidden=torch.zeros(a.num_envs,128,device=device);start=torch.ones(a.num_envs,device=device)
 teacher_actor=None
 if a.oracle_privileged:
  if not a.teacher_checkpoint:p.error('--oracle-privileged requires --teacher-checkpoint')
  teacher=validate_checkpoint(a.teacher_checkpoint,'teacher',spec);teacher_actor=SharedActor(spec.sensor_dim).to(device);teacher_actor.load_state_dict(teacher['shared_actor']);teacher_actor.eval()
 for _ in range(a.steps):
  with torch.no_grad():
   if a.oracle_privileged:
    truth=tensor(env.privileged_vector(),device);action=actor.mean(obs,truth);torch.testing.assert_close(teacher_actor.mean(obs,truth),action,rtol=1e-5,atol=1e-6)
   else:pred,hidden=estimator(obs,pa,pt,hidden,start);action=actor.mean(obs,pred)
  nxt,_,done,_=env_step(env,action);pt=torch.where(done[:,None]>0,nxt[:,42:56],obs[:,42:56]);pa=torch.where(done[:,None]>0,torch.zeros_like(action),action);start=done;obs=nxt
 env.close();print('oracle action comparison passed' if a.oracle_privileged else 'student playback completed')
if __name__=='__main__':main()
