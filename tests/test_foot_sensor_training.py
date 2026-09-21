import numpy as np
import pytest
import mujoco
from types import SimpleNamespace

from lytrobot_amp.foot_force import (CpuFootForceSensor,FootForceSensorConfig,
  FootForceSensorDynamics,FOOT_FORCE_NAMES)
from foot_sensor_training.envs import FootForceEnv


@pytest.mark.parametrize("field,value",[
    ("force_clip",0),("noise_std",-1),("gain_randomization",1),
    ("bias_randomization",-1),("lowpass_alpha",0),("delay_steps",-1),
    ("distribution_sigma",0),
])
def test_invalid_force_dynamics_config_raises(field,value):
    values={field:value}
    with pytest.raises(ValueError):FootForceSensorConfig(**values)


@pytest.mark.parametrize("delay,expected",[
    (0,[1,2,3,4]),(1,[0,1,2,3]),(2,[0,0,1,2]),
])
def test_delay_has_exact_filtered_t_minus_n_semantics(delay,expected):
    dynamics=FootForceSensorDynamics(1,FootForceSensorConfig(delay_steps=delay))
    actual=[float(dynamics.apply(np.full((1,6),x,np.float32))[0,0]) for x in (1,2,3,4)]
    assert actual==expected


def test_gain_bias_lowpass_nonnegative_and_reset_clear():
    dynamics=FootForceSensorDynamics(2,FootForceSensorConfig(lowpass_alpha=.5,delay_steps=1))
    dynamics.gain[:]=2;dynamics.bias[:]=1
    first=dynamics.apply(np.ones((2,6),np.float32));second=dynamics.apply(np.ones((2,6),np.float32))
    assert np.all(first==0) and np.allclose(second,1.5)
    dynamics.reset([0]);after=dynamics.apply(np.ones((1,6),np.float32),[0])
    assert np.all(after==0)
    assert np.all(dynamics.filtered>=0)


def test_force_channel_order_is_stable_left_then_right():
    assert FOOT_FORCE_NAMES==(
      "left_heel_force","left_forefoot_medial_force","left_forefoot_lateral_force",
      "right_heel_force","right_forefoot_medial_force","right_forefoot_lateral_force")


def test_repeated_observation_reads_do_not_advance_filter_or_delay():
    env=FootForceEnv.__new__(FootForceEnv)
    env._force_sensor_output=np.arange(12,dtype=np.float32).reshape(2,6)
    first=env._read_foot_force_sensor().copy();second=env._read_foot_force_sensor().copy()
    np.testing.assert_array_equal(first,second)


def test_partial_autoreset_clears_only_selected_environment_state():
    dynamics=FootForceSensorDynamics(2,FootForceSensorConfig(delay_steps=2))
    dynamics.apply(np.ones((2,6),np.float32));dynamics.apply(np.ones((2,6),np.float32)*2)
    other_filtered=dynamics.filtered[1].copy();other_delay=dynamics.delay[1].copy()
    dynamics.reset([0])
    assert np.all(dynamics.filtered[0]==0) and np.all(dynamics.delay[0]==0)
    np.testing.assert_array_equal(dynamics.filtered[1],other_filtered)
    np.testing.assert_array_equal(dynamics.delay[1],other_delay)


def test_noise_is_applied_and_output_remains_nonnegative():
    dynamics=FootForceSensorDynamics(1,FootForceSensorConfig(noise_std=.25),seed=7)
    samples=np.stack([dynamics.apply(np.ones((1,6),np.float32))[0] for _ in range(8)])
    assert np.var(samples)>0 and np.all(samples>=0)


def _fake_cpu_sensor():
    sensor=CpuFootForceSensor.__new__(CpuFootForceSensor)
    sensor.config=FootForceSensorConfig(distribution_sigma=.02)
    sensor.terrain_geom_ids={0};sensor.geom_side={10:0,20:1};sensor.geom_body={10:1,20:2}
    sensor.anchors=np.asarray([
      [[0,0],[1,-.1],[1,.1]],[[0,0],[1,.1],[1,-.1]]],np.float32)
    return sensor


def _fake_data(contacts):
    return SimpleNamespace(ncon=len(contacts),contact=contacts,
      xmat=np.asarray([np.eye(3),np.eye(3),np.eye(3)],np.float32).reshape(3,9),
      xpos=np.zeros((3,3),np.float32))


def test_cpu_force_no_contact_is_zero():
    np.testing.assert_array_equal(_fake_cpu_sensor().raw(_fake_data([])),np.zeros(6,np.float32))


def test_cpu_support_force_is_nonnegative_and_left_right_mapping_is_stable(monkeypatch):
    forces=iter((120.0,80.0))
    def contact_force(_model,_data,_index,result):
        result[:]=0;result[0]=next(forces)
    monkeypatch.setattr(mujoco,"mj_contactForce",contact_force)
    contacts=[SimpleNamespace(geom1=0,geom2=10,pos=np.asarray([0.,0.,0.])),
              SimpleNamespace(geom1=20,geom2=0,pos=np.asarray([1.,.1,0.]))]
    sensor=_fake_cpu_sensor();sensor.model=object();force=sensor.raw(_fake_data(contacts))
    assert np.all(force>=0) and np.isclose(force[:3].sum(),120) and np.isclose(force[3:].sum(),80)
    assert force[0]>force[1] and force[0]>force[2]
    assert force[4]>force[3] and force[4]>force[5]
