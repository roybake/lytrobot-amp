import numpy as np

from foot_sensor_training.phase_clock import DeployablePhaseClock


def make_clock(**kwargs):
    return DeployablePhaseClock(
        frame_count=100.0, frequency=50.0, control_dt=0.01,
        stand_start_phase_delay_steps=0, stand_start_transition_steps=1,
        **kwargs,
    )


def test_reset_is_unit_circle_and_fixed_phase():
    clock=make_clock(start_phase=17.0,num_envs=3)
    phase=clock.observe()
    np.testing.assert_allclose((phase**2).sum(1),1.0,atol=1e-6)
    clock.advance(progress=np.ones(3,np.float32));clock.reset([1])
    assert clock.phase[1]==17.0 and clock.step_count[1]==0


def test_advance_formula_and_wraparound():
    clock=make_clock(start_phase=99.75)
    clock.advance(progress=np.ones(1,np.float32))
    np.testing.assert_allclose(clock.phase,[0.25],atol=1e-6)
    theta=2*np.pi*.25/100
    np.testing.assert_allclose(clock.observe()[0],[np.sin(theta),np.cos(theta)],atol=1e-6)


def test_numpy_and_torch_clocks_match_for_1000_steps():
    simulation=make_clock(num_envs=2)
    deployment=make_clock(num_envs=2,torch_device="cpu")
    for _ in range(1000):
        np.testing.assert_allclose(simulation.observe(),deployment.observe().numpy(),atol=2e-6)
        simulation.advance();deployment.advance()
    np.testing.assert_allclose(simulation.phase,deployment.phase.numpy(),atol=2e-5)


def test_startup_delay_and_smoothstep_progress():
    clock=DeployablePhaseClock(100,50,.01,num_envs=1,
      stand_start_phase_delay_steps=2,stand_start_transition_steps=4)
    observed=[]
    for _ in range(7):
        observed.append(float(clock.startup_progress()[0]));clock.advance()
    np.testing.assert_allclose(observed,[0,0,0,.15625,.5,.84375,1])
