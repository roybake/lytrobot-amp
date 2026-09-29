import pytest
import torch

from lytrobot_amp.foot_force import FootForceSensorConfig,FootForceSensorDynamics


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA integration test; no CPU fallback")


def test_gpu_partial_autoreset_clears_only_done_environment():
    dynamics=FootForceSensorDynamics(5,FootForceSensorConfig(delay_steps=2),torch_device="cuda")
    dynamics.apply(torch.ones((5,6),device="cuda"))
    dynamics.apply(torch.arange(30,device="cuda",dtype=torch.float32).reshape(5,6)+1)
    filtered=dynamics.filtered.clone();delay=dynamics.delay.clone();gain=dynamics.gain.clone();bias=dynamics.bias.clone()
    dynamics.reset(torch.tensor([3],device="cuda"))
    keep=torch.tensor([0,1,2,4],device="cuda")
    torch.testing.assert_close(dynamics.filtered[keep],filtered[keep])
    torch.testing.assert_close(dynamics.delay[keep],delay[keep])
    torch.testing.assert_close(dynamics.gain[keep],gain[keep])
    torch.testing.assert_close(dynamics.bias[keep],bias[keep])
    assert torch.count_nonzero(dynamics.filtered[3])==0
    assert torch.count_nonzero(dynamics.delay[3])==0
    assert dynamics.cursor[3]==0


def test_gpu_cached_force_reads_do_not_advance_dynamics():
    dynamics=FootForceSensorDynamics(2,FootForceSensorConfig(delay_steps=2),torch_device="cuda")
    dynamics.apply(torch.ones((2,6),device="cuda"))
    filtered=dynamics.filtered.clone();delay=dynamics.delay.clone();cursor=dynamics.cursor.clone()
    # Observation reads consume the cached result; dynamics.apply is the only
    # operation allowed to mutate these states.
    cached=dynamics.delay.clone()
    for _ in range(3):torch.testing.assert_close(dynamics.delay,cached)
    torch.testing.assert_close(dynamics.filtered,filtered)
    torch.testing.assert_close(dynamics.delay,delay)
    torch.testing.assert_close(dynamics.cursor,cursor)
