import torch

from distillation_training.networks import SharedActor,HistoryEstimator


def _rollout(actor,estimator,initial_hidden,steps=8):
    n=3;hidden=initial_hidden.clone();previous_action=torch.zeros(n,14)
    previous_tendon=torch.zeros(n,14);rows=[]
    generator=torch.Generator().manual_seed(9)
    for _ in range(steps):
        sensor=torch.randn(n,64,generator=generator);sensor[:,42:56]=torch.randn(n,14,generator=generator)
        predicted,hidden=estimator(sensor,previous_action,previous_tendon,hidden,torch.zeros(n))
        distribution=actor.distribution(sensor,predicted)
        action=distribution.sample();logp=distribution.log_prob(action).sum(-1)
        rows.append((sensor,previous_action,previous_tendon,action,logp))
        previous_action=action;previous_tendon=sensor[:,42:56]
    return rows


def _replay(actor,estimator,rows,hidden):
    values=[]
    for sensor,previous_action,previous_tendon,action,_ in rows:
        predicted,hidden=estimator(sensor,previous_action,previous_tendon,hidden,torch.zeros(sensor.shape[0]))
        values.append(actor.distribution(sensor,predicted).log_prob(action).sum(-1))
    return torch.stack(values)


def test_nonzero_rollout_initial_hidden_replays_exact_log_probability():
    torch.manual_seed(4);actor=SharedActor(64).eval();estimator=HistoryEstimator(64).eval()
    initial=torch.randn(3,128);rows=_rollout(actor,estimator,initial)
    old=torch.stack([row[-1] for row in rows])
    torch.testing.assert_close(_replay(actor,estimator,rows,initial.clone()),old,rtol=1e-5,atol=1e-6)
    assert not torch.allclose(_replay(actor,estimator,rows,torch.zeros_like(initial)),old)
