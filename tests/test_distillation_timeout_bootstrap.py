import torch

from distillation_training.returns import compute_gae_with_time_limit_bootstrap


def _one_step(done, timeout, terminal_value, rollout_next_value=7.0):
    advantages, returns = compute_gae_with_time_limit_bootstrap(
        rewards=torch.tensor([[1.0]]),
        values=torch.tensor([[0.0]]),
        dones=torch.tensor([[done]]),
        time_outs=torch.tensor([[timeout]]),
        timeout_bootstrap_values=torch.tensor([[terminal_value]]),
        final_next_value=torch.tensor([rollout_next_value]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    return advantages, returns


def test_teacher_and_student_true_termination_does_not_bootstrap():
    _, returns = _one_step(done=1.0, timeout=0.0, terminal_value=999.0)
    torch.testing.assert_close(returns, torch.tensor([[1.0]]))


def test_teacher_and_student_time_limit_uses_terminal_pre_reset_value():
    _, returns = _one_step(done=1.0, timeout=1.0, terminal_value=5.0)
    torch.testing.assert_close(returns, torch.tensor([[1.0 + 0.99 * 5.0]]))


def test_normal_transition_uses_next_rollout_value():
    _, returns = _one_step(
        done=0.0, timeout=0.0, terminal_value=999.0, rollout_next_value=3.0
    )
    torch.testing.assert_close(returns, torch.tensor([[1.0 + 0.99 * 3.0]]))
