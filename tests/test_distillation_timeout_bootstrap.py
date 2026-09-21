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


def _two_step_advantages(first_done, first_timeout, second_reward):
    advantages, _ = compute_gae_with_time_limit_bootstrap(
        rewards=torch.tensor([[1.0], [second_reward]]),
        values=torch.zeros(2, 1),
        dones=torch.tensor([[first_done], [0.0]]),
        time_outs=torch.tensor([[first_timeout], [0.0]]),
        timeout_bootstrap_values=torch.tensor([[5.0], [0.0]]),
        final_next_value=torch.tensor([0.0]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    return advantages


def test_timeout_bootstrap_does_not_leak_next_episode_advantage():
    advantages = _two_step_advantages(
        first_done=1.0, first_timeout=1.0, second_reward=100.0
    )
    changed_next_episode = _two_step_advantages(
        first_done=1.0, first_timeout=1.0, second_reward=10_000.0
    )
    expected = torch.tensor([1.0 + 0.99 * 5.0])
    torch.testing.assert_close(advantages[0], expected)
    torch.testing.assert_close(changed_next_episode[0], expected)


def test_true_terminal_does_not_leak_next_episode_advantage():
    advantages, _ = compute_gae_with_time_limit_bootstrap(
        rewards=torch.tensor([[2.0], [10_000.0]]),
        values=torch.tensor([[0.5], [0.0]]),
        dones=torch.tensor([[1.0], [0.0]]),
        time_outs=torch.zeros(2, 1),
        timeout_bootstrap_values=torch.zeros(2, 1),
        final_next_value=torch.tensor([0.0]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    torch.testing.assert_close(advantages[0], torch.tensor([2.0 - 0.5]))


def test_normal_transition_keeps_gae_trace():
    advantages = _two_step_advantages(
        first_done=0.0, first_timeout=0.0, second_reward=100.0
    )
    expected = torch.tensor([1.0 + 0.99 * 0.95 * 100.0])
    torch.testing.assert_close(advantages[0], expected)


def test_trace_boundary_is_independent_for_each_environment():
    advantages, _ = compute_gae_with_time_limit_bootstrap(
        rewards=torch.tensor([[1.0, 1.0], [2.0, 2.0], [100.0, 3.0]]),
        values=torch.zeros(3, 2),
        dones=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        time_outs=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        timeout_bootstrap_values=torch.tensor(
            [[0.0, 0.0], [5.0, 0.0], [0.0, 0.0]]
        ),
        final_next_value=torch.zeros(2),
        gamma=0.99,
        gae_lambda=0.95,
    )
    discount = 0.99 * 0.95
    env0_at_boundary = 2.0 + 0.99 * 5.0
    expected_env0 = torch.tensor(
        [1.0 + discount * env0_at_boundary, env0_at_boundary, 100.0]
    )
    expected_env1 = torch.tensor(
        [1.0 + discount * (2.0 + discount * 3.0), 2.0 + discount * 3.0, 3.0]
    )
    torch.testing.assert_close(advantages[:, 0], expected_env0)
    torch.testing.assert_close(advantages[:, 1], expected_env1)
