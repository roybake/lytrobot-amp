"""PPO/BPO config type scoped to sensor-only policy training."""

from __future__ import annotations

from dataclasses import dataclass

from lytrobot_amp.ppo_amp import PPOAMPConfig


@dataclass
class SensorPPOAMPConfig(PPOAMPConfig):
    """BPO defaults for sensor-history policies."""

    target_kl: float = 0.025
    target_kl_stop_multiplier: float = 2.0
    # Recurrent policies can enter a bad state distribution after a single
    # large update.  A tighter gradient cap and slightly stronger exploration
    # make that failure mode less likely without adding privileged inputs.
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.003
    # Rollout is safe to compile as a no-grad one-step CUDA graph.  Compile
    # the recurrent backward step too: its first build is expensive, but it
    # is the dominant steady-state speedup for long PPO training runs.
    compile_rollout: bool = True
    compile_policy_update: bool = True
    # Compile several recurrent steps per graph invocation.  Gradients remain
    # connected between chunks, but an N-step rollout no longer needs N Python
    # calls for every minibatch.
    compile_policy_chunk_steps: int = 16
    recurrent_hidden_dim: int = 128
    # Residual MoE is reserved for the serial multi-terrain curriculum.
    use_moe: bool = False
    num_experts: int = 4
    gate_hidden_dims: tuple[int, ...] = (128, 64)
    expert_hidden_dims: tuple[int, ...] = (128, 64)
    residual_scale: float = 0.35
    gate_change_coef: float = 0.002
    expert_load_balance_coef: float = 0.002
    residual_l2_coef: float = 0.001
    privileged_context_loss_coef: float = 0.20
    # Expert AMP clips represent steady walking.  While a stand-start episode
    # is still accelerating, fade in AMP according to the target-speed ramp so
    # it does not contradict the balance/slow-start task objective.
    amp_startup_gate: bool = True
