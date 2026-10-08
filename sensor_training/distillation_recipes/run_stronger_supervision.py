#!/usr/bin/env python3
"""
调整蒸馏配置 - 针对当前教师的改进版本

问题诊断：
- teacher_action_mae从0.23上升到0.45（学生越来越不像教师）
- episode_length只有372（太短，频繁摔倒）
- state loss很低但动作质量差

改进策略：
1. 大幅提高teacher_action_loss权重（让学生更严格模仿）
2. 提高teacher_action_probability（增加教师行为引导）
3. 降低task_reward_lerp（减少RL自主探索的负面影响）
4. 增加episode_steps（给更多时间学习）
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEACHER_CHECKPOINT = REPO_ROOT / "logs/stage1/10-02-20:39:54/model_best.pt"

TRAINING_ARGS = [
    "sensor_training/train_distill_stage1.py",
    "--teacher-checkpoint", str(TEACHER_CHECKPOINT),

    # 环境配置
    "--num-envs", "4096",
    "--num-steps", "24",
    "--decimation", "2",
    "--device", "cuda",
    "--sim-backend", "warp",

    # 增加episode长度，给更多时间学习
    "--episode-steps", "3000",  # 增加（原来2000）

    # === 状态估计器 ===
    "--state-estimator-enabled",
    "--state-estimator-include-foot-position",
    "--actor-use-estimated-state",
    "--state-joint-position-loss-weight", "2.0",
    "--state-joint-velocity-loss-weight", "1.0",
    "--state-root-velocity-loss-weight", "1.5",
    "--state-contact-loss-weight", "1.5",
    "--state-foot-position-loss-weight", "1.0",

    # === 历史窗口 ===
    "--frame-count", "8",
    "--recurrent-hidden-dim", "192",
    "--include-tendon-velocity",
    "--include-tendon-tracking-error",

    # === 关键改进：更强的教师监督 ===
    "--teacher-action-loss-coef", "1.0",              # 提高（原来0.25）
    "--teacher-action-loss-final-coef", "0.50",       # 提高（原来0.05）
    "--teacher-action-loss-decay-iterations", "15000",# 延长衰减
    "--teacher-action-loss-scale", "0.10",            # 提高（原来0.05）
    "--teacher-action-probability", "0.70",           # 提高（原来0.40）
    "--teacher-action-probability-final", "0.10",     # 保持一定引导（原来0.0）
    "--teacher-action-probability-decay-iterations", "8000",

    # === RL权重：降低，让教师监督占主导 ===
    "--task-reward-lerp", "0.60",                     # 降低（原来0.85）
    "--amp-reward-coef", "1.0",
    "--sensor-natural-gait-reward-coef", "2.0",
    "--reference-gait-style-reward-coef", "3.0",

    # === 探索（降低，减少随机性）===
    "--init-std", "0.15",                             # 降低（原来0.18）
    "--max-action-std", "0.22",                       # 降低（原来0.28）
    "--min-action-std", "0.04",
    "--entropy-coef", "0.002",                        # 降低（原来0.004）

    # === 学习率（更保守）===
    "--learning-rate", "5e-5",                        # 降低（原来8e-5）
    "--max-learning-rate", "5e-5",
    "--min-learning-rate", "1e-5",
    "--target-kl", "0.008",                           # 更严格（原来0.012）
    "--target-kl-stop-multiplier", "2.0",

    # === 训练配置 ===
    "--max-iterations", "25000",                      # 增加时间
    "--save-interval", "500",
    "--compile-policy-chunk-steps", "8",

    # === 日志 ===
    "--log-root", "logs/distillation",
    "--log-group", "stage1_stronger_supervision",
    "--wandb-tags", "lytrobot,sensor-distillation,stronger-supervision",
    "--wandb",
]

def main():
    print("=" * 60)
    print("传感器策略蒸馏 - 更强监督版本")
    print("=" * 60)
    print()

    if not TEACHER_CHECKPOINT.exists():
        print(f"❌ 教师checkpoint不存在: {TEACHER_CHECKPOINT}")
        sys.exit(1)

    print(f"✓ 教师: {TEACHER_CHECKPOINT.name}")
    print()
    print("核心改进（针对上次失败）:")
    print("  • teacher_action_loss: 0.25 → 1.0（更强监督）")
    print("  • teacher_action_probability: 0.40 → 0.70（更多引导）")
    print("  • task_reward_lerp: 0.85 → 0.60（减少RL干扰）")
    print("  • learning_rate: 8e-5 → 5e-5（更保守）")
    print("  • episode_steps: 2000 → 3000（更长学习时间）")
    print()
    print("预期效果:")
    print("  • teacher_action_mae应该下降而不是上升")
    print("  • episode_length应该>600")
    print()
    print("预计时间: ~10小时")
    print()
    print("=" * 60)
    print()

    try:
        result = subprocess.run(
            ["uv", "run", "python"] + TRAINING_ARGS,
            cwd=REPO_ROOT,
            check=False
        )

        if result.returncode == 0:
            print()
            print("=" * 60)
            print("✓ 训练完成")
            print("=" * 60)
        else:
            sys.exit(result.returncode)

    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(130)

if __name__ == "__main__":
    main()
