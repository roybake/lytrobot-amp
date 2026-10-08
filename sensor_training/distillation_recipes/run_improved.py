#!/usr/bin/env python3
"""
传感器策略蒸馏 - 改进方案（一键启动）

核心改进：
1. 启用状态估计器 - 从传感器历史重建关节位置/速度/接触状态
2. 更长历史窗口 - frame_count=8
3. 降低教师依赖 - teacher_action_probability 0.40→0.0
4. 温和监督 - teacher_action_loss_coef 0.25→0.05
5. 强化RL - task_reward_lerp=0.85

训练轮次：50000轮
"""

import sys
import subprocess
from pathlib import Path

# 配置
REPO_ROOT = Path(__file__).resolve().parents[2]
TEACHER_CHECKPOINT = REPO_ROOT / "logs/stage1/10-02-20:39:54/model_best.pt"

# 训练参数
TRAINING_ARGS = [
    "sensor_training/train_distill_stage1.py",

    # 教师配置
    "--teacher-checkpoint", str(TEACHER_CHECKPOINT),

    # 环境配置
    "--num-envs", "4096",
    "--num-steps", "24",
    "--decimation", "2",  # 传感器策略要求100Hz控制
    "--device", "cuda",
    "--sim-backend", "warp",

    # === 核心改进：状态估计器 ===
    "--state-estimator-enabled",
    "--state-estimator-include-foot-position",
    "--actor-use-estimated-state",
    "--state-joint-position-loss-weight", "3.0",
    "--state-joint-velocity-loss-weight", "1.5",
    "--state-root-velocity-loss-weight", "3.0",
    "--state-contact-loss-weight", "2.5",
    "--state-foot-position-loss-weight", "2.0",

    # === 历史窗口 ===
    "--frame-count", "8",
    "--recurrent-hidden-dim", "192",
    "--include-tendon-velocity",
    "--include-tendon-tracking-error",

    # === 教师蒸馏配置（弱监督）===
    "--teacher-action-loss-coef", "0.25",
    "--teacher-action-loss-final-coef", "0.05",
    "--teacher-action-loss-decay-iterations", "8000",
    "--teacher-action-loss-scale", "0.05",
    "--teacher-action-probability", "0.40",
    "--teacher-action-probability-final", "0.0",
    "--teacher-action-probability-decay-iterations", "3000",

    # === RL目标权重 ===
    "--task-reward-lerp", "0.85",
    "--amp-reward-coef", "1.0",
    "--sensor-natural-gait-reward-coef", "3.0",
    "--reference-gait-style-reward-coef", "5.0",

    # === 探索设置 ===
    "--init-std", "0.18",
    "--max-action-std", "0.28",
    "--min-action-std", "0.04",
    "--entropy-coef", "0.004",

    # === 学习率 ===
    "--learning-rate", "8e-5",
    "--max-learning-rate", "8e-5",
    "--min-learning-rate", "2e-5",
    "--target-kl", "0.012",
    "--target-kl-stop-multiplier", "1.8",

    # === 训练配置 ===
    "--max-iterations", "50000",
    "--save-interval", "500",
    "--compile-policy-chunk-steps", "8",

    # === 日志 ===
    "--log-root", "logs/distillation",
    "--log-group", "stage1_gait_improved",
    "--wandb-tags", "lytrobot,sensor-distillation,state-estimator,improved",
    "--wandb",
]

def main():
    print("=" * 60)
    print("传感器策略蒸馏 - 改进方案")
    print("=" * 60)
    print()

    # 检查教师checkpoint
    if not TEACHER_CHECKPOINT.exists():
        print(f"❌ 错误: 教师checkpoint不存在")
        print(f"   路径: {TEACHER_CHECKPOINT}")
        print()
        print("请先训练教师或修改TEACHER_CHECKPOINT路径")
        sys.exit(1)

    print(f"✓ 教师checkpoint: {TEACHER_CHECKPOINT}")
    print(f"  大小: {TEACHER_CHECKPOINT.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    print("核心改进:")
    print("  • 状态估计器: 从传感器历史重建关节/速度/接触")
    print("  • 历史窗口: 8帧（原来5帧）")
    print("  • 教师依赖: 40%→0%（原来80%）")
    print("  • 监督强度: 0.25→0.05（原来1.0）")
    print("  • RL权重: 85%（原来75%）")
    print()

    print("训练轮次: 50000轮")
    print("按 Ctrl+C 可随时中断")
    print()
    print("=" * 60)
    print()

    # 执行训练
    try:
        result = subprocess.run(
            ["uv", "run", "python"] + TRAINING_ARGS,
            cwd=REPO_ROOT,
            check=False
        )

        if result.returncode == 0:
            print()
            print("=" * 60)
            print("✓ 训练完成！")
            print("=" * 60)
            print()
            print("检查结果:")
            print("  cd logs/distillation/<run>/stage1/student")
            print("  tail -100 */train.log")
        else:
            print()
            print(f"训练中断，退出码: {result.returncode}")
            sys.exit(result.returncode)

    except KeyboardInterrupt:
        print()
        print("用户中断训练")
        sys.exit(130)

if __name__ == "__main__":
    main()
