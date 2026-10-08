#!/usr/bin/env python3
"""
传感器策略蒸馏 - 两阶段训练（最稳健）

阶段1: 预训练状态估计器（50000轮）
  - 高权重状态loss
  - 不监督教师动作
  - 让GRU学会从传感器重建状态

阶段2: 策略蒸馏（50000轮）
  - 基于预训练的状态估计器
  - 启用温和的教师监督
  - 完成策略蒸馏

每阶段50000轮
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEACHER_CHECKPOINT = REPO_ROOT / "logs/stage1/10-02-20:39:54/model_best.pt"
LOG_ROOT = REPO_ROOT / "logs/distillation"

def run_phase1():
    """阶段1: 预训练状态估计器"""
    print("=" * 60)
    print("阶段1: 预训练状态估计器（50000轮）")
    print("=" * 60)
    print()
    print("目标: 让GRU从传感器历史重建关节、速度、接触状态")
    print("方法: 高权重状态loss，暂不监督教师动作")
    print()

    args = [
        "sensor_training/train_distill_stage1.py",
        "--teacher-checkpoint", str(TEACHER_CHECKPOINT),
        "--num-envs", "4096",
        "--num-steps", "24",
        "--decimation", "2",  # 传感器策略要求100Hz控制
        "--device", "cuda",
        "--sim-backend", "warp",

        # 状态估计器：高权重
        "--state-estimator-enabled",
        "--state-estimator-include-foot-position",
        "--actor-use-estimated-state",
        "--state-joint-position-loss-weight", "5.0",
        "--state-joint-velocity-loss-weight", "2.5",
        "--state-root-velocity-loss-weight", "4.0",
        "--state-contact-loss-weight", "4.0",
        "--state-foot-position-loss-weight", "3.0",

        # 关键：暂时不监督教师动作
        "--teacher-action-loss-coef", "0.0",
        "--teacher-action-probability", "0.0",

        # 让策略通过RL自由探索
        "--task-reward-lerp", "0.70",
        "--amp-reward-coef", "1.2",
        "--sensor-natural-gait-reward-coef", "2.5",
        "--reference-gait-style-reward-coef", "4.0",

        # 历史窗口
        "--frame-count", "8",
        "--recurrent-hidden-dim", "192",
        "--include-tendon-velocity",
        "--include-tendon-tracking-error",

        # 探索
        "--init-std", "0.20",
        "--max-action-std", "0.30",
        "--entropy-coef", "0.005",

        # 学习率
        "--learning-rate", "1.5e-4",
        "--max-learning-rate", "1.5e-4",
        "--target-kl", "0.015",

        # 训练配置
        "--max-iterations", "50000",
        "--save-interval", "500",

        # 日志
        "--log-root", str(LOG_ROOT),
        "--log-group", "phase1_estimator",
        "--wandb-tags", "lytrobot,distillation,phase1-estimator",
        "--wandb",
    ]

    result = subprocess.run(
        ["uv", "run", "python"] + args,
        cwd=REPO_ROOT,
        check=False
    )

    if result.returncode != 0:
        print(f"\n阶段1失败，退出码: {result.returncode}")
        sys.exit(result.returncode)

    # 找到最佳checkpoint
    phase1_checkpoints = list((LOG_ROOT / "phase1_estimator").glob("*/model_best.pt"))
    if not phase1_checkpoints:
        print("\n错误: 找不到阶段1的checkpoint")
        sys.exit(1)

    phase1_best = max(phase1_checkpoints, key=lambda p: p.stat().st_mtime)
    print()
    print("=" * 60)
    print("✓ 阶段1完成")
    print("=" * 60)
    print(f"Checkpoint: {phase1_best}")
    print()
    print("检查状态估计质量:")
    print("  state/joint_position_loss 应该 < 0.12")
    print("  state/root_velocity_loss 应该 < 0.35")
    print()

    return phase1_best

def run_phase2(phase1_checkpoint):
    """阶段2: 策略蒸馏"""
    print("=" * 60)
    print("阶段2: 策略蒸馏（50000轮）")
    print("=" * 60)
    print()
    print("目标: 基于良好的状态估计，模仿教师策略")
    print(f"恢复: {phase1_checkpoint.name}")
    print()

    args = [
        "sensor_training/train_distill_stage1.py",
        "--teacher-checkpoint", str(TEACHER_CHECKPOINT),
        "--resume-checkpoint", str(phase1_checkpoint),
        "--resume-reset-progress",
        "--num-envs", "4096",
        "--num-steps", "24",
        "--decimation", "2",  # 传感器策略要求100Hz控制
        "--device", "cuda",
        "--sim-backend", "warp",

        # 保持状态估计器训练（较低权重）
        "--state-estimator-enabled",
        "--state-estimator-include-foot-position",
        "--actor-use-estimated-state",
        "--state-joint-position-loss-weight", "2.0",
        "--state-joint-velocity-loss-weight", "1.0",
        "--state-root-velocity-loss-weight", "1.5",
        "--state-contact-loss-weight", "1.5",
        "--state-foot-position-loss-weight", "1.0",

        # 现在启用教师监督（温和）
        "--teacher-action-loss-coef", "0.30",
        "--teacher-action-loss-final-coef", "0.08",
        "--teacher-action-loss-decay-iterations", "10000",
        "--teacher-action-loss-scale", "0.06",
        "--teacher-action-probability", "0.35",
        "--teacher-action-probability-final", "0.0",
        "--teacher-action-probability-decay-iterations", "3500",

        # RL权重
        "--task-reward-lerp", "0.82",
        "--amp-reward-coef", "1.0",
        "--sensor-natural-gait-reward-coef", "3.0",
        "--reference-gait-style-reward-coef", "5.0",

        # 网络配置（与阶段1相同）
        "--frame-count", "8",
        "--recurrent-hidden-dim", "192",
        "--include-tendon-velocity",
        "--include-tendon-tracking-error",

        # 探索（略微降低）
        "--init-std", "0.16",
        "--max-action-std", "0.26",
        "--entropy-coef", "0.0035",

        # 学习率（更保守）
        "--learning-rate", "6e-5",
        "--max-learning-rate", "6e-5",
        "--min-learning-rate", "1.5e-5",
        "--target-kl", "0.010",
        "--target-kl-stop-multiplier", "1.6",

        # 训练配置
        "--max-iterations", "50000",
        "--save-interval", "500",
        "--compile-policy-chunk-steps", "8",

        # 日志
        "--log-root", str(LOG_ROOT),
        "--log-group", "phase2_policy",
        "--wandb-tags", "lytrobot,distillation,phase2-policy",
        "--wandb",
    ]

    result = subprocess.run(
        ["uv", "run", "python"] + args,
        cwd=REPO_ROOT,
        check=False
    )

    if result.returncode != 0:
        print(f"\n阶段2失败，退出码: {result.returncode}")
        sys.exit(result.returncode)

    print()
    print("=" * 60)
    print("✓ 两阶段训练完成！")
    print("=" * 60)
    print()

def main():
    print()
    print("=" * 60)
    print("两阶段传感器策略蒸馏")
    print("=" * 60)
    print()

    if not TEACHER_CHECKPOINT.exists():
        print(f"❌ 教师checkpoint不存在: {TEACHER_CHECKPOINT}")
        sys.exit(1)

    print(f"✓ 教师: {TEACHER_CHECKPOINT.name}")
    print()
    print("流程:")
    print("  阶段1: 预训练状态估计器（50000轮）")
    print("  阶段2: 策略蒸馏（50000轮）")
    print("  每阶段50000轮")
    print()
    print("=" * 60)
    print()

    try:
        # 阶段1
        phase1_checkpoint = run_phase1()

        # 阶段2
        run_phase2(phase1_checkpoint)

        print("最终模型:")
        phase2_best = list((LOG_ROOT / "phase2_policy").glob("*/model_best.pt"))
        if phase2_best:
            print(f"  {phase2_best[0]}")

    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(130)

if __name__ == "__main__":
    main()
