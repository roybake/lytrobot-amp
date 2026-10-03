#!/usr/bin/env python3
"""
传感器策略蒸馏 - 快速验证（30分钟）

用1000步测试状态估计器方案是否有效。

成功标准：
- teacher_action_mae 从 0.24 降到 < 0.20
- mean_episode_length > 400
- state/joint_position_loss < 0.15
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEACHER_CHECKPOINT = REPO_ROOT / "logs/stage1/10-02-20:39:54/model_best.pt"

TRAINING_ARGS = [
    "sensor_training/train_distill_stage1.py",
    "--teacher-checkpoint", str(TEACHER_CHECKPOINT),

    # 小规模快速测试
    "--num-envs", "2048",
    "--num-steps", "24",
    "--decimation", "2",  # 传感器策略要求100Hz控制
    "--device", "cuda",
    "--sim-backend", "warp",

    # 状态估计器
    "--state-estimator-enabled",
    "--state-estimator-include-foot-position",
    "--actor-use-estimated-state",
    "--state-joint-position-loss-weight", "4.0",
    "--state-joint-velocity-loss-weight", "2.0",
    "--state-root-velocity-loss-weight", "3.5",
    "--state-contact-loss-weight", "3.0",
    "--state-foot-position-loss-weight", "2.5",

    # 历史窗口
    "--frame-count", "8",
    "--recurrent-hidden-dim", "192",
    "--include-tendon-velocity",
    "--include-tendon-tracking-error",

    # 弱监督
    "--teacher-action-loss-coef", "0.20",
    "--teacher-action-loss-scale", "0.05",
    "--teacher-action-probability", "0.35",
    "--teacher-action-probability-final", "0.0",
    "--teacher-action-probability-decay-iterations", "500",

    # RL
    "--task-reward-lerp", "0.85",
    "--sensor-natural-gait-reward-coef", "3.0",
    "--reference-gait-style-reward-coef", "5.0",

    # 探索
    "--init-std", "0.18",
    "--max-action-std", "0.28",
    "--entropy-coef", "0.004",

    # 学习率
    "--learning-rate", "1e-4",
    "--target-kl", "0.012",

    # 快速测试：仅1000步
    "--max-iterations", "1000",
    "--save-interval", "200",

    # 日志（不用wandb加快速度）
    "--log-root", "logs/sensor_policy/quick_validation",
    "--log-group", "state_estimator_test",
    "--no-wandb",
]

def main():
    print("=" * 60)
    print("快速验证 - 状态估计器方案（1000步，~30分钟）")
    print("=" * 60)
    print()

    if not TEACHER_CHECKPOINT.exists():
        print(f"❌ 教师checkpoint不存在: {TEACHER_CHECKPOINT}")
        sys.exit(1)

    print(f"✓ 教师: {TEACHER_CHECKPOINT.name}")
    print()
    print("测试目标:")
    print("  • teacher_action_mae: 0.24 → < 0.20")
    print("  • episode_length: > 400")
    print("  • state losses: 收敛")
    print()
    print("如果验证通过，用 run_improved.py 跑完整训练")
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
            print("✓ 快速验证完成")
            print("=" * 60)
            print()
            print("查看结果:")
            print("  cd logs/sensor_policy/quick_validation/state_estimator_test")
            print("  tail -50 */train.log | grep -E '(teacher_action_mae|episode_length)'")
            print()
            print("如果指标达标，运行完整训练:")
            print("  uv run python sensor_training/distillation_recipes/run_improved.py")
        else:
            sys.exit(result.returncode)

    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(130)

if __name__ == "__main__":
    main()
