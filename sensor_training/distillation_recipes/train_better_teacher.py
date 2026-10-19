#!/usr/bin/env python3
"""
重新训练Stage1教师 - 更强的姿态跟踪

改进点：
1. 提高pose_track奖励权重
2. 增加足部位置跟踪
3. 降低速度要求（专注姿态质量）
4. 更长训练时间
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TRAINING_ARGS = [
    "scripts/train_stage1.py",

    # 基础配置
    "--num-envs", "4096",
    "--device", "cuda",
    "--sim-backend", "warp",
    "--decimation", "2",

    # === 改进：更强的姿态跟踪 ===
    "--reference-joint-position-reward-coef", "3.0",  # 提高（原来可能较低）
    "--reference-joint-velocity-reward-coef", "1.5",
    "--reference-root-velocity-reward-coef", "1.0",
    "--reference-foot-position-reward-coef", "2.5",   # 提高足部跟踪

    # 步态风格奖励
    "--reference-gait-style-reward-coef", "5.0",      # 增加自然步态

    # 速度设置（降低要求，专注质量）
    "--target-velocity", "0.75",                      # 降低（原来0.87）
    "--forward-velocity-reward-coef", "1.5",
    "--forward-velocity-deficit-penalty-coef", "2.0",

    # 足部跟踪
    "--foot-vel-reward-coef", "0.3",
    "--stance-foot-reward-coef", "0.5",
    "--contact-reward-coef", "0.2",

    # 控制平滑
    "--torque-penalty-coef", "0.0004",
    "--action-rate-penalty-coef", "0.10",

    # 训练配置
    "--max-iterations", "50000",                      # 增加训练时间
    "--save-interval", "500",
    "--episode-steps", "2000",

    # 学习率
    "--learning-rate", "3e-4",
    "--target-kl", "0.015",

    # 日志
    "--log-root", "logs/stage1_improved",
    "--log-group", "better_teacher",
    "--wandb-tags", "lytrobot,stage1,improved-teacher,high-quality",
    "--wandb",
]

def main():
    print("=" * 60)
    print("重新训练Stage1教师 - 提高姿态质量")
    print("=" * 60)
    print()
    print("改进点:")
    print("  • 提高关节位置跟踪权重")
    print("  • 增加足部位置跟踪")
    print("  • 增加步态风格奖励")
    print("  • 降低速度要求到0.75（专注质量）")
    print("  • 训练50000轮")
    print()
    print("训练轮次: 50000轮")
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
            print("✓ 教师训练完成")
            print("=" * 60)
            print()
            print("检查质量:")
            print("  cd logs/stage1_improved/better_teacher")
            print("  tail -100 */train.log")
            print()
            print("目标指标:")
            print("  • episode_length > 800")
            print("  • pose_track > 0.75")
            print("  • forward_velocity > 0.70")
            print("  • success_rate > 0.85")
        else:
            sys.exit(result.returncode)

    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(130)

if __name__ == "__main__":
    main()
