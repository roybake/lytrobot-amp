#!/usr/bin/env python3
"""
全自动方案B - 训练更好的教师 + 自动蒸馏

流程：
1. 训练新教师（35000步，~12小时）
2. 自动找到最佳checkpoint
3. 用新教师蒸馏学生（25000步，~10小时）

总时间：~22小时，无需人工干预
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from training_paths import new_run_dir
RUN_DIR = None

def train_better_teacher():
    global RUN_DIR
    if RUN_DIR is None:
        RUN_DIR = new_run_dir(REPO_ROOT / "logs/distillation")
    """阶段1：训练更好的教师"""
    print("=" * 70)
    print("阶段1: 训练更好的Stage1教师（35000步，~12小时）")
    print("=" * 70)
    print()
    print("改进点:")
    print("  • 提高关节位置跟踪权重")
    print("  • 增加足部位置跟踪")
    print("  • 增加步态风格奖励")
    print("  • 降低速度要求到0.75（专注质量）")
    print()

    args = [
        "scripts/train_stage1.py",
        "--num-envs", "4096",
        "--device", "cuda",
        "--sim-backend", "warp",
        "--decimation", "2",

        # 更强的姿态跟踪
        "--reference-joint-position-reward-coef", "3.0",
        "--reference-joint-velocity-reward-coef", "1.5",
        "--reference-root-velocity-reward-coef", "1.0",
        "--reference-foot-position-reward-coef", "2.5",
        "--reference-gait-style-reward-coef", "5.0",

        # 速度设置
        "--target-velocity", "0.75",
        "--forward-velocity-reward-coef", "1.5",
        "--forward-velocity-deficit-penalty-coef", "2.0",

        # 足部跟踪
        "--foot-vel-reward-coef", "0.3",
        "--stance-foot-reward-coef", "0.5",
        "--contact-reward-coef", "0.2",

        # 控制平滑
        "--ctrl-penalty-coef", "0.04",
        "--action-rate-penalty-coef", "0.10",

        # 训练配置
        "--max-iterations", "35000",
        "--save-interval", "500",
        "--episode-steps", "2000",
        "--learning-rate", "3e-4",
        "--target-kl", "0.015",

        # 日志
        "--log-dir", str(RUN_DIR / "stage1" / "teacher"),
        "--wandb-tags", "lytrobot,stage1,improved-teacher,auto-pipeline",
        "--wandb",
    ]

    result = subprocess.run(
        ["uv", "run", "python"] + args,
        cwd=REPO_ROOT,
        check=False
    )

    if result.returncode != 0:
        print(f"\n❌ 教师训练失败，退出码: {result.returncode}")
        sys.exit(result.returncode)

    # 找到最佳checkpoint
    teacher_checkpoints = list((RUN_DIR / "stage1" / "teacher").glob("model_best.pt"))
    if not teacher_checkpoints:
        print("\n❌ 找不到教师checkpoint")
        sys.exit(1)

    teacher_best = max(teacher_checkpoints, key=lambda p: p.stat().st_mtime)

    print()
    print("=" * 70)
    print("✓ 阶段1完成 - 教师训练完成")
    print("=" * 70)
    print(f"最佳checkpoint: {teacher_best}")
    print()

    # 显示教师质量
    metrics_file = teacher_best.parent / "metrics.csv"
    if metrics_file.exists():
        print("检查教师质量...")
        subprocess.run([
            "python3", "-c",
            f"""
import csv, numpy as np
rows = list(csv.DictReader(open('{metrics_file}')))
def sf(v):
    try: return float(v) if v not in ('', 'nan') else float('nan')
    except: return float('nan')
metrics = ['mean_episode_length', 'task_forward_velocity', 'task_pose_track', 'mean_episode_success_rate']
print("教师最终指标:")
for m in metrics:
    if m in rows[0]:
        avg = np.nanmean([sf(r[m]) for r in rows[-50:]])
        print(f"  {{m:30s}}: {{avg:8.4f}}")
"""
        ])
        print()

    return teacher_best

def distill_with_teacher(teacher_checkpoint):
    """阶段2：用新教师蒸馏学生"""
    print("=" * 70)
    print("阶段2: 传感器策略蒸馏（25000步，~10小时）")
    print("=" * 70)
    print()
    print(f"使用教师: {teacher_checkpoint.name}")
    print()
    print("蒸馏配置:")
    print("  • 强监督: teacher_action_loss=1.0")
    print("  • 高引导: teacher_action_probability=0.70→0.10")
    print("  • 降低RL权重: task_reward_lerp=0.60")
    print()

    args = [
        "sensor_training/train_distill_stage1.py",
        "--log-dir", str(RUN_DIR / "stage1" / "student"),
        "--teacher-checkpoint", str(teacher_checkpoint),

        # 环境配置
        "--num-envs", "4096",
        "--num-steps", "24",
        "--decimation", "2",
        "--device", "cuda",
        "--sim-backend", "warp",
        "--episode-steps", "3000",

        # 状态估计器
        "--state-estimator-enabled",
        "--state-estimator-include-foot-position",
        "--actor-use-estimated-state",
        "--state-joint-position-loss-weight", "2.0",
        "--state-joint-velocity-loss-weight", "1.0",
        "--state-root-velocity-loss-weight", "1.5",
        "--state-contact-loss-weight", "1.5",
        "--state-foot-position-loss-weight", "1.0",

        # 历史窗口
        "--frame-count", "8",
        "--recurrent-hidden-dim", "192",
        "--include-tendon-velocity",
        "--include-tendon-tracking-error",

        # 强监督
        "--teacher-action-loss-coef", "1.0",
        "--teacher-action-loss-final-coef", "0.50",
        "--teacher-action-loss-decay-iterations", "15000",
        "--teacher-action-loss-scale", "0.10",
        "--teacher-action-probability", "0.70",
        "--teacher-action-probability-final", "0.10",
        "--teacher-action-probability-decay-iterations", "8000",

        # RL权重
        "--task-reward-lerp", "0.60",
        "--amp-reward-coef", "1.0",
        "--sensor-natural-gait-reward-coef", "2.0",
        "--reference-gait-style-reward-coef", "3.0",

        # 探索
        "--init-std", "0.15",
        "--max-action-std", "0.22",
        "--min-action-std", "0.04",
        "--entropy-coef", "0.002",

        # 学习率
        "--learning-rate", "5e-5",
        "--max-learning-rate", "5e-5",
        "--min-learning-rate", "1e-5",
        "--target-kl", "0.008",
        "--target-kl-stop-multiplier", "2.0",

        # 训练配置
        "--max-iterations", "25000",
        "--save-interval", "500",
        "--compile-policy-chunk-steps", "8",

        # 日志
        "--log-root", "logs/distillation",
        "--log-group", "with_better_teacher_auto",
        "--wandb-tags", "lytrobot,sensor-distillation,better-teacher,auto-pipeline",
        "--wandb",
    ]

    result = subprocess.run(
        ["uv", "run", "python"] + args,
        cwd=REPO_ROOT,
        check=False
    )

    if result.returncode != 0:
        print(f"\n❌ 蒸馏失败，退出码: {result.returncode}")
        sys.exit(result.returncode)

    print()
    print("=" * 70)
    print("✓ 阶段2完成 - 蒸馏完成")
    print("=" * 70)

def main():
    print()
    print("=" * 70)
    print("全自动流程：训练更好的教师 + 蒸馏传感器策略")
    print("=" * 70)
    print()
    print("流程:")
    print("  阶段1: 训练更好的教师（35000步，~12小时）")
    print("  阶段2: 用新教师蒸馏学生（25000步，~10小时）")
    print()
    print("总时间: ~22小时")
    print("完全自动，无需人工干预")
    print()
    print("=" * 70)
    print()

    try:
        # 阶段1：训练教师
        teacher_checkpoint = train_better_teacher()

        # 阶段2：蒸馏学生
        distill_with_teacher(teacher_checkpoint)

        print()
        print("=" * 70)
        print("🎉 全流程完成！")
        print("=" * 70)
        print()
        print("最终模型:")
        student_checkpoints = list((RUN_DIR / "stage1" / "student").glob("model_best.pt"))
        if student_checkpoints:
            student_best = max(student_checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"  学生: {student_best}")
            print()
            print("测试模型:")
            print(f'  uv run python scripts/play.py --checkpoint "{teacher_checkpoint}" --device cuda --viewer --steps 0')
            print(f'  uv run python sensor_training/play.py --checkpoint "{student_best}" --device cuda --viewer --steps 0')

    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(130)

if __name__ == "__main__":
    main()
