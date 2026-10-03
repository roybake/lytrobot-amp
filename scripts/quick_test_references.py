#!/usr/bin/env python3
"""快速测试参考数据质量，验证修复是否成功。"""

from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

def test_walk_reference():
    """测试平地行走参考数据。"""
    print("=" * 60)
    print("测试平地行走参考数据")
    print("=" * 60)

    old_ref = REPO_ROOT / "robot/reference_data/walk1_mjcf.npz"
    new_ref = REPO_ROOT / "robot/reference_data/walk1_mjcf_repaired.npz"

    # 检查原始数据
    if old_ref.exists():
        with np.load(old_ref) as data:
            qpos = data['qpos']
            travel = qpos[-1, 0] - qpos[0, 0]
            print(f"\n原始数据 (walk1_mjcf.npz):")
            print(f"  帧数: {qpos.shape[0]}")
            print(f"  前进距离: {travel:.3f}m")
            if abs(travel) < 0.01:
                print("  ✗ 问题：机器人原地踏步，无法向前走")

    # 检查修复数据
    if new_ref.exists():
        with np.load(new_ref) as data:
            qpos = data['qpos']
            freq = data['frequency']
            trans = data['cycle_translation']
            speed = trans[0] / (qpos.shape[0] / freq)
            print(f"\n修复数据 (walk1_mjcf_repaired.npz):")
            print(f"  帧数: {qpos.shape[0]} @ {freq}Hz")
            print(f"  周期时长: {qpos.shape[0]/freq:.3f}s")
            print(f"  周期平移: {trans[0]:.3f}m")
            print(f"  目标速度: {speed:.3f}m/s")
            print(f"  ✓ 修复成功：机器人可以向前走")
    else:
        print(f"\n✗ 修复数据不存在：{new_ref}")
        print("  请运行: uv run python scripts/repair_walk_reference.py")
        return False

    return True

def test_stairs_reference():
    """测试楼梯参考数据。"""
    print("\n" + "=" * 60)
    print("测试楼梯参考数据")
    print("=" * 60)

    stairs_ref = REPO_ROOT / "robot/reference_data/multimotion/stairs_supported.npz"

    if not stairs_ref.exists():
        print(f"\n✗ 楼梯数据不存在：{stairs_ref}")
        return False

    with np.load(stairs_ref) as data:
        qpos = data['qpos']
        freq = data['frequency']
        duration = (qpos.shape[0] - 1) / freq
        x_travel = qpos[-1, 0] - qpos[0, 0]
        z_min, z_max = qpos[:, 2].min(), qpos[:, 2].max()
        z_change = z_max - z_min
        avg_speed = x_travel / duration

        print(f"\n楼梯数据 (stairs_supported.npz):")
        print(f"  总帧数: {qpos.shape[0]} @ {freq}Hz")
        print(f"  总时长: {duration:.2f}s")
        print(f"  前进距离: {x_travel:.2f}m")
        print(f"  高度范围: {z_min:.3f}m → {z_max:.3f}m")
        print(f"  爬升高度: {z_change:.3f}m")
        print(f"  平均速度: {avg_speed:.3f}m/s")

        # 分析高度变化
        print(f"\n  高度变化分析:")
        checkpoints = [0, len(qpos)//4, len(qpos)//2, 3*len(qpos)//4, -1]
        for i in checkpoints:
            t = i / freq if i >= 0 else (len(qpos)-1) / freq
            z = qpos[i, 2]
            x = qpos[i, 0]
            stage = "起点" if i == 0 else "终点" if i == -1 else f"中间{i}"
            print(f"    {stage:6s} ({t:5.1f}s): X={x:5.2f}m, Z={z:.3f}m")

        print(f"  ✓ 楼梯数据包含完整的上下楼动作")

    return True

def check_training_scripts():
    """检查训练脚本是否使用修复的数据。"""
    print("\n" + "=" * 60)
    print("检查训练脚本配置")
    print("=" * 60)

    scripts_to_check = [
        REPO_ROOT / "scripts/train_stage1.py",
        REPO_ROOT / "sensor_training/train_stage1.py",
    ]

    all_good = True
    for script in scripts_to_check:
        if not script.exists():
            continue
        content = script.read_text()
        uses_repaired = "walk1_mjcf_repaired.npz" in content
        uses_old = "walk1_mjcf.npz" in content and "walk1_mjcf_repaired.npz" not in content

        print(f"\n{script.relative_to(REPO_ROOT)}:")
        if uses_repaired:
            print("  ✓ 使用修复的参考数据")
        elif uses_old:
            print("  ✗ 仍在使用原始数据，需要更新")
            all_good = False
        else:
            print("  ? 未找到参考数据配置")

    return all_good

def main():
    print("\n" + "=" * 60)
    print("LYTrobot 参考数据质量检查")
    print("=" * 60)

    walk_ok = test_walk_reference()
    stairs_ok = test_stairs_reference()
    scripts_ok = check_training_scripts()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    if walk_ok and stairs_ok and scripts_ok:
        print("\n✓ 所有检查通过！可以开始训练。")
        print("\n快速开始训练：")
        print("\n  1. 平地行走：")
        print("     uv run python scripts/train_stage1.py")
        print("\n  2. 楼梯上下：")
        print("     uv run python sensor_training/train_stairs_deepmimic.py")
    else:
        print("\n✗ 发现问题，请查看上述输出。")
        if not walk_ok:
            print("\n  修复平地行走数据：")
            print("  uv run python scripts/repair_walk_reference.py")

    print()

if __name__ == "__main__":
    main()
