# 传感器策略蒸馏 - 改进方案

## 🎯 问题诊断

你的蒸馏一直失败的根本原因：**观测信息差距过大**

- **教师**: 能看到关节角度、足部位置、接触状态（~100维）
- **学生**: 只能看到IMU+肌腱长度（~20维）
- **差距**: 缺失80%的决策信息

这导致学生无法理解教师为什么做出某个动作。

## ✅ 解决方案

启用**状态估计器**，让GRU从传感器历史重建缺失信息：

```
传感器历史(8帧) → GRU → 估计状态 → Actor决策
                          ↓
                    关节位置/速度
                    足部位置
                    接触状态
```

## 🚀 执行命令（三选一）

### 1️⃣ 完整训练（推荐）⭐

```bash
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/distillation_recipes/run_improved.py
```

- 20000步，约8小时
- 一键自动执行
- 包含所有核心改进

### 2️⃣ 快速验证（30分钟）

```bash
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/distillation_recipes/quick_validate.py
```

- 1000步测试方案有效性
- 看结果后再决定是否完整训练

### 3️⃣ 两阶段训练（最稳健）

```bash
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/distillation_recipes/run_two_stage.py
```

- 阶段1: 预训练状态估计器（3000步）
- 阶段2: 策略蒸馏（15000步）
- 总计约8小时

## 📊 核心改进

| 配置 | 之前（失败） | 现在（改进） | 效果 |
|------|------------|------------|------|
| 状态估计器 | ❌ 未启用 | ✅ 启用 | 弥补观测差距 |
| 历史窗口 | 5帧 | **8帧** | 更多信息 |
| 教师依赖 | 80% | **40%→0%** | 独立学习 |
| 监督强度 | 1.0 | **0.25→0.05** | 温和引导 |
| RL权重 | 75% | **85%** | 强化探索 |

## 📈 预期效果

```
指标                    之前    改进后    目标
teacher_action_mae      0.24 →  0.15    < 0.16
episode_length          200  →  750     > 700
forward_velocity        0.60 →  0.78    > 0.75
success_rate            0.0  →  0.65    > 0.60
```

## 📁 文件说明

```
sensor_training/distillation_recipes/
├── quick_validate.py    # 快速验证（30分钟）
├── run_improved.py      # 单阶段完整训练（8小时）
├── run_two_stage.py     # 两阶段训练（8小时，最稳健）
└── README.md            # 本文件
```

## 🔍 监控训练

另开终端实时查看：

```bash
# 查看日志
tail -f logs/sensor_policy/distillation/*/train.log

# 关键指标
watch -n 10 'tail logs/sensor_policy/distillation/*/train.log | grep -E "(teacher_action_mae|episode_length)"'
```

## 🐛 故障排查

### teacher_action_mae不下降

```bash
# 增加状态估计权重
--state-joint-position-loss-weight 5.0
--state-root-velocity-loss-weight 4.0
```

### episode_length太短

```bash
# 增加历史窗口
--frame-count 10

# 降低学习率
--learning-rate 5e-5
```

## 🎓 理论依据

这个方法基于：

1. **NVIDIA Isaac Gym**: 四足机器人用状态估计器成功部署
2. **MIT Cheetah**: 先训练估计器再训练策略
3. **RMA**: Rapid Motor Adaptation for Legged Robots

参考论文：
- "Learning to Walk in Minutes Using Massively Parallel Deep RL"
- "RMA: Rapid Motor Adaptation for Legged Robots"

## ⏱️ 时间规划

### 单阶段（run_improved.py）
```
00:00 - 08:00  完整训练（20000步）
```

### 两阶段（run_two_stage.py）
```
00:00 - 01:30  阶段1: 预训练估计器（3000步）
01:30 - 08:00  阶段2: 策略蒸馏（15000步）
```

## 📝 详细参数

查看脚本内的完整参数配置：

```bash
cat sensor_training/distillation_recipes/run_improved.py
```

## 🚀 立即开始

**推荐命令**（最简单）：

```bash
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/distillation_recipes/run_improved.py
```

或先30分钟快速验证：

```bash
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/distillation_recipes/quick_validate.py
```

---

**预祝训练成功！** 🎉

训练完成后测试模型：

```bash
uv run python scripts/play.py \
  --checkpoint logs/sensor_policy/distillation/stage1_gait_improved/*/model_best.pt \
  --num-envs 1
```
