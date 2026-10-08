# 传感器策略蒸馏 - 诊断和改进方案

## 🔍 训练结果诊断

你的训练已完成，但效果不好：

```
teacher_action_mae:  0.23 → 0.45  ❌ 越来越差（应该下降）
episode_length:      372          ❌ 太短（目标>700）
forward_velocity:    0.76         ⚠️  速度还可以
success_rate:        0%           ❌ 完全失败
```

教师质量分析：
```
episode_length:      919   ✓
success_rate:        90%   ✓
forward_velocity:    0.67  ❌ 偏慢（目标>0.80）
pose_track:          0.69  ⚠️  姿态一般（目标>0.75）
```

## 🎯 根本原因

1. **教师本身质量不够**
   - forward_velocity只有0.67，走得慢
   - pose_track只有0.69，姿态不够像人
   - 学生蒸馏的上限就是教师的质量

2. **蒸馏配置太弱**
   - 之前的配置（teacher监督0.25→0.05，概率0.40→0）太弱
   - 导致学生RL自主探索偏离了教师

## 💡 改进方案（两选一）

### 方案A：加强现有教师的监督 ⭐（推荐先试）

不重新训练教师，直接改蒸馏配置：

```bash
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/distillation_recipes/run_stronger_supervision.py
```

**改进点**：
- teacher_action_loss: 0.25 → 1.0（更强监督）
- teacher_action_probability: 0.40 → 0.70（更多引导）
- task_reward_lerp: 0.85 → 0.60（减少RL干扰）
- learning_rate: 8e-5 → 5e-5（更保守）
- episode_steps: 2000 → 3000（更长学习时间）

**预期**：teacher_action_mae应该下降而不是上升

**训练轮次**：50000轮

---

### 方案B：先训练更好的教师（治本）

教师质量是蒸馏上限。训练一个走得更好的教师：

```bash
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/distillation_recipes/train_better_teacher.py
```

**改进点**：
- 提高姿态跟踪权重（joint_position, foot_position）
- 增加步态风格奖励
- 降低速度要求到0.75（专注质量而非速度）
- 训练50000轮

**目标指标**：
- episode_length > 800
- pose_track > 0.75
- forward_velocity > 0.70
- success_rate > 0.85

**训练轮次**：50000轮

训练完成后再用新教师蒸馏：
```bash
# 假设新教师在 logs/stage1_improved/better_teacher/xxx/model_best.pt
# 修改 run_stronger_supervision.py 的 TEACHER_CHECKPOINT 路径
# 然后运行蒸馏
```

---

## 📊 两个方案对比

| 方案 | 训练轮次 | 难度 | 效果预期 | 适用场景 |
|------|------|------|---------|----------|
| **方案A** | Student 50000轮 | 低 | 中等改善 | 快速验证，时间紧 |
| **方案B** | Teacher、Student各50000轮 | 中 | 根本改善 | 追求质量，长期方案 |

## 🚀 立即行动

### 推荐路径：先A后B

**第1步**：试试加强监督（50000轮）
```bash
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/distillation_recipes/run_stronger_supervision.py
```

监控关键指标：
```bash
# 另开终端
tail -f logs/distillation/*/stage1/student/train.log | grep teacher_action_mae
```

**成功标准**：
- teacher_action_mae < 0.25（且下降趋势）
- episode_length > 600
- 学生行为更像教师

**如果方案A还是不行**，说明教师质量是瓶颈，再用方案B训练更好的教师。

---

## 🔍 问题根源总结

你的问题链：
```
参考动作 → 教师 → 学生
(人体动作)  (质量中等) (蒸馏失败)
            ↓
       forward_velocity=0.67
       pose_track=0.69
            ↓
       蒸馏配置太弱
       (监督0.25, 概率0.40)
            ↓
       学生RL探索偏离教师
       (mae: 0.23→0.45)
```

**解决思路**：
1. 先加强监督，让学生严格跟随现有教师（方案A）
2. 如果还不行，提升教师质量（方案B）

---

## 📁 新增文件

```
sensor_training/distillation_recipes/
├── run_stronger_supervision.py   # 方案A: 加强监督
├── train_better_teacher.py        # 方案B: 训练更好的教师
├── TROUBLESHOOTING.md             # 本文件
└── ...
```

---

## 🎓 为什么之前会失败

你之前的配置假设：
- 教师质量很好 → 实际只有0.67速度，0.69姿态
- 学生能自主RL探索改进 → 实际RL让学生偏离教师
- 弱监督就够了 → 实际需要更强监督

现在的改进：
- 承认教师质量一般，要么提升教师，要么加强监督
- 降低RL自主探索，让学生更严格模仿
- 更保守的学习率和更长的训练时间

---

**现在开始吧！推荐先试方案A：**

```bash
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/distillation_recipes/run_stronger_supervision.py
```
