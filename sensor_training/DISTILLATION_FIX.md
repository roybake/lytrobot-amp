# 传感器蒸馏失败诊断与改进方案

## 问题诊断

### 根本原因：观测信息差距过大

你的蒸馏一直失败的核心原因：

**教师 vs 学生的观测维度差距**
- **教师策略**: ~100维特权观测
  - 14个关节位置 + 14个关节速度
  - 根部位置、速度、朝向
  - 4个足部的3D位置和速度
  - 参考轨迹的关节、足部目标
  - 接触力传感器（是否着地）

- **学生策略**: ~20-30维传感器观测
  - IMU: 6维（3轴加速度 + 3轴角速度）
  - 肌腱长度: 14维
  - 历史动作: 14维
  - **缺失**: 关节角度、足部位置、接触状态、参考轨迹

这个信息差距让学生根本无法理解教师"为什么"做出某个动作决策。

### 你的训练日志显示的症状

从 `logs/sensor_policy/distillation/stage1_gait/20260930_sensor_walk_v3`:
```
teacher_action_mae 一直在 0.24-0.39 之间
mean_episode_length 从 985 下降到 519（越来越差）
task_forward_velocity 在 0.61-0.71 波动（不稳定）
```

**分析**：
- `teacher_action_mae > 0.24` 说明学生完全学不会教师的动作模式
- episode_length 下降说明学生越来越依赖教师行为（`teacher_action_probability=0.80`太高）
- 当教师行为逐渐退出，学生自己的策略崩溃

## 改进方案（3个选择）

### 方案1: 快速验证（推荐先做）⭐

**目的**: 用1000步小规模测试状态估计器是否有效

```bash
./sensor_training/quick_validate.sh
```

**关键改进点**:
1. **启用状态估计器**: 让GRU从传感器历史重建关节位置、速度、接触状态
2. **更长历史窗口**: frame_count=8（给GRU更多信息）
3. **弱监督**: teacher_action_loss_coef=0.20, 降低对教师的依赖
4. **更强RL**: task_reward_lerp=0.85, 让学生更多依靠奖励学习

**成功标准**（1000步后）:
- teacher_action_mae 下降到 < 0.20
- state/joint_position_loss < 0.15
- mean_episode_length > 400
- task_forward_velocity > 0.65

**时间**: ~30分钟

### 方案2: 单阶段改进蒸馏

**适用**: 快速验证通过后，跑完整训练

```bash
./sensor_training/distill_improved.sh
```

**核心配置**:
```python
# 状态估计器（让学生"看到"更多）
state_estimator_enabled = True
state_joint_position_loss_weight = 3.0
state_root_velocity_loss_weight = 3.0
state_contact_loss_weight = 2.5

# 弱监督（避免过度依赖教师）
teacher_action_loss_coef = 0.25 → 0.05
teacher_action_probability = 0.40 → 0.0 (3000步衰减)

# 更强RL目标
task_reward_lerp = 0.85  # 85%任务奖励, 15%AMP
sensor_natural_gait_reward_coef = 3.0
reference_gait_style_reward_coef = 5.0
```

**预期结果** (20000步):
- teacher_action_mae < 0.18
- mean_episode_length > 600
- task_forward_velocity > 0.75

**时间**: ~8小时

### 方案3: 两阶段蒸馏（最稳健）⭐⭐

**适用**: 如果方案2效果仍不理想

```bash
./sensor_training/distill_two_stage.sh
```

**阶段1**: 纯状态估计预训练（3000步）
- 目标: 让GRU学会从传感器历史重建特权状态
- 方法: 高权重状态loss，**暂时不监督教师动作**
- 学生通过RL自由探索，同时学习状态估计

**阶段2**: 基于状态估计的策略蒸馏（15000步）
- 目标: 在良好的状态估计基础上，模仿教师策略
- 方法: 降低状态loss权重，启用温和的教师监督

**为什么更稳健**:
- 信息差距分解成两步：先弥补观测差距，再学习策略
- 每一步都可独立验证
- 避免状态估计和策略学习互相干扰

**时间**: ~10小时（可中断检查）

## 技术原理

### 状态估计器的作用

你的代码里已经实现了状态估计器（[networks.py:287-293](src:networks.py:287-293)），但可能没启用或权重太低：

```python
# 网络结构
GRU(IMU+肌腱历史) → hidden state (128/192维)
                    ↓
        ┌──────────┴──────────┐
        ↓                     ↓
   State Heads            Actor Head
   ↓                          ↓
关节位置/速度/接触        动作决策
```

**工作原理**:
1. GRU从8帧传感器历史中积累信息
2. State heads从GRU隐状态重建：
   - 14个关节位置
   - 14个关节速度
   - 根部3D速度
   - 左右足接触状态（0/1）
   - 可选：足部相对骨盆的位置

3. Actor可以选择使用估计的状态作为输入

**关键**: 有了这些估计状态，学生的"视野"接近教师，才能理解教师的决策逻辑。

### 为什么之前的配置失败

你之前的配置（从distillation_entrypoints.py推断）：
```python
teacher_action_loss_coef = 1.0  # 太强
teacher_action_probability = 0.80  # 太高
state_estimator_enabled = False  # 未启用！
```

**问题**:
1. 学生80%时间执行教师动作，只有20%自己探索
2. 强行模仿教师动作，但学生"看不到"教师的决策依据
3. 教师行为衰减后，学生的策略完全没学会，立即崩溃

**新配置如何解决**:
1. 启用状态估计，弥补观测差距
2. 降低teacher_action_probability到0.35-0.40，让学生更多自主探索
3. 降低teacher_action_loss_coef到0.20-0.30，作为"温和引导"而非"强制模仿"
4. 提高task_reward_lerp到0.85，让RL目标占主导

## 执行建议

### 第1步: 快速验证（必须）

```bash
# 确保教师checkpoint存在
ls -lh logs/stage1/10-02-20:39:54/model_best.pt

# 运行快速测试（~30分钟）
./sensor_training/quick_validate.sh

# 等待完成，检查输出的指标对比
```

**判断**:
- ✅ teacher_action_mae下降明显 → 继续方案2
- ❌ 指标无明显改善 → 跳到方案3

### 第2步: 选择完整方案

**如果快速验证通过**:
```bash
# 先试方案2（单阶段，更快）
./sensor_training/distill_improved.sh
```

**如果方案2仍不理想，或你想最稳健**:
```bash
# 方案3（两阶段，更可控）
./sensor_training/distill_two_stage.sh

# 可在阶段1后中断检查状态估计质量
# 按提示检查metrics后决定是否继续
```

### 第3步: 监控训练

**关键指标**（每500步检查）:

```bash
# 实时监控
cd logs/sensor_policy/distillation_v2/*/
tail -f train.log

# 或查看wandb dashboard
```

**良好训练的信号**:
```
迭代     teacher_mae  episode_len  forward_vel  state_joint_loss
500      0.22         350          0.68         0.18
1000     0.19         480          0.72         0.14
2000     0.17         590          0.76         0.11
4000     0.15         680          0.78         0.09
```

**失败信号**（需要调整）:
- teacher_mae 不下降或上升
- episode_length < 300 持续
- state losses 不收敛（> 0.30）

## 如果所有方案都失败

### 终极方案：分层训练

```bash
# Level 0: 直接训练sensor policy，允许特权critic
python sensor_training/train_stage1.py \
  --state-estimator-enabled \
  --actor-use-estimated-state \
  # ... 完整传感器观测，但critic可用特权信息

# Level 1: 用Level 0作为teacher蒸馏
# 此时信息差距小很多（都是传感器，只是critic不同）
```

这种方法虽然多一步，但几乎保证成功，因为每一步的信息差距都很小。

## 理论背景

这个问题在模仿学习领域叫 **"observation mismatch"** 或 **"perception gap"**:

**经典解决方案**:
1. **Privileged learning** (你正在用)
2. **State estimation/reconstruction** (我们新加的)
3. **Domain adaptation**
4. **Curriculum learning** (逐步缩小差距)

参考文献:
- NVIDIA Isaac Gym: 使用状态估计器训练四足机器人
- Deep Mimic: 用phase variable弥补观测差距
- Learning to Walk in Minutes: 明确分离特权训练和传感器部署

## 总结

**核心洞察**: 你的问题不是超参数调整，而是**根本的信息差距**。

**解决方案**: 不是降低学习率或调整reward权重，而是**让学生能"看到"教师的决策依据**。

**实现方式**: 状态估计器 + 弱监督 + 强RL引导。

**行动**: 先跑 `quick_validate.sh`，30分钟知道方向对不对，再决定完整方案。

祝训练成功！有问题随时问。
