# 改进的传感器策略蒸馏方案

## 问题分析
当前蒸馏失败的根本原因是**观测信息差距过大**：
- 教师可以看到完整的关节位置、速度、足部接触、参考轨迹
- 学生只能看到IMU角速度/加速度 + 肌腱长度
- 这种信息不对称导致学生无法理解教师为什么做出某个动作

## 改进策略

### 方案1: 渐进式状态估计蒸馏（推荐）

不直接模仿教师动作，而是分两步：
1. **训练状态估计器**：让学生从传感器历史重建特权状态
2. **策略蒸馏**：基于估计状态模仿教师策略

```python
# 已有的状态估计功能 - 需要启用
state_estimator_enabled = True
state_estimator_include_foot_position = True
actor_use_estimated_state = True

# 状态估计loss权重（需要调高）
state_joint_position_loss_weight = 2.0
state_joint_velocity_loss_weight = 1.0
state_root_velocity_loss_weight = 2.0
state_contact_loss_weight = 2.0
state_foot_position_loss_weight = 1.5
```

### 方案2: 更长的历史窗口 + 更大的GRU

```python
# 增加历史帧数，让GRU能积累更多信息
frame_count = 10  # 当前可能是5，增加到10
recurrent_hidden_dim = 256  # 从128增加到256

# 确保测量层包含速度估计
include_tendon_velocity = True
include_tendon_tracking_error = True
tendon_velocity_filter_alpha = 0.95
```

### 方案3: 改进的蒸馏超参数

```bash
# 核心改动
--teacher-action-loss-coef 0.30  # 降低监督强度
--teacher-action-loss-final-coef 0.05  # 最终几乎不监督
--teacher-action-loss-scale 0.05  # 降低scale（原来0.10）

# 行为概率 - 更激进的衰减
--teacher-action-probability 0.50  # 降低初始值（原来0.80）
--teacher-action-probability-final 0.0
--teacher-action-probability-decay-iterations 2000  # 更快衰减

# RL目标权重
--task-reward-lerp 0.85  # 提高任务奖励权重（原来0.75）
--amp-reward-coef 1.0

# 探索
--init-std 0.20  # 增加初始探索
--max-action-std 0.30
--entropy-coef 0.005  # 鼓励探索

# 学习率 - 更保守
--learning-rate 5e-5
--max-learning-rate 5e-5
--target-kl 0.008  # 更严格的KL约束
```

### 方案4: 预训练状态估计器（最有效）

```bash
# 第一阶段：纯状态估计预训练（2000 iterations）
python sensor_training/train_stage1.py \
  --teacher-checkpoint logs/teachers/stage1_gait/09-30-17:33:59/model_best.pt \
  --state-estimator-enabled \
  --state-estimator-include-foot-position \
  --actor-use-estimated-state \
  --state-joint-position-loss-weight 4.0 \
  --state-joint-velocity-loss-weight 2.0 \
  --state-root-velocity-loss-weight 4.0 \
  --state-contact-loss-weight 3.0 \
  --state-foot-position-loss-weight 2.0 \
  --teacher-action-loss-coef 0.0 \  # 暂时不监督动作
  --task-reward-lerp 0.60 \
  --max-iterations 2000 \
  --log-group stage1_state_estimator_pretrain

# 第二阶段：基于状态估计的蒸馏
python sensor_training/train_distill_stage1.py \
  --teacher-checkpoint logs/teachers/stage1_gait/09-30-17:33:59/model_best.pt \
  --resume-checkpoint logs/.../stage1_state_estimator_pretrain/model_2000.pt \
  --teacher-action-loss-coef 0.20 \
  --teacher-action-probability 0.30 \
  --state-joint-position-loss-weight 2.0 \
  --state-root-velocity-loss-weight 2.0 \
  ...
```

## 调试检查项

### 1. 状态估计质量
```python
# 在训练日志中查找这些指标
state/joint_position_loss  # 应该 < 0.1
state/root_velocity_loss   # 应该 < 0.3
state/contact_loss         # 应该 < 0.2
state/foot_position_loss   # 应该 < 0.05
```

### 2. 教师-学生动作差异
```python
teacher_action_mae  # 目标 < 0.15（当前0.24）
```

### 3. 行为表现
```python
mean_episode_length     # 目标 > 800
task_forward_velocity   # 目标 > 0.70
mean_episode_success_rate  # 目标 > 0.60
```

## 终极方案：分层蒸馏

如果上述方案仍然失败，考虑分层训练：

1. **Level 0**: 直接训练sensor policy with privileged critic（已证明可行）
2. **Level 1**: 用Level 0作为teacher蒸馏到纯sensor（信息差距较小）
3. **Level 2**: Fine-tune with hardware sensor noise

这种方法信息差距更平滑，每一步都可验证。
