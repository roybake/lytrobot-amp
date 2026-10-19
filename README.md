# LYTrobot 训练与测试

## 师生蒸馏：转身与变速行走（0.6–1.28 m/s）

### 1. 训练 Teacher

```bash
cd /home/yeah/code/lytrobot-amp
.venv/bin/python scripts/train_stage1.py \
  --command-walking \
  --velocity-range 0.6 1.28 \
  --target-velocity 1.28 \
  --yaw-rate-range -0.8 0.8 \
  --num-envs 2048
```

此模式从头训练带速度和转向指令的策略，每个环境每 4–8 秒切换目标，包含直行、左右转、速度区间内部和两个端点；速度以 0.30 m/s²、转向角速度以 0.80 rad/s² 平滑变化。策略跟踪当前指令，步态节奏随速度变化；前进速度、姿态和 AMP 特征使用与朝向无关的表示，转身后仍能继续前进。训练结果位于 `logs/command_walking/<训练时间>/teacher/`。

### 2. 蒸馏 Student

Teacher 训练完成后，将 `<训练时间>` 替换为实际目录；蒸馏使用 `.pt` 教师权重：

```bash
.venv/bin/python sensor_training/train_distill_stage1.py \
  --teacher-checkpoint "logs/command_walking/<训练时间>/teacher/model_final.pt" \
  --command-walking \
  --num-envs 2048
```

Student 继承 Teacher 的速度闭区间、转向角速度范围和任务奖励。Teacher 在 Student 实际访问的状态下，用相同的速度／转向指令生成监督动作；Student 的输入为 IMU、肌腱、动作历史及可部署的步态／速度／转向指令。Student 的预瞄步态和 AMP 历史节奏同步随速度变化。输出位于同一训练目录的 `student/`，导出的 Student ONNX 独立推理。

### 3. 在 MuJoCo 中键盘控制 Student

```bash
.venv/bin/python sensor_training/play.py \
  --checkpoint "logs/command_walking/<训练时间>/student/model_final.onnx" \
  --keyboard --viewer --steps 0 \
  --target-velocity 0.6 \
  --device cpu
```

点击 MuJoCo 窗口使其获得键盘焦点：

| 按键 | 操作 |
| --- | --- |
| ↑ | 单击提速 0.05 m/s；长按持续加速，最大 1.28 m/s |
| ↓ | 单击降速 0.05 m/s；长按持续减速，最小 0.6 m/s |
| ← / → | 向左／右转身；长按持续转向并增大转向角速度，最大绝对值 0.8 rad/s |
| 松开方向键 | 保持当前速度，转向角速度指令归零，沿当前朝向前进 |
| Esc | 退出 |

上下与左右键可以组合使用。窗口显示指令速度和实测速度。闭区间约束作用于行走速度指令；从静止起步以及加减速时，实际速度存在动力学跟踪误差。此功能需要本模式训练并蒸馏得到的模型。

## 重定向动作测试：

cd /home/yeah/code/lytrobot-amp
.venv/bin/python scripts/view_reference.py \
  --motion-file robot/reference_data/stairs.npz \
  --loop

cd /home/yeah/code/lytrobot-amp
.venv/bin/python scripts/view_stairs_reference.py \
  --reference robot/reference_data/multimotion/stairs.npz \
  --viewer --loop

## 最佳：传感器串行蒸馏 Stage1（GRU128，696D，BPTT96）

蒸馏训练的 `--teacher-checkpoint` 必须指向包含教师权重和训练元数据的 `.pt` 文件；`.onnx` 用于推理播放。

```bash
cd /home/yeah/code/lytrobot-amp

# 训练 Teacher（定速）
cd /home/yeah/code/lytrobot-amp
.venv/bin/python scripts/train_stage1.py \
  --target-velocity 1.28 --num-envs 2048

# 测试 Teacher
.venv/bin/python scripts/play.py \
  --checkpoint "logs/sensor_policy/distillation/10-19-09:47:59/teacher/model_best.onnx" \
  --device cuda --viewer --steps 0
  --model-xml robot/terrains/rough.xml 
  --no-domain-randomization

# 训练 Student（定速）
cd /home/yeah/code/lytrobot-amp
.venv/bin/python sensor_training/train_distill_stage1.py \
  --teacher-checkpoint "logs/sensor_policy/distillation/10-15-09:04:28/teacher/model_best.pt" \
  --target-velocity 1.28 \
  --teacher-action-probability-decay-iterations 10000 \
  --num-envs 2048

# 测试 Student（最佳权重）
.venv/bin/python sensor_training/play.py \
  --checkpoint "logs/sensor_policy/distillation/10-06-13:24:19/student_best_plus30000/model_best.onnx" \
  --device cuda --viewer --steps 0
  --model-xml robot/terrains/rough.xml 
  --no-domain-randomization

```


## 基础行走 Stage1

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python scripts/train_stage1.py \
  --device cuda --sim-backend warp \
  --num-envs 1024 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python scripts/play.py \
  --checkpoint "logs/stage1/10-02-20:39:54/model_1500.onnx" \
  --device cuda --viewer --steps 0 --seed 42
```

## 鲁棒行走 Stage2

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python scripts/train_stage2.py \
  --stage1-checkpoint "logs/stage1/<run>/model_final.pt" \
  --device cuda --sim-backend warp \
  --num-envs 4096 --max-iterations 7500

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python scripts/play.py \
  --checkpoint "logs/stage2/<run>/model_final.onnx" \
  --device cuda --viewer --steps 0 --seed 42
```

## IMU＋肌腱＋动作历史行走 Stage1

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/train_stage1.py \
  --device cuda --sim-backend warp --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/play.py \
  --checkpoint "logs/sensor_policy/stage1/<run>/model_final.onnx" \
  --device cuda --viewer --steps 0
```

## IMU＋肌腱＋动作历史行走 Stage2

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/train_stage2.py \
  --stage1-checkpoint "logs/sensor_policy/stage1/<run>/model_final.pt" \
  --device cuda --sim-backend warp --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/play.py \
  --checkpoint "logs/sensor_policy/stage2/<run>/model_final.onnx" \
  --device cuda --viewer --steps 0
```


## DeepMimic 连续上下楼

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/train_stairs_deepmimic.py \
  --device cuda --sim-backend warp --num-envs 4096 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/play.py \
  --checkpoint "logs/stairs/10-06-13:01:44/model_final.onnx" --viewer --steps 0
```

## 蹲起传感器策略 Stage1

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python squat_training/train_stage1.py \
  --device cuda --sim-backend warp --num-envs 4096 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python squat_training/play.py \
  --checkpoint "logs/squat_sensor_policy/stage1/<run>/model_final.pt" --viewer --steps 0
```

## 蹲起传感器策略 Stage2

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python squat_training/train_stage2.py \
  --stage1-checkpoint "logs/squat_sensor_policy/stage1/<run>/model_final.pt" \
  --device cuda --sim-backend warp --num-envs 4096 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python squat_training/play.py \
  --checkpoint "logs/squat_sensor_policy/stage2/<run>/model_final.pt" --viewer --steps 0
```

## 多动作／全地形行走（两参考）

```bash
# 平地预训练
cd /home/yeah/code/lytrobot-amp
uv run python scripts/train_multimotion_walk.py \
  --reference-set two --device cuda --sim-backend warp \
  --num-envs 1024 --max-iterations 30000

# 从预训练权重训练全地形课程
cd /home/yeah/code/lytrobot-amp
uv run python scripts/train_multimotion.py \
  --reference-set two \
  --resume-checkpoint "logs/multimotion_walk_warmup/<run>/model_final.pt" \
  --resume-reset-progress --resume-reset-optimizer \
  --device cuda --sim-backend warp --num-envs 4096 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python scripts/play.py \
  --checkpoint "logs/multimotion_two_ref/<run>/model_final.onnx" \
  --stage multimotion --device cuda --viewer --num-robots 3 --steps 0 --seed 42
```

## 多动作／全地形行走（四参考）

```bash
# 平地预训练
cd /home/yeah/code/lytrobot-amp
uv run python scripts/train_multimotion_walk.py \
  --reference-set four --device cuda --sim-backend warp \
  --num-envs 1024 --max-iterations 30000

# 从预训练权重训练全地形课程
cd /home/yeah/code/lytrobot-amp
uv run python scripts/train_multimotion.py \
  --reference-set four \
  --resume-checkpoint "logs/multimotion_walk_warmup/<run>/model_final.pt" \
  --resume-reset-progress --resume-reset-optimizer \
  --device cuda --sim-backend warp --num-envs 4096 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python scripts/play.py \
  --checkpoint "logs/multimotion_four_ref/<run>/model_final.onnx" \
  --stage multimotion --device cuda --viewer --num-robots 3 --steps 0 --seed 42
```

## 无窗口测试

```bash
cd /home/yeah/code/lytrobot-amp
uv run python scripts/play.py \
  --checkpoint "logs/stage1/<run>/model_final.onnx" \
  --device cuda --no-viewer --steps 1000 --seed 42
```

#  以下内容忽略

## 传感器串行蒸馏（一键）

```bash
# 训练自然步态；同时训练鲁棒阶段时增加 --include-robust-stage
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/train_serial_distillation.py \
  --final-sensor-gait

# 测试第一阶段教师（将 <run> 替换为训练时间戳）
uv run python scripts/play.py \
  --checkpoint "logs/sensor_policy/distillation/10-06-13:24:19/teacher/model_final.pt" \
  --device cuda --viewer --steps 0

# 测试第一阶段学生
uv run python sensor_training/play.py \
  --checkpoint "logs/sensor_policy/distillation/10-06-13:24:19/student/model_best.pt" \
  --device cuda --viewer --steps 0

# 测试第二阶段教师
uv run python scripts/play.py \
  --checkpoint "logs/distillation/<run>/stage2/teacher/model_final.onnx" \
  --device cuda --viewer --steps 0

# 测试第二阶段学生
uv run python sensor_training/play.py \
  --checkpoint "logs/distillation/<run>/stage2/student/model_final.onnx" \
  --device cuda --viewer --steps 0
```

## 传感器串行蒸馏（分阶段）

以下命令的 `<run>` 使用同一个训练时间戳，各阶段目录会自动创建。

```bash
# Teacher Stage1
cd /home/yeah/code/lytrobot-amp
uv run python scripts/train_stage1.py \
  --device cuda --sim-backend warp --reference-ctrl-blend 1.0 \
  --max-iterations 50000 --log-dir "logs/distillation/<run>/stage1/teacher"

# Teacher Stage2
cd /home/yeah/code/lytrobot-amp
uv run python scripts/train_stage2.py \
  --stage1-checkpoint "logs/distillation/<run>/stage1/teacher/model_final.pt" \
  --device cuda --sim-backend warp --reference-ctrl-blend 1.0 \
  --max-iterations 50000 --log-dir "logs/distillation/<run>/stage2/teacher"

# Student Stage1
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/train_distill_stage1.py \
  --teacher-checkpoint "logs/distillation/<run>/stage1/teacher/model_final.pt" \
  --device cuda --sim-backend warp --max-iterations 50000 \
  --log-dir "logs/distillation/<run>/stage1/student"

# Student Stage2
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/train_distill_stage2.py \
  --student-checkpoint "logs/distillation/<run>/stage1/student/model_final.pt" \
  --teacher-checkpoint "logs/distillation/<run>/stage2/teacher/model_final.pt" \
  --device cuda --sim-backend warp --max-iterations 50000 \
  --log-dir "logs/distillation/<run>/stage2/student"

# 测试 Teacher
cd /home/yeah/code/lytrobot-amp
uv run python scripts/play.py \
  --checkpoint "logs/sensor_policy/distillation/10-06-13:24:19/teacher/model_final.onnx" \
  --device cuda --viewer --steps 0

# 测试 Student
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/play.py \
  --checkpoint "logs/distillation/<run>/stage2/student/model_final.onnx" \
  --device cuda --viewer --steps 0
```

## 传感器直接训练（无足底力，56D）

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python foot_sensor_training/train_no_force_stage1.py \
  --device cuda --sim-backend warp \
  --num-envs 1024 --num-steps 96 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python foot_sensor_training/play.py \
  --checkpoint "logs/foot_sensor_no_force/stage1/<run>/model_final.pt" --viewer --steps 0
```

## 传感器直接训练（有足底力，62D）

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python foot_sensor_training/train_stage1.py \
  --device cuda --sim-backend warp \
  --num-envs 1024 --num-steps 96 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python foot_sensor_training/play.py \
  --checkpoint "logs/foot_sensor/stage1/<run>/model_final.pt" --viewer --steps 0
```

## 传感器直接训练（相位＋无足底力，58D）

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python foot_sensor_training/train_phase_no_force_stage1.py \
  --device cuda --sim-backend warp \
  --num-envs 1024 --num-steps 96 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python foot_sensor_training/play.py \
  --checkpoint "logs/foot_sensor_phase/stage1/<run>/model_final.pt" --viewer --steps 0
```

## 传感器直接训练（相位＋足底力，64D）

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python foot_sensor_training/train_phase_force_stage1.py \
  --device cuda --sim-backend warp \
  --num-envs 1024 --num-steps 96 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python foot_sensor_training/play.py \
  --checkpoint "logs/foot_sensor_phase_force/stage1/<run>/model_final.pt" --viewer --steps 0
```
