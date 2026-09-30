# LYTrobot 训练与测试

所有命令在仓库根目录执行。将 `<run>` 替换为实际运行目录，`<threshold>` 替换为选定的蒸馏验收阈值。

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
  --checkpoint "logs/stage1/<run>/model_final.onnx" \
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

## 传感器串行蒸馏（一键）

```bash
# 训练自然步态；同时训练鲁棒阶段时增加 --include-robust-stage
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/train_serial_distillation.py \
  --run-name imu_walk_100hz_v1

# 测试：使用 logs/serial_distillation/imu_walk_100hz_v1/pipeline.json 中的 Student 权重路径
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/play.py \
  --checkpoint "<student-checkpoint.onnx>" --device cuda --viewer --steps 0
```

## 传感器串行蒸馏（分阶段）

```bash
# Teacher Stage1
cd /home/yeah/code/lytrobot-amp
uv run python scripts/train_stage1.py \
  --device cuda --sim-backend warp --reference-ctrl-blend 1.0 \
  --max-iterations 30000 --log-root logs/teachers --log-group stage1_gait

# Teacher Stage2
cd /home/yeah/code/lytrobot-amp
uv run python scripts/train_stage2.py \
  --stage1-checkpoint "logs/teachers/stage1_gait/<run>/model_final.pt" \
  --device cuda --sim-backend warp --reference-ctrl-blend 1.0 \
  --max-iterations 30000 --log-root logs/teachers --log-group stage2_robust

# Student Stage1
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/train_distill_stage1.py \
  --teacher-checkpoint "logs/teachers/stage1_gait/<run>/model_final.pt" \
  --device cuda --sim-backend warp --max-iterations 30000

# Student Stage2
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/train_distill_stage2.py \
  --student-checkpoint "logs/sensor_policy/distillation/stage1_gait/<run>/model_final.pt" \
  --teacher-checkpoint "logs/teachers/stage2_robust/<run>/model_final.pt" \
  --device cuda --sim-backend warp --max-iterations 30000

# 测试 Teacher
cd /home/yeah/code/lytrobot-amp
uv run python scripts/play.py \
  --checkpoint "logs/teachers/stage2_robust/<run>/model_final.onnx" \
  --device cuda --viewer --steps 0

# 测试 Student
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/play.py \
  --checkpoint "logs/sensor_policy/distillation/stage2_robust/<run>/model_final.onnx" \
  --device cuda --viewer --steps 0
```

## Teacher → Student 蒸馏（无足底力，58D）

```bash
# 训练 Teacher
cd /home/yeah/code/lytrobot-amp
uv run python distillation_training/train_teacher.py \
  --sensor-variant phase_no_force --device cuda --sim-backend warp \
  --num-envs 1024 --num-steps 96 \
  --output logs/distillation/teacher_phase_no_force.pt

# 测试 Teacher
cd /home/yeah/code/lytrobot-amp
uv run python distillation_training/play_teacher.py \
  --checkpoint logs/distillation/teacher_phase_no_force.pt \
  --steps 1000 --seed 42 --camera-mode fixed

# 依次训练 BC、DAgger、Student RL
cd /home/yeah/code/lytrobot-amp
uv run python distillation_training/train_pipeline.py \
  --teacher-checkpoint logs/distillation/teacher_phase_no_force.pt \
  --run-dir logs/distillation/phase_no_force \
  --device cuda --sim-backend warp --num-envs 1024 --max-iterations 1000 \
  --max-validation-normalized-action-mse "<threshold>"

# 测试 Student
cd /home/yeah/code/lytrobot-amp
uv run python distillation_training/play.py \
  --checkpoint logs/distillation/phase_no_force/student_rl.pt \
  --device cuda --sim-backend warp --steps 1000 --seed 42
```

## Teacher → Student 蒸馏（有足底力，64D）

```bash
# 训练 Teacher
cd /home/yeah/code/lytrobot-amp
uv run python distillation_training/train_teacher.py \
  --sensor-variant phase_force --device cuda --sim-backend warp \
  --num-envs 1024 --num-steps 96 \
  --output logs/distillation/teacher_phase_force.pt

# 测试 Teacher
cd /home/yeah/code/lytrobot-amp
uv run python distillation_training/play_teacher.py \
  --checkpoint logs/distillation/teacher_phase_force.pt \
  --steps 1000 --seed 42 --camera-mode fixed

# 依次训练 BC、DAgger、Student RL
cd /home/yeah/code/lytrobot-amp
uv run python distillation_training/train_pipeline.py \
  --teacher-checkpoint logs/distillation/teacher_phase_force.pt \
  --run-dir logs/distillation/phase_force \
  --device cuda --sim-backend warp --num-envs 1024 --max-iterations 1000 \
  --max-validation-normalized-action-mse "<threshold>"

# 测试 Student
cd /home/yeah/code/lytrobot-amp
uv run python distillation_training/play.py \
  --checkpoint logs/distillation/phase_force/student_rl.pt \
  --device cuda --sim-backend warp --steps 1000 --seed 42
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

## DeepMimic 连续上下楼

```bash
# 训练
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/train_stairs_deepmimic.py \
  --device cuda --sim-backend warp --num-envs 4096 --max-iterations 30000

# 测试
cd /home/yeah/code/lytrobot-amp
uv run python sensor_training/play.py \
  --checkpoint "logs/stairs/<run>/model_best.onnx" --viewer --steps 0
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
