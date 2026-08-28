# LYTrobot AMP 双足机器人训练

## 环境与约定

```bash
cd /home/yeah/code/LYTrobot_amp
uv sync
```

以下命令均在项目根目录执行。将 `<run>` 替换为对应训练目录名；训练用
`model_final.pt`，播放可替换为同目录的 `model_best.pt` 或 ONNX 文件。Stage 1
关闭域随机化，Stage 2 必须加载同一任务的 Stage 1 checkpoint 并启用域随机化。

所有 Actor 的输入均为 56 维传感器帧（42 维 IMU + 14 维肌腱长度），不包含参考动作、
相位、地形或仿真真值。`--viewer --steps 0` 会持续播放并自动录屏，录像保存在 checkpoint
所属任务的 `recordings/` 目录。

## 任务命令

### 1. 普通行走

```bash
# Stage 1 训练
uv run python scripts/train_stage1.py

# Stage 1 测试
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage1/<run>/model_final.pt \
  --stage stage1 --viewer --steps 0

# Stage 2 训练
uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/<run>/model_final.pt

# Stage 2 测试
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage2/<run>/model_final.pt \
  --stage stage2 --viewer --steps 0
```

### 2. 蹲起

```bash
# Stage 1 训练
uv run python squat_training/train_stage1.py

# Stage 1 测试
uv run python squat_training/play.py \
  --checkpoint logs/squat_sensor_privileged_v2/stage1/<run>/model_final.pt \
  --stage stage1 --viewer --steps 0

# Stage 2 训练
uv run python squat_training/train_stage2.py \
  --stage1-checkpoint logs/squat_sensor_privileged_v2/stage1/<run>/model_final.pt

# Stage 2 测试
uv run python squat_training/play.py \
  --checkpoint logs/squat_sensor_privileged_v2/stage2/<run>/model_final.pt \
  --stage stage2 --viewer --steps 0
```

### 3. 全地形多动作（平地、崎岖地、坡道、上下楼梯）

```bash
# Stage 1 训练
uv run python scripts/train_multimotion_stage1.py

# Stage 1 测试
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/stage1/<run>/model_final.pt \
  --stage multimotion --viewer --steps 0

# Stage 2 训练
uv run python scripts/train_multimotion_stage2.py \
  --stage1-checkpoint logs/multimotion/stage1/<run>/model_final.pt

# Stage 2 测试
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/stage2/<run>/model_final.pt \
  --stage multimotion --viewer --steps 0
```

### 4. 无阶梯单参考 AMP（平地、崎岖地、上下坡）

```bash
# Stage 1 训练
uv run python scripts/train_multimotion_no_stairs_stage1.py

# Stage 1 测试
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/no_stairs_amp_single/stage1/<run>/model_final.pt \
  --stage stage1 --viewer --steps 0

# Stage 2 训练
uv run python scripts/train_multimotion_no_stairs_stage2.py \
  --stage1-checkpoint \
  logs/multimotion/no_stairs_amp_single/stage1/<run>/model_final.pt

# Stage 2 测试
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/no_stairs_amp_single/stage2/<run>/model_final.pt \
  --stage stage2 --viewer --steps 0
```

### 5. 无阶梯 Sensor Teacher

```bash
# 训练
uv run python scripts/train_multimotion_no_stairs_amp_teacher.py

# 测试
uv run python sensor_training/play.py \
  --checkpoint \
  logs/multimotion/multimotion_no_stairs_amp_sensor_teacher_v2/<run>/model_final.pt \
  --viewer --steps 0
```

### 6. 无阶梯 Student 蒸馏

`--teacher-checkpoint` 只接受改造前保存的冻结特权 Teacher，不要传入新的 Sensor Teacher checkpoint。

```bash
# 训练
uv run python scripts/train_multimotion_no_stairs_amp_student.py \
  --teacher-checkpoint \
  logs/multimotion/multimotion_no_stairs_amp_teacher/<run>/model_best.pt

# 测试
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/multimotion_no_stairs_amp_student/<run>/model_final.pt \
  --viewer --steps 0
```

### 7. 独立连续上下楼梯

```bash
# Stage 1 训练
uv run python scripts/train_stairs_stage1.py

# Stage 1 测试
uv run python sensor_training/play.py \
  --checkpoint logs/stairs/stage1/<run>/model_final.pt \
  --stage stage1 --viewer --steps 0

# Stage 2 训练
uv run python scripts/train_stairs_stage2.py \
  --stage1-checkpoint logs/stairs/stage1/<run>/model_final.pt

# Stage 2 测试
uv run python sensor_training/play.py \
  --checkpoint logs/stairs/stage2/<run>/model_final.pt \
  --stage stage2 --viewer --steps 0
```

### 8. 多动作网络布局的平地预训练

```bash
# Stage 1 训练
uv run python scripts/train_multimotion_walk_stage1.py

# Stage 1 测试
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/multimotion_walk_warmup/stage1/<run>/model_final.pt \
  --stage stage1 --no-serial-terrain-course --viewer --steps 0

# Stage 2 训练
uv run python scripts/train_multimotion_walk_stage2.py \
  --stage1-checkpoint \
  logs/sensor_policy/multimotion_walk_warmup/stage1/<run>/model_final.pt

# Stage 2 测试
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/multimotion_walk_warmup/stage2/<run>/model_final.pt \
  --stage stage2 --no-serial-terrain-course --viewer --steps 0
```

### 9. 旧无阶梯超参数复现

```bash
# 训练
uv run python scripts/train_no_stairs_old.py

# 测试
uv run python sensor_training/play.py \
  --checkpoint \
  logs/sensor_policy/no_stairs_old_sensor_privileged_v2/<run>/model_final.pt \
  --viewer --steps 0
```

## 其他检查

```bash
# 不加载 checkpoint，查看机器人 MJCF
uv run python scripts/view_model.py --model robot/MJCFmodelposition.xml --steps 2000

# 完整测试集
uv run python -m pytest -q

# 录屏逻辑测试
uv run python -m pytest -q tests/test_play_recording.py
```

任务实现、参考动作和重定向说明分别见 [multimotion/README.md](multimotion/README.md)、
[sensor_training/README.md](sensor_training/README.md) 与 [squat_training/README.md](squat_training/README.md)。
