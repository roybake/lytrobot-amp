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

### 3. 08-09 多地形串联（崎岖地、上下坡；无楼梯）

```bash
# Stage 1 训练
uv run python scripts/train_multimotion_stage1.py

# Stage 1 测试
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/no_stairs_amp_single/stage1/<run>/model_final.onnx \
  --stage stage1 --viewer --steps 0

# Stage 2 训练
uv run python scripts/train_multimotion_stage2.py \
  --stage1-checkpoint logs/multimotion/no_stairs_amp_single/stage1/<run>/model_final.pt

# Stage 2 测试
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/no_stairs_amp_single/stage2/<run>/model_final.pt \
  --stage stage2 --viewer --steps 0
```

### 4. 四个独立地形任务

独立地形训练现在只保留一套 canonical 配置，新训练和回放统一使用
`logs/independent_terrain/<terrain>/`。历史 checkpoint 不会被任何正式 launcher 自动选用。
本轮修复了独立坡面高度基准、下坡静止出生和起步检查时机，并调整了最佳模型选择。
从头训练及验收说明见 [独立训练任务](docs/independent_training_tasks.md)。

平地、上坡、下坡和不平路面现在分别作为独立任务训练；每个任务只使用自己的参考
动作和地形，不共享串联课程的 terrain 序列。四个任务默认把运行时地形铺满约
`24 m × 24 m` 的训练区域：不平路面平铺原始高度图，上坡/下坡使用连续斜面，平地
使用原有无限平面；原始地形文件只读、不被改写。默认直接训练 IMU + 肌腱长度传感器
Actor，不使用 Teacher action distillation；训练期 asymmetric critic 仍然保留。

```bash
# 平地训练（Stage 1，无随机化）
uv run python scripts/train_flat_stage1.py
# 平地训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_flat_stage2.py \
  --stage1-checkpoint logs/independent_terrain/flat/stage1/<run>/model_final.pt

# 平地测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/flat/stage1/<run>/model_best.onnx \
  --stage stage1 --viewer --steps 0
# 平地测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/flat/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0

# 不平路面训练（Stage 1，无随机化）
uv run python scripts/train_rough_stage1.py
# 不平路面训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_rough_stage2.py \
  --stage1-checkpoint logs/independent_terrain/rough/stage1/<run>/model_best.pt

# 不平路面测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/rough/stage1/<run>/model_best.onnx \
  --stage stage1 --viewer --steps 0
# 不平路面测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/rough/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0

# 上坡训练（Stage 1，无随机化）
uv run python scripts/train_slope_up_stage1.py
# 上坡训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_slope_up_stage2.py \
  --stage1-checkpoint logs/independent_terrain/slope_up/stage1/<run>/model_best.pt

# 上坡测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/slope_up/stage1/<run>/model_best.pt \
  --stage stage1 --viewer --steps 0
# 上坡测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/slope_up/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0

# 下坡训练（Stage 1，无随机化）
uv run python scripts/train_slope_down_stage1.py
# 下坡训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_slope_down_stage2.py \
  --stage1-checkpoint logs/independent_terrain/slope_down/stage1/<run>/model_best.pt

# 下坡测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/slope_down/stage1/<run>/model_best.pt \
  --stage stage1 --viewer --steps 0
# 下坡测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/slope_down/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0
```

上面命令中的 `<run>` 要替换成实际运行目录名。如果只生成了最新 checkpoint，
可以把测试命令里的 `model_best.pt` 换成同目录的 `model_final.pt`。

代码接口快速测试（不启动训练）：

```bash
PYTHONPATH=. uv run pytest -q tests/test_independent_terrain_tasks.py tests/test_more_terrains.py
```

建议先用不蒸馏版本建立四个基线。单地形没有技能选择歧义，直接传感器闭环训练通常比
把特权 Teacher 的参考条件/残差迁移给 Student 更干净；仅当同一地形、同一模型和同一
控制时钟下的 Teacher 能显著改善探索时，再用独立 task ID 做小权重蒸馏对照。详细契约和
实验准则见 [`docs/independent_training_tasks.md`](docs/independent_training_tasks.md)。

### 5. 无阶梯单策略 AMP（平地、崎岖地、上下坡）

这是一个共享单策略、使用四段 terrain-aligned AMP 专家动作的任务：平地使用
`walk`，崎岖地使用 `rough_walk`，上坡和下坡分别使用 `slope_up` 与
`slope_down`。动作权重均为 0.25，目标速度统一为 `0.862295 m/s`。

```bash
# Stage 1 训练
uv run python scripts/train_multimotion_no_stairs_stage1.py

# Stage 1 测试
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/no_stairs_amp_single/stage1/09-01-23:13:40/model_best.pt\
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

### 6. 无阶梯 AMP Teacher

```bash
# 训练
uv run python scripts/train_multimotion_no_stairs_amp_teacher.py
.venv/bin/python scripts/train_multimotion_no_stairs_amp_teacher_lane_hold.py
# 测试（Teacher 是参考条件化的特权策略，obs_dim=131，
# 必须用 scripts/play.py 回放；sensor_training/play.py 只适用于
# 仅 IMU+肌腱长度输入的传感器策略，obs_dim=56）
uv run python scripts/play.py \
  --checkpoint \
  logs/multimotion/multimotion_no_stairs_amp_teacher/09-01-23:14:51/model_final.onnx\
  --viewer --steps 0
```

历史 08-06 多动作 checkpoint 的公平回放使用冻结模型与地形兼容配置，例如：

```bash
uv run python scripts/play.py \
  --checkpoint logs/multimotion/08-06-10:56:09/model_final.pt \
  --legacy-0806-compat --seed 42 --no-episode-random-seed \
  --num-robots 1 --viewer --steps 0 --no-record
```

### 7. 无阶梯 Student 蒸馏

Teacher 使用当前无阶梯配置训练 `walk/rough_walk/slope_up/slope_down` 四段
动作，四段权重均为 0.25；其中上下坡动作来自平地人体动作的 terrain-relative
full-foot IK 重定向。Student 蒸馏时使用与 Teacher 完全相同的动作顺序、权重、
地形类型和控制时钟；Student launcher 会拒绝仍记录
`stair_up/stair_down` 的旧 Teacher checkpoint。Teacher 所需标签只保留在底层
环境，Student Actor 输入仍严格只有 IMU 和肌腱长度。

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

### 8. 独立连续上下楼梯

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

### 9. 多动作网络布局的平地预训练

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

### 10. 旧无阶梯超参数复现

```bash
# 训练
uv run python scripts/train_no_stairs_old.py

# 测试
uv run python sensor_training/play.py \
  --checkpoint \
  logs/sensor_policy/no_stairs_old_0806_teacher/<run>/model_final.pt \
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
