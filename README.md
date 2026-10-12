# LYTrobot 训练与测试

所有命令在仓库根目录执行。将 `<run>` 替换为实际运行目录，`<threshold>` 替换为选定的蒸馏验收阈值。

平地单动作行走默认使用 `robot/reference_data/walk1_mjcf_repaired.npz`：45 帧、40 Hz、周期 1.125 秒，循环位移 0.97875 米，配套的目标速度为 `0.87 m/s`，且 `--motion-loop true`。该文件按 `robot/MJCFmodelposition.xml` 的实际关节布局修正了被动髌骨、足底网格接触、关节限位和周期速度。原始人体录制 `walk1.npz` 及长序列 `walk1_mjcf.npz` 保留作来源与对照；如显式使用长序列，需同时显式指定匹配的速度、循环方式并重新审核足底接触。Teacher、Student、Stage1、Stage2 应使用同一参考版本，已按旧序列训练的权重不能直接视为新参考的已验收结果。

## 行走参考与地面接触验收

```bash
# 用 MuJoCo 的实际碰撞网格逐帧检查几何接触、关节限位、髌骨约束和周期速度
uv run python scripts/audit_walk_contact.py --output logs/walk_ground_contact_audit.json

# 静态参考姿态与完整步态回放
uv run python robot/Launch.py --headless --steps 1
uv run python scripts/view_reference.py --motion-file robot/reference_data/walk1_mjcf_repaired.npz --loop

# 自由基座、受力受限的动力学回放，单独检查跌倒、持续支撑与前进能力
uv run python scripts/test_reference_dynamics.py \
  --motion-file robot/reference_data/walk1_mjcf_repaired.npz \
  --source training --seconds 10 --trials 8 \
  --output logs/walk_reference_dynamics.json
```

当前仓库已有修复轨迹对应的运动学审计结果，最大支撑间隙约 0.82 毫米、最小足底间隙约 -0.095 毫米，但动力学稳定性需要通过上述独立实验及最终策略评估确认，几何接触合格并不意味着电机驱动下能够长期保持平衡。

## 最佳：传感器串行蒸馏 Stage1（GRU128，696D，BPTT96）

沿用 `logs/sensor_policy/distillation/10-06-13:24:19` 的配置和目录布局，Teacher、Student 各训练 50000 轮。默认保存到 `logs/sensor_policy/distillation/<run>/teacher` 和 `<run>/student`。Teacher 启动时自动生成时间戳（`MM-DD-HH:MM:SS`），Student 根据教师权重路径复用该时间戳；将后续命令中的 `<run>` 替换为 Teacher 输出的时间戳。
W&B 名称为 `<训练时间戳>-teacher`、`<训练时间戳>-student`。

```bash
cd /home/yeah/code/lytrobot-amp

# 训练 Teacher
.venv/bin/python scripts/train_stage1.py \
  --num-envs 1024 --max-iterations 50000 \
  --evaluate-gait --gait-fidelity-selection --gait-evaluation-seconds 30

# 测试 Teacher
.venv/bin/python scripts/play.py \
  --checkpoint "logs/sensor_policy/distillation/<run>/teacher/model_final.pt" \
  --device cuda --viewer --steps 0

# 训练 Student
.venv/bin/python sensor_training/train_distill_stage1.py \
  --teacher-checkpoint "logs/sensor_policy/distillation/<run>/teacher/model_final.pt" \
  --num-envs 1024 --num-steps 96

# 测试 Student（最佳权重）
.venv/bin/python sensor_training/play.py \
  --checkpoint "logs/sensor_policy/distillation/<run>/student/model_best.pt" \
  --device cuda --viewer --steps 0
```

## 推荐：最终纯传感器人体步态

实机现有7组IMU和14条肌腱编码器；Teacher → Student Stage1 → Student Stage2 DR → ONNX。
默认GRU256、45D状态估计、统一测量肌腱派生特征、BPTT96，关闭future prediction。

```bash
# Stage1：先建立合格Teacher，再训练Student
.venv/bin/python sensor_training/train_final_sensor_gait.py --phase stage1

# 或一次执行全部阶段
.venv/bin/python sensor_training/train_final_sensor_gait.py
```

Stage2接续、导出和验收详见 [最终训练说明](sensor_training/STATE_ESTIMATION.md)。
以下入口保留用于基础策略、专项任务或历史兼容。

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
  --device cuda --sim-backend warp --num-envs 1024 --max-iterations 50000 \
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
  --device cuda --sim-backend warp --num-envs 1024 --max-iterations 50000 \
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