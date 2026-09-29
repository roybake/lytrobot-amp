# 传感器串行蒸馏

部署策略始终只使用 IMU 与肌肉长度；特权 Teacher 仅在训练期间读取
仿真完整状态，并且不会写入导出的 ONNX Actor。

蒸馏与直接行走训练是两条隔离路径：`train_stage1.py`/`train_stage2.py` 使用普通
`SensorHistoryPPOAMPTrainer`，其配置、rollout、loss 和 checkpoint 中均无 Teacher；只有
`train_distill_stage1.py`/`train_distill_stage2.py` 会进入独立的 distillation 环境、配置与
trainer。普通入口不接受 `--teacher-*` 参数，蒸馏入口则强制要求 Teacher checkpoint。

Teacher 和传感器 Student 必须保持相同的 `--decimation`。Teacher 可以带参考控制
基线：蒸馏器会把 Teacher 动作换算成绝对腱长目标，再映射为 Student 的实测腱长残差。
因此 Student 和导出的 ONNX Actor 仍使用 `reference_ctrl_blend=0.0`，不需要参考相位。
所有入口固定使用 `walk1_omniretarget.npz` 和其平均速度 `0.87 m/s`。不要复用由旧
`walk1_mjcf.npz` 训练的 Teacher；该数据脚趾关节静止且允许完全伸膝，会把直腿踮脚
步态原样蒸馏给 Student。

## 1. 训练特权 Teacher

```bash
python3 scripts/train_stage1.py \
  --reference-ctrl-blend 0.0 \
  --max-iterations 30000 \
  --log-root logs/teachers --log-group stage1_gait

python3 scripts/train_stage2.py \
  --stage1-checkpoint logs/teachers/stage1_gait/<run>/model_final.pt \
  --reference-ctrl-blend 0.0 \
  --max-iterations 30000 \
  --log-root logs/teachers --log-group stage2_robust
```

stage2 的任务奖励与 stage1 相同；它只增加域随机化来学习恢复行为。

## 2. 蒸馏传感器 Student

```bash
python3 sensor_training/train_distill_stage1.py \
  --teacher-checkpoint logs/teachers/stage1_gait/<run>/model_final.pt

python3 sensor_training/train_distill_stage2.py \
  --student-checkpoint logs/sensor_policy/distillation/stage1_gait/<run>/model_final.pt \
  --teacher-checkpoint logs/teachers/stage2_robust/<run>/model_final.pt
```

每一阶段默认训练 30000 轮。第一阶段的 Teacher 动作 Huber 损失系数从
`1.0` 退火到 `0.1`，第二阶段从 `0.5` 退火到 `0.1`。训练数据由 Student
自己执行并访问到的状态组成，Teacher 只给这些 on-policy 状态打动作标签，
因此不会破坏 PPO 的 on-policy 动作似然。Teacher 观测中的上一动作也是该状态里真实执行过的
Student 动作，不再注入未作用于动力学的 Teacher 假想动作历史。

Student 强制使用100 Hz控制和48步 recurrent BPTT，并显式读取当前帧加前5帧
IMU/肌腱历史，梯度窗口为0.48秒；其他控制频率会在训练启动前报错。GRU状态跨rollout
持续保留，且每步都有Teacher动作监督。直接蒸馏入口使用2048个环境，使每轮仍为98,304个
transition。串行总控会根据 Teacher
的 `--num-envs` 自动推导 Student 环境数；可用 `--student-num-envs` 和
`--student-num-steps` 单独覆盖。

## 日志目录

```text
logs/
├── teachers/
│   ├── stage1_gait/<run>/
│   └── stage2_robust/<run>/
└── sensor_policy/
    └── distillation/
        ├── stage1_gait/<run>/
        └── stage2_robust/<run>/
```

`metrics.csv` 中新增：`teacher_action_loss`、`teacher_action_mae` 和
`teacher_action_loss_coef`。每个 Student checkpoint 的 `distillation` 字段
记录 Teacher 路径、迭代号及当前蒸馏系数，便于复盘训练来源。

## 一键运行全部四个阶段

```bash
uv run python sensor_training/train_serial_distillation.py \
  --run-name imu_walk_100hz_v1
```

总控入口依次训练 `Teacher Stage1 → Teacher Stage2 → Student Stage1 → Student
Stage2`，所有阶段均为 100 Hz（`--decimation 2`）、无参考控制基线和 30000 轮。
每一阶段完成后会更新：

```text
logs/serial_distillation/imu_walk_100hz_v1/pipeline.json
```

若训练中断，使用相同 `--run-name` 再次执行同一命令即可继续未完成阶段；已完成阶段会被
跳过。先检查将要运行的命令可使用 `--dry-run`，离线运行可使用 `--no-wandb`。
