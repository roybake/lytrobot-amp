# 传感器串行蒸馏

## 推荐入口

本次最终流程使用 `train_final_sensor_gait.py`，固定 GRU256、45D状态估计、共享
encoder测量派生特征、BPTT96，并关闭 future prediction。完整说明和三组推荐命令
见 [STATE_ESTIMATION.md](STATE_ESTIMATION.md)。

```bash
.venv/bin/python sensor_training/train_final_sensor_gait.py
```

以下为已有独立蒸馏入口的兼容说明；其48步/128D等旧默认不代表最终推荐配置。


部署策略始终只使用 IMU 与肌肉长度；特权 Teacher 仅在训练期间读取
仿真完整状态，并且不会写入导出的 ONNX Actor。

蒸馏与直接行走训练是两条隔离路径：`train_stage1.py`/`train_stage2.py` 使用普通
`SensorHistoryPPOAMPTrainer`，其配置、rollout、loss 和 checkpoint 中均无 Teacher；只有
`train_distill_stage1.py`/`train_distill_stage2.py` 会进入独立的 distillation 环境、配置与
trainer。普通入口不接受 `--teacher-*` 参数，蒸馏入口则强制要求 Teacher checkpoint。

Teacher 和传感器 Student 必须保持相同的 `--decimation`。新训练默认让两者都使用
`reference_ctrl_blend=1.0`：控制器按同一参考腱长表输出前馈，Actor 学习反馈残差。
蒸馏器仍会把 Teacher 动作换算成绝对腱长目标，再映射为 Student 动作；相同的控制基准
可以减少目标动作被裁剪。该参考表由控制器时钟读取，不需要额外硬件传感器。
所有行走入口统一使用 `walk1_mjcf_repaired.npz`（原始人体 mocap，前进速度 0.87 m/s），目标速度配置为 `0.87 m/s`。
Teacher 和 Student 必须使用相同参考数据；使用其他参考训练的 Teacher 需要重新训练。

## 1. 训练特权 Teacher

Teacher 采用任务一 `scripts/train_stage1.py` 的全状态 PPO/BPO+AMP 配置为基础。
固定步行参考无需设置 `--motion-loop`（原始 npz 无循环字段，35200 帧序列足够任意回合长度）。
总控配方使用 20 秒回合，并增加相位对齐的关节/足端跟踪与抑制跳跃、踮脚、滑步的奖励。
这些训练真值不会进入 Student Actor。教师必须先通过连续回放，再用于蒸馏。

```bash
python3 scripts/train_stage1.py \
  --reference-ctrl-blend 1.0 \
  --motion-loop true --episode-steps 2000 --evaluate-gait \
  --max-iterations 50000 \
  --log-dir "logs/distillation/<run>/stage1/teacher"

python3 scripts/train_stage2.py \
  --stage1-checkpoint "logs/distillation/<run>/stage1/teacher/model_final.pt" \
  --reference-ctrl-blend 1.0 \
  --motion-loop true --episode-steps 2000 --evaluate-gait \
  --max-iterations 50000 \
  --log-dir "logs/distillation/<run>/stage2/teacher"
```

stage2 的任务奖励与 stage1 相同；它只增加域随机化来学习恢复行为。

## 2. 蒸馏传感器 Student

```bash
python3 sensor_training/train_distill_stage1.py \
  --teacher-checkpoint "logs/distillation/<run>/stage1/teacher/model_final.pt" \
  --log-dir "logs/distillation/<run>/stage1/student"

python3 sensor_training/train_distill_stage2.py \
  --student-checkpoint "logs/distillation/<run>/stage1/student/model_final.pt" \
  --teacher-checkpoint "logs/distillation/<run>/stage2/teacher/model_final.pt" \
  --log-dir "logs/distillation/<run>/stage2/student"
```

每一阶段默认最多训练 50000 轮。第一阶段的 Teacher 动作 Huber 损失系数从
`1.0` 退火到 `0.3`，第二阶段从 `0.5` 退火到 `0.3`。动作误差先除以 `0.1` 再求 Huber
损失，使细小但会改变肌腱控制的偏差获得足够监督。标签描述 Student 渐入之前的 Actor
输出，修正了教师与学生启动渐入被重复相乘的问题。

每个回合开始时抽取控制者：Stage1 教师执行概率在前 4000 次更新内从 `0.8` 降到 `0`，
Stage2 在前 2000 次更新内从 `0.3` 降到 `0`。教师只看到实际执行动作的历史，并为两种
回合都提供动作标签。教师执行的样本用于监督学习，策略似然损失、熵与 KL 统计只使用学生
自己执行的样本。此处参考了 [Instinct TPPO 的教师接管与 DAgger 机制](https://github.com/project-instinct/instinct_rl/blob/main/instinct_rl/algorithms/tppo.py)，
并结合本项目的优化器做了样本屏蔽。

Student 的奖励同时跟踪板载时钟对应的人体关节动作和足端轨迹；相位来自软件命令，仿真
关节与足端真值只参与奖励。部署输入继续为机身/腿部 IMU、14 条肌肉长度、自身动作历史和
板载命令，保持现有 696 维。只有单个机身 IMU 的硬件可通过总控 `--student-root-imu-only`
或直接入口 `--sensor-root-imu-only` 使用 480 维输入；两种观测布局不能直接共用 Student 权重。

Student 强制使用100 Hz控制和48步 recurrent BPTT，并显式读取当前帧加前5帧
IMU/肌腱历史，梯度窗口为0.48秒；其他控制频率会在训练启动前报错。GRU状态跨rollout
持续保留，且每步都有Teacher动作监督。直接蒸馏入口使用2048个环境，使每轮仍为98,304个
transition。串行总控会根据 Teacher
的 `--num-envs` 自动推导 Student 环境数；可用 `--student-num-envs` 和
`--student-num-steps` 单独覆盖。

## 日志目录

```text
logs/distillation/<MM-DD-HH:MM:SS>/
├── stage1/
│   ├── teacher/
│   └── student/
└── stage2/
    ├── teacher/
    └── student/
```

训练启动时自动创建目录。同一阶段的教师和学生放在一起；独立蒸馏入口会将
使用的教师权重复制到同阶段的 `teacher/`。新布局从下次启动新训练时生效，
已有训练目录不迁移。显式指定 `--log-dir` 时使用传入的路径，例如
`logs/distillation/<run>/stage1/student`。

`metrics.csv` 中记录：`teacher_action_loss`、`teacher_action_mae`、
`teacher_action_loss_coef`、`teacher_behavior_probability` 与 `student_on_policy_fraction`。
每个 Student checkpoint 的 `distillation` 字段
记录 Teacher 路径、迭代号及当前蒸馏系数，便于复盘训练来源。

## 一键训练自然步态

```bash
uv run python sensor_training/train_serial_distillation.py \
  --run-name 09-30-10:47:29
```

总控默认执行 `Teacher Stage1 → Student Stage1`，两者均为 100 Hz（`--decimation 2`），
使用同一参考控制基线。Teacher 和 Student 分别通过步态验收后可提前结束；不合格的
Teacher 不会交给 Student。需要后续鲁棒训练时加 `--include-robust-stage`，扩展为
`Teacher Stage1 → Student Stage1 → Teacher Stage2 → Student Stage2`。
阶段数量是本项目的工程流程，不是开源蒸馏算法的固定要求。
`--teacher-stage1-iterations`、`--teacher-stage2-iterations` 可分别设置教师预算。
每一阶段完成后会更新：

```text
logs/serial_distillation/imu_walk_100hz_v1/pipeline.json
```

若训练中断，使用相同 `--run-name` 再次执行同一命令即可继续未完成阶段；已完成阶段会被
跳过。先检查将要运行的命令可使用 `--dry-run`，离线运行可使用 `--no-wandb`。

## 连续步态验收

`--evaluate-gait` 会在每次保存 checkpoint 时用三个种子（42、123、2026）分别从站立起点
确定性回放 20 秒，禁止周期结束重置，所有种子都须通过验收。
非循环参考也可评估，只要剩余序列足够覆盖评估时长；`walk1_mjcf_repaired.npz` 包含 1.125 秒、45 帧的可循环数据，
训练应启用 `--motion-loop true`。若参考提前结束触发重置，评估仍会报错。
合格条件为：全程无跌倒、平均速度与命令相差不超过 0.15 m/s、膝关节相位误差 RMS 不超过
0.20 rad、足端局部位置 RMS 不超过 0.06 m、膝屈伸幅度至少达到参考的 70%，锁膝采样比例
不超过 10%。`model_best` 按验收结果和连续跟踪评分选择；仅增加回合长度不算合格。
评估记录保存在 `gait_evaluations.jsonl`、`best_gait_evaluation.json` 和 `gait_evaluation_best.csv`，
同时上传 W&B 的 `eval/gait_*` 指标。确定性评估通过仍需后续噪声与实机验证。

```bash
uv run python sensor_training/gait_evaluation.py \
  --checkpoint "logs/distillation/<run>/stage1/teacher/model_best.onnx" \
  --output recording/distillation/teacher_gait_evaluation.json
```

旧权重缺少循环标记时，诊断可加 `--force-loop`；它用于揭示跨周期问题，不会修改旧模型。

## 可视化测试师生模型

将 `<run>` 替换为训练时间戳。第一阶段权重位于 `stage1/teacher/`、`stage1/student/`；第二阶段位于 `stage2/teacher/`、`stage2/student/`。

```bash
# 第一阶段教师
uv run python scripts/play.py \
  --checkpoint "logs/distillation/<run>/stage1/teacher/model_final.onnx" \
  --device cuda --viewer --steps 0

# 第一阶段学生
uv run python sensor_training/play.py \
  --checkpoint "logs/distillation/<run>/stage1/student/model_final.onnx" \
  --device cuda --viewer --steps 0

# 第二阶段教师
uv run python scripts/play.py \
  --checkpoint "logs/distillation/<run>/stage2/teacher/model_final.onnx" \
  --device cuda --viewer --steps 0

# 第二阶段学生
uv run python sensor_training/play.py \
  --checkpoint "logs/distillation/<run>/stage2/student/model_final.onnx" \
  --device cuda --viewer --steps 0
```

测试最佳权重时将 `model_final.onnx` 换成 `model_best.onnx`。

开启 `--viewer` 时，师生录屏分别保存到
`recording/distillation/<stage>/<run>-teacher/` 和
`recording/distillation/<stage>/<run>-student/`。`<stage>` 为 `stage1` 或 `stage2`，
`<run>` 使用训练目录的时间戳，例如 `10-08-13:12:25`。
