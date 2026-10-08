# 58D/64D variant-aware Teacher → Student 蒸馏

同一套蒸馏代码支持两条可部署路线：

| variant | Actor observation（固定顺序） | estimator input |
|---|---|---:|
| `phase_no_force` | IMU `[0:42]`、tendon `[42:56]`、phase sin/cos `[56:58]` | 58+14+14 = 86D |
| `phase_force` | IMU `[0:42]`、tendon `[42:56]`、6D plantar force `[56:62]`、phase sin/cos `[62:64]` | 64+14+14 = 92D |

两条路线只相差 6D 足底力。SharedActor 的输入层由 variant 决定，其余 SensorEncoder 128、PrivilegedEncoder 64、ActorHead 256/128、14D action 完全相同。HistoryEstimator 都使用 GRU hidden 128 和 96-step BPTT。部署 recurrent state 都是 `hidden 128 + previous tendon 14 = 142D`，不随 sensor variant 改变。

Teacher 的 training-only privileged vector 对两条路线完全相同：body-frame root linear velocity 3、projected gravity 3、true tendon velocity 14，共 20D。它不含 contact、foot force truth、phase truth、reference state 或 terrain label。无足底 Teacher 的 actor 输入严格为 58D sensor + 20D privileged；有足底 Teacher 为 64D sensor + 20D privileged。Critic 可使用相同的 20D training-only truth。

Teacher 和 Student RL 共用 `AMPTrainingModule`、expert motion、AMP reward、discriminator、gradient penalty 和 optimizer。AMP feature 只用于 discriminator/reward，不进入 actor observation 或 ONNX。Student RL 优先恢复 Student checkpoint 中的 discriminator，否则使用 Teacher checkpoint；默认继续训练，`--freeze-amp-discriminator` 仅供诊断。

正式任务默认使用 `--task-reward-mode gated_reference`、`walk1_mjcf_repaired.npz` 和 `target_velocity=0.87`。该模式在基础环境 Stage1 参考奖励之上，按躯干直立及高度缩放正向跟踪奖励，并单独惩罚后退；终止时给予固定跌倒代价。`--task-reward-mode stage1_reference` 保留原始参考奖励作对照，`--task-reward-mode sensor_stability` 保留为显式消融。三种模式都保持 `reference_ctrl_blend=0.0`，reference tendon 只能作为 reward target，不能进入 Actor 或控制前馈。

`gated_forward_gait_v5` 配置额外强调速度跟踪、向前位移和速度不足，后退速度惩罚系数为 8，并设置较高的跌倒/停滞代价；AMP 默认权重为 task 1.0、style 0.2。Teacher 动作探索从标准差 0.35 开始，限制在 0.05–0.60；学习率默认 `1e-4`，用 `target_kl=0.025` 控制 PPO 更新，rollout 默认 96 步。行走参考统一使用 `walk1_mjcf_repaired.npz`（45 帧、40 Hz，周期 1.125 秒，每周期前进 0.97875 米）；phase 时钟按该步态周期归一化，速度目标为 0.87 m/s。旧 `logs/distillation/teacher_phase_no_force.pt` 是此前 `sensor_stability` + 长参考训练出的权重，修改配置不会修复它，需要重新训练。

重新训练 58D Teacher：

```bash
uv run python distillation_training/train_teacher.py \
  --sensor-variant phase_no_force --device cuda --sim-backend warp \
  --num-envs 1024 --num-steps 96 --no-wandb \
  --output logs/distillation/teacher_phase_no_force_gated_v3.pt
uv run python distillation_training/play_teacher.py \
  --checkpoint logs/distillation/teacher_phase_no_force_gated_v3.pt \
  --steps 1000 --seed 42 --camera-mode fixed
```

评估时检查输出的 `mean_forward_velocity` 和 `reset_steps`，并用多个 seed 观察至少 10 秒的连续步行；不能仅凭训练 reward 判定步态稳定。
无界面批量评估可给 `play_teacher.py` 添加 `--headless --steps 1000`，无需打开 MuJoCo 窗口。

训练中断后可传 `--resume <同一任务配置的 Teacher checkpoint>` 并把 `--max-iterations` 设为新的总迭代数；优化器和 AMP 判别器状态会一同恢复。旧长周期参考的 checkpoint 不兼容 `gated_forward_gait_v5`，不能作为新训练的续训起点。

## 正式命令

Teacher 显式选择 variant；BC、DAgger、Student RL 从 checkpoint 自动继承并验证 variant，禁止 silent mismatch。

Teacher、BC、DAgger、Student RL 每个阶段默认训练 50000 轮；`train_pipeline.py` 的默认预算也是每个 Student 阶段 50000 轮。

```bash
# Route A: no force
uv run python distillation_training/train_teacher.py --sensor-variant phase_no_force --device cuda --sim-backend warp
uv run python distillation_training/train_student_bc.py --teacher-checkpoint teacher_no_force.pt --device cuda --sim-backend warp --num-envs 1024 --num-steps 96 --max-validation-normalized-action-mse <threshold>
uv run python distillation_training/train_student_dagger.py --teacher-checkpoint teacher_no_force.pt --student-checkpoint student_no_force_bc.pt --device cuda --sim-backend warp --num-envs 1024 --num-steps 96
uv run python distillation_training/train_student_rl.py --teacher-checkpoint teacher_no_force.pt --student-checkpoint student_no_force_dagger.pt --device cuda --sim-backend warp --num-envs 1024 --num-steps 96 --amp

# Route B: force
uv run python distillation_training/train_teacher.py --sensor-variant phase_force --device cuda --sim-backend warp
uv run python distillation_training/train_student_bc.py --teacher-checkpoint teacher_force.pt --device cuda --sim-backend warp --num-envs 1024 --num-steps 96 --max-validation-normalized-action-mse <threshold>
uv run python distillation_training/train_student_dagger.py --teacher-checkpoint teacher_force.pt --student-checkpoint student_force_bc.pt --device cuda --sim-backend warp --num-envs 1024 --num-steps 96
uv run python distillation_training/train_student_rl.py --teacher-checkpoint teacher_force.pt --student-checkpoint student_force_dagger.pt --device cuda --sim-backend warp --num-envs 1024 --num-steps 96 --amp
```

显存不足时优先把 `--num-envs` 从 1024 降到 512/256，保持 `--num-steps 96`。BC 正式输出必须通过 manual gate（action MSE + KL）或 relative normalized-action-MSE gate；只有短 smoke 可用 `--allow-unvalidated-output`，其 checkpoint 明确记录 `validation.status=unresolved`。

Teacher 和 Student RL 的 time-limit transition 使用 autoreset 前 `terminal_observation` 与 `terminal_privileged_vector` 计算 critic bootstrap；fall 仍是真 terminal，不 bootstrap。两者调用相同的 `returns.py` GAE 实现。

## Oracle、ONNX 与 GPU smoke

```bash
uv run python distillation_training/play.py --checkpoint student.pt --teacher-checkpoint teacher.pt --oracle-privileged --device cuda --sim-backend warp
uv run python distillation_training/play.py --checkpoint student.pt --export-onnx student.onnx
uv run python scripts/smoke_distillation_gpu.py
```

ONNX 自动从 checkpoint 选择 observation 58D 或 64D；两者输入还包括 previous action 14D、recurrent state 142D、episode start。metadata 记录 sensor variant、58/64、86/92、hidden 128、state 142、phase clock，并明确 privileged/AMP/reference tendon 不进入 actor。

## Paired ablation

用相同 seed（推荐 42、43、44）分别训练 `phase_no_force` 与 `phase_force`，其余 reward、AMP、phase、reset、action、网络容量、BPTT、motion 和优化参数保持一致。记录 episode length、fall rate、forward speed、task reward、AMP reward、gait quality、action smoothness、Student/Teacher KL 与 privileged estimator MAE。

| Policy | Force | Phase | BPTT | Distillation | AMP | Score |
|---|---|---:|---:|---:|---:|---:|
| Direct-58 | No | Yes | 96 | No | Yes | |
| Direct-64 | Yes | Yes | 96 | No | Yes | |
| Student-58 | No | Yes | 96 | Yes | Yes | |
| Student-64 | Yes | Yes | 96 | Yes | Yes | |
