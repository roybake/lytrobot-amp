# 足底力 Teacher → Student 蒸馏

Teacher actor：62D 物理传感器帧加训练期 20D 可解释向量：body root linear velocity (3)、projected gravity (3)、true tendon rate (14)。不包含 phase/reference/qpos/qvel/contact。Teacher 与 Student 都固定 `reference_ctrl_blend=0.0`。

共享模块为 `SensorEncoder`、`PrivilegedEncoder`、`ActorHead`；Student 只训练 `HistoryEstimator`，输入 `sensor(62)+previous action(14)+causal tendon velocity(14)=90D`，GRU hidden 为 128。导出输入为 `observation(62), previous_action(14), recurrent_state(142), episode_start`，没有特权输入。

```bash
uv run python distillation_training/train_teacher.py
uv run python distillation_training/train_student_bc.py --teacher-checkpoint <path>
uv run python distillation_training/train_student_dagger.py --teacher-checkpoint <path> --student-checkpoint <path>
uv run python distillation_training/train_student_rl.py --teacher-checkpoint <path> --student-checkpoint <path>
uv run python distillation_training/play.py --checkpoint <path>
uv run python distillation_training/play.py --checkpoint <path> --teacher-checkpoint <teacher.pt> --oracle-privileged
```

Student BC、DAgger、RL 默认均为 `--num-envs 1024 --num-steps 96`，优化器以完整 environment sequence 做 0.96 s truncated BPTT，不把 timestep flatten。GPU/Warp 是默认执行路径；CPU 仅用于 CPU force 验收或显式 smoke 回退。

## 推荐：自动编排

Teacher 训练完成并通过单独评估后，以下命令自动顺序执行 BC、DAgger、RL；每一阶段的路径会写入 `pipeline.json`，便于恢复和审计：

```bash
uv run python distillation_training/train_pipeline.py \
  --teacher-checkpoint logs/distillation/teacher/<run>/teacher.pt \
  --run-dir logs/distillation/<run>
```

若 BC validation 的 reconstruction MAE / action MSE / KL 不达标，请停止在 BC 阶段检查可观测性，不要继续 DAgger：

```bash
uv run python distillation_training/train_pipeline.py \
  --teacher-checkpoint <teacher.pt> \
  --run-dir logs/distillation/<run> --skip-dagger --skip-rl
```

BC/DAgger 使用 `SmoothL1(p_hat,p) + Gaussian-KL(teacher,student)` 并记录 reconstruction MAE/action MSE/KL。DAgger 以 student 控制状态、teacher 仅生成标签，replay 有上限；RL 后期退火两个蒸馏系数。
