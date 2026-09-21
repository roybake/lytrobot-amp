# 64D phase-enabled Teacher → Student 蒸馏

Teacher 与 Student 看到相同的 64D 部署输入：`IMU 42 + tendon 14 + foot force 6 + phase sin/cos 2`。Teacher 额外得到训练期 20D truth：body root linear velocity 3、projected gravity 3、true tendon velocity 14。phase 是 controller-known input，既不放进 privileged target，也不由 estimator 蒸馏。

Student estimator 输入为 `sensor 64 + previous action 14 + causal tendon velocity 14 = 92D`，GRU hidden 保持 128D。部署 recurrent state 仍为 `hidden 128 + previous tendon 14 = 142D`；phase 由外部 `DeployablePhaseClock` 维护。BC/DAgger/RL 默认完整 96-step BPTT，沿 env 维采样，不 flatten 时间维。

```bash
uv run python distillation_training/train_teacher.py --device cuda --sim-backend warp
uv run python distillation_training/train_student_bc.py --teacher-checkpoint <teacher.pt> --device cuda --sim-backend warp --num-envs 1024 --num-steps 96 --max-validation-action-mse <chosen_threshold> --max-validation-kl <chosen_threshold>
uv run python distillation_training/train_student_dagger.py --teacher-checkpoint <teacher.pt> --student-checkpoint <student_bc.pt> --device cuda --sim-backend warp --num-envs 1024 --num-steps 96
uv run python distillation_training/train_student_rl.py --teacher-checkpoint <teacher.pt> --student-checkpoint <student_dagger.pt> --device cuda --sim-backend warp --num-envs 1024 --num-steps 96 --amp
uv run python distillation_training/play.py --checkpoint <student_rl.pt> --teacher-checkpoint <teacher.pt> --oracle-privileged
uv run python distillation_training/play.py --checkpoint <student_rl.pt> --export-onnx student.onnx
```

Student RL stage A 只训练 HistoryEstimator、critic 与 `log_std`，冻结 SensorEncoder、PrivilegedEncoder、ActorHead。rollout 起点的非零 hidden 会被 detach/clone 保存，并用作 PPO sequence replay 的初始状态。导出的 ONNX 输入严格为 `observation(64), previous_action(14), recurrent_state(142), episode_start`，metadata 包含 64/92/128/142 schema 和完整 phase clock 参数。

Teacher 和 Student RL 共用 `AMPTrainingModule`：默认 reward 为 `0.5 * task + 0.6 * AMP`，gradient penalty 与参考采样实现一致。Student 优先从 Student RL checkpoint 恢复 discriminator/optimizer，否则从 Teacher checkpoint 加载；默认继续训练 discriminator，`--freeze-amp-discriminator` 仅用于诊断。AMP feature 只供 discriminator/reward 使用，不进入 actor 或 ONNX。

BC/DAgger replay 默认直接保留在 CUDA，sequence 始终为 `[N,96,...]`，训练时转为 `[96,N,...]`。BC 不再使用无限阈值：必须显式配置 action-MSE+KL manual gate，或配置 `--max-validation-normalized-action-mse` relative gate；否则 checkpoint 的 `validation.status` 为 `unresolved`，pipeline 不会进入 DAgger。两轮 smoke 可显式传 `--allow-unvalidated-output`。
