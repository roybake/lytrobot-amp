# 64D phase-enabled Teacher → Student 蒸馏

Teacher 与 Student 看到相同的 64D 部署输入：`IMU 42 + tendon 14 + foot force 6 + phase sin/cos 2`。Teacher 额外得到训练期 20D truth：body root linear velocity 3、projected gravity 3、true tendon velocity 14。phase 是 controller-known input，既不放进 privileged target，也不由 estimator 蒸馏。

Student estimator 输入为 `sensor 64 + previous action 14 + causal tendon velocity 14 = 92D`，GRU hidden 保持 128D。部署 recurrent state 仍为 `hidden 128 + previous tendon 14 = 142D`；phase 由外部 `DeployablePhaseClock` 维护。BC/DAgger/RL 默认完整 96-step BPTT，沿 env 维采样，不 flatten 时间维。

```bash
uv run python distillation_training/train_teacher.py
uv run python distillation_training/train_student_bc.py --teacher-checkpoint <teacher.pt> --num-envs 1024 --num-steps 96
uv run python distillation_training/train_student_dagger.py --teacher-checkpoint <teacher.pt> --student-checkpoint <student_bc.pt> --num-envs 1024 --num-steps 96
uv run python distillation_training/train_student_rl.py --teacher-checkpoint <teacher.pt> --student-checkpoint <student_dagger.pt> --num-envs 1024 --num-steps 96
uv run python distillation_training/play.py --checkpoint <student_rl.pt> --teacher-checkpoint <teacher.pt> --oracle-privileged
uv run python distillation_training/play.py --checkpoint <student_rl.pt> --export-onnx student.onnx
```

Student RL stage A 只训练 HistoryEstimator、critic 与 `log_std`，冻结 SensorEncoder、PrivilegedEncoder、ActorHead。rollout 起点的非零 hidden 会被 detach/clone 保存，并用作 PPO sequence replay 的初始状态。导出的 ONNX 输入严格为 `observation(64), previous_action(14), recurrent_state(142), episode_start`，metadata 包含 64/92/128/142 schema 和完整 phase clock 参数。

默认 task reward 为 phase-independent sensor stability reward，AMP 仍使用 `robot/reference_data/walk1_mjcf.npz` 提供自然步态先验。
