# 62D 足底力 Sensor Stage1

这是与旧 `sensor_training` 完全隔离的 ablation：Actor 当前帧严格为 `42 IMU + 14 tendon length + 6 virtual plantar normal force = 62D`，不含 phase、reference tendon、root truth、terrain 或 contact flag。执行器固定使用 `reference_ctrl_blend=0.0`，即 `ctrl=current_tendon_length+action_residual`。

```bash
uv run python foot_sensor_training/train_no_force_stage1.py
uv run python foot_sensor_training/train_stage1.py
uv run python foot_sensor_training/train_no_force_stage1.py --num-envs 1024 --num-steps 96
uv run python foot_sensor_training/train_stage1.py --num-envs 1024 --num-steps 96
uv run python foot_sensor_training/play.py --checkpoint <path>
```

前两组分别是 56D/62D、4096×24，后两组分别是 56D/62D、1024×96；每次 rollout 都是 98304 个样本。两边奖励、reset、动作语义、AMP、PPO、GRU 和 seed 策略相同，唯一变量为 6D 足力。

6 路顺序：`left_heel, left_forefoot_medial, left_forefoot_lateral, right_heel, right_forefoot_medial, right_forefoot_lateral`。CPU 用 `mj_contactForce` 的连续 solver normal force；Warp 调用当前安装版本的官方 `mujoco_warp.contact_force`，由它处理 world、EFC address、摩擦锥和 normal row 解码。三个锚点统一位于各自 foot body frame；articulated toe 的 geom-local 坐标不会被误当作 foot-local 坐标。每个 runtime contact 先转换到 foot frame，再做 Gaussian 分配。归一化为 `clip(F/(mg/2),0,4)/4`。

可选传感器动态依次执行 per-channel gain、bias、Gaussian noise、低通和 ring-buffer delay；reset 清空 filter/delay。默认参数全为恒等设置，不改变第一轮 ablation。
