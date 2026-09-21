# 62D 足底力 Sensor Stage1

这是与旧 `sensor_training` 完全隔离的 ablation：Actor 当前帧严格为 `42 IMU + 14 tendon length + 6 virtual plantar normal force = 62D`，不含 phase、reference tendon、root truth、terrain 或 contact flag。执行器固定使用 `reference_ctrl_blend=0.0`，即 `ctrl=current_tendon_length+action_residual`。

```bash
uv run python foot_sensor_training/train_stage1.py
uv run python foot_sensor_training/play.py --checkpoint <path>
```

6 路顺序：`left_heel, left_forefoot_medial, left_forefoot_lateral, right_heel, right_forefoot_medial, right_forefoot_lateral`。CPU 用 `mj_contactForce` 的连续 solver normal force；Warp 从 `wp_data.efc.force`（与 `mujoco_warp.contact_force` 相同 solver-force 来源）读取连续法向约束力。每个接触按现有脚部 collision geom 推导的局部锚点做 Gaussian 分配。归一化为 `clip(F/(mg/2),0,4)/4`。
