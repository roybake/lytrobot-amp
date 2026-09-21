# 足力 / 内部 phase 消融实验

四个入口共享 reward、reset、动作语义、AMP/PPO、seed、`walk1_mjcf.npz` 与 100 Hz 控制；Actor 不接收 reference tendon 或 simulator truth：

| 实验 | 入口 | 固定 observation 顺序 |
|---|---|---|
| A | `train_no_force_stage1.py` | `IMU[0:42], tendon[42:56]` = 56D |
| B | `train_stage1.py` | 上述 + `foot_force[56:62]` = 62D |
| C | `train_phase_stage1.py` | 56D + `sinθ[56], cosθ[57]` = 58D |
| D | `train_phase_force_stage1.py` | 62D + `sinθ[62], cosθ[63]` = 64D |

phase 是控制器内部固定 reset 时钟，不是完整 gait command：`φ₀=start_phase`，每个控制 tick 执行 `φ ← (φ + frequency·control_dt·α) mod frame_count`，输出 `θ=2πφ/frame_count` 的 `[sinθ, cosθ]`。`α` 使用与仿真一致的 delayed smoothstep stand-start schedule；训练和部署均调用 `DeployablePhaseClock`。第一版强制 `stand_start_prob=1`、`motion_reset_prob=0`、固定 120-step transition，contact correction 关闭。

推荐实验矩阵：

```bash
# 24-step：分别量化 Δphase 与 Δforce（每轮 4096*24=98304）
uv run python foot_sensor_training/train_no_force_stage1.py
uv run python foot_sensor_training/train_phase_stage1.py
uv run python foot_sensor_training/train_stage1.py
uv run python foot_sensor_training/train_phase_force_stage1.py

# 96-step：量化 ΔBPTT（每轮 1024*96=98304）
uv run python foot_sensor_training/train_phase_stage1.py --num-envs 1024 --num-steps 96
uv run python foot_sensor_training/train_phase_force_stage1.py --num-envs 1024 --num-steps 96
```

6 路足力顺序固定为 left heel/forefoot medial/forefoot lateral，再到对应 right 三路。CPU 使用 `mj_contactForce`，Warp 使用 `mujoco_warp.contact_force`。传感器 dynamics 每个 policy tick 只更新一次，observation/logging 只读缓存；autoreset 按 env 清 filter/delay。`delay_steps=N` 的精确定义为 `output_t=filtered_(t-N)`。

比较实验时不要把 `Δforce`、`Δphase` 与 `ΔBPTT` 同时改变。
