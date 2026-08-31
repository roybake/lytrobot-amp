# 独立训练任务

现有 `scripts/train_multimotion.py` 保持原有四动作和串联地形。本文涉及的训练入口统一
使用传感器 GRU actor：每帧只有 42 维 IMU 和 14 维肌腱长度，不输入技能标签、命令、
参考相位或地形真值。蹲起任务另行使用反馈校正的 `sin/cos` 相位，不改变这里的独立
任务。checkpoint 和日志目录彼此隔离。

## 去掉阶梯的多动作训练

该入口继续复用普通行走 `scripts/train_stage1.py` / `scripts/train_stage2.py` 的传感器
Actor、奖励主体、PPO/AMP 默认值和两阶段规则，同时保留原来的无阶梯串联课程。
每条路线恰好包含一次 `flat`、`rough` 和相邻的 `slope_up → slope_down`，三个地形块共有
6 种唯一排列；训练 episode 随机分配路线，测试启动时也用新 seed 随机生成路线。
Actor 仍看不到地形顺序，只能依靠 IMU 和肌腱长度适应地形。

```bash
uv run python scripts/train_multimotion_no_stairs_stage1.py \
  --num-envs 4096 --max-iterations 30000

uv run python scripts/train_multimotion_no_stairs_stage2.py \
  --stage1-checkpoint \
    logs/multimotion/no_stairs/stage1/<run>/model_best.pt \
  --num-envs 4096 --max-iterations 7500
```

Stage 1 和 Stage 2 的领域随机化、恢复优化器及训练轮数规则与普通行走入口一致。
当前课程不把机械能计入奖励；功率和机械能指标仍正常记录，仅用于离线分析。奖励优先推动持续前进，同时保留足滑惩罚。
训练前 18000 轮全部从入口开始，先学会持续前进；随后用 8000 轮把随机后续阶段
reset 概率缓慢提高到 25%，为后段地形补充直接样本，同时至少保留 75% 的完整路线
episode，避免后段 reset 淹没入口行走并导致遗忘。完整路线成功率仍只统计入口起步的
episode。测试始终从路线入口开始。输出分别写入
`logs/multimotion/no_stairs/stage1/<timestamp>/` 和
`logs/multimotion/no_stairs/stage2/<timestamp>/`。

## 独立上下楼梯训练

该任务使用 `walk`、`stair_up`、`stair_down` 三个参考动作组成一条不可拆分的连续路线：
地面走近楼梯、完整上楼、越过顶部、完整下楼，再在地面走出。训练前 15000 轮只从入口
起步，先把连续平地步态训练稳定；随后用 7500 轮把随机后续阶段 reset 概率提高到 35%，
补足上楼、下楼和走出阶段，同时保留多数入口样本，避免困难阶段导致行走遗忘。测试仍
强制从入口开始。动作切换只推进参考阶段，不重置或搬运机器人状态。

```bash
uv run python scripts/train_stairs_stage1.py \
  --num-envs 4096 --max-iterations 30000

uv run python scripts/train_stairs_stage2.py \
  --stage1-checkpoint \
    logs/stairs/stage1/<run>/model_best.pt \
  --num-envs 4096 --max-iterations 7500
```

Stage 1 强制关闭域随机化，默认使用较易的 `0.13 m` 级高和 `0.32 m` 踏深；
Stage 2 强制开启自适应域随机化，并默认切换到更困难的 `0.20 m` 级高和 `0.22 m` 踏深。
两个阶段均不将能耗指标计入奖励优化。
命令行显式给出的楼梯尺寸仍可覆盖默认值。输出分别写入
`logs/stairs/stage1/<timestamp>/` 和
`logs/stairs/stage2/<timestamp>/`。`--reward-mode stairs` 使用独立的
楼梯奖励组合；串联楼梯的奖励表面与实际碰撞几何使用同一个世界坐标配置，且
`stair_min_root_clearance=0.60 m` 会终止陷入台阶的轨迹。重点指标为：

- `task_stair_vertical_velocity_track`：跟踪上楼/下楼参考动作各自的垂直速度；
- `task_stair_swing_clearance`：摆动脚相对当前台阶表面的净空；
- `task_stair_tread_support`：参考支撑脚是否在踏面上形成有效接触；
- `task_stair_min_swing_clearance_value`：摆动侧脚/趾几何体的最小实际净空（米）。
- `task_stair_root_clearance_value`：骨盆相对当前台阶表面的实际净空；
- `task_stair_terrain_penetration`：因低于最小净空而终止的穿模/陷入标记。

两个任务的阶段入口均固定 `--no-skill-conditioning --reference-ctrl-blend 0`，通过
`sensor_training.train` 构建严格传感器策略。命令行不能重新启用技能输入或参考控制
混合。正式训练前仍应确认 CUDA、显存、
输出目录和恢复 checkpoint；不同任务之间不要交叉恢复 checkpoint。

每个 run 同时保留两组原子 checkpoint：`model_final.pt/.onnx` 是最新状态，用于中断
恢复；`model_best.pt/.onnx` 只有在滚动窗口内真实跨过至少一个路线阶段后才生成，再按
完整路线成功率、后缀/阶段完成度、路线进度，最后按奖励和滑移择优，用于 Stage 1 晋级
和测试。选择依据写入 `model_best.json`。

所有动作的可视化测试都会自动录屏，并在各自日志树内按两阶段隔离：
`logs/<task>/stage1/recordings/` 与 `logs/<task>/stage2/recordings/`。录像文件名包含训练
run 和 checkpoint iteration；重复录制会追加序号，不覆盖已有文件。
