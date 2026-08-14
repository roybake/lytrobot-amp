# 独立训练任务

现有 `scripts/train_multimotion.py` 保持原有四动作和串联地形。本页三个多动作/楼梯
任务统一使用传感器 GRU actor：每帧只有 42 维 IMU 和 14 维肌腱长度，不输入技能标签、
命令、参考相位或地形真值。蹲起任务另行使用反馈校正的 `sin/cos` 相位，不改变这里的
三个独立任务。checkpoint 和日志目录彼此隔离。

## 去掉阶梯的多动作训练

该任务只使用 `walk`、`rough_walk`，串联路线只包含 rough、6° 上坡和 6° 下坡：

```bash
uv run python scripts/train_multimotion_no_stairs_stage1.py \
  --num-envs 4096 --max-iterations 30000

uv run python scripts/train_multimotion_no_stairs_stage2.py \
  --stage1-checkpoint \
    logs/multimotion/no_stairs/stage1/<run>/model_final.pt \
  --num-envs 4096 --max-iterations 7500
```

Stage 1 强制关闭域随机化，Stage 2 强制开启自适应域随机化。输出分别写入
`logs/multimotion/no_stairs/stage1/<timestamp>/` 和
`logs/multimotion/no_stairs/stage2/<timestamp>/`。不要用全地形多动作任务的
checkpoint 恢复，因为两个任务的动作标签数量（策略条件维度）不同。

## 独立上下楼梯训练

该任务只使用 `stair_up`、`stair_down` 两个参考动作和楼梯地形：

```bash
uv run python scripts/train_stairs_stage1.py \
  --num-envs 4096 --max-iterations 30000

uv run python scripts/train_stairs_stage2.py \
  --stage1-checkpoint \
    logs/stairs/stage1/<run>/model_final.pt \
  --num-envs 4096 --max-iterations 7500
```

Stage 1 强制关闭域随机化，Stage 2 强制开启自适应域随机化。输出分别写入
`logs/stairs/stage1/<timestamp>/` 和
`logs/stairs/stage2/<timestamp>/`。`--reward-mode stairs` 使用独立的
楼梯奖励组合，重点指标为：

- `task_stair_vertical_velocity_track`：跟踪上楼/下楼参考动作各自的垂直速度；
- `task_stair_swing_clearance`：摆动脚相对当前台阶表面的净空；
- `task_stair_tread_support`：参考支撑脚是否在踏面上形成有效接触；
- `task_stair_min_swing_clearance_value`：摆动侧脚/趾几何体的最小实际净空（米）。

两个任务的阶段入口均固定 `--no-skill-conditioning --reference-ctrl-blend 0`，通过
`sensor_training.train` 构建严格传感器策略。命令行不能重新启用技能输入或参考控制
混合。正式训练前仍应确认 CUDA、显存、
输出目录和恢复 checkpoint；不同任务之间不要交叉恢复 checkpoint。
