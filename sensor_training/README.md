# IMU + 肌腱长度独立训练

动力学、奖励和 AMP 判别器复用现有实现；Actor 使用 recurrent BPO 更新，AMP 判别器和 AMP reward 保留。所有训练统一使用纯位置控制。通用 `scripts/train.py` 也已直接绑定本目录的传感器环境和 recurrent trainer。行走、多动作、楼梯、蹲起、Teacher 和旧超参数复现入口全部遵守本页的严格传感器输入约束。

sensor stage1 默认与 `scripts/train_stage1.py` 使用相同的奖励系数和训练模式；stage2 入口会显式追加鲁棒训练和速度奖励相关参数。环境收到动作时仍统一裁剪到 `[-1, 1]`。

## 观测定义

Actor 每帧只读取：

- 7个三轴加速度计：21维，除以 `9.81`
- 7个三轴陀螺仪：21维，除以 `10 rad/s`
- 14条受控肌腱实际长度：14维，按执行器控制范围中心和半宽归一化

传感器单帧56维。行走策略与 checkpoint 只传输当前56维传感器帧，并由跨
episode 持久化的128维 GRU 隐状态保存因果历史。除此之外不拼接任何命令或仿真状态，因此不包含目标速度、动作/技能标签、
关节角度、关节速度、根部真值、参考轨迹、步态相位、足端状态、地形标签或上一动作。
`target_velocity` 只参与训练奖励计算，不进入策略网络；该策略学习固定训练速度，不是
速度命令跟踪策略。

AMP 判别器使用单帧51维特征：根高、根姿态、根线/角速度、14维关节角度、14维关节速度和足端局部位置；连续6帧拼接后为306维。AMP 参考历史按 `policy_dt` 从参考动作中连续采样并统一进行左右镜像。

AMP 判别器仍维护6帧历史；episode 重置时全部填充为当前 AMP 帧，同时清零对应环境的
GRU 隐状态，避免跨 episode 泄漏状态。

策略保持14维肌腱位置目标修正量：单一 Actor MLP 直接把 GRU 隐状态映射为动作，并经过
`tanh` 限幅。网络中不再包含专家分支、门控或专家混合。Critic 和 BPO median critic
使用 GRU 隐状态以及 v2 富特权上下文：无噪声完整仿真/参考观测、执行器控制/力/速度、
详细足底接触、连续地形参数、motion/课程状态、质量与肌腱随机化、摩擦/重力/延迟和外力。
上下文维度会随动作库大小和任务特有状态变化，字段及切片写入 checkpoint 元数据。
GRU 的独立 11 维 estimator head 只预测可由本体历史推断的根部线速度、左右脚接触和连续
地形参数，避免让 Actor 表征拟合不可观测的 motion ID 或随机质量。特权上下文和 estimator
目标都不进入 Actor 或 ONNX。训练按环境序列做截断 BPTT。

传感器多地形入口固定 `--reference-ctrl-blend 0`，执行器基准来自实测肌腱长度，
模拟器选择的动作参考只用于训练奖励和 AMP，不进入部署控制链。地形过渡完全由 GRU
根据本体反馈产生；动作 rate 正则负责抑制接缝抖动。论文依据、
上下文字段、限制和验证方法见
[`docs/proprioceptive_terrain_transition.md`](../docs/proprioceptive_terrain_transition.md)。

传感器 stage1 的启动参数刻意对齐 `scripts/train_stage1.py`：`stage1`、`cuda`、`warp`、`4096` 环境、关闭 domain randomization、开启 wandb，并使用相同的 stage1 奖励默认值。训练日志额外记录 `task_termination`，用于区分真实跌倒与正常超时。

训练、测试、ONNX 回放和录屏统一采用 `0.005 s` 物理步长、`decimation=2`，即严格
`100 Hz` 策略控制周期。50 Hz 旧 checkpoint 会被训练恢复与播放入口拒绝，避免用不同控制
周期测试。窗口与无窗口测试都按墙钟实时节拍运行；录屏按仿真时间采样到目标 fps，不通过
逐控制步写帧改变视频速度。

## 训练

```bash
uv run python sensor_training/train_stage1.py

uv run python sensor_training/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/<run>/model_final.pt

uv run python sensor_training/train_multimotion_stage1.py \
  --num-envs 4096 --max-iterations 30000

uv run python scripts/train_multimotion_no_stairs_stage2.py \
  --stage1-checkpoint \
    logs/multimotion/no_stairs_amp_single/stage1/<run>/model_final.pt \
  --num-envs 4096 --max-iterations 7500
```

`sensor_training/train_multimotion_stage1.py`、`sensor_training/train_multimotion_stage2.py`、
`scripts/train_multimotion_no_stairs_stage1.py` 和 `scripts/train_multimotion_no_stairs_stage2.py`
都指向唯一 canonical 任务 `multimotion_no_stairs_amp_single`，使用相同的四段参考
动作、6 个无阶梯课程变体和 `logs/multimotion/no_stairs_amp_single/` 日志树。
Stage 2 只改变阶段、随机化、checkpoint/reset 和训练迭代数；从 Stage 1 checkpoint
启动时同时重置 progress 与 optimizer。AMP 判别器按 motion ID 条件化，但该条件不进入
Student Actor。

所有 Stage 1 入口关闭域随机化，所有 Stage 2 入口必须加载对应任务的 Stage 1
checkpoint，并强制开启逐环境随机化以及定期重采样的 Warp 共享质量、惯量、摩擦和
重力参数。训练与测试每次启动以及每个 episode/reset 默认重采样 seed；仅在复现实验时
同时传入 `--seed N --no-episode-random-seed`。多动作 Stage 2 另外使用自适应随机化
强度课程。
全地形多动作使用
`scripts/train_multimotion_stage1.py` / `scripts/train_multimotion_stage2.py`。

多场景入口的平地与普通行走参考保留 CMU 69_01 经机器人尺度重定向后的原始速度
`0.8623 m/s`（`3.10 km/h`）；转弯和楼梯同样保留各自的原始参考速度。当前阶段优先
训练自然节奏下的稳定性；如需提高速度，应在稳定模型上增加独立的速度课程。
默认串联路线仅覆盖平地、5 cm 不平路、6°上/下坡和上下楼；楼梯使用
`0.13 m` 级高、`0.22 m` 踏面训练。无导航命令时左/右转在本体观测上不可辨识，因此
转向参考不混入这个直线地形策略的 AMP 教师分布。自适应领域随机化只在 Stage 2 启用，
并定期重采样共享质量、摩擦和重力参数。执行器功率、机械能和比能耗仍写入训练日志，
当前课程不据此判定模型是否合格，也不把比能耗加入奖励。

如果需要从 stage1 checkpoint 继续做速度微调，可以显式传入奖励和优化参数：

```bash
uv run python sensor_training/train_stage1.py \
  --resume-checkpoint logs/sensor_policy/stage1/<run>/model_final.pt \
  --resume-reset-optimizer \
  --run-name sensor-stage1-speed-ft \
  --learning-rate 2e-5 \
  --forward-velocity-reward-coef 0.5 \
  --forward-progress-reward-coef 0.25 \
  --forward-velocity-deficit-penalty-coef 0.5 \
  --forward-velocity-excess-penalty-coef 0.5 \
  --max-iterations 34000 \
  --save-interval 100
```

快速冒烟训练：

```bash
uv run python sensor_training/train_stage1.py \
  --num-envs 32 --max-iterations 1 --num-steps 4 \
  --no-wandb --run-name smoke
```

默认 checkpoint 写入 `logs/sensor_policy/`，不会与原训练日志混在一起。新训练只能
恢复同时带有相同 `sensor_observation` 和 `sensor_policy_architecture` 元数据的
100 Hz single-GRU/MLP checkpoint；旧 11 维 Critic 上下文、旧50 Hz recurrent、旧336/337维前馈传感器模型、
完整状态策略和历史力位联控模型均不兼容，必须使用新的运行目录从头训练。

导出的 ONNX 使用显式状态接口：输入为 `observation`、`recurrent_state` 和
`episode_start`，输出为 `action` 和 `next_recurrent_state`。项目播放入口会自动维护并
在 episode 结束后清零状态。

## 播放

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage1/<run>/model_final.pt \
  --viewer

uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/stage2/<run>/model_final.pt \
  --stage multimotion \
  --viewer
```

viewer 默认无限重复测试，直到手动关闭窗口；无需额外传 `--steps 0`。每次重置从稳定
动作起点开始，普通 episode 超时和随机动作相位默认关闭。需要有限测试时显式传入
`--steps <N>`，需要随机相位恢复测试时显式传入 `--motion-reset-prob 1.0`。
