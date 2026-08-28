# LYTrobot AMP 双足行走训练

本项目参考 `AMP_mjlab` 的训练组织方式，但针对当前机器人模型重新实现了一个轻量的 MuJoCo + PyTorch BPO-AMP 训练栈。训练模型固定使用：

- 机器人 MJCF：`robot/MJCFmodelposition.xml`
- 人体参考步态：`robot/reference_data/walk1.npz`或`robot/reference_data/walk1_mjcf.npz`
- 目标任务：用 AMP 风格奖励和前向速度跟踪奖励训练机器人向前行走

所有训练统一使用 14 个肌腱位置执行器，策略动作固定为 14 维位置目标修正量。

## 统一策略输入约束

项目所有训练入口统一构建传感器 GRU actor。行走、多动作和楼梯策略每个控制帧只包含
42 维 IMU 和 14 维实际肌腱长度，共 56 维；不接收关节状态、根部真值、目标速度、
技能标签或地形参数。蹲起是唯一例外：在相同的 336 维传感器历史后附加当前控制器的
`sin/cos` 相位，因此为 338 维。该相位由肌腱长度估计的蹲起深度和 IMU 姿态反馈校正，
不是仿真真值。行走 Actor 只传输当前 56 维帧，由 128 维 GRU 隐状态形成因果记忆；
6 帧历史仅为 AMP 判别器保留，不再作为冗余 Actor 输入搬运。
AMP、任务奖励和 asymmetric critic 可以使用训练期仿真真值，但这些值不会进入 actor
或导出的 ONNX。普通行走策略日志写入 `logs/sensor_policy/`，全地形和无阶梯串联课程日志
统一写入 `logs/multimotion/`，独立阶梯训练写入 `logs/stairs/`；旧完整状态 checkpoint、
旧 336 维无相位蹲起 checkpoint 与当前 338 维策略不兼容。

## 命令速查

所有训练命令都显式进入项目根目录；每个 run 仅保留最新的
`model_final.pt/.onnx` 与最佳的 `model_best.pt/.onnx`。`model_final` 用于续训，
`model_best` 用于播放/部署；`model_best.json` 仅记录最佳模型的选择指标。

### 训练与权重位置

普通行走：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage1.py
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage2.py --stage1-checkpoint logs/sensor_policy/stage1/<run>/model_final.pt
```

权重：`logs/sensor_policy/stage1/<run>/`、`logs/sensor_policy/stage2/<run>/`。

蹲起：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python squat_training/train_stage1.py
cd /home/yeah/code/LYTrobot_amp && uv run python squat_training/train_stage2.py --stage1-checkpoint logs/squat_sensor_policy/stage1/<run>/model_final.pt
```

权重：`logs/squat_sensor_policy/stage1/<run>/`、`logs/squat_sensor_policy/stage2/<run>/`。

全地形多动作（含楼梯）：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_stage1.py
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_stage2.py --stage1-checkpoint logs/multimotion/stage1/<run>/model_final.pt
```

权重：`logs/multimotion/stage1/<run>/`、`logs/multimotion/stage2/<run>/`。

无阶梯、单个 OmniRetarget 数据、纯 AMP 传感器策略：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_no_stairs_stage1.py
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_no_stairs_stage2.py --stage1-checkpoint logs/multimotion/no_stairs_amp_single/stage1/<run>/model_final.pt
```

权重：`logs/multimotion/no_stairs_amp_single/stage1/<run>/`、`logs/multimotion/no_stairs_amp_single/stage2/<run>/`。

无阶梯 privileged Teacher（08-06 当前安全近似）：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_no_stairs_amp_teacher.py
```

权重：`logs/multimotion/multimotion_no_stairs_amp_teacher/<run>/`。

无阶梯 IMU+肌腱 Student（冻结 Teacher 后蒸馏）：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_no_stairs_amp_student.py --teacher-checkpoint logs/multimotion/multimotion_no_stairs_amp_teacher/<run>/model_best.pt
```

权重：`logs/multimotion/multimotion_no_stairs_amp_student/<run>/`。

独立上下楼梯：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stairs_stage1.py
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stairs_stage2.py --stage1-checkpoint logs/stairs/stage1/<run>/model_final.pt
```

权重：`logs/stairs/stage1/<run>/`、`logs/stairs/stage2/<run>/`。

多动作网络布局的平地预训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_walk_stage1.py
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_walk_stage2.py --stage1-checkpoint logs/sensor_policy/multimotion_walk_warmup/stage1/<run>/model_final.pt
```

权重：`logs/sensor_policy/multimotion_walk_warmup/stage1/<run>/`、`logs/sensor_policy/multimotion_walk_warmup/stage2/<run>/`。

旧无阶梯参考条件基线（仅用于对照）：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_no_stairs_old.py
```

权重：`logs/sensor_policy/stage1/<run>/`；不要与 Teacher/Student 或纯 AMP 无阶梯任务混用。

### 测试、录屏与完整测试集

传感器行走、多动作、无阶梯、楼梯和 Student：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python sensor_training/play.py --checkpoint logs/<task>/<stage>/<run>/model_best.onnx --stage <stage> --viewer --steps 0 --record-fps 30
```

蹲起：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python squat_training/play.py --checkpoint logs/squat_sensor_policy/<stage>/<run>/model_best.onnx --stage <stage> --viewer --steps 0 --record-fps 30
```

Teacher 和旧参考条件基线：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/play.py --checkpoint logs/multimotion/multimotion_no_stairs_amp_teacher/<run>/model_best.pt --stage stage1 --device cuda --viewer --steps 0 --record-fps 30
```

录屏位于对应 checkpoint 的 `recordings/` 目录。播放固定按 100 Hz 仿真节拍等待；录屏按
仿真时间重采样 30 fps，所以播放与 MP4 均为真实仿真速度（1 秒视频 = 1 秒仿真）。

```bash
cd /home/yeah/code/LYTrobot_amp && .venv/bin/python -m pytest -q
cd /home/yeah/code/LYTrobot_amp && .venv/bin/python -m pytest -q tests/test_play_recording.py
```

`sensor_training/train*.py` 是对应 `scripts/` 传感器任务的兼容入口；除排查兼容性外，使用上面的正式入口。

## 所有任务的两阶段训练与测试命令

所有正式任务遵守同一约束：Stage 1 强制关闭域随机化；Stage 2 必须加载同任务的
Stage 1 checkpoint，并强制开启完整域随机化。Stage 2 包含逐环境的肌腱长度、动作延迟、
观测/动作噪声和随机推力，还开启 Warp 批次共享的质量、惯量、摩擦和重力随机化，默认
每 250 iteration 重采样共享参数。Stage 2 不能从随机权重启动，也不能用
`--no-domain-randomization` 关闭随机化。

训练和测试都默认随机：每次启动自动生成新的基础 seed，每个 episode/reset 再重采样
episode seed。所以下面的正式命令不要添加 `--seed`。只有复现实验时才同时添加：

```bash
--seed 42 --no-episode-random-seed
```

以下命令都从项目根目录执行；将 `<run>` 替换为实际时间目录：

```bash
cd /home/yeah/code/LYTrobot_amp
```

所有带 `--viewer` 的动作测试会自动录像。录像不再写到仓库根目录，而是按任务和阶段
分别保存到对应的 `logs/<task>/stage1/recordings/` 或
`logs/<task>/stage2/recordings/`，同名文件会自动追加序号且不会覆盖。



### 1. 普通行走

Stage 1 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage1.py
```

Stage 1 有界面测试：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage1/<run>/model_final.pt \
  --stage stage1 \
  --device cuda \
  --viewer \
  --steps 0
```

Stage 2 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/<run>/model_final.pt
```

Stage 2 有界面测试：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage2/<run>/model_final.pt \
  --stage stage2 \
  --device cuda \
  --viewer \
  --steps 0
```

`sensor_training/train_stage1.py`、`sensor_training/train_stage2.py` 是同一传感器行走任务的
兼容入口，不需要与上面的正式入口重复训练。

### 2. 蹲起

Stage 1 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python squat_training/train_stage1.py
```

Stage 1 有界面测试：

```bash
uv run python squat_training/play.py \
  --checkpoint logs/squat_sensor_policy/stage1/<run>/model_final.pt \
  --stage stage1 \
  --device cuda \
  --viewer \
  --steps 0
```

Stage 2 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python squat_training/train_stage2.py \
  --stage1-checkpoint logs/squat_sensor_policy/stage1/<run>/model_final.pt
```

Stage 2 有界面测试：

```bash
uv run python squat_training/play.py \
  --checkpoint logs/squat_sensor_policy/stage2/<run>/model_final.pt \
  --stage stage2 \
  --device cuda \
  --viewer \
  --steps 0
```

### 3. 全地形多动作

Stage 1 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_stage1.py
```

Stage 1 有界面测试（默认同屏 3 台机器人和随机路线）：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/stage1/<run>/model_final.pt \
  --stage multimotion \
  --device cuda \
  --viewer \
  --steps 0
```

Stage 2 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_stage2.py \
  --stage1-checkpoint logs/multimotion/stage1/<run>/model_final.pt
```

Stage 2 有界面测试（默认同屏 3 台机器人和随机路线）：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/stage2/<run>/model_final.pt \
  --stage multimotion \
  --device cuda \
  --viewer \
  --steps 0
```

`sensor_training/train_multimotion_stage2.py` 是全地形 Stage 2 的兼容别名。全地形
Stage 1 请使用上面的 `scripts/train_multimotion_stage1.py`；传感器目录下同名 Stage 1
入口已改为下节的无阶梯任务。

### 4. 无阶梯随机串联行走

该入口沿用普通行走的传感器 Actor、奖励主体、PPO/AMP 参数和两阶段流程；路线包含
平地、不平路面、6° 上坡和 6° 下坡，四类地形以 6 种随机串联顺序训练。每次测试启动
也会使用新的随机 seed 生成路线。训练经过入口预热后逐步加入随机后续阶段 reset，避免
后段地形样本饥饿；测试和完整路线成功率仍强制从入口连续走到出口。

当前配置以“先稳定走完全程”为目标，使用已验证的
`multimotion/walk_omniretarget.npz` 与 `multimotion/rough_walk_omniretarget.npz`，目标速度约
`0.8623 m/s`，不再强制单一 1.39 m/s 快走参考。Stage 1 reset 中 50% 使用随机步态
姿态、速度和相位学习步态延续，另外 50% 保持回放一致的“路线入口、phase 0、静止”
起点；后续地形随机 reset 从 5000 轮开始开放，到 15000 轮达到 80%，使粗糙地面和坡面
获得足够训练样本。参考状态的局部 X/Y 位移会重定位到路段入口，不会虚增路线进度。
Actor 和 ONNX 输入仍严格只有 IMU 与肌肉长度，参考动作、相位、目标速度和地形标签均
不进入网络或执行器基准。

机器人所有实体几何现在都只与静态地形碰撞，机器人内部不产生自碰撞；地面、平台、
坡面和高度场统一使用硬接触参数。Warp 每个 world 的接触/约束容量提高到 `128/512`，
原生凸体碰撞保持开启，随机 reset 会按实际碰撞面上抬直至零嵌入。训练对每毫米地形
嵌入施加惩罚，深度超过 1 cm 立即终止；支撑脚切向速度另有独立滑移惩罚。

`08-25-11:38:09` 运行使用了失效的串联 reset（配置的 `motion_reset_prob=0.5` 实际被
强制覆盖成 phase 0 + 零速度），不能通过继续增加旧训练轮数修复，应从头启动新 run。

Stage 1 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_no_stairs_stage1.py
# 等价的传感器目录入口：
cd /home/yeah/code/LYTrobot_amp && uv run python sensor_training/train_multimotion_stage1.py
```

Stage 1 有界面测试：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/no_stairs/stage1/08-27-10:25:51/model_best.onnx\
  --stage stage1 \
  --device cuda \
  --viewer \
  --steps 0
```

Stage 2 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_no_stairs_stage2.py \
  --stage1-checkpoint \
    logs/multimotion/no_stairs/stage1/<run>/model_best.pt
```

Stage 2 有界面测试：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/no_stairs/stage2/<run>/model_final.pt \
  --stage stage2 \
  --device cuda \
  --viewer \
  --steps 0
```

### 5. 独立上下楼梯

Stage 1 楼梯默认级高/踏深为 `0.13/0.32 m`，Stage 2 切换到更困难的
`0.20/0.22 m`。训练路线固定为“地面平地走近 → 连续上楼 → 顶部继续前进
→ 连续下楼 → 平地走出”，上楼和下楼不作为两个独立 episode。测试强制从路线入口
开始，关闭随机阶段 reset，机器人状态在动作阶段切换时不会被搬运或跳变。阶梯碰撞、
奖励和失败检测共用同一套世界坐标表面；骨盆陷入踏面以下会立即终止，不能靠穿模得分。
Stage 1 前 `15000` 轮只从路线入口训练稳定步态，再用 `7500` 轮把非零阶段 reset
概率线性升到 `0.35`，避免楼梯失败样本过早淹没平地行走能力。`model_best` 只有在滚动
窗口内真实跨过至少一个路线阶段后才会生成；仅有入口进度但从未跨阶段的策略只保留为
`model_final`，不会再被误标为最佳模型。

Stage 1 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stairs_stage1.py \
  --stair-step-height 0.13
```

Stage 1 有界面测试：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/stairs/stage1/08-19-15:47:30/model_final.onnx \
  --stage stage1 \
  --device cuda \
  --viewer \
  --steps 0
```

Stage 2 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stairs_stage2.py \
  --stage1-checkpoint logs/stairs/stage1/<run>/model_best.pt \
  --stair-step-height 0.13
```

Stage 2 有界面测试：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/stairs/stage2/<run>/model_final.pt \
  --stage stage2 \
  --device cuda \
  --viewer \
  --steps 0
```

### 6. 多动作网络布局的平地预训练

Stage 1 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_walk_stage1.py
```

Stage 1 有界面测试：

```bash
uv run python sensor_training/play.py \
  --checkpoint \
    logs/multimotion/no_stairs/stage1/08-18-08:42:56/model_best.onnx\
  --stage stage1 \
  --no-serial-terrain-course \
  --device cuda \
  --viewer \
  --steps 0
```

Stage 2 训练：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_walk_stage2.py \
  --stage1-checkpoint \
    logs/sensor_policy/multimotion_walk_warmup/stage1/<run>/model_final.pt
```

Stage 2 有界面测试：

```bash
uv run python sensor_training/play.py \
  --checkpoint \
    logs/sensor_policy/multimotion_walk_warmup/stage2/<run>/model_final.pt \
  --stage stage2 \
  --no-serial-terrain-course \
  --device cuda \
  --viewer \
  --steps 0
```

所有测试命令都强制打开 MuJoCo viewer，并用 `--steps 0` 持续运行到手动关闭窗口。
Stage 2 checkpoint 测试默认继承训练配置中的域随机化；标称参数诊断可额外添加
`--no-domain-randomization`，鲁棒性测试可显式添加 `--domain-randomization`。默认测试
seed 每次启动、每次 reset 都不同；如需复现，在对应的有界面测试命令末尾同时添加
`--seed 42 --no-episode-random-seed`，不要删除 `--viewer --steps 0`。

正式实验不要直接运行无阶段后缀的任务脚本；它们只定义动作、地形、奖励和公共网络参数。
恢复中断的 Stage 2 时使用 `--resume-checkpoint <stage2-checkpoint>`，不要同时传
`--stage1-checkpoint`。所有训练原子更新最新状态 `model_final.pt/.onnx`，并额外保留按
任务完成质量选择的 `model_best.pt/.onnx` 和选择依据 `model_best.json`。

## 直线全地形多动作训练

当前训练策略按要求暂不训练转身，只使用 walk、rough、pit、gap、stair_up、
stair_down 六种参考动作，连续通过不平路、沟坑、间隙、坡道和楼梯。
左/右转参考文件仍保留给后续独立转身课程使用，不再拼入直线地形路线：

```bash
cd /home/yeah/code/LYTrobot_amp
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_stage1.py \
  --num-envs 4096 --max-iterations 30000

cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_multimotion_stage2.py \
  --stage1-checkpoint \
    logs/multimotion/stage1/<run>/model_best.pt \
  --num-envs 4096 --max-iterations 7500
```

该入口已内置 4096 环境配置：每轮 24 步、98,304 条 transition，PPO 使用 8 个
minibatch（每个 12,288 条），AMP 每轮执行 2 次 4096 样本更新。

若需要额外平地课程，先用 `scripts/train_multimotion_walk_stage1.py` 完成兼容多动作网络布局的平地
稳定预训练，再把该 checkpoint 传给上面的完整课程入口；具体命令和晋级指标见
[`multimotion/README.md`](multimotion/README.md)。
旧的八动作（含 turn_left/turn_right）checkpoint 使用 8 维技能条件，不能直接加载到
当前六动作策略；需要从六动作平地 warmup 新开训练。

多动作训练的 checkpoint、ONNX、指标和 W&B 文件统一保存在
`logs/multimotion/stage1|stage2/<月-日-时:分:秒>/`，每次启动自动创建新的时间目录。
每个训练机器人面前都是包含 rough、pit、gap、6°上/下坡和 stairs 的完整直线串联路线；启动时预生成
多种随机地形顺序，episode 会随机分配路线并连续执行全部动作/地形段，走完整条路线
后才成功重置。默认前 5000 轮全部从路线入口训练，随后用 10000 轮把随机后续阶段
reset 的概率线性提高到 80%，避免尚未学会平地稳定行走时就被直接投放到困难地形；
成功率始终只统计从路线入口开始并完整走到终点的 episode。
周期保存、内存保护保存和正常退出始终原子覆盖最新的 `model_final.pt/.onnx`；每个保存
周期还会按完整路线成功率、阶段进度、奖励和滑移更新 `model_best.pt/.onnx`，不再生成
按迭代编号的历史权重。

可视化测试最新模型（打开 viewer 后会同步录屏）：

```bash
cd /home/yeah/code/LYTrobot_amp
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/stage2/<run>/model_final.pt \
  --stage multimotion \
  --device cuda \
  --viewer \
  --steps 0
```

录像默认保存为
`logs/multimotion/stage2/recordings/<run>-<iteration>.mp4`；重复测试会追加编号，
不会覆盖已有录像。多动作测试默认同屏显示 3 个机器人，每个机器人使用一条独立的
随机顺序完整路线；可用 `--num-robots` 调整数量。需要复现路线顺序时同时使用
`--seed N --no-episode-random-seed`。

动作数据来源、重定向复现、阶梯尺寸、Stage2 命令和实现说明见
[`multimotion/README.md`](multimotion/README.md)。

## 代码结构

```text
scripts/train.py              # 训练入口，默认打开 wandb
scripts/train_stage1.py       # 第一阶段训练入口，GPU/Warp，关闭领域随机化
scripts/train_stage2.py       # 第二阶段训练入口，GPU/Warp，开启领域随机化
scripts/train_multimotion_stage*.py           # 全地形多动作两阶段入口
scripts/train_multimotion_no_stairs_stage*.py # 平地/不平路/上下坡随机串联两阶段入口
scripts/train_stairs_stage*.py                # 连续平地-上楼-下楼-平地两阶段入口
scripts/two_stage_task.py                     # 两阶段强制参数与checkpoint衔接
scripts/view_model.py         # 机器人 MJCF 本地 MuJoCo 可视化/加载测试
sensor_training/play.py       # 行走、多动作和楼梯传感器 checkpoint 播放
squat_training/play.py        # 蹲起传感器 checkpoint 播放
scripts/play.py               # 上述播放入口共用的底层实现
src/lytrobot_amp/env.py       # MuJoCo 向量环境、动作映射、奖励和终止
src/lytrobot_amp/env_warp.py  # MuJoCo Warp GPU batched 环境
src/lytrobot_amp/motion.py    # robot/reference_data/walk1.npz 到机器人下肢关节的参考数据映射
src/lytrobot_amp/networks.py  # Actor-Critic 和 AMP discriminator
src/lytrobot_amp/ppo_amp.py   # rollout、GAE、BPO policy update、AMP update
```

`robot/reference_data/walk1.npz` 中的下肢关节名与机器人一致，加载时会自动映射 14 个公共下肢关节；机器人特有的 patella slide joints 保留 MJCF 默认值和 equality 约束。

## 环境

推荐直接使用项目自己的 `uv` 环境，依赖已固定到当前 NVIDIA 570 驱动可用的 CUDA 12.8 组合：

```bash
cd /home/yeah/code/LYTrobot_amp
uv sync
```

当前核心版本为：

- `mujoco==3.9.0`
- `mujoco-warp==3.9.0.1`
- `warp-lang==1.14.0`
- `torch==2.10.0`（CUDA 12.8）
- `numpy==2.4.6`
- `wandb==0.27.2`

旧版 MuJoCo-Warp/Warp 会出现长期训练时主内存持续增长，不要混用其他虚拟环境中的旧包。

## 正常训练

训练必须使用 GPU/Warp：`--device cuda --sim-backend warp`。Stage 1 默认 30000
iteration，Stage 2 通常默认 7500 iteration；任务入口默认使用 4096 个环境。每个
iteration 写本地日志，W&B 默认每 10 轮上报一次。第一阶段强制关闭域随机化，第二阶段
强制加载 checkpoint 并开启域随机化。每次训练启动和每个 episode 默认使用新随机 seed；
需要复现时必须同时传 `--seed N --no-episode-random-seed`。完整命令统一放在本文开头
“所有任务的两阶段训练与测试命令”。

训练默认启用 Warp CUDA Graph，物理积分周期为 `0.005 s`、策略 `decimation=2`，因此训练、测试和部署 ONNX 均为 `100 Hz` 控制。入口拒绝其他控制频率以及旧 50 Hz checkpoint，防止训练/测试动力学错配。所有普通、传感器、多动作、无阶梯和楼梯训练入口共享同一套快速物理默认值。新训练默认采用 `--robot-collision-profile feet`：脚掌和脚趾继续使用四个完整 STL 网格、三维摩擦接触、`5 mm` 预测 margin 和原有硬接触参数，其余身体网格不进入地形碰撞 broadphase，跌倒仍由根高度和姿态终止；求解器使用 `iterations=10`、`ls_iterations=20`，每个 world 预留 32 个 contact 和 128 个 constraint。重置只在仍有地形穿透时追加全批 Warp forward，传感器历史也复用当步已计算的 IMU、肌腱和 AMP 帧。旧 checkpoint 缺少碰撞模式字段时按 `full_body` 解释，并自动保留至少 `20` 次求解迭代和 `128/512` 容量，以维持原动力学回放。精确续训可显式传 `--robot-collision-profile full_body`；要从旧权重迁移到快速足部模式，需传 `--allow-collision-profile-migration` 且不能指定旧 `--log-dir`，以免把两种动力学写进同一日志。可分别用 `--solver-iterations`、`--solver-ls-iterations`、`--warp-nconmax`、`--warp-njmax` 显式增加精度/容量，也可用 `--no-warp-cuda-graph` 关闭 CUDA Graph 调试。默认不主动 sleep；与桌面共用 GPU 时可显式增加 `--iteration-sleep-ms 10`。

`full_body` 模式下全身 STL 凸碰撞体均与静态地形建立约束；默认快速 `feet` 模式仅让脚掌和脚趾的 4 个具名完整网格参与地形碰撞及接触奖励。两种模式都禁止机器人内部自碰撞。旧版 MuJoCo-Warp 3.8.0.2/Warp 1.12.0 存在长期训练主内存增长问题；当前版本默认关闭周期性 Python GC，但仍保留 32GB RSS 安全停止保护，必要时可通过 `--memory-cleanup-interval` 恢复定期回收。

`--stage1-checkpoint` 用来指定第一阶段权重 `model_final.pt`，并把第二阶段的 iteration 与 `global_step` 从 0 重新计数。蹲起 Stage2 还会重置优化器，并根据成功率、下蹲深度和异常终止率自适应调整域随机化强度；不要把旧 Stage2 checkpoint 传给这个参数。也可以使用等价别名 `--stage1-model`。如果需要恢复第二阶段自身的中断训练，使用底层参数 `--resume-checkpoint logs/sensor_policy/stage2/<run>/model_final.pt`，此时会保留第二阶段已有的训练进度、域随机化强度和优化器状态。旧版训练产生的编号 checkpoint 仍可加载，但新训练不会再生成。

第二阶段默认随机化包括：负重质量、各 body 质量、肌腱长度比例、地面摩擦系数、重力加速度 `9.75~9.85m/s^2`、随机方向/大小推力、观测噪声、动作噪声、`2~5ms` action delay。范围均可用命令行覆盖，例如：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/<stage1_run>/model_final.pt \
  --payload-mass-min 0.0 \
  --payload-mass-max 3.0 \
  --body-mass-scale-min 0.85 \
  --body-mass-scale-max 1.15 \
  --tendon-length-scale-min 0.95 \
  --tendon-length-scale-max 1.05 \
  --ground-friction-min 0.55 \
  --ground-friction-max 1.40 \
  --gravity-min 9.75 \
  --gravity-max 9.85 \
  --push-force-min 0.0 \
  --push-force-max 80.0 \
  --obs-noise-std 0.02 \
  --action-noise-std 0.01 \
  --action-delay-min-ms 2.0 \
  --action-delay-max-ms 5.0
```

为了让策略具备物理样机可用的“静止起步”能力，训练环境默认会把一部分 episode 设为双脚静止站立起步：先保持站立相位，再用 80~160 个 policy step 平滑接入周期步态，起步前 100 step 不施加随机推力。相关参数可以覆盖：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/<stage1_run>/model_final.pt \
  --stand-start-prob 0.5 \
  --stand-start-phase-delay-steps 25 \
  --stand-start-transition-min-steps 80 \
  --stand-start-transition-max-steps 160 \
  --stand-start-push-delay-steps 100
```

测试样机起步效果时建议强制从静止起步开始：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage2/<run>/model_final.pt \
  --device cuda \
  --viewer \
  --steps 0 \
  --stand-start-prob 1.0
```


使用 `--viewer` 打开 MuJoCo 测试时会自动录制 MP4，默认从 checkpoint 路径识别任务
和阶段，保存到该任务的 `logs/<task>/stage1|stage2/recordings/`。命名来自权重所在
run 和 iteration，例如：

```text
logs/sensor_policy/stage1/recordings/06-06-21-19-41-2049.mp4
```

默认录制为 1080p、30fps、H.264、`yuv420p`、`faststart` MP4，兼容 PPT 和微信播放。播放循环固定使用 100 Hz 实时墙钟节拍，录屏按照仿真时间重采样为 30fps，因此 1 秒视频严格对应 1 秒仿真，不存在倍速或逐控制步写帧造成的慢放。MuJoCo 窗口和录屏镜头都会始终跟随机器人 `pelvis` 移动。重复测试同一个权重时会自动追加 `-2`、`-3`，避免覆盖已有视频。

所有 checkpoint 播放入口统一采用持续重复测试语义：只要 viewer 没有关闭就一直运行。
循环动作直接继续下一个周期；非循环动作、完整多地形路线或单次蹲起完成后，从稳定的
动作起点开始下一轮；真实摔倒也会重置后继续测试。viewer 模式省略 `--steps` 时等价于
`--steps 0`，显式传入正整数才限制步数。回放默认禁用普通 episode 超时、reset noise
和随机参考相位，但每次启动及每次 reset 仍使用新随机 seed；需要专项测试随机参考相位
恢复时可显式传入 `--motion-reset-prob 1.0`。
终端汇总中的 `repetitions` 表示动作/路线完成次数，`terminations` 表示真实失稳次数，
两者不会再混为平均奖励。

首次在线记录前需要：

```bash
wandb login
```

常用参数：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage1.py \
  --num-envs 1024 \
  --target-velocity 1.0 \
  --motion-reset-prob 0.5 \
  --action-residual-scale 0.5 \
  --distal-action-scale 1.0 \
  --reference-ctrl-blend 0.0 \
  --disc-learning-rate 3e-5 \
  --amp-grad-penalty-coef 2.0 \
  --amp-label-smoothing 0.20 \
  --amp-obs-noise-std 0.05 \
  --amp-reward-coef 1.2 \
  --task-reward-lerp 0.50 \
  --max-iterations 30000 \
  --save-interval 500
```

控制固定为纯位置模式：策略动作保持 14 维，只控制肌腱位置目标。
历史 28 维力位联控 checkpoint 不再兼容；续训和播放需使用 14 维位控 checkpoint。

足端稳定奖励分为三个互不冲突的部分：参考局部足速只约束摆动脚；支撑脚按脚掌/脚趾的真实接触掩码约束世界坐标水平速度；参考要求支撑但实际离地时单独扣分。参考接触标签只用于判断当前是否应当落地，不再把参考局部足速错误地当成支撑脚世界零速。默认参数为：

```bash
--foot-vel-reward-coef 0.35 \
--stance-foot-vel-sigma 0.15 \
--stance-foot-reward-coef 1.0 \
--contact-reward-coef 0.70 \
--missed-contact-penalty-coef 0.80
```

其中 `task_stance_foot` 越接近 1 越好，`task_stance_foot_speed` 是接触足端的世界水平速度（m/s），`task_missed_contact` 是参考支撑却未接触的比例，后两项应持续下降。

日志和权重写入 `logs/<stage>/<时间>/`，包括：

- `params/config.json`
- `model_final.pt`
- `model_final.onnx`
- `model_best.pt`
- `model_best.onnx`
- `model_best.json`

## 从已有 checkpoint 稳定化续训

如果策略已经能走但脚踝/双脚高频抖动，优先从最新稳定行走模型续训，而不是从头训练。第一阶段不要立刻缩小控制幅度，只恢复 actor/discriminator 和 `global_step`，重置 Adam 优化器状态，用新的足踝平滑奖励和有限探索标准差先把动作抖动压下来：

```bash
cd /home/yeah/code/LYTrobot_amp
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage1.py \
  --resume-checkpoint logs/06-06-21:19:41/model_final.pt \
  --resume-reset-optimizer \
  --run-name 06-07-stabilize-ankle-stage1 \
  --learning-rate 1e-4 \
  --disc-learning-rate 2e-5 \
  --action-residual-scale 0.5 \
  --distal-action-scale 1.0 \
  --max-iterations 2600 \
  --save-interval 25
```

第二阶段鲁棒性训练从第一阶段最好的 checkpoint 继续，并启用领域随机化：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/06-07-stabilize-ankle-stage1/model_final.pt \
  --resume-reset-optimizer \
  --run-name 06-07-stabilize-ankle-stage2 \
  --learning-rate 7e-5 \
  --disc-learning-rate 2e-5 \
  --action-residual-scale 0.45 \
  --distal-action-scale 0.9 \
  --max-iterations 3500 \
  --save-interval 25
```

`--max-iterations` 是最终 iteration 编号上限，不是额外训练轮数；恢复时会从 `model_final.pt` 内记录的 iteration 继续，设置 `2600` 表示训练到 iteration 2599。重点观察 wandb 中的 `task_action_rate`、`task_distal_action_rate`、`task_stance_foot`、`task_stance_foot_speed`、`task_missed_contact`、`task_distal_vel`、`mean_length` 和 `mean_amp_reward`。

针对当前支撑脚前后蹭动，先从与第二阶段兼容的 14 维 position stage1 策略低学习率续训约 4000 iteration：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage1.py \
  --resume-checkpoint logs/sensor_policy/stage1/06-18-10:01:10/model_final.pt \
  --resume-reset-optimizer \
  --run-name 06-30-no-slip-stage1 \
  --learning-rate 5e-5 \
  --disc-learning-rate 2e-5 \
  --max-iterations 34000 \
  --save-interval 100
```

选择 `task_stance_foot_speed`、`task_missed_contact` 已下降且步态未退化的 checkpoint，再进入第二阶段；不要直接用旧 stage2 权重消除已经形成的腾空蹭步：

```bash
cd /home/yeah/code/LYTrobot_amp && uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/06-30-no-slip-stage1/model_final.pt \
  --resume-reset-optimizer \
  --run-name 06-30-no-slip-stage2 \
  --learning-rate 3e-5 \
  --disc-learning-rate 1e-5 \
  --max-iterations 10000 \
  --save-interval 100
```

## 模型文件可视化测试

不依赖训练 checkpoint，直接加载 `robot/MJCFmodelposition.xml` 做 MuJoCo 模型测试。

本地打开 MuJoCo viewer：

```bash
cd /home/yeah/code/LYTrobot_amp
uv run python scripts/view_model.py \
  --model robot/MJCFmodelposition.xml \
  --steps 2000
```

默认会使用 `robot/reference_data/walk1.npz` 第一帧初始化模型姿态，并把 14 个肌腱位置执行器设置到当前肌腱长度附近，便于本地 viewer 中稳定查看模型。

## 播放 checkpoint

每次保存都会原子覆盖 `model_final.pt`（PyTorch 完整训练 checkpoint）和
`model_final.onnx`（确定性 Actor），并在任务完成质量提高时原子更新
`model_best.pt/.onnx`。只有 PT/ONNX 两者都成功写入并通过 ONNX 校验后才替换旧版本。
`.pt` 可用于续训和 MuJoCo 播放，
`.onnx` 内嵌环境/网络配置元数据，可直接用于 MuJoCo 播放。

PyTorch checkpoint 有界面测试：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage2/<run>/model_final.pt \
  --stage stage2 \
  --device cuda \
  --viewer --steps 0
```

ONNX 模型使用同一个传感器播放入口：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage2/<run>/model_final.onnx \
  --stage stage2 --viewer --steps 0
```

当前 `08-27-15:38:23` Stage 1 ONNX 模型的有界面测试命令：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage1/08-27-15:38:23/model_final.onnx\
  --stage stage1 \
  --viewer \
  --steps 0
```

该 ONNX 内嵌 56 维传感器输入、128 维 GRU 状态和环境配置，播放入口会自动维护并在
episode 重置时清零循环状态。模型元数据启用了平地、不平路面、上坡和下坡组成的无阶梯
串联路线，因此上面的命令会按训练配置测试完整路线。`--steps 0` 表示持续播放，关闭
MuJoCo 窗口即可结束；带 `--viewer` 时会自动录制视频。ONNX 推理固定使用 CPU
Execution Provider，无需指定 `--device cuda`。

录像会根据 checkpoint 路径保存到对应阶段的 `recordings/` 目录；例如当前模型写入
`logs/sensor_policy/stage1/recordings/08-27-15-38-23-<iteration>.mp4`，同名录像会自动
追加序号而不会覆盖。其他动作同样保存到各自 checkpoint 所在任务的
`stage1/recordings/` 或 `stage2/recordings/`。
