# LYTrobot AMP 双足行走训练

本项目使用 MuJoCo + PyTorch BPO-AMP 训练双足机器人稳定、自然地向前行走。所有策略统一使用 14 个肌腱位置执行器，策略动作是 14 维位置目标修正量。

## 快速索引

| 任务 | 训练入口 | 测试入口 | 权重目录 |
| --- | --- | --- | --- |
| Stage1 基础行走 | `scripts/train_stage1.py` | `scripts/play.py` | `logs/stage1/<run>/` |
| Stage2 鲁棒行走 | `scripts/train_stage2.py` | `scripts/play.py` | `logs/stage2/<run>/` |
| 仅 IMU+肌腱传感器策略 | `sensor_training/train_stage1.py` / `train_stage2.py` | `sensor_training/play.py` | `logs/sensor_policy/` |
| 蹲起传感器策略 | `squat_training/train_stage1.py` / `train_stage2.py` | `squat_training/play.py` | `logs/squat_sensor_policy/` |
| 多动作/全地形行走 | `scripts/train_multimotion.py` | `scripts/play.py` | `logs/multimotion/` |
| GPU/Warp 冒烟验证 | `scripts/train_stage1.py` / `train_stage2.py` | `scripts/play.py` | `logs/smoke_gpu/` |
| 仅测试 MJCF 模型 | 无 | `scripts/view_model.py` | 无 |

以下命令都从仓库根目录执行：

```bash
cd /home/yeah/code/LYTrobot_amp_recovered_20260806
```

## 环境安装

推荐使用项目自己的 `uv` 环境：

```bash
uv sync
```

训练必须使用 GPU/Warp：`--device cuda --sim-backend warp`。当前主要依赖为 `mujoco==3.9.0`、`mujoco-warp==3.9.0.1`、`warp-lang==1.14.0`、`torch==2.10.0`（CUDA 12.8）和 `wandb==0.27.2`。首次在线记录前执行：

```bash
wandb login
```

机器人模型为 `robot/MJCFmodelposition.xml`，参考步态为 `robot/reference_data/walk1.npz` 或 `walk1_mjcf.npz`。

控制回路默认 100 Hz（5 ms 物理步长、`--decimation 2`）。为复现已有的 50 Hz
策略或继续其训练，可显式传入 `--decimation 4`；回放程序会优先采用 checkpoint
中保存的训练配置。

## 任务一：Stage1 基础稳定行走

Stage1 关闭领域随机化，先让机器人学会稳定站立和周期性向前行走。默认 1024 个环境、30000 iterations，权重保存到 `logs/stage1/<run>/`。

### 从头训练

```bash
uv run python scripts/train_stage1.py \
  --device cuda \
  --sim-backend warp \
  --num-envs 1024 \
  --max-iterations 30000
```

### 从已有 Stage1 权重续训

```bash
uv run python scripts/train_stage1.py \
  --resume-checkpoint logs/stage1/<run>/model_<iter>.pt \
  --resume-reset-optimizer \
  --run-name <new-run-name> \
  --max-iterations 34000
```

脚踝抖动或支撑脚蹭地时，可使用较低学习率续训：

```bash
uv run python scripts/train_stage1.py \
  --resume-checkpoint logs/stage1/06-18-10:01:10/model_final.pt \
  --resume-reset-optimizer \
  --run-name 06-30-no-slip-stage1 \
  --learning-rate 5e-5 \
  --disc-learning-rate 2e-5 \
  --max-iterations 34000 \
  --save-interval 100
```

### 测试 Stage1 checkpoint

无窗口测试：

```bash
uv run python scripts/play.py \
  --checkpoint logs/stage1/09-08-23:06:21/model_27000.onnx \
  --device cuda \
  --viewer \
  --steps 0 \
  --seed 42
```

打开 MuJoCo viewer 并录制视频：

```bash
uv run python scripts/play.py \
  --checkpoint logs/stage1/09-08-23:06:21/model_final.onnx \
  --device cuda \
  --viewer \
  --steps 0 \
  --seed 42
```

`.pt` 和 `.onnx` 都可以播放；ONNX 是确定性 Actor，使用同一个 `scripts/play.py` 入口。

## 任务二：Stage2 鲁棒行走

Stage2 从 Stage1 权重开始，开启领域随机化，默认 4096 个环境、7500 iterations，权重保存到 `logs/stage2/<run>/`。Stage1 和 Stage2 默认都使用 7.3728 亿个 policy sample。

### 从 Stage1 开始训练

```bash
uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/stage1/09-08-23:06:21/model_final.pt \
  --resume-reset-optimizer \
  --device cuda \
  --sim-backend warp
```

`--stage1-checkpoint` 会把 Stage2 的 iteration 和 `global_step` 从 0 开始计数。不要把旧 Stage2 权重作为这个参数；恢复中断的 Stage2 训练应使用 `--resume-checkpoint`。

### 恢复中断的 Stage2 训练

```bash
uv run python scripts/train_stage2.py \
  --resume-checkpoint logs/stage2/<run>/model_<iter>.pt \
  --device cuda \
  --sim-backend warp
```

### 测试 Stage2 checkpoint

```bash
uv run python scripts/play.py \
  --checkpoint logs/stage2/09-10-15:43:15/model_3500.onnx \
  --device cuda \
  --viewer \
  --steps 0 \
  --seed 42
```

测试静止起步能力时，可强制所有 episode 从静止开始：

```bash
uv run python scripts/play.py \
  --checkpoint logs/stage2/<run>/model_final.pt \
  --device cuda \
  --viewer \
  --steps 0 \
  --stand-start-prob 1.0
```

领域随机化包括负重、body 质量、肌腱长度、地面摩擦、重力、随机推力、观测噪声、动作噪声和动作延迟；参数可用 `--payload-mass-*`、`--body-mass-scale-*`、`--ground-friction-*`、`--push-force-*`、`--obs-noise-std`、`--action-noise-std` 和 `--action-delay-*-ms` 覆盖。

## 任务三：仅 IMU+肌腱长度传感器策略

该任务使用 `sensor_training/` 下的独立训练和播放入口。

### Stage1 训练与测试

```bash
uv run python sensor_training/train_stage1.py

uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage1/09-09-15:44:13/model_final.onnx\
  --device cuda \
  --viewer \
  --steps 0
```

### Stage2 训练与测试

```bash
uv run python sensor_training/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/<run>/model_final.pt

uv run python sensor_training/play.py \
  --checkpoint logs/stage2/09-10-15:43:15/model_final.onnx \
  --device cuda \
  --viewer \
  --steps 0
```

## 任务四：蹲起传感器策略

### Stage1 训练与测试

```bash
uv run python squat_training/train_stage1.py

uv run python squat_training/play.py \
  --checkpoint logs/squat_sensor_policy/stage1/<run>/model_final.pt \
  --viewer \
  --steps 0
```

### Stage2 训练与测试

```bash
uv run python squat_training/train_stage2.py \
  --stage1-checkpoint logs/squat_sensor_policy/stage1/<run>/model_final.pt

uv run python squat_training/play.py \
  --checkpoint logs/squat_sensor_policy/stage2/<run>/model_final.pt \
  --no-domain-randomization \
  --seed 42 \
  --no-episode-random-seed \
  --viewer \
  --steps 0
```

## 任务五：多动作/全地形行走

多动作训练支持两套可切换的参考布局，均连续通过平地、不平路和上下坡，不把转身和阶梯拼入直线路线：

- `--reference-set two`：原始的 `walk + rough_walk`，各 50%；坡地沿用 `rough_walk` 专家。
- `--reference-set four`：`walk + rough_walk + slope_up + slope_down`；坡地使用专属参考，默认模式。

两套策略的 skill one-hot 维度不同，checkpoint **不可互相续训**。日志分别写入 `logs/multimotion_two_ref/` 与 `logs/multimotion_four_ref/`，不会混淆。

### 推荐训练流程

先进行平地 warmup，使网络布局与多动作策略兼容：

```bash
uv run python scripts/train_multimotion_walk.py \
  --reference-set two \
  --num-envs 1024 \
  --max-iterations <warmup-iterations>
```

再进入完整全地形课程：

```bash
uv run python scripts/train_multimotion.py \
  --reference-set two \
  --num-envs 4096 \
  --max-iterations 30000
```

四参考坡地专属训练：

```bash
uv run python scripts/train_multimotion_walk.py \
  --reference-set four \
  --num-envs 1024 \
  --max-iterations <warmup-iterations>

uv run python scripts/train_multimotion.py \
  --reference-set four \
  --num-envs 4096 \
  --max-iterations 30000
```

也可以从 warmup checkpoint 进入完整课程：

```bash
uv run python scripts/train_multimotion.py \
  --reference-set four \
  --resume-checkpoint logs/multimotion_walk_warmup/<run>/model_final.pt \
  --resume-reset-progress \
  --resume-reset-optimizer \
  --num-envs 4096 \
  --max-iterations 30000
```

### 测试多动作 checkpoint

```bash
uv run python scripts/play.py \
  --checkpoint logs/multimotion_four_ref/09-10-12:37:53/model_4500.onnx\
  --stage multimotion \
  --device cuda \
  --viewer \
  --num-robots 3 \
  --steps 0 \
  --seed 42
```

多动作测试默认显示 3 个机器人，每个机器人使用独立的随机路线；可用 `--num-robots 1` 改成单机器人。具体动作数据、重定向和课程说明见 [`multimotion/README.md`](multimotion/README.md)。

旧八动作/六动作 checkpoint 的技能 one-hot 维度与当前两动作策略不同，不能直接加载；需要重新进行两动作 warmup。

## 任务六：GPU/Warp 冒烟验证

用于快速确认训练环境、模型导出和 CUDA/Warp 配置正常。冒烟测试关闭 W&B，不用于评价策略质量。

### Stage1 冒烟训练与测试

```bash
WANDB_MODE=disabled uv run python scripts/train_stage1.py \
  --num-envs 64 \
  --max-iterations 1 \
  --num-steps 4 \
  --epochs 1 \
  --minibatches 1 \
  --amp-minibatch-size 64 \
  --amp-disc-updates 1 \
  --save-interval 1 \
  --no-wandb \
  --log-root logs/smoke_gpu

uv run python scripts/play.py \
  --checkpoint logs/smoke_gpu/stage1/<run>/model_final.onnx \
  --device cuda \
  --steps 200
```

### Stage2 冒烟训练与测试

```bash
WANDB_MODE=disabled uv run python scripts/train_stage2.py \
  --num-envs 64 \
  --max-iterations 1 \
  --num-steps 4 \
  --epochs 1 \
  --minibatches 1 \
  --amp-minibatch-size 64 \
  --amp-disc-updates 1 \
  --save-interval 1 \
  --no-wandb \
  --log-root logs/smoke_gpu

uv run python scripts/play.py \
  --checkpoint logs/smoke_gpu/stage2/<run>/model_final.onnx \
  --device cuda \
  --steps 200
```

## 任务七：直接测试 MJCF 模型

不依赖训练 checkpoint，只检查机器人模型能否加载和仿真：

```bash
uv run python scripts/view_model.py \
  --model robot/MJCFmodelposition.xml \
  --headless \
  --steps 200
```

打开 MuJoCo viewer：

```bash
uv run python scripts/view_model.py \
  --model robot/MJCFmodelposition.xml \
  --steps 2000
```

## 训练和测试的通用参数

常用 Stage1 参数示例：

```bash
uv run python scripts/train_stage1.py \
  --num-envs 1024 \
  --target-velocity 1.0 \
  --motion-reset-prob 0.5 \
  --action-residual-scale 0.5 \
  --distal-action-scale 1.0 \
  --reference-ctrl-blend 1.0 \
  --disc-learning-rate 3e-5 \
  --amp-grad-penalty-coef 2.0 \
  --amp-label-smoothing 0.20 \
  --amp-obs-noise-std 0.05 \
  --amp-reward-coef 1.2 \
  --task-reward-lerp 0.50 \
  --max-iterations 30000 \
  --save-interval 500
```

训练默认启用 Warp CUDA Graph；调试时可使用 `--no-warp-cuda-graph`、`--solver-iterations`、`--solver-ls-iterations`、`--warp-nconmax`、`--warp-njmax` 和 `--iteration-sleep-ms`。CPU 后端仅用于加载和调试，不作为训练方案。

训练默认开启 episode 级随机 seed；需要完全复现实验时增加 `--no-episode-random-seed`。

## 日志、权重和录像

每个训练 run 目录通常包含：

```text
logs/<stage>/<run>/
├── params/config.json
├── model_<iter>.pt
├── model_<iter>.onnx
└── model_final.pt / model_final.onnx
```

新 checkpoint 的 `.pt` 和 `.onnx` 成功写入并通过 ONNX 校验后，旧的 `model_*` 权重会自动清理。`--viewer` 播放时自动录制 MP4，按任务保存到 `recording/stage1/`、`recording/stage2/`、`recording/multimotion/` 或对应策略目录；重复测试不会覆盖已有录像。

## 代码结构

```text
scripts/train_stage1.py       # Stage1 基础行走
scripts/train_stage2.py       # Stage2 领域随机化
scripts/train_multimotion.py  # 多动作/全地形课程
scripts/train_multimotion_walk.py # 多动作平地 warmup
scripts/play.py               # 普通、Stage1、Stage2、多动作 checkpoint 播放
scripts/view_model.py         # 直接加载 MJCF 模型
sensor_training/              # 仅 IMU+肌腱长度策略
squat_training/               # 蹲起传感器策略
src/lytrobot_amp/env.py       # MuJoCo 向量环境
src/lytrobot_amp/env_warp.py  # MuJoCo Warp GPU 环境
src/lytrobot_amp/networks.py  # Actor-Critic 和 AMP discriminator
src/lytrobot_amp/ppo_amp.py   # rollout、GAE、PPO/BPO 和 AMP 更新
```

## 历史训练恢复

恢复 `logs/multimotion/08-06-10:56:09` 的历史四动作训练时，从恢复目录执行：

```bash
.venv/bin/python recovery/run.py train --dry-run
.venv/bin/python recovery/run.py train

.venv/bin/python recovery/run.py play \
  --checkpoint logs/multimotion/08-06-10:56:09/model_final.onnx \
  --steps 1000 \
  --num-robots 3 \
  --seed 42
```

恢复训练输出隔离到 `logs/multimotion_recovered_20260806/`，不会覆盖原始模型。同期测试：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  recovery/test_recovery.py \
  tests/test_multimotion_log_paths.py \
  tests/test_multimotion_stairs.py \
  tests/test_serial_terrain_course.py \
  tests/test_play_recording.py \
  tests/test_model_export.py
```

## 常规代码检查

```bash
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
```

旧版 28 维力位联控 checkpoint 不兼容当前 14 维纯位置策略；续训和播放请使用 14 维 position checkpoint。
