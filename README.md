# LYTrobot AMP 双足行走训练

本项目参考 `AMP_mjlab` 的训练组织方式，但针对当前机器人模型重新实现了一个轻量的 MuJoCo + PyTorch BPO-AMP 训练栈。训练模型固定使用：

- 机器人 MJCF：`robot/MJCFmodelposition.xml`
- 人体参考步态：`robot/reference_data/walk1.npz`或`robot/reference_data/walk1_mjcf.npz`
- 目标任务：用 AMP 风格奖励和前向速度跟踪奖励训练机器人向前行走

所有训练统一使用 14 个肌腱位置执行器，策略动作固定为 14 维位置目标修正量。

## 代码结构

```text
scripts/train.py              # 训练入口，默认打开 wandb
scripts/train_stage1.py       # 第一阶段训练入口，GPU/Warp，关闭领域随机化
scripts/train_stage2.py       # 第二阶段训练入口，GPU/Warp，开启领域随机化
scripts/view_model.py         # 机器人 MJCF 本地 MuJoCo 可视化/加载测试
scripts/play.py               # checkpoint headless/可视化播放
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

训练必须使用 GPU/Warp：`--device cuda --sim-backend warp`。stage1 默认 1024 环境、30000 iteration；stage2 默认 4096 环境、7500 iteration，两者都是 7.3728 亿个 policy sample，避免 stage2 因环境数扩大而误跑 4 倍样本。每个 iteration 写本地日志，W&B 默认每 10 轮上报一次。第一阶段不启用领域随机化，权重写入 `logs/stage1/<run>/`。两个阶段训练默认开启 episode 级随机 seed；需要完全复现实验时可加 `--no-episode-random-seed`。

训练默认启用 Warp CUDA Graph，求解器使用 `iterations=10`、`ls_iterations=20`，每个 world 预留 32 个 contact 和 64 个 constraint。可分别用 `--no-warp-cuda-graph`、`--solver-iterations`、`--solver-ls-iterations`、`--warp-nconmax`、`--warp-njmax` 做回退或容量调试。默认不主动 sleep；与桌面共用 GPU 时可显式增加 `--iteration-sleep-ms 10`。

机器人 STL 只用于显示和惯量，碰撞仅保留双脚掌和脚趾的 4 个简化 box。旧版 MuJoCo-Warp 3.8.0.2/Warp 1.12.0 存在长期训练主内存增长问题；当前版本默认关闭周期性 Python GC，但仍保留 32GB RSS 安全停止保护，必要时可通过 `--memory-cleanup-interval` 恢复定期回收。

```bash
cd /home/yeah/code/LYTrobot_amp
uv run python scripts/train_stage1.py

only IMU+
cd /home/yeah/code/LYTrobot_amp
uv run python sensor_training/train_stage1.py

蹲起
uv run python squat_training/train_stage1.py
```

第二阶段是独立训练入口，在第一阶段基础配置上启用领域随机化，默认同样开启 wandb，权重写入 `logs/stage2/<run>/`：

```bash
cd /home/yeah/code/LYTrobot_amp
uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/stage1/06-26-08:03:21/model_final.pt

only IMU+
cd /home/yeah/code/LYTrobot_amp
uv run python sensor_training/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/<运行目录>/model_final.pt

蹲起
  uv run python squat_training/train_stage2.py \
  --stage1-checkpoint logs/squat_sensor_policy/stage1/07-17-08:46:52/model_final.pt
```

`--stage1-checkpoint` 用来指定接着哪个第一阶段权重模型训练，例如 `model_1000.pt` 或 `model_final.pt`。它会加载模型和优化器状态，但把第二阶段的 iteration 与 `global_step` 从 0 重新计数，因此第一阶段已经达到 30000 iteration 时仍可直接开始第二阶段。也可以使用等价别名 `--stage1-model`。如果不传这个参数，第二阶段会从头训练；如果需要恢复第二阶段自身的中断训练，也可以继续使用底层参数 `--resume-checkpoint logs/stage2/<run>/model_<iter>.pt`，此时会保留第二阶段已有的训练进度。

也可以显式调用统一入口：

```bash
uv run python scripts/train.py \
  --stage stage2 \
  --domain-randomization \
  --device cuda \
  --sim-backend warp
```

第二阶段默认随机化包括：负重质量、各 body 质量、肌腱长度比例、地面摩擦系数、重力加速度 `9.75~9.85m/s^2`、随机方向/大小推力、观测噪声、动作噪声、`2~5ms` action delay。范围均可用命令行覆盖，例如：

```bash
uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/stage1/<stage1_run>/model_final.pt \
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
uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/stage1/<stage1_run>/model_final.pt \
  --stand-start-prob 0.5 \
  --stand-start-phase-delay-steps 25 \
  --stand-start-transition-min-steps 80 \
  --stand-start-transition-max-steps 160 \
  --stand-start-push-delay-steps 100
```

测试样机起步效果时建议强制从静止起步开始：

```bash
uv run python scripts/play.py \
  --checkpoint logs/stage2/06-30-09:08:47/model_14750.pt\
  --device cuda \
  --viewer \
  --steps 0 \
  --stand-start-prob 1.0
```


指定模型可视化测试命令：
stage1
```bash
cd /home/yeah/code/LYTrobot_amp
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage1/07-08-09:14:39/model_30000.pt \
 --device cuda \
  --viewer \
  --steps 0

uv run python scripts/play.py \
  --checkpoint logs/sensor_policy/stage1<run>/model_final.onnx \
  --viewer \
  --steps 0

蹲起
uv run python squat_training/play.py \
  --checkpoint logs/squat_sensor_policy/stage2/07-18-09:36:29/model_final.pt\
  --viewer \
  --steps 0
```
06-18-10:01:10/model_final.pt stage1最好

stage2
```bash
cd /home/yeah/code/LYTrobot_amp
uv run python scripts/play.py \
  --checkpoint logs/stage2/06-30-09:08:47/model_15000.pt \
  --device cuda \
  --viewer \
  --steps 0

uv run python scripts/play.py \
  --checkpoint logs/squat_sensor_policy/stage1/07-17-08:46:52/model_final.pt \
  --viewer \
  --steps 0


```

使用 `--viewer` 打开 MuJoCo 测试时会自动录制 MP4，按阶段保存到 `recording/stage1/` 或 `recording/stage2/`。命名来自权重所在 run 和 iteration，例如：

```text
recording/stage1/06-06-21-19-41-2049.mp4
```

默认录制为 1080p、30fps、H.264、`yuv420p`、`faststart` MP4，兼容 PPT 和微信播放。MuJoCo 窗口和录屏镜头都会始终跟随机器人 `pelvis` 移动。重复测试同一个权重时会自动追加 `-2`、`-3`，避免覆盖已有视频。

Headless 加载测试命令：

```bash
cd /home/yeah/code/LYTrobot_amp
uv run python scripts/view_model.py \
  --model robot/MJCFmodelposition.xml \
  --headless \
  --steps 200
```


首次在线记录前需要：

```bash
wandb login
```

常用参数：

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
- `model_500.pt`
- `model_<iter>.pt`
- `model_final.pt`

## 从已有 checkpoint 稳定化续训

如果策略已经能走但脚踝/双脚高频抖动，优先从最新稳定行走模型续训，而不是从头训练。第一阶段不要立刻缩小控制幅度，只恢复 actor/discriminator 和 `global_step`，重置 Adam 优化器状态，用新的足踝平滑奖励和有限探索标准差先把动作抖动压下来：

```bash
cd /home/yeah/code/LYTrobot_amp
uv run python scripts/train_stage1.py \
  --resume-checkpoint logs/06-06-21:19:41/model_2049.pt \
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
uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/stage1/06-07-stabilize-ankle-stage1/model_<best>.pt \
  --resume-reset-optimizer \
  --run-name 06-07-stabilize-ankle-stage2 \
  --learning-rate 7e-5 \
  --disc-learning-rate 2e-5 \
  --action-residual-scale 0.45 \
  --distal-action-scale 0.9 \
  --max-iterations 3500 \
  --save-interval 25
```

`--max-iterations` 是最终 iteration 编号上限，不是额外训练轮数；从 `model_2049.pt` 恢复且设置 `2600` 时，会训练 `2050..2599`。重点观察 wandb 中的 `task_action_rate`、`task_distal_action_rate`、`task_stance_foot`、`task_stance_foot_speed`、`task_missed_contact`、`task_distal_vel`、`mean_length` 和 `mean_amp_reward`。

针对当前支撑脚前后蹭动，先从与第二阶段兼容的 14 维 position stage1 策略低学习率续训约 4000 iteration：

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

选择 `task_stance_foot_speed`、`task_missed_contact` 已下降且步态未退化的 checkpoint，再进入第二阶段；不要直接用旧 stage2 权重消除已经形成的腾空蹭步：

```bash
uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/stage1/06-30-no-slip-stage1/model_<best>.pt \
  --resume-reset-optimizer \
  --run-name 06-30-no-slip-stage2 \
  --learning-rate 3e-5 \
  --disc-learning-rate 1e-5 \
  --max-iterations 10000 \
  --save-interval 100
```

## GPU 冒烟测试

冒烟测试也跑 GPU/Warp，并且必须关闭 wandb：

```bash
cd /home/yeah/code/LYTrobot_amp
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
```

第二阶段随机化冒烟：

```bash
cd /home/yeah/code/LYTrobot_amp
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
```

CPU 后端仅保留为模型加载/调试路径，不作为训练方案。

## 模型文件可视化测试

不依赖训练 checkpoint，直接加载 `robot/MJCFmodelposition.xml` 做 MuJoCo 模型测试。

Headless 加载和仿真测试：

```bash
cd /home/yeah/code/LYTrobot_amp
uv run python scripts/view_model.py \
  --model robot/MJCFmodelposition.xml \
  --headless \
  --steps 200
```

本地打开 MuJoCo viewer：

```bash
cd /home/yeah/code/LYTrobot_amp
uv run python scripts/view_model.py \
  --model robot/MJCFmodelposition.xml \
  --steps 2000
```

默认会使用 `robot/reference_data/walk1.npz` 第一帧初始化模型姿态，并把 14 个肌腱位置执行器设置到当前肌腱长度附近，便于本地 viewer 中稳定查看模型。

## 播放 checkpoint

每次保存都会同时生成同名的 PyTorch 完整训练 checkpoint（`.pt`）和 ONNX
确定性 Actor（`.onnx`）。新模型成功写入并通过 ONNX 校验后，当前 run 目录中的旧
`model_*.pt/.onnx` 会自动删除，因此始终只保留最新的一对模型。`.pt` 可用于续训和
MuJoCo 播放，`.onnx` 内嵌环境/网络配置元数据，可直接用于 MuJoCo 播放。

Headless 检查：

```bash
python scripts/play.py \
  --checkpoint logs/smoke_gpu/stage1/<run>/model_final.pt \
  --steps 200 \
  --device cuda
```

有图形环境时可打开 MuJoCo viewer：

```bash
python scripts/play.py \
  --checkpoint logs/stage2/<run>/model_final.pt \
  --device cuda \
  --viewer
```

ONNX 模型使用同一个播放入口（传感器策略仍使用 `sensor_training/play.py`）：

```bash
python scripts/play.py \
  --checkpoint logs/stage2/<run>/model_final.onnx \
  --viewer
```

这条命令也会自动生成 `recording/stage2/<run>-<iteration>.mp4`。测试第一阶段 checkpoint 时会保存到 `recording/stage1/`。

## 已验证

本机已完成以下检查：

```text
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python robot/Launch.py --headless --steps 20
stage1 GPU/Warp 冒烟训练通过，保存 model_1.pt/model_final.pt
stage2 GPU/Warp 领域随机化冒烟训练通过，保存 model_1.pt/model_final.pt
旧 position stage1 checkpoint 使用新足端奖励续训加载通过
stage1/stage2 model_final.pt 播放加载通过
```

已生成的冒烟目录：

```text
logs/smoke_gpu/stage1/stage1-base-smoke/
logs/smoke_gpu/stage2/stage2-domain-rand-smoke_1/
```
