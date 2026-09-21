# LYTrobot AMP 双足行走训练

本项目使用 MuJoCo + PyTorch BPO-AMP 训练双足机器人稳定、自然地向前行走。所有策略统一使用 14 个肌腱位置执行器，策略动作是 14 维位置目标修正量。

## 快速索引

| 任务 | 训练入口 | 测试入口 | 权重目录 |
| --- | --- | --- | --- |
| Stage1 基础行走 | `scripts/train_stage1.py` | `scripts/play.py` | `logs/stage1/<run>/` |
| Stage2 鲁棒行走 | `scripts/train_stage2.py` | `scripts/play.py` | `logs/stage2/<run>/` |
| IMU+肌腱+自身动作历史策略（不蒸馏） | `sensor_training/train_stage1.py` / `train_stage2.py` | `sensor_training/play.py` | `logs/sensor_policy/` |
| 传感器串行蒸馏（独立可选） | `sensor_training/train_distill_stage1.py` / `train_distill_stage2.py` | `sensor_training/play.py` | `logs/sensor_policy/distillation/<phase>/<run>/` |
| **足底力 Teacher→Student 蒸馏（新）** | `distillation_training/train_teacher.py`；`train_pipeline.py` | `distillation_training/play.py` | `logs/distillation/` |
| **足底力 62D sensor-only（新）** | `foot_sensor_training/train_stage1.py` | `foot_sensor_training/play.py` | `logs/foot_sensor/` |
| 一键串行蒸馏 | `sensor_training/train_serial_distillation.py` | `sensor_training/play.py` | `logs/serial_distillation/<run>/pipeline.json` |
| 蹲起传感器策略 | `squat_training/train_stage1.py` / `train_stage2.py` | `squat_training/play.py` | `logs/squat_sensor_policy/` |
| 多动作/全地形行走 | `scripts/train_multimotion.py` | `scripts/play.py` | `logs/multimotion/` |
| 固定上楼专项 | `scripts/train_stair_up.py` | `scripts/play.py` | `logs/stair_up/<run>/` |
| 固定下楼专项 | `scripts/train_stair_down.py` | `scripts/play.py` | `logs/stair_down/<run>/` |
| 连续上下楼专项 | `scripts/train_stair_up_down.py` | `scripts/play.py` | `logs/stair_up_down/<run>/` |
| GPU/Warp 冒烟验证 | `scripts/train_stage1.py` / `train_stage2.py` | `scripts/play.py` | `logs/smoke_gpu/` |
| 仅测试 MJCF 模型 | 无 | `scripts/view_model.py` | 无 |

以下命令都从仓库根目录执行：

```bash
cd /home/yeah/code/LYTrobot_amp_recovered_20260806
```

## 足底力训练：两个独立项目

这两个目录与旧 `sensor_training/` 相互独立，旧 checkpoint 不会被读取或覆盖。虚拟足底传感器不改 XML/MJCF，输出顺序固定为：左脚 `heel, forefoot_medial, forefoot_lateral`，右脚同序。

### A. 自动化 Teacher → Student 蒸馏

先训练和评估 teacher；确认其 checkpoint 已稳定后，一键依次启动 BC、DAgger、Student RL，并在运行目录写入 `pipeline.json`：

```bash
uv run python distillation_training/train_teacher.py --device cuda --sim-backend warp

uv run python distillation_training/train_pipeline.py \
  --teacher-checkpoint logs/distillation/teacher/<run>/teacher.pt \
  --run-dir logs/distillation/<run> \
  --device cuda --sim-backend warp \
  --max-validation-action-mse <chosen_threshold> \
  --max-validation-kl <chosen_threshold>
```

只做可观测性诊断（BC）可使用：

```bash
uv run python distillation_training/train_pipeline.py \
  --teacher-checkpoint <teacher.pt> \
  --run-dir logs/distillation/<run> \
  --device cuda --sim-backend warp \
  --max-validation-normalized-action-mse <chosen_threshold> \
  --skip-dagger --skip-rl
```

完整说明见 [distillation_training/README.md](distillation_training/README.md)。

### B. 无蒸馏的 62D 足底力 ablation

```bash
uv run python foot_sensor_training/train_stage1.py --device cuda --sim-backend warp
uv run python foot_sensor_training/play.py --checkpoint <checkpoint.pt>
```

完整说明见 [foot_sensor_training/README.md](foot_sensor_training/README.md)。

## 环境安装

推荐使用项目自己的 `uv` 环境。仓库内的 `.venv` 必须是本项目创建的目录，不能软链接到其他项目：

```bash
uv sync
```

项目运行所需的机器人模型、地形、参考动作和重定向源码均在本仓库内。复现 LAFAN walk1 时，BVH 文件及解析器位于 `squat_training/retargeting/`，无需安装或读取 GMR 项目；其第三方许可证见 `lafan_vendor/license.txt`。首次安装仍需从 Python 包索引获取 `uv.lock` 锁定的第三方软件包。

训练必须使用 GPU/Warp：`--device cuda --sim-backend warp`。当前主要依赖为 `mujoco==3.9.0`、`mujoco-warp==3.9.0.1`、`warp-lang==1.14.0`、`torch==2.10.0`（CUDA 12.8）和 `wandb==0.27.2`。首次在线记录前执行：

```bash
wandb login
```

机器人模型为 `robot/MJCFmodelposition.xml`；本轮正式 Teacher、Student 与 AMP 训练统一使用 `robot/reference_data/walk1_mjcf.npz`。

控制回路默认 100 Hz（5 ms 物理步长、`--decimation 2`）。传感器策略的所有训练入口
强制使用该频率，传入其他 decimation 会在启动前报错；回放程序仍会优先采用 checkpoint
中保存的训练配置，以兼容历史模型。

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

## 任务三：固定楼梯上楼/下楼专项

项目内置的 CMU subject 114 trial 07 人体数据经 OmniRetarget 重定向后，保存为
`robot/reference_data/multimotion/stair_up_omniretarget.npz` 和
`stair_down_omniretarget.npz`。另有 `stair_up_down_omniretarget.npz` 将五级上楼、
顶部平台和五级下楼组成一个不循环、无拼接跳变的连续 episode。

专项课程固定使用上下连通的 5 级楼梯：每级高度 `0.13 m`，前进方向踏步宽度为 `0.32 m`，
横向净宽为 `2.0 m`。两段楼梯在最高 `0.65 m` 处由 `0.64 m` 的短过渡平面连接。每次 reset 从人体片段第 0 帧开始，
关闭动作起点、状态噪声和动作噪声；策略只学习固定参考动作周围的有界残差。非循环
参考到达末帧后必须按末帧动作稳定保持 `30` 个控制步，再检查根部 X 位移（容差 `0.05 m`），未达到固定距离不会计为成功；
根部横向偏离楼梯中心线超过 `0.30 m` 会立即失败，不能绕开台阶走平地。
当前 CMU 片段的固定根部位移为：上楼 `1.8209 m`、下楼 `2.0503 m`；训练和回放均以
各自片段的末帧位移作为目标。

```bash
# 上楼（独立训练）
uv run python scripts/train_stair_up.py \
  --device cuda --sim-backend warp --num-envs 4096 --max-iterations 30000

# 下楼（独立训练）
uv run python scripts/train_stair_down.py \
  --device cuda --sim-backend warp --num-envs 4096 --max-iterations 30000

# 上楼 -> 顶部短平台 -> 下楼（连续训练）
uv run python scripts/train_stair_up_down.py \
  --device cuda --sim-backend warp --num-envs 4096 --max-iterations 30000
```

两个 checkpoint 均可直接用 `scripts/play.py` 回放；回放会读取 checkpoint 中保存的
楼梯几何和动作配置。参考数据可按 `multimotion/README.md` 的 OmniRetarget 流程从
项目内置人体数据重新生成。

## 任务四：IMU+肌腱长度+自身动作历史策略

该任务部署时 Actor **只**使用 IMU、肌腱长度、最近6次自身控制动作和 GRU 隐状态；不读取关节真值、
接触真值、参考相位或参考控制。训练期间 Critic 可读取少量特权上下文以降低价值估计
方差，但该信息不会进入导出的 ONNX Actor。

普通传感器训练不使用 Teacher，奖励与已验收的特权 Stage1 步态一致：跟踪参考动作、
前向速度与自然步态，跌倒直接获得大额负奖励。AMP 判别器继续提供自然步态约束，
Critic 在训练期间读取根速度、足接触、关节速度和地形上下文；这些仿真真值不会进入部署 Actor。
参考动作使用 `walk1_mjcf.npz`（已验收步态视频所用的 clip）。执行器基准使用与 Actor
gait command 同源的板载参考肌腱长度（`reference_ctrl_blend=1.0`、`action_residual_scale=0.5`），
Actor 只输出反馈残差；该参考表由控制器本地查表产生，不引入仿真真值或额外传感器。

### 特权 Teacher → 传感器 Student 串行蒸馏

蒸馏与普通传感器行走训练完全分开：普通 `sensor_training/train_stage1.py` 和
`train_stage2.py` 不加载 Teacher，也不计算或记录任何蒸馏数据；下面的专用入口才启用
Teacher 标签与蒸馏损失。普通入口不会接受 `--teacher-*` 参数。

先准备两位使用**相同直接控制接口**的特权 Teacher。所有 Teacher 和 Student 都必须
显式使用 `--reference-ctrl-blend 0.0`；同时保持 `--decimation`、
`--action-residual-scale` 和 `--distal-action-scale` 一致。训练入口会校验这些设置，
防止把动作语义不一致的 Teacher 蒸馏给 Student。

```bash
# Teacher 1：自然基础步态
uv run python scripts/train_stage1.py \
  --reference-ctrl-blend 0.0 \
  --max-iterations 30000 \
  --log-root logs/teachers --log-group stage1_gait

# Teacher 2：从 Teacher 1 继续，训练抗扰恢复
uv run python scripts/train_stage2.py \
  --stage1-checkpoint logs/teachers/stage1_gait/<run>/model_final.pt \
  --reference-ctrl-blend 0.0 \
  --max-iterations 30000 \
  --log-root logs/teachers --log-group stage2_robust
```

再训练同一个传感器 Student。第一阶段学习自然步态，第二阶段加载第一阶段 Student
并在域随机化下学习恢复；两阶段默认均训练 30000 轮。

```bash
# Student 1：只看 IMU + 肌腱长度，模仿自然步态 Teacher
uv run python sensor_training/train_distill_stage1.py \
  --teacher-checkpoint logs/teachers/stage1_gait/<run>/model_final.pt

# Student 2：从 Student 1 接续，模仿鲁棒 Teacher
uv run python sensor_training/train_distill_stage2.py \
  --student-checkpoint logs/sensor_policy/distillation/stage1_gait/<run>/model_final.pt \
  --teacher-checkpoint logs/teachers/stage2_robust/<run>/model_final.pt
```

Teacher 只为 Student **自己执行、自己访问到的 on-policy 状态**生成动作标签，Student
仍执行自己的采样动作，因此不会破坏 PPO 的动作似然。Teacher 动作 Huber 损失会逐渐
减弱：Student 1 为 `1.0 → 0.1`，Student 2 为 `0.5 → 0.1`。`metrics.csv` 会记录
`teacher_action_loss`、`teacher_action_mae` 和 `teacher_action_loss_coef`；每个 Student
checkpoint 也会记录所用 Teacher 的绝对路径与迭代信息。

日志目录按角色和阶段隔离，便于筛选：

```text
logs/
├── teachers/
│   ├── stage1_gait/<run>/
│   └── stage2_robust/<run>/
└── sensor_policy/
    └── distillation/
        ├── stage1_gait/<run>/
        └── stage2_robust/<run>/
```

更详细的约束与命令见 [sensor_training/DISTILLATION.md](sensor_training/DISTILLATION.md)。

### 一键完成四阶段蒸馏

不想逐个启动 Teacher 和 Student 时，使用总控入口。它会自动按
`Teacher Stage1 → Teacher Stage2 → Student Stage1 → Student Stage2` 顺序执行，
统一使用 100 Hz（`--decimation 2`）、`reference_ctrl_blend=0.0` 和每阶段 30000 轮。

```bash
uv run python sensor_training/train_serial_distillation.py \
  --run-name imu_walk_100hz_v1
```

中断后以**同一个** `--run-name` 再次运行即可从最近阶段 checkpoint 继续；流水线状态、
命令、阶段目录和错误原因保存在：

```text
logs/serial_distillation/imu_walk_100hz_v1/pipeline.json
```

首次确认时可只打印完整命令而不创建日志或启动 GPU 训练：

```bash
uv run python sensor_training/train_serial_distillation.py \
  --run-name imu_walk_100hz_v1 \
  --dry-run
```

可用 `--num-envs`、`--student-num-envs`、`--student-num-steps`、
`--stage1-iterations`、`--stage2-iterations` 覆盖资源和迭代数；Student 强制100 Hz
控制并使用48步 BPTT，同时自动调整并行环境数以保持原24步 rollout
的总样本量；
离线训练时增加 `--no-wandb`。

### 直接训练（不使用 Teacher）

以下是本方案的非蒸馏训练入口。Stage1/Stage2 都使用可部署 gait command，从站立双支撑
相位缓慢加速到目标速度；Stage2 再加入领域随机化。新观测为696维，旧420/336维检查点
不能恢复，须从头训练。

### Stage1 训练与测试

```bash
cd /home/yeah/code/LYTrobot_amp_recovered_20260806
uv run python sensor_training/train_stage1.py 


uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage1/gait-feedforward-v2/model_10500.onnx \
  --device cpu \
  --viewer \
  --steps 0
```

传感器训练默认编译 recurrent BPTT 策略更新以降低每轮耗时；首轮编译较慢，后续轮次复用优化内核。可用 `--no-compile-policy-update` 回退到 eager 路径进行调试。

### Stage2 训练与测试

```bash
uv run python sensor_training/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/<run>/model_final.pt

uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage2/<run>/model_final.onnx \
  --device cuda \
  --viewer \
  --steps 0
```

## 任务四：蹲起传感器策略

### Stage1 训练与测试

```bash
cd /path/to/LYTrobot_amp_recovered_20260806
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

新 checkpoint 的 `.pt` 和 `.onnx` 成功写入并通过 ONNX 校验后，旧的编号权重会自动清理；按最新 rollout 稳定率选出的 `model_best.pt/.onnx` 会保留，建议优先用于评测和阶段衔接。`--viewer` 播放时自动录制 MP4，按任务保存到 `recording/stage1/`、`recording/stage2/`、`recording/multimotion/` 或对应策略目录；重复测试不会覆盖已有录像。

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
