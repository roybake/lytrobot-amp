# IMU + 肌腱长度独立训练

## 纯传感器上下楼梯 DeepMimic

`sensor_training/train_stairs_deepmimic.py` 使用 GRU Actor。楼梯专项每帧输入为机身一个 IMU 的三轴加速度与三轴角速度、14 条肌腱实际长度、上一时刻 14 维动作，以及原始 DeepMimic 定义的一个归一化相位标量；6 帧历史共 210 维。相位由控制器单调时钟产生，不是仿真真值；参考动作、目标速度、世界坐标和脚端接触均不进入 Actor。仿真全状态只参与训练奖励或 critic。默认训练匹配 `robot/reference_data/multimotion/stairs.npz` 与 `robot/terrains/stairs.xml`：三阶、0.13 m 立板、0.32 m 踏步，连续完成上楼—平台—下楼。

训练使用纯 DeepMimic 模仿目标：五组姿态、速度、key point、根部姿态和根部速度指数项，PPO clipped objective、GAE、Reference State Initialization（RSI）和跌倒提前终止；关闭 AMP、LCP、根部世界坐标 XY 惩罚以及自然步态/接触/抬脚塑形奖励。训练回合从完整参考中均匀采样相位，避免固定从第 0 帧起步导致策略永远看不到后半程；播放和验收仍固定从第 0 帧开始。逐帧参考只用于 DeepMimic 奖励和控制器前馈肌腱目标，不作为 Actor 的参考姿态输入。旧 204/696/708 维 checkpoint 不能直接用于新的 210 维输入。

```bash
uv run python sensor_training/train_stairs_deepmimic.py --check-data
uv run python sensor_training/train_stairs_deepmimic.py \
  --max-iterations 30000

uv run python sensor_training/play.py \
  --checkpoint logs/stairs/<时间戳>/model_best.onnx \
  --viewer
```

楼梯专项默认使用 4096 个并行环境，可通过 `--num-envs` 覆盖。`model_best.pt/.onnx` 和 `model_final.pt/.onnx` 保存在 `logs/stairs/<时间戳>/`。每次训练自动按现有训练器的 `MM-DD-HH:MM:SS` 格式命名；同秒启动时自动附加数字后缀，避免覆盖。

楼梯专项训练会在运行时把参考根节点 X 平移 `-0.39 m`，与 MuJoCo 中验证的机器人和 `robot/terrains/stairs.xml` 距离一致；原始 `stairs.npz` 的每帧关节姿态、速度和文件内容都不改。对齐后，377 个踏步着地帧无脚部外伸，脚网格最小台面余量约 3.1 cm，足部碰撞盒最大穿透约 1.0 cm。参考仍有最长 1.10 秒连续无接触，训练入口会警告这一点，不能保证严格物理重现。`--check-data` 可查看对齐审计；`--skip-reference-audit` 仅供代码冒烟测试。详细结果见 `multimotion/retargeting/diagnostics/stairs_offset_minus_039_audit.json`。

默认 `--direction exact` 使用上述精确场景数据。`--direction up`、`down` 或 `up_down` 仍保留旧的程序化五级楼梯课程。`scripts/train_stairs_deepmimic.py` 是此纯传感器入口的兼容别名。

动力学、奖励和 AMP 判别器复用现有实现；Actor 使用 recurrent BPO 更新，AMP 判别器和 AMP reward 保留。所有训练统一使用纯位置控制。

sensor stage1 默认使用独立 PPO+BPO+AMP 训练，不启用教师蒸馏。参考动作使用含完整脚跟—脚掌—脚趾滚动的 `walk1_omniretarget.npz`，目标速度为 `0.87 m/s`。Actor 获得可在实机控制器生成的 gait command，执行器同时使用该参考表作为板载前馈肌腱目标；Actor 只学习 IMU/肌腱反馈残差。任务奖励同时跟踪参考动作、前向速度和实测步态质量，AMP 作为辅助风格先验。训练回合中70%从站立开始，固定选择参考动作第15帧的双支撑相位：先站立0.2秒，再用0.8～1.0秒 smoothstep 渐入步态。其余30%从随机参考相位及其对应物理状态开始，只用于覆盖完整步态周期。Stage2 保留同一起步分布并加入领域随机化；实机和默认播放始终100%从站立开始。

Stage1/Stage2 的接触奖励直接读取地面接触点，记录每只脚的脚跟→脚掌→脚趾状态、左右触地顺序、步幅和支撑时间。先碰脚趾、长期仅脚趾支撑、双脚同时腾空或同时向前蹦都受到惩罚；左右步幅、接触占比、足部横向位置和前后摆动则用于镜像对称评分。Warp 接触缓冲只读取当步 `nacon` 个活跃接触，避免把旧接触当成着地；脚跟和脚趾同时受地面支撑时，将中间脚掌视为已加载，以补偿 Warp 对网格中央接触点的漏检。单动作 Stage1/Stage2 显式循环播放闭合行走参考，避免训练回合在加速结束前就被截断。上述奖励没有把参考相位或教师动作送进 Actor。

## 观测定义

普通行走任务的 Actor 每帧输入如下；楼梯专项只选用 `rootacc` 和 `rootgyro` 这一组机身 IMU，关闭 gait command，并额外输入一个 DeepMimic 归一化相位：

- 7个三轴加速度计：21维，除以 `9.81`
- 7个三轴陀螺仪：21维，除以 `10 rad/s`
- 14条受控肌腱实际长度：14维，按执行器控制范围中心和半宽归一化
- 控制器最近一次发出的14维动作，保持精确值，不叠加传感器噪声

此外，每帧包含由控制器单调时钟和本地参考表产生的46维 gait command：相位
`sin/cos`、启动进度、归一化目标速度，以及当前/未来0.1秒/未来0.2秒的3组14维
归一化肌腱参考。它不是传感器，不依赖仿真真值，也不需要增加实机硬件。

单帧116维。环境按“过去第5帧 → 当前帧”的顺序维护6帧、696维历史，因此Actor
同时获得最近6次动作，能够区分相同肌腱长度下不同的控制输入与动力学趋势。Actor
每步显式读取全部6帧，并由跨 rollout 持久化的128维 GRU 隐状态继续保存更长的因果
历史。在强制100 Hz控制下显式历史覆盖0.05秒；实机控制器
只需用同样顺序维护6帧环形
缓冲区。Actor 不包含关节角度、关节速度、根部真值、足端真值或地形标签；物理传感器
仍然只有 IMU 和肌腱长度。步态相位、渐入进度、速度目标和参考肌腱前视由板载软件明确
提供，训练与部署使用同一生成逻辑。

AMP 判别器沿用原有单帧45维特征，其中已经包含14维关节角度和14维关节速度；同样拼接连续6帧，共270维。专家历史按 `policy_dt` 从参考动作中连续采样并统一进行左右镜像。

episode 重置时，6个历史位置都填充为重置后的当前帧，同时清零对应环境的 GRU
隐状态，避免跨 episode 泄漏状态。

普通 Stage1/Stage2 策略保持14维肌腱位置目标修正量，由 GRU 后的基础 MLP 直接生成
`tanh(base)` 动作，不含 gate 或 expert 网络。只有 `train_multimotion.py` 的串联地形课程
启用4个 residual expert MLP；gate MLP 用 softmax 连续加权专家，最终动作经过
`tanh(base + residual_scale * residual)`，不存在 one-hot 地形动作硬切换。Critic 和
BPO median critic 使用 GRU 隐状态和仅训练期可见的25维特权上下文。GRU 同时预测根部
线速度、左右脚接触、14维关节速度和连续地形参数；加权 Smooth-L1 辅助损失提高根速度
与接触监督的权重，促使隐状态从 IMU/肌腱/动作历史形成
隐式地形估计。特权上下文不进入 Actor 或 ONNX。训练按环境序列做截断 BPTT；串联地形
MoE 额外使用 gate 变化、专家负载均衡和 residual L2 正则抑制路由抖动与专家塌缩。

单步 Stage1/Stage2 固定 `--reference-ctrl-blend 1`：执行器基准是与 Actor gait command
同源的板载参考肌腱长度，Actor 输出是叠加其上的反馈残差，不依赖额外传感器或仿真状态。
多地形入口仍可使用自己的控制语义；地形过渡由 GRU +
soft MoE 根据本体反馈产生；动作 rate 和 gate 变化正则负责抑制接缝抖动。论文依据、
上下文字段、限制和验证方法见
[`docs/proprioceptive_terrain_transition.md`](../docs/proprioceptive_terrain_transition.md)。

传感器 stage1 默认使用100 Hz（`decimation=2`）、64步 BPTT、3072个环境，关闭 domain randomization 并开启 wandb。PPO 预算与已验收的特权 Stage1 步态保持一致：自适应学习率 `3e-4`（下限 `1e-5`）、`clip_coef=0.2`、`target_kl=0.025`（`target_kl_stop_multiplier=2.0`）、`entropy_coef=0.002`、`min/max_action_std=0.05/0.6`、`task_reward_lerp=0.5`、`amp_reward_coef=1.2`、`fall_penalty=2.0`。参考动作使用 `walk1_mjcf.npz`（产生已验收步态视频的同一个 clip），目标速度 `1.0 m/s`。执行器基准是与 gait command 同源的板载参考肌腱长度（`--reference-ctrl-blend 1.0`），Actor 输出叠加其上的反馈残差（`--action-residual-scale 0.5`）；参考表来自板载查表，因此该控制语义可以直接部署。回合长 1200 步（12 秒），其中 50% 从站立平滑渐入步态、50% 从随机参考相位开始；速度不足终止检查在启动 4 秒后开始。

该配方与特权 Stage1 每次 PPO 更新模拟相同的物理步数（`3072 × 64 × decimation 2 = 4096 × 24 × decimation 4 = 393,216`），也就是每轮梯度更新对应的数据量与已验证曲线一致，收敛节奏可以直接对照：`task_forward_velocity` 约 1200 轮达到 `0.3 m/s`、约 4700 轮达到 `0.5 m/s`、约 20000 轮达到 `0.8 m/s`，回合长度同期从约 120 步升到 800 步以上。RTX 3060 上单轮约 6 秒，因此 1200 轮约 2 小时、30000 轮约 50 小时，`--save-interval 500` 便于中途取 checkpoint 做回放评测。前 400 轮只需确认回合长度、跌倒率与参考跟踪是否单调改善；用几百轮的外观判断步态质量会得出错误结论。训练日志重点查看 `task_forward_velocity`、`task_pose_track`、`task_foot_pos`、`task_termination` 和回合长度；仅回合长度上升而速度不升不能视为成功。

传感器 trainer 默认用 `torch.compile` 融合 no-grad 的 rollout Actor、AMP reward 和单步
recurrent BPTT 更新；这些图在每个环境步复用，能加速100 Hz采样和PPO反向传播。首次 policy
backward 编译在消费级 GPU 上可能需要数分钟，但只需一次，后续更新会显著更快。在
100 Hz下默认使用0.64秒梯度窗口；`3072 × 64 = 196,608` 个 transition，与特权 Stage1
每次更新模拟的 393,216 个物理步相同，同时减少串行仿真步数并提高GPU占用。所有传感器训练入口
都强制 `--decimation 2`；其他值会在启动前直接报错。GRU隐状态仍跨
rollout传递，只在边界处detach，所以0.64秒是梯度窗口而不是策略记忆上限。可用
`--no-compile-rollout` 或 `--no-compile-policy-update` 回退对应eager路径；默认每个编译图处理
16个recurrent step，避免逐步调用和编译完整序列。Teacher 蒸馏入口使用实测更快的8步块。

## 训练

本方案不使用蒸馏。从头训练独立传感器策略：

```bash
uv run python sensor_training/train_stage1.py

uv run python sensor_training/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/<run>/model_final.pt
```

当前控制语义与 `deployable-gait-command-v1` 的实测长度基准不同；即使输入同为696维，
该旧 checkpoint 也不能恢复。相同传感器布局的近期 checkpoint 可以只调整站立过渡时序后续训：

```bash
uv run python sensor_training/train_stage1.py \
  --resume-checkpoint logs/sensor_policy/stage1/gait-feedforward-v2/model_final.pt \
  --resume-reset-optimizer \
  --run-name gait-forward-recovery
```

用训练输出的 ONNX 在 CPU MuJoCo 做固定种子步态核验，报告写入 checkpoint 目录；`natural_gait_pass` 同时检查无重置、腾空、踮脚、足部滚动、镜像、步幅、落脚节奏、摆动脚离地、交替支撑、躯干姿态、骨盆高度和前向速度。门槛按 `walk1_omniretarget.npz` 的统计标定：

```bash
uv run python scripts/evaluate_sensor_gait.py \
  --checkpoint logs/sensor_policy/stage1/<run>/model_final.onnx \
  --steps 800 --seed 42
```

多地形策略可使用单独入口：

```bash
uv run python sensor_training/train_multimotion.py \
  --num-envs 1024 \
  --max-iterations 30000
```

多场景入口的平地与普通行走参考保留 CMU 69_01 经机器人尺度重定向后的原始速度
`0.8623 m/s`（`3.10 km/h`）。该训练目标
有意低于现有 `5 km/h` 验收门槛，当前阶段优先验证自然节奏下的稳定性；后续若恢复
高速指标，应在稳定模型上增加独立的速度课程。
默认串联路线仅覆盖平地、5 cm 不平路和 6°上/下坡；楼梯数据与地形实现保留为可选能力，当前默认训练不启用。无导航命令时左/右转在本体观测上不可辨识，因此
转向参考不混入这个直线地形策略的 AMP 教师分布。默认启用从 0 强度开始的自适应领域随机化，
并定期重采样共享质量、摩擦和重力参数。训练与验收均不计算或约束能耗。

训练完成后运行仿真预验收：

```bash
uv run python scripts/evaluate_acceptance.py \
  --checkpoint logs/sensor_policy/multimotion/<run>/model_final.pt \
  --device cuda
```

脚本只有在平地平均速度达到 `5 km/h`、锁膝采样比例不超过 `10%`、平均膝屈曲跟踪不低于 `90%`、六场景均无跌倒，并且楼梯高度/踏面四个边界组合下的完整串联路线都至少走完一次时才返回成功，并生成同目录
`acceptance_report.json`。该报告是仿真预验收；系统集成仍须实物测量。

只有同一套前馈控制语义产生的 Stage1 checkpoint 才可继续训练；不要恢复旧
`deployable-gait-command-v1` 或 `natural-gait-ft` 模型，也不要移除前进奖励：

```bash
uv run python sensor_training/train_stage1.py \
  --resume-checkpoint logs/sensor_policy/stage1/gait-feedforward-v2/model_<iteration>.pt \
  --run-name gait-feedforward-v2-resumed
```

快速冒烟训练：

```bash
uv run python sensor_training/train_stage1.py \
  --num-envs 32 --max-iterations 1 --num-steps 4 \
  --no-wandb --run-name smoke
```

普通传感器训练默认 checkpoint 写入 `logs/sensor_policy/`，楼梯专项写入 `logs/stairs/`。新训练只能
恢复同时带有相同 `sensor_observation`、`sensor_policy_architecture` 元数据和相同 MoE 开关的
6帧输入 context recurrent checkpoint；旧420维历史策略、旧单帧输入 recurrent MoE、旧336/337维前馈传感器模型、
完整状态策略和历史力位联控模型均不兼容，必须使用新的运行目录从头训练。

导出的 ONNX 使用显式状态接口：输入为696维6帧 `observation`、`recurrent_state` 和
`episode_start`，输出为 `action` 和 `next_recurrent_state`。项目播放入口会自动维护
传感器帧缓冲和隐状态，并在 episode 结束后重置二者。ONNX 的
`lytrobot_amp.metadata.sensor_observation.gait_command` 内嵌归一化参考表和全部启动参数；
实机侧可用 `DeployableGaitCommandGenerator.from_metadata(...)` 复现普通行走的46维命令；楼梯专项不输入该 gait command，也不需要里程计，但会把同一复位时钟归一化为 `[0,1]` 的 DeepMimic 相位标量，并按该时钟运行参考肌腱前馈。复位时清零
GRU状态和6帧缓冲，调用生成器 `reset()`；每执行一次100 Hz动作后调用 `advance()`。
实机将 ONNX 原始输出传给 `generator.tendon_control_target(action)`，得到包含参考前馈、
残差和最初0.5秒动作渐入的物理肌腱长度目标，不能再二次乘 `generator.action_scale`。
写入下一帧历史的上一动作是 `action * generator.action_scale`。默认部署过渡取训练范围中点
200步，即站立0.5秒后约2秒进入完整行走。

## 播放

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage1/<run>/model_final.pt \
  --viewer

uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/multimotion/<run>/model_final.pt \
  --stage multimotion \
  --viewer
```
