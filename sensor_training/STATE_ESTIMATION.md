# 最终纯传感器人体步态工作流

主要入口是 `train_final_sensor_gait.py`，它使用已有串行蒸馏流程、环境和训练器：

```
Privileged Teacher Stage1
  → Sensor Student Stage1（Teacher + PPO + state supervision）
  → Teacher takeover = 0 的纯 Student 验收
  → Sensor Student Stage2（渐进物理/测量 DR）
  → 纯 Student nominal + DR 验收
  → deployment.onnx
```

不要求 A–G 实验。`train_ablation.py` 仅作为历史工具保留，不是推荐入口。
默认完整流程自动训练 Teacher；也可以提供一个通过严格验收的现有 Teacher。
不会把“短时间不倒”当成人体轨迹复现已经完成。

## 硬件与固定部署结构

本次已确认实机具备全部 7 组 IMU：pelvis、左右大腿、小腿和脚。不增加传感器，
不依赖电机 Iq、足底力、外部里程计或地形标签。

| 组成 | 单帧维度 |
|---|---:|
| 7 组三轴 accelerometer + gyro | 42 |
| 当前肌腱测量长度 | 14 |
| 测量长度的因果差分/低通速度 | 14 |
| 测量长度减当前板载参考长度 | 14 |
| previous action | 14 |
| 板载 gait command | 46 |
| 总计 | **144** |

6 帧短显式历史：**864D observation**，跨度 0.05 秒。GRU hidden：**256D**。
BPTT 默认 **96**（100 Hz 下 0.96 秒），也支持 128；BPTT 不改变单步部署接口。

```
864D measured-sensor history
  → GRUCell(864, 256)
  → 5 个独立状态 heads（各自 256→64→output）
  → [q14, qdot14, root_velocity3, contact2, feet12]，共 45D
  → Actor([hidden256, estimates45])：301→256→128→14
  → bounded tendon residual
```

Contact head 输出 logits，进入 Actor 前取 sigmoid；连续估计使用 normalized
physical representation。Actor loss 默认可反向更新估计器，没有 `.detach()`。
Critic 可读取真值 context；真实关节、root、接触、足端和地形信息均不进入 Actor。

## 统一测量：避免理想肌腱信号绕过实机噪声

`measurement.py` 提供 CPU/torch 共用的 `CausalMeasurement`。Warp 和 mjlab 均使用
同一个 sensor wrapper。每个控制 tick 只采样一次，然后缓存；读取不推进滤波器、
延迟队列或 RNG。所有肌腱通道来自这个缓存：

```
l_true → encoder gain/bias → additive noise → quantization/clipping
       → causal delay buffer → l_measured

length          = l_measured
velocity        = lowpass((l_measured[t] - l_measured[t-1]) / dt)
tracking_error  = l_measured - l_reference_current
```

不再出现“length 有噪声，velocity/error 用裸真值”的旁路。旧的 `obs_noise_std`
也在测量层施加到长度，再生成派生特征，不能绕过测量层。

支持固定 bias、reset bias、encoder scale error、量化、0/1/2 tick latency、clipping。
Scale error 围绕已知 encoder/control 中心作用；bias 单独建模。IMU 也支持噪声、
reset bias 和因果延迟。CPU/torch 使用相同公式、单位、reset、quantization 和 delay
语义；两种 RNG 不要求相同 seed 产生逐位相同的随机数。

Reset 重新填满选定环境的延迟缓存，用新测量重置速度滤波器；第一帧速度为零。
Timeout bootstrap 在 autoreset 前采样并复用同一 tick，保持继续运行的环境采样一致。

最终 profile 的传感器设置在 Stage1/Stage2 完全相同，DR 强度改变采样幅度，因此
Stage2 可以严格加载 Stage1 checkpoint：

| 测量参数 | 固定/基础值 | DR 满强度增加/范围 |
|---|---:|---:|
| tendon noise | 0.1 mm | 再增加 0.5 mm |
| tendon reset bias | 0 | ±2 mm |
| tendon gain error | 0 | ±1% |
| tendon quantization | 0.05 mm | 同上 |
| tendon/IMU delay | 0 tick | 0–2 tick |
| normalized IMU noise | 0.001 | 再增加 0.01 |
| normalized IMU bias | 0 | ±0.02 |

肌腱速度低通 alpha=0.25，归一化为 1 m/s；tracking error 除以 0.05 m。
IMU acceleration 除以 9.81，gyro 除以 10 rad/s。值按 metadata 中 clip 配置裁剪。
测量范围是原控制范围加 0.02 m margin。这些是初始工程设置，尚未用实机测量数据校准。

**实机部署不再人为注入仿真 noise/bias/delay。** 直接使用真实 encoder/IMU 输出，
用 `CausalTendonFeatures` 计算派生量，采用真实控制 tick 的 dt 和相同的归一化、
clipping、history 顺序。参考表由 `DeployableGaitCommandGenerator` 和板载时钟生成。
当前参考 normalized tendon target 用 `ctrl_mid/ctrl_half_range` 转回米，再算 error。
参考表的未来 0.1/0.2 秒采样是已知软件命令，不是未来真实传感器。

## 状态监督与 loss

训练标签只有：q14、qdot14、root velocity3、right/left contact2、四个 pelvis-local
foot contact geom × xyz12。关节顺序来自 `common_joint_names`；contact 顺序为 right/left。
Root velocity 是 world-frame 标签，由网络估计，不从真值直接输入 Actor。

| 量 | 归一化 scale | loss 权重 |
|---|---:|---:|
| joint position | 1 rad | 1.0 |
| joint velocity | 10 rad/s | 0.5 |
| root linear velocity | 3 m/s | 0.5 |
| contact logits / 0–1 target | BCEWithLogits | 0.5 |
| pelvis-local foot position | 1 m | 0.5 |

```
L_state = 1.0 SmoothL1(q_hat, q/1)
        + 0.5 SmoothL1(qdot_hat, qdot/10)
        + 0.5 SmoothL1(v_hat, v/3)
        + 0.5 BCEWithLogits(contact_logits, contact_truth)
        + 0.5 SmoothL1(feet_hat, feet/1)

L_total = existing PPO/value/entropy/context loss
        + lambda_action SmoothL1(student_mean, remapped_teacher_action)
        + L_state
```

Teacher 动作 remapping、startup ramp 和 episode takeover 机制保留。最终配方使用
未经额外 action scale 缩放的 SmoothL1：Stage1 lambda_action 从 5→1，takeover 从
0.8→0；Stage2 lambda_action 从 2→0.5，takeover 从 0.2→0。正常预算下 takeover
分别在前 4000 / 2000 轮降到零。缩短预算时也会缩短 schedule；质量门槛保持不变。

AMP 保留为 style prior，最终配方 coefficient=0.3、task_reward_lerp=0.9；主要
监督仍是 Teacher action、逐相位 reference reward 和 state labels。最终参考奖励的
q/qdot/root velocity/foot coefficients 为4.0/1.5/1.5/2.5（兼容入口旧默认不变）。原本的 AMP
discriminator 不变。Future prediction 默认关闭，代码保留但不进入最终配置。

每个 minibatch 的 `evaluate_sequence_shared` **只执行一次 GRU trajectory**，并
复用 Actor means、state predictions、context predictions；PPO、Teacher action loss
和 state loss 共享梯度图。编译 chunk 间 hidden 连接保留，reset mask 切断上一回合。
部署 inference 结果未改变。

保存并打印独立的 state MAE/contact accuracy；CSV/W&B 保留 loss 和 Teacher 指标。
`state_supervision_metadata` 与当前 target layout 同步：可选配置 33D/45D，最终固定45D。

## Teacher 质量与最终验收

每个保存周期记录 phase-aligned q/qdot/foot/root RMSE、左右 contact mismatch、
step frequency、stride length、左右 stance ratio、knee ROM/reference、pelvis height、
torso orientation、fall rate、episode completion 和 forward velocity。

参考 RMSE 使用起步后 2 秒以后的采样；不足 2 秒的测试使用所有可用采样。Stride
length 是同侧连续 touchdown 的 world-x 位移；频率是左右总 touchdown/s。Contact
error 是逐 tick 的标签不一致率。模型没有独立 torso body，torso orientation 使用
pelvis/root 姿态相对参考的 quaternion angle。

最终检查为 **30 秒、seeds 42/123/2026、全部种子通过**；full-reference 门槛是：

| 最差 seed 指标 | 最大值 |
|---|---:|
| 全 14 关节位置 RMSE | 0.20 rad |
| 关节速度 RMSE | 2.0 rad/s |
| 足端 local RMSE | 0.06 m |
| root velocity RMSE | 0.15 m/s |
| contact mismatch | 25% |
| pelvis height RMSE | 0.05 m |
| torso orientation RMSE | 0.15 rad |

还必须通过原有连续不倒、速度和 knee ROM/lock 检查。最终配方的最佳 Teacher
按 `reference_fidelity_accepted/score` 排序，Student handoff 使用 **model_final.pt**，
防止使用 takeover 尚未归零的早期 best checkpoint。最终配方不在首次过短 gait gate
后停训。未通过 gate 的 Teacher 会阻止 Student 开始，错误信息明确指向 Teacher。

Student 的 ONNX evaluation 不创建 takeover trainer，也不执行 Teacher 动作：
`eval/teacher_execution_probability=0`。Teacher 在单独 rollout 评估，仅生成对比指标。
Stage2 另外评估满强度 DR 下独立 Student 的连续行走。Teacher/Student 使用相同
reference/control 与物理扰动参数，输出 `teacher/*`、`student/*`、`student_teacher_gap/*`。

验收结果保存在 pipeline manifest 和 `final_gait_evaluation.json`。30 秒、3 个种子
仍不能证明实机安全或长期可靠性；实际部署需逐步验证时间同步、校准和 100 Hz 周期。

## 推荐命令

### Stage1：训练 Teacher，再训练 Student

```bash
.venv/bin/python sensor_training/train_final_sensor_gait.py --phase stage1
```

默认各 30000 轮。已有合格 Teacher 时可以用：

```bash
.venv/bin/python sensor_training/train_final_sensor_gait.py --phase stage1 \
  --teacher-checkpoint logs/teachers/stage1_gait/<teacher-run>/model_best.pt
```

Teacher checkpoint 旁必须有对应 ONNX。默认 `--run-name` 自动生成时间戳；用同一个
`--run-name MM-DD-HH:MM:SS` 恢复已有最终流程。不要复用旧 workflow 的 manifest。

### Stage2：从已验收的最终 Stage1 Student 接续

```bash
.venv/bin/python sensor_training/train_final_sensor_gait.py --phase stage2 \
  --teacher-checkpoint logs/teachers/stage1_gait/<teacher-run>/model_best.pt \
  --student-checkpoint logs/sensor_policy/distillation/stage1_gait/<student-run>/model_final.pt
```

Stage2 沿用已验收的 privileged Stage1 Teacher 作为当前 Student 状态上的 DAgger
标签源。它增加线性 curriculum（前半段预算从0到1）：mass ±15%、payload 0–2 kg、
friction 0.65–1.4、gravity 9.6–10.05、tendon calibration ±2%、actuator PD strength
±10%、action delay 0–20 ms，以及上述 IMU/encoder uncertainties 和 mild pushes。

Warp/mjlab 的 mass/friction/gravity/actuator strength 属于批次共享模型，每250轮
完整重置并重采样；sensor bias/delay 属于每环境。CPU 在逐环境 stepping 前应用
该环境的模型参数。Strength 同时缩放 actuator gain/bias，保持 PD 控制语义。

一次运行全部阶段：

```bash
.venv/bin/python sensor_training/train_final_sensor_gait.py
```

`train_serial_distillation.py --final-sensor-gait` 是等价入口。
加 `--dry-run` 查看完整配置而不训练、不创建目录。

### ONNX 导出 / 播放

通过完整 pipeline 后自动生成 `deployment.onnx` 和 `.deployment.json`。
手动重导出：

```bash
.venv/bin/python sensor_training/export_sensor_policy.py \
  --checkpoint logs/sensor_policy/distillation/stage2_robust/<run>/model_final.pt \
  --output deployment/final_sensor_gait.onnx

.venv/bin/python sensor_training/play.py \
  --checkpoint deployment/final_sensor_gait.onnx --viewer --steps 3000 --seed 42
```

手动导出检查结构/零 takeover；性能验收由 pipeline 完成，不会将“能够导出”等同于
“步态通过验收”。外部输入固定为 **observation、recurrent_state、episode_start**；
输出固定为 **action、next_recurrent_state**。Critic、真值、Teacher 和 terrain 不在
部署图中；导出不需要启动仿真或读取 Teacher 文件。

## 兼容性与后续 terrain fine-tuning

Legacy 默认与模型结构解释保留。新增 measurement 配置进入 checkpoint；带派生
肌腱特征的旧 checkpoint 缺少 `shared_cached_measurement_v1` 会明确拒绝 resume，
因为旧通道使用过理想 truth，不能静默当成新测量模型。全新功能关闭的旧 layout
仍兼容；维度、history、hidden、state layout/scales 不兼容时拒绝加载。

Metadata 保存 IMU layout、tendon measurement/feature normalization、frame order、
history、gait command/control/reference tables、GRU/state config、state scales/loss
weights、Teacher metadata、control dt 和 residual semantics。

最终 Actor 可以从 Stage2 checkpoint 继续 fine-tune rough/slope；保持 sensor/layout/
reference/control 不变，更新 terrain task 或 reference command 表时需要显式验证。
Actor 不增加 terrain label，context 只进入 Critic。新的地形训练仍需匹配的 Teacher
和任务验收；本次没有训练或声称完成地形适应。

## 单步规模与风险

固定最终网络的 deployable 部分约 **1,058,875 参数**，FP32 权重约4.04 MiB；
每步矩阵计算 **1,056,576 MAC**（约2.11 MFLOP），100 Hz 约0.106 GMAC/s。
不包含 Critic/discriminator、模拟器、encoder/IMU读数和控制器开销。实际 Orin
batch=1 延迟依赖 runtime、kernel launch、线程/功耗/频率与系统负载，尚未测量；
这些数字是按网络矩阵形状计算的规模估计，不是硬件时延承诺。

目前未发现 Actor/ONNX 的 simulator-truth 输入旁路。剩余部署风险是实际传感器与
actuator 校准、时间戳对齐、measurement DR 范围未实测、reference clock/ramp 实现
以及 Orin latency。以上均需要实机测量；本次没有新增硬件依赖。

当前已有 `20260930_sensor_walk_v3` Teacher 的30秒复测未过严格门槛（最短23.05秒），
因此推荐从不带旧 Teacher 的 Stage1 命令开始。新代码测试不能替代新策略训练结果。

## 本次修改与测试结果

修改文件：

- 配置/测量：`config.py`、`experiment_options.py`、`envs.py`、新增 `measurement.py`。
- 网络/训练：`networks.py`、`trainer.py`、`distillation_trainer.py`。
- 最终流程：`train_serial_distillation.py`、`train_stage2.py`、新增 `final_profile.py`、
  `train_final_sensor_gait.py`、`export_sensor_policy.py`。
- 奖励/DR：`../scripts/train.py`、`../src/lytrobot_amp/env.py`、
  `../src/lytrobot_amp/env_warp.py`。
- 评估：`gait_evaluation.py`。
- 测试：新增 `tests/test_final_sensor_gait.py`，更新 `tests/test_stage1_reward_profile.py`。
- 文档：`README.md`、`DISTILLATION.md`、本文件、`../README.md`。

验证命令：

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q
```

最终全库结果：**164 passed，15 subtests passed，3 failed**。本次最终方案及相关
Sensor/Warp/mjlab/蒸馏/ONNX测试均通过。3个失败分别为：

- `recovery/test_recovery.py::test_recorded_launcher_and_all_effective_arguments`
- `recovery/test_recovery.py::test_playback_uses_historical_assets_without_mutating_metadata`
- `squat_training/tests/test_squat_task.py::SquatTaskTests::test_actor_receives_sensor_history_and_internal_phase_clock`

三项均在未修改的 main（c55dc5f）隔离快照中复现；涉及历史参数/缺失地形资产，以及
旧 squat 观测期望。未扩大本次平地人体步态任务去修改这些独立流程。

测试覆盖：统一 noisy length/velocity/error、causal delay/quantization/reset、CPU/torch
一致语义、33/45D metadata、Actor不读真值、end-to-end estimator梯度、一次共享GRU
forward、跨编译chunk梯度/reset、BPTT96真实训练、CPU/Warp/mjlab Stage2加载/DR更新、
Teacher takeover=0、action remapping、拒绝旧架构、reference RMSE与Teacher质量门槛。

另外运行了真实 CUDA/Warp CLI 的一次96步更新，确认状态指标打印、PT/ONNX保存。
重新导出的完整864D模型只有三个外部输入，多步/部分reset情况下 ONNX 与 PyTorch
最大数值差为 **8.20e-8**。`git diff --check` 通过。

该 smoke 模型仅用于实现验证，不是训练完成的策略；未进行新的30000轮正式训练，
也未进行 Orin 上的时延或实机控制测试。
