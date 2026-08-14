# 机器人蹲起专项训练

本目录是完全隔离的蹲起任务入口。它复用项目现有的 MuJoCo/Warp 动力学、BPO-AMP、奖励和两阶段训练流程，但不会覆盖行走任务的动作文件、入口或日志。

## 动作来源与重定向

动作选用 [CMU MoCap Subject 22 / Trial 14](https://mocap.cs.cmu.edu/search.php?subjectnumber=22&trinum=14) 的 `alternating squats`，截取 Subject A 的第 340–620 帧作为完整站立→下蹲→站立周期。原始 ASF/AMC 保存在 `retargeting/source_data/`；CMU 数据的使用说明见 [CMU MoCap 首页](https://mocap.cs.cmu.edu/)，单位换算遵循其 [ASF/AMC 格式说明](https://mocap.cs.cmu.edu/info.php)。

下载文件校验值：`22.asf` 的 SHA-256 为 `13a2303cca57f85775efc113f91acd313115e4a7bf02102879a3f62006d0e79f`，`22_14.amc` 为 `3dcc742b315eed3b58e99c7f9bb4e8ee60a17db6f62740bea48ec2d74b2771ce`。

重定向使用 [OmniRetarget](https://omniretarget.github.io/) 发布的官方 [Holosoma 实现](https://github.com/amazon-far/holosoma)，固定提交 `79d728c2c40e7f9ce8c28f9d9cab117ed2350bb5`。`retargeting/omniretarget_project/` 是独立的官方源码工作树；`models/lyt/` 是与本项目 14 个下肢转动关节轴、关节限位和连杆尺寸一致的无肌腱运动学模型。

人体到机器人的关键点映射如下：

| 人体关键点 | 机器人连杆 |
| --- | --- |
| Hips | pelvis |
| RightUpLeg / LeftUpLeg | right_hip / left_hip |
| RightLeg / LeftLeg | right_knee / left_knee |
| RightFoot / LeftFoot | right_foot / left_foot |
| RightToeBase / LeftToeBase | right_toe / left_toe |

官方 interaction-mesh SQP 同时施加机器人关节限位、地面非穿透、双脚水平位置固定和双脚高度锁定，并用对称的可行蹲姿作为名义轨迹，避免优化进入单腿下蹲的局部极小值。重定向前以首尾站立姿态为基准，将原动作的下蹲深度缩放为 `0.5`，腿部屈曲与骨盆下降同步变浅。最终参考轨迹位于 `reference_data/squat_cmu_22_14_omniretarget.npz`：240 帧、40 Hz；下蹲 3 秒、起身 3 秒，完整周期精确为 6 秒，最低蹲姿位于第 120 帧（3 秒）。骨盆高度范围为 `0.723–0.874 m`，下降 `0.151 m`（原动作重定向结果下降 `0.310 m`），并带有显式的双脚全程接触标签。该文件的 SHA-256 为 `3f76df5dfd61ce0bcbc7ba7000b8e5531debdd38ef1d706ba9afd5ff2e910383`。

该浅蹲文件已原位替换旧参考数据，`train.py`、`train_stage1.py`、`train_stage2.py` 和 `play.py` 的默认动作路径均指向这一新文件。旧 checkpoint 学到的是原深蹲目标，浅蹲训练应从 Stage 1 重新开始，不要直接用旧 Stage 1 checkpoint 启动 Stage 2。

复现重定向：

```bash
cd squat_training/retargeting
uv sync
uv run python prepare_cmu_squat.py \
  --asf source_data/22.asf --amc source_data/22_14.amc \
  --output prepared/cmu_22_14_squat_cycle.npz
uv run python run_omniretarget.py \
  --input prepared/cmu_22_14_squat_cycle.npz \
  --output results/cmu_22_14_squat_lyt_raw.npz \
  --squat-depth-scale 0.5
cd ../..
uv run python squat_training/retargeting/build_training_reference.py \
  --input squat_training/retargeting/results/cmu_22_14_squat_lyt_raw.npz \
  --output squat_training/reference_data/squat_cmu_22_14_omniretarget.npz
```

## 策略输入与训练

Actor 不接收关节角、关节速度、速度指令或仿真根部真值。每帧姿态反馈只有 7 个三轴加速度计、7 个三轴陀螺仪和 14 个肌腱长度，共 `42 + 14 = 56` 维；当前帧加前 5 帧历史组成 `56 × 6 = 336` 维。其后附加控制器当前相位的 `sin/cos` 两维，总观测为 `338` 维；GRU 每步读取最新 56 维传感器和当前 2 维相位，共 58 维。AMP 判别器继续使用训练期特权运动特征。

相位不是盲目固定时钟。训练环境以 14 个实际肌腱长度在“站立→最低点”的参考肌腱方向上投影，得到物理下蹲深度，并用骨盆 IMU 重力方向检查躯干直立性：

- 实际动作落后参考深度超过 `0.08` 时逐步减慢相位；
- 未达到 `0.85` 深度时不允许从下蹲切换到起立；
- 未恢复到 `0.15` 深度以内时不允许开始下一次下蹲；
- IMU 直立度低于 `0.75` 时冻结相位。

为了防止策略通过冻结参考相位延长回合并反复领取静态跟踪奖励，推进倍率低于
`0.25` 的控制步会被记为停滞步，且该步任务奖励保证为负。每个物理蹲起周期最多
允许累计 40 个停滞步（40 Hz 下为 1 秒）；超过预算会以失败终止。完成一次由传感器
确认的“站立→深度至少 0.85→恢复到 0.15 以内”周期只记录完成事件，默认不会立即
结束 episode。策略必须在整个回合内持续站稳并完成至少一次蹲起，普通 timeout 才算
成功；完成蹲起后又摔倒仍然是失败。
推进倍率在 `(0.25, 1)` 内时还会按 `2 × (1-scale)`
连续扣除减速代价；到回合上限仍未完成周期会作为异常终止处理，而不是可 bootstrap
的正常 timeout。

策略观测、奖励参考和执行目标共用这一反馈校正后的相位。实机控制器必须采用相同的肌腱长度归一化和 IMU 姿态估计，并在稳定站立启动时同时复位相位与 GRU 状态。训练日志额外记录 `task_phase_feedback_scale`、`task_phase_feedback_frozen` 和 `task_sensor_depth_progress`。
checkpoint 和 ONNX 元数据中的 `sensor_observation.phase_feedback` 会保存控制周期、参考频率、站立肌腱长度、归一化半宽、深度投影方向、参考深度曲线和全部门控阈值，实机端应直接读取这些值，避免手工复制参数造成训练/部署不一致。

默认入口会启用 `--squat-phase-feedback`。阈值可用 `--squat-phase-lag-tolerance`、`--squat-phase-min-scale`、`--squat-bottom-depth-threshold`、`--squat-stand-depth-threshold`、`--squat-upright-threshold`、`--squat-phase-stall-threshold`、`--squat-phase-max-stall-steps`、`--squat-phase-stall-penalty` 和 `--squat-phase-slowdown-penalty-coef` 调整。由于观测从 336 维变为 338 维，旧无相位 checkpoint 不能恢复，必须从 Stage 1 重新训练。

蹲起环境使用独立奖励 profile，不复用行走任务的摆动脚和交替步态奖励。任务奖励由参考姿态/骨盆高度/运动速度跟踪、双脚固定与持续接触、左右同步性和动作平滑性组成。默认完整周期奖励为 5 分，跌倒或失败终止额外扣除 50 分；训练从稳定站立开始，躯干直立终止阈值提高到 `0.65`，保证生存优先于蹲起次数。训练日志中的 `task_squat_track` 和 `task_support_track` 分别表示蹲起轨迹跟踪质量与双支撑质量。

第一阶段关闭域随机化：

```bash
uv run python squat_training/train_stage1.py
```

两个阶段都使用蹲起专用的保守更新边界：初始/最大学习率 `1e-4`、KL 目标 `0.01`、
KL 硬停止 `0.015`、BPO ratio 范围 `0.10`，每轮只更新 2 个 epoch。命令行显式传参
仍可覆盖这些默认值。

第二阶段从第一阶段权重开始，并启用质量、摩擦、肌腱长度、重力、传感噪声、动作延迟和外力随机化：

```bash
uv run python squat_training/train_stage2.py \
  --stage1-checkpoint logs/squat_sensor_policy/stage1/<run-name>/model_final.pt
```

Stage2 默认重置优化器并使用 `1e-4` 学习率。域随机化每 250 轮评估一次：成功率和下蹲深度达标且异常终止率足够低时强度增加 `0.025`，性能明显下降时回退，不再按训练轮数强行增加。外力最终范围仍为 0–80 N，但间隔放宽为 75–200 个控制步；所有默认回合都从稳定站立状态开始，随机化动力学和扰动而不随机化部署初态。

Warp 的质量、惯量、地面摩擦和重力属于整个批次共享参数。Stage 2 现在强制启用这些
共享参数的域随机化，并默认每 250 iteration 重采样；逐环境的肌腱长度、动作延迟，
以及逐步的观测/动作噪声和外力也同时参与随机化。Stage 1 仍强制使用标称参数。
训练与测试每次启动以及每个 episode/reset 默认重采样 seed；仅在复现实验时同时传入
`--seed N --no-episode-random-seed`。

训练时重点观察 `mean_reward`、`episode/reward_per_step`、`bottom_depth_success_rate`、`episode_success_rate`、`rollout_episode_success_rate`、`task_phase_stalled`、`task_phase_stall_timeout`、`task_incomplete_squat_cycle`、`task_squat_cycle_completed`、`termination_fraction` 和 `domain_randomization_scale`。`mean_reward` 与 `episode/reward_per_step` 都是按控制步归一化的主奖励曲线，已消除回合长度对累计奖励的直接放大；原累计值只保留在 `mean_episode_reward` 和 `episode/total_reward`。`episode_success_rate` 是最近已完成回合中真正完成蹲起周期的滑动成功率，`rollout_episode_success_rate` 是当前 rollout 的即时值。

短程验证和回放：

```bash
uv run python squat_training/train_stage1.py \
  --num-envs 32 --num-steps 4 --max-iterations 1 --no-wandb --run-name smoke
uv run python squat_training/play.py \
  --checkpoint <checkpoint.pt> --viewer --steps 0
```

上面的正常测试默认使用随机 seed，并让 Stage 2 checkpoint 保持域随机化。固定种子、
标称参数的复现诊断可额外添加
`--no-domain-randomization --seed 42 --no-episode-random-seed`。

`play.py` 默认从稳定站立进入动作。只有专门评估任意动作相位恢复能力时，才显式传入
`--stand-start-prob 0.0`；这不是正常蹲起的部署条件。默认 viewer 回放关闭普通 episode
超时，并反复执行站立→下蹲→起立，直到关闭 viewer 或达到显式的 `--steps`。旧的
“单周期成功即结束”checkpoint 会在每次站起后从稳定站立开始下一轮；使用当前训练配置
生成的生存型 checkpoint 则保持同一个物理状态连续循环。终端汇总中的 `squat_cycles`
是实际完成的蹲起次数，`terminations` 是真实失稳次数。可显式传入
`--no-squat-terminate-on-cycle-completion` 对旧模型执行无物理重置的极限测试，但旧模型
没有接受过这种训练，不能将该结果当作正常回放能力。

不经过策略网络，将 14 自由度 OmniRetarget 结果映射到完整机器人 MJCF 后直接可视化：

```bash
uv run python squat_training/play_reference.py
```

所有训练日志写入 `logs/squat_sensor_policy/`，录像按 checkpoint 阶段写入
`logs/squat_sensor_policy/stage1/recordings/` 或
`logs/squat_sensor_policy/stage2/recordings/`。
W&B 的 `episode` 分组中会记录 `episode/total_reward`（已完成回合的平均累计总回报）
和 `episode/length`（已完成回合的平均长度），两条曲线统一使用 `global_step` 作为横轴。
