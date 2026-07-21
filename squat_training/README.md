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

Actor 不接收关节角、关节速度或速度指令。每帧姿态反馈只有 7 个三轴加速度计、7 个三轴陀螺仪和 14 个肌腱长度，共 `42 + 14 = 56` 维；当前帧加前 5 帧历史组成 `56 × 6 = 336` 维。控制器再附加软件内部深蹲时钟的 `sin(phase), cos(phase)` 两维，因此 Actor 总输入为 `338` 维。相位不是实机姿态传感器，部署时由控制周期和参考轨迹发生器生成。AMP 判别器继续使用原训练代码的特权运动特征。

蹲起环境使用独立奖励 profile，不复用行走任务的摆动脚和交替步态奖励。任务奖励由参考姿态/骨盆高度/运动速度跟踪、双脚固定与持续接触、左右同步性和动作平滑性组成；失去足底接触会受罚，跌倒额外扣除 10 分，保证即使其余跟踪项满分时终止步奖励仍为负。训练日志中的 `task_squat_track` 和 `task_support_track` 分别表示蹲起轨迹跟踪质量与双支撑质量。

第一阶段关闭域随机化：

```bash
uv run python squat_training/train_stage1.py
```

第二阶段从第一阶段权重开始，并启用质量、摩擦、肌腱长度、重力、传感噪声、动作延迟和外力随机化：

```bash
uv run python squat_training/train_stage2.py \
  --stage1-checkpoint logs/squat_sensor_policy/stage1/<run-name>/model_final.pt
```

短程验证和回放：

```bash
uv run python squat_training/train_stage1.py \
  --num-envs 32 --num-steps 4 --max-iterations 1 --no-wandb --run-name smoke
uv run python squat_training/play.py --checkpoint <checkpoint.pt>
```

不经过策略网络，将 14 自由度 OmniRetarget 结果映射到完整机器人 MJCF 后直接可视化：

```bash
uv run python squat_training/play_reference.py
```

所有训练日志写入 `logs/squat_sensor_policy/`，录像写入 `recording/squat_sensor_policy/`。
