# CMU 69_01 行走重定向数据

`walk_cmu_69_01_omniretarget.npz` 使用 CMU Graphics Lab Motion Capture Database
Subject 69 / Trial 01 的 `walk forward` 动作。原始 ASF/AMC 保存在
`squat_training/retargeting/source_data/`。

## 选择与处理

- 官方原始频率为 120 Hz，选择第 263–407 帧的稳定直线行走完整周期。
- 每 3 帧采样为 40 Hz，共 49 帧。
- 人体骨盆高度按机器人腿长缩放，比例为 `0.9121`。
- 脚趾高度和速度共同判断支撑相位，并保留 2 帧自然双支撑；独立周期中右脚支撑 24 帧、
  左脚支撑 26 帧。
- 每个支撑段拆分为 heel / flat / toe 三阶段；OmniRetarget 使用脚跟、脚掌、脚趾分阶段
  Z 约束，允许足部绕接触点滚动，不再整脚刚性锁定。
- 主模型接触 IK 使用真实 STL 脚跟底点和脚趾底点；每次 heel strike 前的 5 帧中，
  脚趾网格均离地至少 10 mm；heel 阶段脚趾平均离地约 7.4 mm，toe 阶段脚跟平均
  抬升约 9.0 mm。支撑段采用 -0.6 mm 网格补偿以抵消脚跟标记点与 STL 最低点偏差；
  48/49 帧存在严格网格接触，剩余过渡帧的支撑脚间隙仅 0.022 mm。
- 14 个主动关节映射到完整机器人；两侧被动髌骨滑块保留主模型默认值。
- 根节点 `x/y` 固定，源动作前向速度保存在 `qvel[:, 0]`，均值约 `0.863 m/s`。
- 最终 `qpos=(49, 23)`、`qvel=(49, 22)`，首尾姿态、速度和接触标签闭合。
- OmniRetarget 最终代价为 `0.2166`。
- 最终 NPZ 的 SHA-256 为
  `354be99dece4fb770fc50419039bd5655aed797e8463807fcf360fb58d78350c`。

原始文件校验值：

- `69.asf`: `c3019031ae5789b1d43d3bba424229ac544ccc68bb8c06e7d67bf75b1c41605f`
- `69_01.amc`: `db7976c321c0a637049ca28e2f663caead4ab23e11930837d986150b4c2b6d75`

## 复现

```bash
cd squat_training/retargeting
.venv/bin/python prepare_cmu_walk.py \
  --output prepared/cmu_69_01_walk_cycle.npz
.venv/bin/python run_omniretarget_walk.py \
  --input prepared/cmu_69_01_walk_cycle.npz \
  --output results/cmu_69_01_walk_lyt_raw.npz
.venv/bin/python build_walk_reference.py \
  --input results/cmu_69_01_walk_lyt_raw.npz \
  --output ../../robot/reference_data/walk_cmu_69_01_omniretarget.npz \
  --ground-clearance -0.0006
```

可视化完整机器人映射结果：

```bash
uv run python squat_training/play_reference.py \
  --motion-file robot/reference_data/walk_cmu_69_01_omniretarget.npz
```

数据来源：<https://mocap.cs.cmu.edu/search.php?subjectnumber=69>。
