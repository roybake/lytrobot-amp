# walk1 OmniRetarget 动作数据

`walk1_omniretarget.npz` 是从原始 LAFAN `walk1_subject1.bvh` 人体骨架提取单个闭合步态周期后，
使用 OmniRetarget/Holosoma interaction-mesh SQP 重定向得到的训练参考动作。
旧版从 `walk1_mjcf.npz` 反算机器人关键点的二次映射方式已弃用，原输入文件未改动。

## 数据约定

- 原始周期使用 LAFAN 第 210–254 帧（含端点）、30 Hz，重采样为 40 Hz。
- 最终为 60 帧，`qpos=(60, 23)`，`qvel=(60, 22)`。
- 该区间是直线行走片段中首尾人体局部姿态误差较小的完整左右步态周期。
- 根节点 `x/y` 固定为零；源动作前向速度写入 `qvel[:, 0]`，与现有行走训练约定一致。
- `foot_contact[:, 0]` 为右脚、`foot_contact[:, 1]` 为左脚；保留源动作中 outgoing toe 与
  incoming heel 同时接触的自然双支撑，40 Hz 结果中共有 5 个双支撑帧。
- `foot_roll_mode` 使用 `0=摆动、1=脚跟、2=全脚掌、3=脚趾`，`foot_roll_phase` 保存每侧
  连续支撑段的 `[0, 1]` 相位。主模型接触 IK 从真实 STL 自动提取脚跟底点和脚趾底点。
- heel 阶段脚跟先接触且脚趾保持间隙，flat 阶段脚掌和脚趾共同承重，toe 阶段脚跟平均抬升
  约 9.4 mm 后由脚趾完成蹬离；60/60 帧均有 MuJoCo 地面接触。
- 14 个重定向关节映射到主模型对应关节；`patella_r_j1`、`patella_l_j1` 是主模型被动滑块，保留主模型默认值。
- 周期边界先使用 6 帧闭合，再对完整周期做 17 帧循环 Savitzky–Golay 平滑，最后执行
  平滑根高度校正和局部接触 IK；首尾 `qpos/qvel` 完全一致。

关节链映射为：

| 输入人体链 | 重定向机器人链 |
| --- | --- |
| Hips | pelvis |
| RightUpLeg / RightLeg | right_hip / right_knee |
| RightFoot / RightToeBase | right_foot / right_toe |
| LeftUpLeg / LeftLeg | left_hip / left_knee |
| LeftFoot / LeftToeBase | left_foot / left_toe |

## 复现

在仓库根目录执行：

```bash
squat_training/retargeting/.venv/bin/python squat_training/retargeting/prepare_walk1.py \
  --bvh /home/yeah/code/unitreeG1/GMR-master/data/lafan1/walk1_subject1.bvh \
  --output squat_training/retargeting/prepared/walk1_cycle.npz

cd squat_training/retargeting
.venv/bin/python run_omniretarget_walk.py \
  --input prepared/walk1_cycle.npz \
  --output results/walk1_cycle_lyt_raw.npz \
  --step-size 0.32 --foot-tolerance 0.03

.venv/bin/python build_walk_reference.py \
  --input results/walk1_cycle_lyt_raw.npz \
  --output ../../robot/reference_data/walk1_omniretarget.npz \
  --temporal-weight 0.06
```

重定向使用固定的 Holosoma 提交 `79d728c2c40e7f9ce8c28f9d9cab117ed2350bb5`；
完整元数据也保存在 NPZ 的 `metadata` 字段中。

当前 NPZ 的 SHA-256 为
`14abd6ec89c5b75f380dbf3e97b8b3e4ddad414fd89a126cad341cb4bd2e44c8`。
