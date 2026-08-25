# 直线全地形 MoRE 多动作 AMP

本目录保存八动作/地形训练所需的可复现数据流水线。训练数据不是手工拼接的机器人姿态，而是公开 CMU MoCap 动作经项目内置的官方 OmniRetarget/Holosoma interaction-mesh SQP 重定向到 LYT 14-DoF 下肢模型后，再映射到主项目 14 个肌腱位置执行器。rough、pit、gap 数据以已重定向的走路动作作为初值，再用实际 MoRE 地形射线高度和完整脚部 STL 执行接触 IK。

## 动作来源

| 动作 | CMU 数据 | 选用帧（原始 120 Hz） | 训练帧率 | 是否循环 |
|---|---|---:|---:|---|
| 走路 | subject 69, trial 01 | 263–407 / stride 3 | 40 Hz | 是 |
| 左转 | subject 69, trial 06（walk and turn repeated） | 2260–2520 / stride 6 | 40 Hz（2×时间拉伸） | 否，89.7° / 4.3 s |
| 右转 | subject 69, trial 06（walk and turn repeated） | 3550–3790 / stride 6 | 40 Hz（2×时间拉伸） | 否，88.0° / 4.0 s |
| 崎岖地面走路 | 走路重定向数据×3 周期 | MoRE 128×128 高度图 | 40 Hz | 否 |
| pit 高台走路 | 走路重定向数据×3 周期 | 0.30 m 高台 | 40 Hz | 否 |
| gap 跨越 | 走路重定向数据×4 周期 | 0.20 m 训练间隙 | 40 Hz | 否 |
| 上楼 | subject 114, trial 07（walking up and down stairs） | 300–720 / stride 6 | 40 Hz | 否 |
| 下楼 | subject 114, trial 07 | 900–1370 / stride 6 | 40 Hz | 否 |

平地 walk 及其派生的 rough、pit、gap 参考保留 CMU 69_01 经 LYT 尺度映射后的
原始重定向速度 `0.8623 m/s`，不再把时间轴压缩到 `1.4 m/s`。转身与上下楼继续
使用各自片段的原始速度，当前训练先解决自然节奏下的稳定性。

原始 ASF/AMC 文件位于 `retargeting/source_data/`，训练文件位于 `robot/reference_data/multimotion/`。CMU 数据库允许研究和商业项目使用，但不能直接转售转换后的数据；使用时应按 CMU 页面要求致谢。来源页面：

- <https://mocap.cs.cmu.edu/search.php?subjectnumber=69>
- <https://mocap.cs.cmu.edu/search.php?subjectnumber=114>

OmniRetarget 源码快照来自 `amazon-far/holosoma` 提交 `79d728c2c40e7f9ce8c28f9d9cab117ed2350bb5`，许可证与 NOTICE 保存在 `squat_training/retargeting/omniretarget_project/`。

## 重定向复现

```bash
cd /home/yeah/code/LYTrobot_amp

# 1. ASF/AMC -> Z-up 人体关键点、接触标签和动作片段
uv run --project squat_training/retargeting \
  python multimotion/retargeting/prepare_clips.py

# 2. 对五个片段执行 OmniRetarget SQP
for clip in walk turn_left turn_right stair_up stair_down; do
  uv run --project squat_training/retargeting \
    python multimotion/retargeting/run_omniretarget.py \
      --input multimotion/retargeting/prepared/${clip}_human.npz \
      --output multimotion/retargeting/results/${clip}_lyt_raw.npz
done

# 3. 映射到主 MJCF，并在完整 STL 几何上执行脚底滚动/防穿插 IK
uv run --project squat_training/retargeting \
  python multimotion/retargeting/build_references.py

# 4. 生成 rough、pit、gap 地形接触 IK 数据
uv run --project squat_training/retargeting \
  python multimotion/retargeting/build_terrain_references.py

# 5. 自动校验：腿部无负几何距离、转身峰值 <= 50 deg/s、
#    支撑期满足脚跟 -> 全脚掌 -> 脚趾
uv run python multimotion/retargeting/validate_references.py
```

接触 IK 对踝、距下关节和脚趾使用单帧局部可信域与时间连续性约束，防止欠约束
求解跳到另一组接触姿态。独立楼梯训练所用参考会额外校验：单帧关节跳变不超过
`0.80 rad`、关节速度不超过 `18 rad/s`；超过任一限制时禁止进入楼梯训练。

每个最终 NPZ 都包含 `qpos`、`qvel`、`joint_names`、`frequency`、`foot_contact`、`foot_roll_phase`、`foot_roll_mode`、`task`、`loop` 和 JSON `metadata`；metadata 保留源帧、源网址、OmniRetarget 版本、最终代价及腿间距/脚底接触误差。

## 重定向效果可视化

参考数据仍可生成走路、转身及各地形动作预览；当前强化学习入口暂不采样左右转身：

```bash
MUJOCO_GL=egl uv run python scripts/view_multimotion_retarget.py
```

默认输出：

- `recording/reference/multimotion_more_terrain_test.mp4`
- `recording/reference/multimotion_more_terrain_test.png`
- `recording/reference/all_terrain_walking_test.mp4`（仅 gap/pit/rough/stairs，
  0.8× 速度完整行走）
- `recording/reference/all_terrain_walking_test.png`
- `recording/reference/multimotion_foot_contact_slow.mp4`（0.5×脚部慢放）
- `recording/reference/multimotion_heel_sole_toe.png`（三阶段接触对照）
- `recording/reference/multimotion_left_right_turns.png`（左右转起止姿态对照）
- `recording/reference/multimotion_validation.json`（逐动作数值校验）

视频使用训练时相同的 LYT 模型与地形参数：楼梯位于 `y=0 m`，平地位于 `y=3 m`，rough/pit/gap 依次位于 `y=6/9/12 m`。画面左上角标注动作与来源，MuJoCo 接触点用于检查脚底落点。交互播放或只查看指定动作：

```bash
# 打开 MuJoCo 窗口，同时保存视频
uv run python scripts/view_multimotion_retarget.py --viewer

# 仅交互播放上下楼，不录制
uv run python scripts/view_multimotion_retarget.py \
  --clips stair_up stair_down --viewer --no-record

# 生成脚跟 -> 脚掌 -> 脚趾慢放测试
MUJOCO_GL=egl uv run python scripts/view_multimotion_retarget.py \
  --speed 0.5 \
  --output recording/reference/multimotion_foot_contact_slow.mp4
```

## 当前多动作训练地形

源文件与许可证已复制到 `robot/terrains/more_discolrl/`，来源为本机
`/home/yeah/code/unitreeG1/MORE/MoRE-discoRL`。`src/lytrobot_amp/terrain.py`
地形实现支持 gap、pit、rough、6°上/下坡和 stairs；当前训练入口只把 rough、
6°上/下坡和 stairs 拼入 LYT 连续串联路线。gap、pit 数据与实现仍保留用于离线参考
验证，不参与当前多动作训练。
正式训练把 rough 最大高度差限制为 0.05 m，并使用 3 级、台阶高 0.13 m、
踏面 0.22 m 的训练楼梯；原始 CMU 楼梯参考数据仍来自 0.13 m 高、
0.58 m 深的重定向标定。环境会把参考骨盆升降幅度缩放到实际台阶总高度，并使用
相对参考高度的摔倒边界；踏面长度差异仍由策略通过观测和残差动作适应。

每条完整状态路线只包含 rough、6°上/下坡和 stairs，地形沿前进方向首尾拼接。由于所有
Warp world 共享同一个编译模型，启动时会根据 `--seed` 预生成 8 条随机排列
（可用 `--terrain-course-variants` 修改），每个 episode 随机选择其中一条。机器人
按走路及所选路线的地形动作连续推进；楼梯段自动展开为上楼和下楼，
只有完成整条路线才记作成功结束。每种排列都包含全部地形，不会退化成单地形通道。
为避免策略永远只看到入口动作，默认先用 5000 轮只训练从 stage 0 出发的完整路线，
再用 10000 轮把随机非零 stage reset 的概率线性提高到 80%；跨过 stage 边界时给予
稠密奖励。这样先建立稳定平地入口步态，再逐步覆盖所有困难地形；严格成功率仍只接受
从 stage 0 出发的完整路线。

## 训练

策略网络只允许输入 IMU 和肌腱长度时，使用传感器专用入口。Actor 输入为连续 6 帧
`42 维 IMU + 14 维肌腱长度`，共 336 维；不包含目标速度、动作 one-hot、参考相位、
关节真值、足端状态或地形标签：

```bash
uv run python scripts/train_multimotion_stage1.py \
  --num-envs 4096 --max-iterations 30000

uv run python scripts/train_multimotion_stage2.py \
  --stage1-checkpoint \
    logs/multimotion/stage1/<run>/model_final.pt \
  --num-envs 4096 --max-iterations 7500
```

对应 checkpoint 必须使用传感器播放入口：

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/multimotion/stage2/<run>/model_final.pt \
  --stage multimotion \
  --device cuda \
  --viewer \
  --steps 0
```

`scripts/train_multimotion.py` 现在也是传感器任务公共配置，但不负责选择阶段；正式训练
应使用带 `_stage1.py` / `_stage2.py` 后缀的入口，以保证域随机化状态正确。

第一阶段：

```bash
uv run python scripts/train_multimotion_walk_stage1.py \
  --num-envs 4096 \
  --max-iterations 5000

uv run python scripts/train_multimotion_stage1.py \
  --resume-checkpoint \
    logs/sensor_policy/multimotion_walk_warmup/stage1/<run>/model_final.pt \
  --resume-reset-progress \
  --resume-reset-optimizer \
  --num-envs 4096 \
  --max-iterations 30000
```

完整状态训练入口默认按 4096 个并行环境配置：`num_steps=24`，每轮共 98,304 条
transition；PPO 划分为 8 个 12,288 样本 minibatch，AMP 使用 4096 样本并更新 2 次。
8 条随机路线平均各分配 512 个环境。随机阶段课程前 5000 轮保持为 0，随后 10000 轮
线性升至 0.8；训练日志的 `startp` / `terrain_course_random_start_probability` 可用于
核对当前课程进度。`task_invalid_state` 正常应持续为 0；出现非零时不要把能耗尖峰误判
为真实策略能耗，应先排查仿真状态。

第一条命令保留完整多动作策略的观测维度和 14 维动作接口，但环境与 AMP 专家采样只
使用平地 walk；同时暂不施加 CoT 惩罚，先学习立即纠偏和持续不摔倒。选择
`mean_episode_length` 持续接近 `1000`、`termination_fraction` 接近 0 且速度没有明显
超过参考值的 checkpoint，再执行第二条命令进入完整串联路线。不要从旧的
`1.4 m/s` checkpoint 继续训练。旧八动作或六动作 checkpoint 的技能 one-hot 和当前
四动作观测维度不同，也不能用于续训；先运行四动作 walk warmup。

如果兼容当前四动作布局的旧 checkpoint 中 `forward_velocity_reward_coef`、
`forward_progress_reward_coef` 和 `forward_velocity_deficit_penalty_coef` 都是
0，策略通常会学成原地站立。需要加载权重并用当前的向前奖励重新计数续训；仅修改
`play.py` 的目标速度不会改变已经训练好的策略：

```bash
uv run python scripts/train_multimotion_stage1.py \
  --resume-checkpoint \
    logs/multimotion/stage1/<旧运行目录>/model_final.pt \
  --resume-reset-progress \
  --resume-reset-optimizer \
  --num-envs 4096 \
  --max-iterations 5000
```

续训时应观察 `task_forward_velocity` 和 `mean_episode_success_rate`，不要只看总奖励；
前者应明显离开 0，后者最终应大于 0 后再用于完整路线播放。

模型权重、ONNX、训练指标和 W&B 日志分别写入
`logs/multimotion/stage1/<月-日-时:分:秒>/` 和
`logs/multimotion/stage2/<月-日-时:分:秒>/`，两个阶段不会混写。
每次 checkpoint 成功写入后都会删除旧的 `.pt` 和 `.onnx`，因此训练目录中
始终只保留最新的一对模型文件。

第二阶段领域随机化：

```bash
uv run python scripts/train_multimotion_stage2.py \
  --stage1-checkpoint \
    logs/multimotion/stage1/<run>/model_final.pt \
  --num-envs 4096 \
  --max-iterations 7500
```

## 可视化测试与录屏

使用统一播放入口测试多动作 checkpoint；`--viewer` 会像 Stage1/Stage2 一样同步
录制 MP4。多动作测试默认创建 3 个机器人、3 条不同随机顺序的完整串联路线并同屏
显示：

```bash
cd /home/yeah/code/LYTrobot_amp
uv run python scripts/play.py \
  --checkpoint logs/multimotion/stage2/<run>/model_final.pt \
  --stage multimotion \
  --device cuda \
  --viewer \
  --steps 0
```

录像保存在 `recording/multimotion/`，本例首次运行的文件名为
`07-30-11-52-12-26000.mp4`。省略 `--stage multimotion` 时也会根据 checkpoint
路径和多动作环境配置自动识别；重复测试同一权重会追加 `-2`、`-3`，避免覆盖。
可用 `--num-robots` 覆盖同屏数量；`--seed` 固定三条路线的随机排列。
完整路线结束后会从路线起点自动开始下一轮；只要 viewer 未关闭就持续测试。真实摔倒
只会计入 `terminations` 并触发稳定起点重置，不会结束整个播放进程。

训练机制包括：

- 一个策略共享四种直线动作，actor 观测追加 4 维动作 one-hot；左右转身暂不参与训练；
- reset 后立即启用策略的肌腱残差纠偏，不再用前20步渐增限制压制动态平衡动作；
- 前进速度与进度奖励乘以根部姿态/高度稳定度，倾倒前冲不再获得完整正奖励；
- 速度跟踪带宽收紧到 `0.20 m/s`，超速惩罚系数提高到 `3.0`，抑制超过参考速度后继续前冲；
- 摔倒终止额外扣 `10`，确保高速前冲后立即倒地的回报低于持续稳定行走；
- 完整评测 episode 从走路开始，依次通过随机排列的 rough、6°上/下坡和 stairs 全部地形；
- AMP 判别器从四个直线地形重定向片段的共享归一化混合分布采样；
- PPO/BPO 强化学习奖励继续跟踪根姿态、线/角速度、关节、足端与接触；
- 动作片段结束后自动切换下一路段，完成整条串联路线后才成功重置；
- 台阶和地面均参与足端接触奖励与摩擦领域随机化。

快速验证：

```bash
uv run --with pytest python -m pytest -q \
  tests/test_multimotion_stairs.py tests/test_more_terrains.py \
  tests/test_serial_terrain_course.py
```
