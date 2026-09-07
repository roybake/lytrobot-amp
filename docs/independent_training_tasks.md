# 独立训练任务

现有 `scripts/train_multimotion.py` 保持原有四动作和串联地形。本文涉及的训练入口统一
使用传感器 GRU actor：每帧只有 42 维 IMU 和 14 维肌腱长度，不输入技能标签、命令、
参考相位或地形真值。蹲起任务另行使用反馈校正的 `sin/cos` 相位，不改变这里的独立
任务。checkpoint 和日志目录彼此隔离。

## 四个独立地形任务

2026-09-07 修订（v4）：独立四地形任务统一采用固定目标速度 `0.862295 m/s`、
`sigma=0.35`、前向速度/进度系数 `3.0/2.0`、缺速/超速惩罚 `4.0/3.0`，
`task_reward_lerp=0.75`，AMP 系数 `1.5`。任务奖励使用物理可观测的 anti-collapse
合约，禁止 reference control 混入 Student；`amp_task_only` 保留为训练入口标记。
足滑阈值/系数为 `0.08/0.50`，`motion_reset_prob=0.50`，`stand_start_prob=0.0`。
保留 `forward_step_reward_coef=1.0`：仅在
参考摆动侧实际离地、另一侧接触地面且摆动脚向前运动时奖励，并乘以躯干稳定度。
两脚拖地、原地站立、腾空和反向摆动不获得这项奖励；CPU 测试与 Warp 训练使用
相同公式。原有姿态跟踪、前进速度、停滞终止和足滑惩罚继续启用。

粗糙路面使用原高度图，**最高点与最低点的总高差不超过 0.05 m**，不是正负各
0.05 m。独立 rough 入口拒绝超过该幅值的命令行参数。铺设保持重定向使用的
5 m × 2 m 尺度，并在原始 `(2, 6)` 中心对齐；旧场地的 3 m 横向缩放与平移
会改变脚下地形。测试从 checkpoint 恢复这些几何参数；独立任务默认也恢复训练
的 2000 步回合长度，静止起步验收默认仍为 100%，训练静止起步比例为 50%。

只读参考检查：`PYTHONPATH=.:src uv run python scripts/audit_independent_references.py`。
检查覆盖实际碰撞、接触标签、步距、支撑脚滑移、循环接缝，以及平移 3/9 米后的
地形适配。详见 [本次检查结果](independent_walking_audit_20260907.md)。粗糙参考现已
另存修复版本 `robot/reference_data/multimotion/independent_collision_v1/rough_walk_omniretarget.npz`，
独立 rough 训练默认使用该文件。它将五个步态周期对应到高度图的 5 m 空间周期，
使用训练碰撞几何修正根部净空，并重新计算速度；其余三种地形继续使用原参考。
四种地形在本次采样检查中均无穿透，但不能代替新策略训练与多 seed 回放。
现有动作及 checkpoint 均未覆盖。

2026-09-05 的初始化修订继续保留；新实验默认写入 `logs/independent_terrain_v3/`，任务 ID 和传感器
Actor 输入维度不变。建议 Stage 1 从头训练；旧失败策略的续训不能验证新初始化分布。
原有日志、模型和动作文件保持不变。上下坡高度目标和 AMP 高度现在按当前位置与
参考相位位置之间的坡面高差校正，避免循环片段结束后高度基准跳回原点。
独立任务的 `model_best` 优先比较完成率，再比较终止率、足滑和奖励；这仍是训练
分布上的指标，不能代替从静止起步的确定性回放验收。

当前可读历史日志只有两次平地训练：旧任务末轮平均速度约 0.844 m/s、完成率 94.3%，
补充静止起步后的续训约 0.803 m/s、完成率 80.4%。这些统计混合了参考状态起步、
静止起步及随机动作，不能证明部署能稳定行走。代码回归覆盖坡面前移后的高度目标、
AMP 一致性和下坡静止出生；尚未通过新训练证明四种地形均能稳定行走。

验收时四个任务分别测试 Stage 1/2，并使用多个固定 seed 从静止起步，以 2000 控制步
（默认 20 秒）为一回合；同时记录净前进距离、跌倒/停滞、横向偏移和足滑。可先以
至少 20 个 seed 中 90% 走完且净前进不少于 10 米作为实验门槛，再检查录像步态质量。
这只是建议验收标准，并非已取得的结果。

下面四个入口是真正的单地形任务，不是串联课程的不同日志别名。每个任务只加载一段
对应的 terrain-aligned 参考动作，Actor 仍严格只接收 IMU 和肌腱长度；Stage 1 关闭域
随机化，Stage 2 从 Stage 1 checkpoint 开始并开启域随机化。非串联模式的独立上坡/下坡
会生成实际坡面碰撞体，坡面几何与参考动作的根部高度变化方向一致。四个独立任务默认
启用运行时训练场：不平路面用同一份原始高度图平铺约 `24 m × 24 m` 区域，上坡/下坡
用连续斜面覆盖同一区域，平地使用原有无限平面。原始地形文件以只读方式复用，不会被
移动、覆盖或修改。

四个地形的训练 reset 默认关闭站立起步，使用 50% reference-state reset；部署验收
仍应额外覆盖自然站立起步。这样可以先消除静止吸引子，再单独验证部署起步鲁棒性。
上下坡静止起步按出生位置的真实坡面高度放置，再进行碰撞穿透校正。停滞检查在
400 步后开始计时，晚于默认最长 370 步的站立到行走过渡，首个检查窗口结束于 500 步。

任务奖励同时保留参考姿态、关节速度、足端轨迹、接触和对称性跟踪，AMP 作为额外的
整体动作风格约束；不再使用只含速度、直立和支撑目标的 `amp_task_only` 奖励。这样可
避免策略以锁膝、大跨步和持续倾斜的方式投机满足前向速度目标。

```bash
# 平地训练（Stage 1，无随机化）
uv run python scripts/train_flat_stage1.py
# 平地训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_flat_stage2.py \
  --stage1-checkpoint logs/independent_terrain_v3/flat/stage1/<run>/model_best.pt

# 平地测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain_v3/flat/stage1/<run>/model_best.pt \
  --stage stage1 --viewer --steps 0
# 平地测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain_v3/flat/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0

# 不平路面训练（Stage 1，无随机化）
uv run python scripts/train_rough_stage1.py
# 不平路面训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_rough_stage2.py \
  --stage1-checkpoint logs/independent_terrain_v3/rough/stage1/<run>/model_best.pt

# 不平路面测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain_v3/rough/stage1/<run>/model_best.pt \
  --stage stage1 --viewer --steps 0
# 不平路面测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain_v3/rough/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0

# 上坡训练（Stage 1，无随机化）
uv run python scripts/train_slope_up_stage1.py
# 上坡训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_slope_up_stage2.py \
  --stage1-checkpoint logs/independent_terrain_v3/slope_up/stage1/<run>/model_best.pt

# 上坡测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain_v3/slope_up/stage1/<run>/model_best.pt \
  --stage stage1 --viewer --steps 0
# 上坡测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain_v3/slope_up/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0

# 下坡训练（Stage 1，无随机化）
uv run python scripts/train_slope_down_stage1.py
# 下坡训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_slope_down_stage2.py \
  --stage1-checkpoint logs/independent_terrain_v3/slope_down/stage1/<run>/model_best.pt

# 下坡测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain_v3/slope_down/stage1/<run>/model_best.pt \
  --stage stage1 --viewer --steps 0
# 下坡测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain_v3/slope_down/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0
```

上面命令中的 `<run>` 要替换成实际运行目录名。如果只生成了最新 checkpoint，
可以把测试命令里的 `model_best.pt` 换成同目录的 `model_final.pt`。
测试命令默认会录屏；录像保持与仿真控制步一致的 1 倍时间速度，并统一保存到各自地形的
`logs/independent_terrain_v3/<地形>/recording/`，例如下坡为
`logs/independent_terrain_v3/slope_down/recording/`。如只想看画面、不保存录像，可追加
`--no-record`。

代码接口快速测试（不启动训练）：

```bash
PYTHONPATH=. uv run pytest -q tests/test_independent_terrain_tasks.py tests/test_more_terrains.py
```

对应的稳定任务 ID 分别是 `independent_flat`、`independent_rough`、
`independent_slope_up` 和 `independent_slope_down`。四者均使用直接传感器 Actor + AMP，
不启用 action distillation；`model_final` 用于中断恢复，`model_best` 用于评估和导出。
独立任务显式循环步态参考，使机器人能够走完整个铺满后的训练场；任务速度统一使用固定的
物理目标速度，不把参考片段边界处的速度跳变传给策略。上下坡的出生位姿使用运行时校准
偏移，使参考脚底贴合连续坡面；原始动作文件和地形文件均不修改。

## 蒸馏还是直接训练

对这四个独立任务，建议先直接训练，不蒸馏。原因是每个任务只有一种地形和一种参考动作，
Teacher 的额外动作标签/参考条件并不能提供跨技能选择信息；直接传感器训练可以让 GRU
从真实 IMU、肌腱长度和接触反馈中学习该地形的闭环适应，避免 Teacher 的参考控制残差、
观测分布和 Student 控制空间之间产生偏差。当前实现仍保留 asymmetric critic 和训练期
特权上下文/estimator loss，它们是降低训练方差的训练辅助，不是把 Teacher action 蒸馏进
部署 Actor。

只有在直接训练出现明显的探索困难、而且已经有同一模型/同一地形/同一时钟下稳定的
privileged Teacher 时，才建议为单个任务增加蒸馏对照实验。蒸馏必须使用同任务 Teacher，
不能把现有四地形 Teacher 或其他坡向 Teacher 交叉使用；并单独建立新的 task ID、日志树
和 checkpoint。实验顺序建议是：先以直接训练为基线，再以较小的
`distillation_coef`（例如 `0.1`）做对照，最后只在成功率、滑移和能耗都改善时保留蒸馏
版本。未经这组对照，不建议把蒸馏作为四个独立任务的默认方案。

## 去掉阶梯的多动作训练

该入口使用唯一 canonical `multimotion_no_stairs_amp_single` 传感器 Actor、奖励主体、
PPO/AMP 默认值和两阶段规则，同时保留无阶梯串联课程。
每条路线恰好包含一次 `flat`、`rough` 和相邻的 `slope_up → slope_down`，三个地形块共有
6 种唯一排列；训练 episode 随机分配路线，测试启动时也用新 seed 随机生成路线。
Actor 仍看不到地形顺序，只能依靠 IMU 和肌腱长度适应地形。

```bash
uv run python scripts/train_multimotion_no_stairs_stage1.py \
  --num-envs 4096 --max-iterations 30000

uv run python scripts/train_multimotion_no_stairs_stage2.py \
  --stage1-checkpoint \
    logs/multimotion/no_stairs_amp_single/stage1/<run>/model_best.pt \
  --num-envs 4096 --max-iterations 7500
```

Stage 1 关闭 DR；Stage 2 从 Stage 1 checkpoint 加载权重并默认追加
`--resume-reset-progress --resume-reset-optimizer`，再开启 adaptive DR。当前课程不把
机械能计入奖励；功率和机械能指标仍正常记录，仅用于离线分析。随机起始概率前 750
轮 warmup，随后 4000 轮 ramp 到 0.70；完整路线 success 只统计 `start_stage==0`。
跨地形即时奖励为 `0.50`，进入新 terrain 后连续生存 75 个 policy step 且保持前进
才支付 `2.0` delayed transition bonus。AMP 训练按 motion ID 对齐 expert，condition
只进入 discriminator，不进入 Student Actor。输出分别写入
`logs/multimotion/no_stairs_amp_single/stage1/<timestamp>/` 和
`logs/multimotion/no_stairs_amp_single/stage2/<timestamp>/`。

## 独立上下楼梯训练

该任务使用 `walk`、`stair_up`、`stair_down` 三个参考动作组成一条不可拆分的连续路线：
地面走近楼梯、完整上楼、越过顶部、完整下楼，再在地面走出。训练前 15000 轮只从入口
起步，先把连续平地步态训练稳定；随后用 7500 轮把随机后续阶段 reset 概率提高到 35%，
补足上楼、下楼和走出阶段，同时保留多数入口样本，避免困难阶段导致行走遗忘。测试仍
强制从入口开始。动作切换只推进参考阶段，不重置或搬运机器人状态。

```bash
uv run python scripts/train_stairs_stage1.py \
  --num-envs 4096 --max-iterations 30000

uv run python scripts/train_stairs_stage2.py \
  --stage1-checkpoint \
    logs/stairs/stage1/<run>/model_best.pt \
  --num-envs 4096 --max-iterations 7500
```

Stage 1 强制关闭域随机化，默认使用较易的 `0.13 m` 级高和 `0.32 m` 踏深；
Stage 2 强制开启自适应域随机化，并默认切换到更困难的 `0.20 m` 级高和 `0.22 m` 踏深。
两个阶段均不将能耗指标计入奖励优化。
命令行显式给出的楼梯尺寸仍可覆盖默认值。输出分别写入
`logs/stairs/stage1/<timestamp>/` 和
`logs/stairs/stage2/<timestamp>/`。`--reward-mode stairs` 使用独立的
楼梯奖励组合；串联楼梯的奖励表面与实际碰撞几何使用同一个世界坐标配置，且
`stair_min_root_clearance=0.60 m` 会终止陷入台阶的轨迹。重点指标为：

- `task_stair_vertical_velocity_track`：跟踪上楼/下楼参考动作各自的垂直速度；
- `task_stair_swing_clearance`：摆动脚相对当前台阶表面的净空；
- `task_stair_tread_support`：参考支撑脚是否在踏面上形成有效接触；
- `task_stair_min_swing_clearance_value`：摆动侧脚/趾几何体的最小实际净空（米）。
- `task_stair_root_clearance_value`：骨盆相对当前台阶表面的实际净空；
- `task_stair_terrain_penetration`：因低于最小净空而终止的穿模/陷入标记。

两个任务的阶段入口均固定 `--no-skill-conditioning --reference-ctrl-blend 0`，通过
`sensor_training.train` 构建严格传感器策略。命令行不能重新启用技能输入或参考控制
混合。正式训练前仍应确认 CUDA、显存、
输出目录和恢复 checkpoint；不同任务之间不要交叉恢复 checkpoint。

每个 run 同时保留两组原子 checkpoint：`model_final.pt/.onnx` 是最新状态，用于中断
恢复；`model_best.pt/.onnx` 只有在滚动窗口内真实跨过至少一个路线阶段后才生成，再按
完整路线成功率、后缀/阶段完成度、路线进度，最后按奖励和滑移择优，用于 Stage 1 晋级
和测试。选择依据写入 `model_best.json`。

所有动作的可视化测试都会自动录屏，并在各自日志树内按两阶段隔离：
`logs/<task>/stage1/recordings/` 与 `logs/<task>/stage2/recordings/`。录像文件名包含训练
run 和 checkpoint iteration；重复录制会追加序号，不覆盖已有文件。
