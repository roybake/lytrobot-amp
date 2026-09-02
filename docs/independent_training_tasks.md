# 独立训练任务

现有 `scripts/train_multimotion.py` 保持原有四动作和串联地形。本文涉及的训练入口统一
使用传感器 GRU actor：每帧只有 42 维 IMU 和 14 维肌腱长度，不输入技能标签、命令、
参考相位或地形真值。蹲起任务另行使用反馈校正的 `sin/cos` 相位，不改变这里的独立
任务。checkpoint 和日志目录彼此隔离。

## 四个独立地形任务

下面四个入口是真正的单地形任务，不是串联课程的不同日志别名。每个任务只加载一段
对应的 terrain-aligned 参考动作，Actor 仍严格只接收 IMU 和肌腱长度；Stage 1 关闭域
随机化，Stage 2 从 Stage 1 checkpoint 开始并开启域随机化。非串联模式的独立上坡/下坡
会生成实际坡面碰撞体，坡面几何与参考动作的根部高度变化方向一致。四个独立任务默认
启用运行时训练场：不平路面用同一份原始高度图平铺约 `24 m × 24 m` 区域，上坡/下坡
用连续斜面覆盖同一区域，平地使用原有无限平面。原始地形文件以只读方式复用，不会被
移动、覆盖或修改。

```bash
# 平地训练（Stage 1，无随机化）
uv run python scripts/train_flat_stage1.py
# 平地训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_flat_stage2.py \
  --stage1-checkpoint logs/independent_terrain/flat/stage1/<run>/model_best.pt

# 平地测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/flat/stage1/<run>/model_best.pt \
  --stage stage1 --viewer --steps 0
# 平地测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/flat/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0

# 不平路面训练（Stage 1，无随机化）
uv run python scripts/train_rough_stage1.py
# 不平路面训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_rough_stage2.py \
  --stage1-checkpoint logs/independent_terrain/rough/stage1/<run>/model_best.pt

# 不平路面测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/rough/stage1/<run>/model_best.pt \
  --stage stage1 --viewer --steps 0
# 不平路面测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/rough/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0

# 上坡训练（Stage 1，无随机化）
uv run python scripts/train_slope_up_stage1.py
# 上坡训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_slope_up_stage2.py \
  --stage1-checkpoint logs/independent_terrain/slope_up/stage1/<run>/model_best.pt

# 上坡测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/slope_up/stage1/<run>/model_best.pt \
  --stage stage1 --viewer --steps 0
# 上坡测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/slope_up/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0

# 下坡训练（Stage 1，无随机化）
uv run python scripts/train_slope_down_stage1.py
# 下坡训练（Stage 2，加载 Stage 1 并开启随机化）
uv run python scripts/train_slope_down_stage2.py \
  --stage1-checkpoint logs/independent_terrain/slope_down/stage1/<run>/model_best.pt

# 下坡测试（Stage 1）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/slope_down/stage1/<run>/model_best.pt \
  --stage stage1 --viewer --steps 0
# 下坡测试（Stage 2）
uv run python sensor_training/play.py \
  --checkpoint logs/independent_terrain/slope_down/stage2/<run>/model_best.pt \
  --stage stage2 --viewer --steps 0
```

上面命令中的 `<run>` 要替换成实际运行目录名。如果只生成了最新 checkpoint，
可以把测试命令里的 `model_best.pt` 换成同目录的 `model_final.pt`。
测试命令默认会录屏；录像保持与仿真控制步一致的 1 倍时间速度，并统一保存到各自地形的
`logs/independent_terrain/<地形>/recording/`，例如下坡为
`logs/independent_terrain/slope_down/recording/`。如只想看画面、不保存录像，可追加
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

该入口继续复用普通行走 `scripts/train_stage1.py` / `scripts/train_stage2.py` 的传感器
Actor、奖励主体、PPO/AMP 默认值和两阶段规则，同时保留原来的无阶梯串联课程。
每条路线恰好包含一次 `flat`、`rough` 和相邻的 `slope_up → slope_down`，三个地形块共有
6 种唯一排列；训练 episode 随机分配路线，测试启动时也用新 seed 随机生成路线。
Actor 仍看不到地形顺序，只能依靠 IMU 和肌腱长度适应地形。

```bash
uv run python scripts/train_multimotion_no_stairs_stage1.py \
  --num-envs 4096 --max-iterations 30000

uv run python scripts/train_multimotion_no_stairs_stage2.py \
  --stage1-checkpoint \
    logs/multimotion/no_stairs/stage1/<run>/model_best.pt \
  --num-envs 4096 --max-iterations 7500
```

Stage 1 和 Stage 2 的领域随机化、恢复优化器及训练轮数规则与普通行走入口一致。
当前课程不把机械能计入奖励；功率和机械能指标仍正常记录，仅用于离线分析。奖励优先推动持续前进，同时保留足滑惩罚。
训练前 18000 轮全部从入口开始，先学会持续前进；随后用 8000 轮把随机后续阶段
reset 概率缓慢提高到 25%，为后段地形补充直接样本，同时至少保留 75% 的完整路线
episode，避免后段 reset 淹没入口行走并导致遗忘。完整路线成功率仍只统计入口起步的
episode。测试始终从路线入口开始。输出分别写入
`logs/multimotion/no_stairs/stage1/<timestamp>/` 和
`logs/multimotion/no_stairs/stage2/<timestamp>/`。

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
