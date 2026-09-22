# CMU 楼梯人体动作候选（2026-09-22）

本目录仅保存 CMU 官方源数据、可复现解析脚本、实际测量的轨迹及候选分析；不包含机器人可达性/碰撞/动力学验证结论。

## 推荐组合

对于 `robot/terrains/more_discolrl/stairs.xml` 的 0.20 m 高、0.32 m 深楼梯，优先选 **CMU 143_17 上楼 + 141_08 下楼**。这两位 subject 的官方类别都是普通 General Subject Capture，目标动作无夸张步态标签。原始台阶约 0.20 m，优于 113 的约 0.13 m；上楼和下楼单步都选为左腿支撑、右腿摆动，方便主流程做左右镜像交替。

| 用途 | 文件 | 原始帧范围（从 0 起，含末帧） | 原时长 | 根节点前进 / 高度变化 |
|---|---|---|---|---|
| 上楼稳定单步 | `143_17_up_single_step_aligned_120hz.npz` | 53–100 | 0.3917 s | +0.2877 m / +0.1982 m |
| 上楼两级含靠近/收脚 | `143_17_up_two_steps_aligned_120hz.npz` | 0–155 | 1.2917 s | +0.8845 m / +0.3928 m |
| 下楼稳定单步 | `141_08_down_single_step_aligned_120hz.npz` | 215–245 | 0.2500 s | +0.3993 m / −0.2217 m |
| 下楼两级 | `141_08_down_two_steps_aligned_120hz.npz` | 169–251 | 0.6833 s | +0.7880 m / −0.4421 m |

单步时间建议先设置约 **0.7–0.8 s**，对应 0.32 m 踏步约 **0.40–0.46 m/s** 的平均前进速度；这是目标机器人重定向的初始工程参数，不是源人体原速度，也不是经过动力学验证的安全速度。应在关节速度、接触滑动和动态可行性检查后决定是否接受。

`aligned` 文件只有水平刚体旋转（朝 +X）、根 XY 原点平移和地面趾点高度估计平移。没有做拟合楼梯的逐帧改写、时间重采样或机器人重定向。`fps` 保持原 120 Hz，`source_frames` 存 `[起始帧, 末帧, stride]`。关键点名称及 FK 方法沿用 `squat_training/retargeting/prepare_cmu_squat.py`。

## 接触与几何测量

`143_17`：朝向原 XY 为 `[-0.9999199, 0.01265885]`，地面趾点估计 Z=0.04544264 m。以下为转向 +X 后的趾点接触候选，X 相对整段初始根节点，Z 相对估计地面。区间左闭右开、从 0 起。

| 脚 | 接触帧 | X / Z（m） |
|---|---|---|
| 右 | [10, 63) | 0.3761 / 0.0135 |
| 左 | [51, 106) | 0.7184 / 0.1974 |
| 右 | [100, 199) | 0.9979 / 0.4014 |
| 左 | [146, 274) | 0.9952 / 0.3977 |

前两级有效抬升约 0.184 m、0.204 m，趾点着地点水平间距约 0.342 m、0.280 m；平均约 0.311 m，与目标 0.32 m 很接近。由于踝姿态和趾关节噪声，这些数值是骨骼点测量，不能当作原楼梯的精密尺寸。143_17 中间有倒退下楼，末段有跨过顶面直接落地，因此不应整段直接用于五上五下。

`141_08`：朝向原 XY 为 `[0.9962037, -0.08705305]`，地面趾点估计 Z=0.05262880 m。顶部右趾在 [178, 206) 接触候选的 X/Z 约 1.5576/0.4159 m；中阶左趾在 [216, 234) 为 1.8426/0.2317 m；右脚末级落地约 245–255 帧，245 帧右趾 X/Z 为 2.2590/0.0140 m。原始下降动作较快，足趾旋转噪声显著，简单速度阈值会漏检最后触地；这一段的接触需结合踝高度、足姿态和目标机器人几何校正。

`candidate_analysis.json` 包含全轨低速趾点接触候选：对趾点作 1.5 帧高斯平滑，速度低于 0.32 m/s，5 帧闭运算，至少持续 10 帧。该检测供筛选使用，**不是**机器人接触标签真值。`height` 是原转换器的首末窗口髋高估计；裁剪后的抬高轨迹不能直接拿它当站立腿长。

## 其他候选

- `113_19`：上下同侧、顶端转向，约 0.13 m 台阶；不适合作为直线穿过五级楼梯的完整原型。
- `113_20`：连续向前上下约两级，约 0.12–0.14 m 台阶，水平步长约 0.5 m；可作为较慢的下楼风格备选。官方 subject 类别为 Post pregnant woman。
- `141_07`：上两级后较大的落地跨步，适合做上楼备选。

## 来源与使用条件

文件全部来自 CMU Graphics Lab Motion Capture Database 官方域名；SHA-256、下载 URL 和大小保存在 `download_manifest.json`，没有批量抓取数据库。

- [CMU 首页与使用说明](https://mocap.cs.cmu.edu/)
- [Subject 143：trial 17 “Walk Up Stairs And Over”](https://mocap.cs.cmu.edu/search.php?subjectnumber=143)
- [Subject 141：trial 7 / trial 8](https://mocap.cs.cmu.edu/search.php?subjectnumber=141)
- [Subject 113：trial 19 / trial 20](https://mocap.cs.cmu.edu/search.php?subjectnumber=113)

CMU 首页说明数据可免费用于研究、可纳入商业产品，但不得直接转售数据本身（包括转换后的形式）。发表结果时请按官网要求致谢 CMU 数据库及 NSF EIA-0196217 资助，并可按官网说明反馈论文引用。这里没有将数据重新声明为 MIT/CC0；解析器的代码许可与数据使用条件不同。

官网还说明同一个人可能使用多个 subject 编号；每个 subject 都有独立标定骨架。足趾/手部关节可能有噪声，应按需要平滑。

## 重现

从仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 squat_training/retargeting/.venv/bin/python multimotion/retargeting/source_data/cmu_stairs_20cm/analyze_candidates.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 squat_training/retargeting/.venv/bin/python multimotion/retargeting/source_data/cmu_stairs_20cm/prepare_selected_clips.py
```

适配目标地形时应以实际足底几何的完整支持面、摆动足跨越立面、关节范围和自碰撞为约束。已有源人体动作和关键点落在正确高度本身不能证明机器人不穿模或动态稳定。
