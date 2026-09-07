# 四个独立地形的行走检查（2026-09-07）

## 后续修复与视频

粗糙参考已另存为
`robot/reference_data/multimotion/independent_collision_v1/rough_walk_omniretarget.npz`，
独立 rough 入口现已引用该文件。原始 NPZ 保留。使用原 rough 参考的一个完整左右
步态周期，扩展为五个步态周期/5 m 的地形周期；100 Hz、580 帧，平均速度
0.86207 m/s，与任务 0.862295 m/s 相差不到 0.001 m/s。通过实际训练模型的
碰撞查询修正根部净空（最大上移约 0.03012 m），未更改碰撞 margin、地形尺度或
关节姿态，重算根部及关节速度，并再次检查帧间插值。不是通过隐藏脚部碰撞消除穿透。

完整复查包含 6656 个姿态采样点：粗糙参考在 0/5/10/15 m 的四个周期位置逐帧及
半帧检查，另外三种参考在 0/3/9 m 检查；所有位置包含循环接缝。
四种地形最大检测穿透均为 0 m，非足部穿透接触数均为 0。
粗糙周期位移已经与地形周期对齐，不把任意空间平移当作同一参考相位。

| 地形 | 无脚底支撑面在 3 mm 内的采样比例 | 根部垂直速度峰值 m/s | 关节速度峰值 rad/s |
|---|---:|---:|---:|
| 平地 | 0% | 0.276 | 8.032 |
| 粗糙 | 4.74% | 0.810 | 12.414 |
| 上坡 | 0% | 0.550 | 9.900 |
| 下坡 | 1.04% | 1.186 | 12.000 |

重要解释：修复后 rough 参考保留正净空，MuJoCo 高度场在这些静态 FK 姿态上不产生
实际接触点；因此原始 `airborne_fraction=1.0`，不能把它描述成受力支撑。表中
“3 mm 内支撑”通过仅用于测量的向下探测获得，渲染/保存的姿态没有执行该探测。
它用于区分毫米级净空和明显悬空，不代表动力学接触验收。约 4.74% 的粗糙参考采样
间隙仍超过 3 mm；下坡原参考也有较大的垂直速度峰值，后续动力学跟踪仍需验证。
没有对原始接触标签重标注来隐藏这些误差。

四段视频均为训练几何下的运动学重定向回放（不是策略回放），18 秒、40 fps、
960×720、H.264。PNG 为视频第 1 秒预览，JSON 保存模型配置、源文件哈希和指标：

修复 NPZ 与视频使用的源 NPZ 逐字节一致，SHA-256 为
`b7502c22c130acc25d67437189905b0958d81917d6f08801216bb75346554a1b`。
58 项针对性测试通过，四个视频均经 ffprobe 核实为 720 帧/18 秒；
`git diff --check` 通过。未启动 GPU 训练或策略动力学回放。

- [平地视频](../recording/reference/independent_20260907_videos/flat.mp4)
- [粗糙视频](../recording/reference/independent_20260907_videos/rough.mp4)
- [上坡视频](../recording/reference/independent_20260907_videos/slope_up.mp4)
- [下坡视频](../recording/reference/independent_20260907_videos/slope_down.mp4)
- [完整检查 JSON](../recording/reference/independent_20260907_videos/audit.json)

复现修复、复查及视频（输出目录必须不存在）：

```bash
MUJOCO_GL=egl PYTHONPATH=.:src uv run python scripts/repair_independent_references.py \
  --output-dir recording/reference/independent_new_check --seconds 18
```

## 修复前检查记录（保留对照）

本次检查使用本地现有重定向 NPZ、实际 LYT MJCF 及
`robot/terrains/more_discolrl/height_field_lyt.png`，没有启动 GPU 训练。
当前工作区没有可读取的四个独立任务最新训练日志/checkpoint，因而不能判定用户
观察到的失败策略的具体参数或报告其成功率。历史奖励依据指定 run 的
`logs/multimotion/08-06-10:56:09/params/config.json`，该 run 原本是带技能条件的
串联任务；本次仅参考奖励，不替换独立任务的传感器 Actor 合约。

| 参考 | 平均前进速度 m/s | 平均步长 m | 支撑脚水平速度均值 m/s | 初始位置最大碰撞穿透 m |
|---|---:|---:|---:|---:|
| walk | 0.8623 | 0.5174 | 0.3444 | 0 |
| rough_walk | 0.8623 | 0.5013 | 0.3210 | 0.0163 |
| slope_up | 0.8623 | 0.4966 | 0.4333 | 0 |
| slope_down | 0.8623 | 0.5232 | 0.3733 | 0 |

步长/滑移从真实足部 FK 与 heel/sole/toe 接触阶段计算，并非用根部速度反推。
支撑脚水平速度含滚动、接触阶段切换和拟合误差，不能直接当成策略足滑率。
四段均具有左右交替步态，但这些数值不能证明其动力学可执行性。

粗糙场地旧铺设宽度 3 m、起始边界 (-2,-10)，与参考使用的 2 m 宽度及原中心
(2,6) 不一致；修正网格后，初始穿透由 0.0351 m 降至 0.0163 m。这个残余是真实
MuJoCo 碰撞检测结果，不以 NPZ 元数据中的零穿透覆盖。参考平移 3/9 m 后最大
穿透分别为 0.0348/0.0294 m：空间非周期高度图与循环步态不能逐点保持一致。
因此 rough 参考仍未通过严格接触验收，后续若重新生成参考，需要使用当前碰撞模型
约束脚底，并检查整段行程，不能仅修正出生点。

上下坡沿实际坡度前移 3/9 m 后无穿透，接触标签不匹配比例分别约 4.51%/2.08%；
平地约 10.42%。rough、上下坡的源 NPZ 都标记非循环，而训练循环使用；关节首尾
最大差分别为 0.2283/0.1927/0.1528 rad，平地为 0.2448 rad。首尾差包含正常一个
采样间隔的运动，不能单凭该差断言存在不连续；rough 的空间地形失配则已被检测到。

v3 恢复历史迈步奖励，并加入实际支撑/摆动状态奖励，保留足滑惩罚及停滞终止。
这修正了明确的配置缺失，但不能承诺无需训练即可正常行走。验收应使用新训练模型，
至少 20 个固定 seed、每回合 2000 步，检查净前进距离、跌倒、停滞与录像步态。
训练日志统一写入唯一的 `logs/independent_terrain/<terrain>/stage1|stage2/`。

复现检查：

```bash
PYTHONPATH=.:src uv run python scripts/audit_independent_references.py
PYTHONPATH=.:src uv run pytest -q tests/test_forward_steps.py tests/test_independent_terrain_tasks.py tests/test_play_recording.py tests/test_more_terrains.py
```
