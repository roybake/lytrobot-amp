# 仅本体感知的跨地形平滑过渡方案

## 目标与边界

本方案面向同一个 14 维肌腱位置策略在平地、不平路、上下坡和上下楼梯之间连续行走。
部署 Actor 只能读取 IMU、肌腱长度和自身 GRU 状态；地形标签、动作 one-hot、根部真值、
接触真值、参考相位和参考肌腱目标均不能进入 Actor 或导出的 ONNX。

纯本体感知只能在脚或机体与地形发生相互作用后估计地形，不能可靠预知尚未接触的深坑、
宽沟或突然出现的高障碍。因此本文方案适合连续可探测的粗糙路、坡面和楼梯；需要提前选
落脚点的未知断崖或大间隙仍应使用深度、触觉探测或上层路径规划，不能仅靠训练承诺安全。
同理，没有航向命令时，完全相同的本体状态无法决定应左转还是右转；转向动作不进入本入口
的 AMP 混合分布，应由后续带航向命令的策略处理。

## 论文与作者代码结论

- [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251)
  使用特权教师和读取本体历史的学生网络，证明时间历史可以隐式恢复接触和地形信息。
- [RMA](https://arxiv.org/abs/2107.04034) 与双足版本
  [A-RMA](https://arxiv.org/abs/2205.15299) 将本体历史映射为环境 extrinsics，支持在未知
  动力学和路面上快速在线适应；A-RMA 还指出必须用估计器实际误差继续微调 base policy。
- [DreamWaQ](https://arxiv.org/abs/2301.10602) 在单阶段训练中让 Actor 使用短时本体历史，
  critic 使用完整状态，并显式估计基座速度和隐式环境上下文。其
  [作者代码](https://github.com/Manaro-Alpha/DreamWaQ) 的 CENet 同时进行速度估计、
  观测重构和策略优化。
- [Learning Humanoid Locomotion over Challenging Terrain](https://arxiv.org/abs/2410.03654)
  使用本体观测/动作历史 Transformer，先做平地序列预训练，再在复杂地形上强化学习；论文
  观察到隐状态会按地形自然聚类并产生 in-context adaptation。
- [CTS](https://arxiv.org/abs/2405.10830) 并行训练特权编码器和本体编码器，共用策略与
  asymmetric critic，避免传统两阶段蒸馏的分布偏移。

本项目选择 DreamWaQ/CTS 风格的单阶段方案，而不直接移植大型 Transformer 或完整两阶段
DAgger：当前策略已有因果 GRU、soft residual-MoE、AMP 参考数据和大规模 Warp rollout，
加入训练期上下文监督即可复用现有路径，显著小于另建教师/学生 rollout 的工程和显存成本。

## 实现结构

```text
IMU(42) + 肌腱长度(14)
          │ 每步 56 维，部署唯一输入
          ▼
       GRU(128) ───────────► base actor + soft residual-MoE ─► 14 维动作
          │                                      │
          │                                      └─ gate 时序平滑/负载均衡
          ├─► context estimator（仅训练损失，部署不需要目标）
          └─► asymmetric critic + 训练期特权上下文
```

训练上下文共 11 维，全部归一化到 `[-1, 1]`：

1. 三轴根部线速度；
2. 左右脚接触状态；
3. 连续地形参数：坡度、不平度、有符号台阶高度、踏面深度、沟宽和坑深。

上下文目标不会拼入 Actor 观测。GRU 的 context head 使用 Smooth-L1 损失恢复这些量；critic
可直接读取训练真值以降低 value 方差。导出的 ONNX 仍只有 `observation`、
`recurrent_state` 和 `episode_start` 三个输入。

传感器多地形入口把 `reference_ctrl_blend` 固定为 0。执行器目标以实测肌腱长度为基准，
策略输出 14 维目标修正量；模拟器按当前位置选择的参考动作只用于奖励和 AMP 教师分布，
不再通过控制基准泄漏给部署策略。动作 rate 惩罚、soft gate 变化损失和 residual L2 共同
约束过渡抖动。

## 训练顺序

```bash
uv run python sensor_training/train_multimotion.py \
  --num-envs 1024 \
  --max-iterations 30000
```

默认课程为：前 5000 轮全部从完整路线 stage 0 开始，随后 10000 轮把随机后续 stage
reset 概率从 0 线性升到 0.8；领域随机化使用性能门控的 adaptive 模式。新的 context
critic 与旧 recurrent checkpoint 形状不同，必须新开运行，不能直接加载旧权重。

训练时至少监控：

- `privileged_context_loss`：应下降且保持有限；
- `gate_change_loss`、`task_action_rate`：不应在地形边界形成持续尖峰；
- `task_course_stage_completion` 和 `episode_success_rate`：确认不仅学会站立；
- `task_invalid_state`：应持续为 0；
- `termination_fraction` 与各地形单独成功率。

## 验证

```bash
uv run --with pytest python -m pytest -q \
  sensor_training/tests/test_observation.py \
  sensor_training/tests/test_policy_distribution.py \
  sensor_training/tests/test_recurrent_policy.py \
  sensor_training/tests/test_multimotion_entrypoint.py

uv run python sensor_training/train_multimotion.py \
  --num-envs 8 --num-steps 4 --epochs 1 --minibatches 1 \
  --amp-disc-updates 1 --amp-minibatch-size 32 \
  --max-iterations 1 --no-wandb --run-name proprio-context-smoke
```

短程 smoke 只验证数值、接口、checkpoint 和 CUDA 路径，不证明策略已经收敛。正式结论必须
使用独立随机种子完成长训练，并运行 `scripts/evaluate_acceptance.py` 的完整路线评测。
