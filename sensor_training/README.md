# IMU + 肌腱长度独立训练

动力学、奖励和 AMP 判别器复用现有实现；Actor 使用 BPO 更新，AMP 判别器和 AMP reward 保留。所有训练统一使用纯位置控制。

sensor stage1 默认与 `scripts/train_stage1.py` 使用相同的奖励系数和训练模式；stage2 入口会显式追加鲁棒训练和速度奖励相关参数。环境收到动作时仍统一裁剪到 `[-1, 1]`。

## 观测定义

Actor 每帧只读取：

- 7个三轴加速度计：21维，除以 `9.81`
- 7个三轴陀螺仪：21维，除以 `10 rad/s`
- 14条受控肌腱实际长度：14维，按执行器控制范围中心和半宽归一化
- 当前目标前向速度：1维，默认与 stage1/stage2 一致为 `1.0 m/s`；站立起步时按过渡进度从0平滑增加

传感器单帧56维。按“过去第5帧 → 当前帧”的顺序拼接6帧，再追加当前目标前向速度，Actor 输入共337维，不包含关节角度、关节速度、根部真值、参考轨迹、步态相位或上一动作。

AMP 判别器沿用原有单帧45维特征，其中已经包含14维关节角度和14维关节速度；同样拼接连续6帧，共270维。专家历史按 `policy_dt` 从参考动作中连续采样并统一进行左右镜像。

episode 重置时，6个历史位置都填充为重置后的当前帧，避免跨 episode 泄漏状态。

策略输出14维肌腱位置目标修正量，Actor 输入为337维。

传感器 stage1 的启动参数刻意对齐 `scripts/train_stage1.py`：`stage1`、`cuda`、`warp`、`4096` 环境、关闭 domain randomization、开启 wandb，并使用相同的 stage1 奖励默认值。当前目标前向速度仍保留在 Actor 观测中，保证 checkpoint 输入布局与后续 stage2 保持一致。训练日志额外记录 `task_termination`，用于区分真实跌倒与正常超时。

## 训练

```bash
uv run python sensor_training/train_stage1.py

uv run python sensor_training/train_stage2.py \
  --stage1-checkpoint logs/sensor_policy/stage1/<run>/model_final.pt
```

如果需要从 stage1 checkpoint 继续做速度微调，可以显式传入奖励和优化参数：

```bash
uv run python sensor_training/train_stage1.py \
  --resume-checkpoint logs/sensor_policy/stage1/<run>/model_<best>.pt \
  --resume-reset-optimizer \
  --run-name sensor-stage1-speed-ft \
  --learning-rate 2e-5 \
  --forward-velocity-reward-coef 0.5 \
  --forward-progress-reward-coef 0.25 \
  --forward-velocity-deficit-penalty-coef 0.5 \
  --forward-velocity-excess-penalty-coef 0.5 \
  --max-iterations 34000 \
  --save-interval 100
```

快速冒烟训练：

```bash
uv run python sensor_training/train_stage1.py \
  --num-envs 32 --max-iterations 1 --num-steps 4 \
  --no-wandb --run-name smoke
```

默认 checkpoint 写入 `logs/sensor_policy/`，不会与原训练日志混在一起。该训练只能恢复本次纯位控生成的337维输入、14维输出 checkpoint；现有127维、336维策略 checkpoint，以及历史337维输入、28维输出的力位联控 checkpoint 均不兼容。

## 播放

```bash
uv run python sensor_training/play.py \
  --checkpoint logs/sensor_policy/stage1/<run>/model_final.pt \
  --viewer
```
