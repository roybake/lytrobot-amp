# LYTrobot AMP 项目代理说明

## 项目目标

本项目是基于 MuJoCo、PyTorch 和 MuJoCo-Warp 的双足人型机器人强化学习训练代码，使用 PPO/AMP 风格方法训练行走、转身、上下楼梯、传感器策略和蹲起策略。

与本项目协作时使用中文，优先给出可复现、可验证的工程结果。

## 关键约束

- 默认机器人模型为 `robot/MJCFmodelposition.xml`。
- 标准策略动作固定为 14 维肌腱位置执行器目标修正量；不要在未说明的情况下改变动作维度、关节映射或观测接口。
- 参考运动数据位于 `robot/reference_data/`、`multimotion/` 和各任务目录的 `reference_data/` 中。修改映射时必须检查关节名、数据形状和时间采样。
- 训练优先使用项目自己的 `uv` 环境和已锁定依赖，不要混用系统 Python 或其他虚拟环境中的 MuJoCo/Warp 版本。
- 正式训练使用 GPU/Warp；只做逻辑验证时不要擅自启动长时间训练、W&B 实验或 viewer。
- 不要提交或覆盖已有 checkpoint、录像、W&B 输出和日志；新增实验输出应使用新的运行目录。

## 常用命令

```bash
uv sync
uv run pytest
uv run python scripts/train_stage1.py
uv run python scripts/train_stage2.py --stage1-checkpoint <path>
uv run python scripts/train_multimotion.py --num-envs 4096 --max-iterations 30000
```

执行训练前先确认 CUDA、显存、checkpoint 路径和输出目录。除非用户明确要求，不要运行默认规模的长训练；优先使用单元测试、短步数或静态检查验证改动。

## 修改流程

1. 先阅读相关入口、配置、环境、网络和测试，确认真实调用路径。
2. 保持改动最小，保留用户已有修改，不使用破坏性 Git 命令。
3. 修改奖励、终止条件、随机化、动作映射或模型导出时，补充或更新对应测试。
4. 完成后运行与改动相关的最小测试集，必要时再运行完整 `uv run pytest`。
5. 汇报修改文件、验证命令、测试结果和未验证的风险；训练代码改动还要说明是否实际运行了训练。

## 委派规则

只有任务边界清晰且可以独立完成时才委派子任务。需要委派时优先使用个人自定义智能体 `luna_worker`，适合它的任务包括只读代码定位、日志摘要、测试结果分析和单一模块检查。

委派任务必须明确：目标、允许访问的范围、是否只读、预期输出和禁止扩展的边界。多个智能体不要同时修改同一文件；涉及代码写入时由主智能体统一整合和验证。
