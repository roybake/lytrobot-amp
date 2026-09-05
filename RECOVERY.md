# 2026-08-06 多动作训练恢复说明

已将 `logs/multimotion/08-06-10:56:09` 对应的历史代码、测试、机器人资源和原始训练产物恢复到独立 Git worktree：

`/home/yeah/code/LYTrobot_amp_recovered_20260806`

恢复分支：`recovery/multimotion-20260806`。原工程的当前分支、未提交修改和原始日志均保留。

## 直接使用

在终端执行：

```bash
cd /home/yeah/code/LYTrobot_amp_recovered_20260806

# 查看旧模型，自动使用复制到恢复目录的 model_final.pt。
# 旧版 viewer 会同时录制视频，存入本恢复目录 recording/multimotion/。
.venv/bin/python recovery/run.py play --viewer --steps 3000 --seed 42

# 无窗口测试。
.venv/bin/python recovery/run.py play --steps 1000 --num-robots 3 --seed 42

# 仅检查重训参数，不启动训练、不创建日志或 W&B run。
.venv/bin/python recovery/run.py train --dry-run

# 确认要重新训练时执行：4096 环境、30000 迭代，沿用原始 W&B 设置。
.venv/bin/python recovery/run.py train
```

新训练输出到 `logs/multimotion_recovered_20260806/<时间戳>/`。入口拒绝复用日志目录、固定运行名、resume checkpoint 和已有 W&B run ID。

ONNX 回放：

```bash
.venv/bin/python recovery/run.py play \
  --checkpoint logs/multimotion/08-06-10:56:09/model_final.onnx \
  --steps 1000 --seed 42
```

请使用 `recovery/run.py`：原模型 metadata 内保存了原工程的绝对路径，直接运行历史 `scripts/play.py` 会读取仍然存在的原工程资源。恢复入口只在内存中重定位这些路径，并优先导入恢复目录的源码；模型文件本身没有修改。入口将回放目标速度设为训练记录的 `0.862295 m/s`，仍可通过参数覆盖。指定 `--seed 42` 会按历史回放逻辑关闭每次 episode 随机 seed；不指定则随机回放。

## 定位依据

W&B 本地元数据 `wandb/run-20260806_105611-4ltcse1e/files/wandb-metadata.json` 记录：

- 启动时间：2026-08-06 10:56:11，北京时间。
- 启动入口：`scripts/train_multimotion.py`。
- 当时 HEAD：`b9319bd77e05d502b548be81191dcedf57e3fdd9`（09:22:51 提交）。
- 完整启动参数同时保存在 `params/config.json` 的 `argv` 中。

重要差异：`b9319bd` 的默认入口还是六动作、包含 gap/pit；实际日志已经是四动作版本。这些默认参数在后续提交 `61625708b1035616105cbea866513e3e345a78bc`（11:15:56 提交）中出现。

本恢复目录以 **6162570 的完整仓库快照**为基础。已验证该提交的 `default_train_args()` 加 `--max-iterations 30000` 与原始日志展开后的 argv 完全一致；所有解析后的有效参数也一致（重定位根目录，并按原 main 逻辑将 stage1 的 domain_randomization=None 解析为 False）。

`b9319bd` 与 `6162570` 的整个 `src/`、`robot/`，以及 `scripts/train.py`、`scripts/play.py`、`pyproject.toml`、`uv.lock` 的 Git 对象完全相同。因而恢复的是日志 HEAD 对应的训练核心、回放实现及资源，配上与实际启动记录匹配的四动作入口。

日志没有保存运行时完整的未提交源码差异，因此不能证明当时每一个内存中的源文件都与提交逐字相同；也未找到这次模型当时的具体回放命令。这里恢复的是有提交、完整参数和模型兼容性验证支持的版本，同期回放脚本已实际运行通过。

## 训练契约和附属内容

| 项目 | 恢复内容 |
| --- | --- |
| Actor | 参考动作条件输入，131 维观测、14 维动作，隐藏层 256/128 |
| 动作 | walk / rough_walk / stair_up / stair_down |
| 权重、循环 | 各 0.25；true / false / false / false |
| 地形 | rough / slope_up / slope_down / stairs，8 个串联地形变体 |
| 阶段 | stage1，关闭域随机化；原始日志组 multimotion |
| 训练 | Warp/CUDA，4096 环境，24 rollout steps，seed 42，30000 迭代 |
| 终态 | iteration=29999，global_step=2949120000 |
| 核心 | `src/lytrobot_amp/` 环境、Warp、网络、PPO/AMP、动作与地形实现 |
| 入口 | `scripts/train.py`、`scripts/train_multimotion.py`、`scripts/play.py` |
| 资源 | `robot/` 历史 XML、STL、参考 npz、地形；`multimotion/` 重定向代码及数据 |
| 测试与文档 | 对应提交的 `tests/`、README、docs 和多动作说明 |
| 依赖 | 原 `pyproject.toml`、`uv.lock`；另保存运行时 `recovery/original_requirements.txt` |
| 训练记录 | 原日志目录完整复制，包括 PT、ONNX、config、metrics、train.log、W&B 文件 |

`scripts/evaluate_acceptance.py` 是传感器策略验收入口，不适用于这个 131 维参考条件策略；本模型测试使用同期 `scripts/play.py`，通过恢复入口调用。

本机 `.venv` 链接到原工程现有环境；运行时 requirements 中列出的所有包版本均已核对一致，包括 Python 3.13.13、Torch 2.10.0+cu128、MuJoCo 3.9.0。没有重新安装依赖。跨机器克隆时应创建独立环境，使用历史锁文件和运行时 requirements 核对依赖。

## 校验记录

- 历史相关测试 31 项通过：多动作参数、楼梯、串联地形、回放录制、模型导出。仅有旧 ONNX 导出 API 弃用提示。
- 恢复入口 `recovery/test_recovery.py` 的 7 项测试通过，检查全部历史有效参数、源码导入位置、资源重定位、原 metadata 不变和新实验日志隔离；合计 38 项测试通过。
- 原 PT、ONNX 各运行 50 步 CPU 回放，seed=42、单机器人：两者 mean_reward=5.4109、mean_forward_velocity=0.8280 m/s。
- 默认三机器人 PT 回放也完成 50 步（150 个机器人样本），mean_reward=5.2855、mean_forward_velocity=0.8276 m/s。
- `recovery/manifest.json` 保存 77 个代码、资源和训练产物的 SHA-256；全部核对通过。原工程中的模型和日志哈希也一致。
- `git diff --check` 通过。历史 `scripts/`、`src/`、`robot/`、`tests/` 未做修改；新增适配位于 `recovery/`。
- 没有重新训练，没有启动 GPU 训练、GUI 或录像；短程 CPU 回放仅验证加载与执行，不代表完整地形通关或重训效果已复现。

复查命令：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  recovery/test_recovery.py tests/test_multimotion_log_paths.py \
  tests/test_multimotion_stairs.py tests/test_serial_terrain_course.py \
  tests/test_play_recording.py tests/test_model_export.py
git diff --check
```

原模型、完整日志和虚拟环境仅保存在本机恢复目录中，按仓库规则不提交 Git；源码、测试、配置证据、依赖记录和哈希清单保存在恢复分支。原模型的另一个副本仍在原工程的日志目录。
