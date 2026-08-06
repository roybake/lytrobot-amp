# MoRE-discoRL 地形源文件

本目录从本机以下仓库复制而来：

`/home/yeah/code/unitreeG1/MORE/MoRE-discoRL`

复制日期：2026-07-29。原文件保持不改动，LYT 适配实现在
`src/lytrobot_amp/terrain.py`，不会直接加载这些 G1 场景中的机器人模型。

| 文件 | SHA-256 | 用途 |
|---|---|---|
| `gap.xml` | `f834c8b6124cfe0b33760a09347941617d7f34da86fad7b18e3644debab3adea` | 0.40 m 间隙评测场景 |
| `pit.xml` | `2e168a9b82984816349752478243a728b646dd40d7430e17e120b950ce01fe46` | 被低地包围的高台场景 |
| `rough.xml` | `4291398d5d914ce2cd0bea476f92ddaf23971bb62949d8332186d52ecb95ba83` | 高度场场景 |
| `stairs.xml` | `86d6b756d91e1a3ca12ebb039e3860de7d95985f415dec3eef98f1f24cac7850` | 0.15 m×0.31 m 楼梯评测场景 |
| `height_field.png` | `592c83e4df32e5474b17987d9c66cefa80369e7d5dba5f082de777e2ab4f6929` | 128×128 rough 高度图 |
| `height_field_lyt.png` | `d6ee868f88f8edf5d59ceb5b6323f4610967246c44612ea07b254e60911608bc` | 64×32 LYT/Warp 训练副本 |
| `terrain_source.py` | `61499dd9042d88becd0bfc7ec69961a0c1ffd34a4c194861c54396300f5edb44` | Isaac Gym 程序化地形及课程参数 |

原 `rough.xml` 中的 `../height_field.png` 相对路径与源仓库实际文件位置不一致；
为了保留校验值，本目录中的源 XML 未修改。LYT 适配代码直接解析同目录的
训练副本，因此训练和可视化不受该上游路径问题影响。

LYT 场景的物理尺寸为 5.0 m×2.0 m。若直接使用 128×128 网格，脚底会同时
覆盖超过 MuJoCo Warp 单个高度场几何允许的 50 个接触候选，因此训练加载
`height_field_lyt.png`。该文件用 area filter 从原图确定性降采样到 64×32，
保留地形大尺度轮廓，同时把单脚覆盖的网格数降到安全范围；原文件及其哈希
仍完整保留。

`LICENSE-CC-BY-NC-4.0.md` 是 MoRE-discoRL 仓库许可证；
`terrain_source.py` 文件头声明 BSD-3-Clause，许可证副本保存在
`LICENSE-BSD-3-Clause.txt`。使用或再分发前请遵守对应许可证。

## LYT 训练适配

- gap：保留 0.30 m 高平台，默认课程宽度为 0.20 m；复制的 XML 保留
  0.40 m 最终评测宽度，可用 `--gap-terrain-width` 调整。
- pit：将源高台缩放为 2.0 m 宽的安全通道，周围地面低 0.30 m。
- rough：保留源高度图样本，缩放到 LYT 的 5.0 m×2.0 m 通道，默认最大起伏
  0.10 m。
- stairs：训练继续使用 CMU 楼梯动作标定的 0.13 m×0.58 m×3 级布局；
  原始 0.15 m×0.31 m×10 级 XML 保留作更高难度评测。

四类地形与平地分别放在并行 Y 通道中，同一个 MuJoCo/Warp 模型可同时供
批量环境抽样。动作名中的 `gap`、`pit`、`rough`、`stair` 会自动路由到匹配
通道，避免参考动作在错误地形上启动。
