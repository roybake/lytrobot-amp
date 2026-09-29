# `stairs.xml` 动作重定向

目标场景是 `robot/terrains/more_discolrl/stairs.xml`：5 级上楼、平台、4 级下楼，
踏步高度 0.20 m、深度 0.32 m。人体先验来自 CMU Mocap 143_17（上楼）与
141_08（下楼），来源和处理记录见 `source_data/cmu_stairs_20cm/README_sources.md`。

生成完整参考动作：

```bash
squat_training/retargeting/.venv/bin/python \
  multimotion/retargeting/prepare_stairs_xml_demo.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
squat_training/retargeting/.venv/bin/python \
  multimotion/retargeting/run_stairs_xml_omniretarget.py \
  --ignore-joint-limits --nominal-weight 20 \
  --export robot/reference_data/multimotion/stairs_xml_up_down_omniretarget.npz
```

播放：

```bash
squat_training/retargeting/.venv/bin/python scripts/view_stairs_xml_reference.py
```

验收结果位于 `results/stairs_xml/validation.json`。当前导出包含 737 帧，20 Hz，
36.8 秒；检查了全部存储帧及每帧间两个插值姿态，共 2209 个姿态。地形穿透和
双腿自碰撞均为 0，最大计划接触误差 8.25 mm，最大关节速度 3.84 rad/s，根节点
最大速度 0.834 m/s，平均前进速度 0.104 m/s。

按任务要求，求解未采用当前 XML 的关节限位，XML 文件没有被重定向程序修改。
验证报告中的 `required_joint_envelope` 给出了后续修改机器人模型所需的逐关节范围。
该文件是运动学参考；闭环策略的动力学稳定性仍需在强化学习训练和 rollout 中验证。
