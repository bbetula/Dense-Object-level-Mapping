# 场景图生成与指标计算

本目录在原有语义建图结果上生成场景图，并计算两个验收指标：

- 场景图生成完整度 `>= 90%`
- 结构化数据准确度 `>= 85%`

所有默认路径和阈值都写在：

```text
fused_construction/scene_graph/scene_graph_config.py
```

当前默认输入：

```text
/data1/user/data/fastlivo_output_qs2_03.17/lidar/res/all_raw_qs2_03.17_color_normal.pcd
```

当前默认输出：

```text
/data1/user/data/fastlivo_output_qs2_03.17/lidar/scene_graph/all_raw_qs2_03.17_color_normal_strict
```

## 一键生成

```bash
conda activate dom
cd /data1/user/Dense-Object-level-Mapping
bash fused_construction/scene_graph/run_scene_graph_pipeline.sh
```

默认输出：

```text
scene_graph.json                         # 最终场景图：nodes + edges
visualization/scene_graph_view.html      # 自包含可视化 HTML
scene_graph_coverage_audit.json          # 语义地图类别覆盖审计
expected_contents_template.csv           # 实际场景内容 GT 表头
expected_relations_template.csv          # 实际关系 GT 表头
node_review_template.csv                 # 预测节点人工评审表
edge_review_template.csv                 # 预测边人工评审表
```

## GT 怎么来

GT 不是脚本自动生成的，也不能由模型预测结果反推。正式验收时，GT 需要由人工/专家/验收人员根据真实场景填写：

```text
expected_contents_template.csv   # 实际场景里应该存在的内容，例如 floor、wall、tree、box
expected_relations_template.csv  # 实际场景里应该存在的关系，例如 tree on floor、box adjacent_to wall
node_review_template.csv         # 对每个预测节点判断 correct=1/0
edge_review_template.csv         # 对每条预测边判断 correct=1/0
```

之前的 `flyfield_expected_contents_example.csv` 和 `flyfield_expected_relations_example.csv` 只是用于预检查流程的示例清单，不是正式 GT。现在已删除，避免误用。

## 指标如何计算

场景图生成完整度：

```text
完整度 = 匹配内容数量 / 实际场景内容数量
```

脚本对应字段：

```text
实际场景内容数量 = expected_contents 中 required=1 的有效行数
匹配内容数量 = 这些 GT 内容中已填写 matched_node_id 的数量
```

结构化数据准确度：

```text
Precision = TP / (TP + FP + FN)
```

脚本里：

```text
TP = node_review 中 correct=1 的节点数 + edge_review 中 correct=1 的边数
FP = node_review 中 correct=0 的节点数 + edge_review 中 correct=0 的边数
FN = expected_contents 未匹配数量 + expected_relations 未匹配数量
```

## 计算指标

人工填写 GT/评审 CSV 后运行：

```bash
conda activate dom
cd /data1/user/Dense-Object-level-Mapping
python fused_construction/scene_graph/evaluate_scene_graph.py
```

输出：

```text
/data1/user/data/fastlivo_output_qs2_03.17/lidar/scene_graph/all_raw_qs2_03.17_color_normal_strict/scene_graph_metrics.json
```

## 可视化

打开：

```text
/data1/user/data/fastlivo_output_qs2_03.17/lidar/scene_graph/all_raw_qs2_03.17_color_normal_strict/visualization/scene_graph_view.html
```

HTML 里包含 3D 场景图、节点表和边表。默认使用 Three.js 交互渲染：左键旋转，滚轮缩放，右键/中键平移；节点按 `centroid(x,y,z)` 放置，边按节点三维位置连接。支持节点查询、边查询、关系类型过滤、类别过滤，并可点击节点或边查看详情。

## 完整性检查

一键脚本会自动运行：

```bash
python fused_construction/scene_graph/audit_scene_graph_coverage.py
```

它会从输入 PCD 重新统计所有出现过的语义类别，并检查这些类别是否都在 `scene_graph.json` 中至少有一个节点。审计结果写入：

```text
/data1/user/data/fastlivo_output_qs2_03.17/lidar/scene_graph/all_raw_qs2_03.17_color_normal_strict/scene_graph_coverage_audit.json
```
