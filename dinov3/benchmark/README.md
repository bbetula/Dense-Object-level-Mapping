# DINOv3 指标评估脚本说明

本目录提供机狗臂飞场分割模型的指标评估脚本。脚本输入路径、输出目录、模型权重和阈值均在脚本或 `metric_common.py` 中固定定义，运行时不需要填写大量命令行参数。

精度评估采用「高精度教师推理结果 = GT」的对齐评估方式：教师 (DINOv3 ViT-7B + ADE20K M2F head) 的输出 `res_dinov3_whole/*_mask_id.png` 作为机狗实采场景的事实标签，学生蒸馏模型 (ViT-L) 的输出 `res_robotdog_distill/*_mask_id.png` 按像素与之比较。

## 公共配置

`metric_common.py` 是公共配置和工具模块，不单独作为指标运行。

### 模型注册表与切换接口

公共模块维护一份 M2F 模型注册表，并暴露统一的导出变量供其他评测脚本直接 import：

```python
CHECKPOINT_HEAD   # 当前模型权重 .pth 路径
CONFIG_FILE       # 当前模型 detectron2 配置
MODEL_NAME        # 当前模型名称字符串
NUM_CLASSES       # 当前模型类别数
```

注册表中现有两个键，默认评测 `robotdog`：

| MODEL_KEY | 用途 | 类别数 |
|---|---|---:|
| `city` | Cityscapes 19 类 M2F 学生 | 19 |
| `robotdog` | 机狗 ADE20K 150 类蒸馏学生（推荐） | 150 |

切换方式（三选一）：

1. 直接修改 `metric_common.py` 顶部的 `MODEL_KEY`。
2. 设置环境变量：`BENCHMARK_MODEL=city python ...`。
3. 评测脚本里给 `build_m2f_runner` 显式传 `head=` / `config_file=` / `model_name=` / `num_classes=`，跳过注册表。

为兼容历史代码，旧名 `CITY_HEAD / CITY_CONFIG_FILE / build_city_runner` 等仍可用，指向同一对象。

### 固定输入路径

```text
延迟评测原图目录:                 /data1/user/data/2026.05.10_机狗二次数据采集结果_标定/image
教师伪标签 (作为分割精度 GT):     /data1/user/data/2026.05.10_机狗二次数据采集结果_标定/image/res_dinov3_whole
学生蒸馏推理 (作为分割精度 Pred): /data1/user/data/2026.05.10_机狗二次数据采集结果_标定/image/res_robotdog_distill
```

### 固定阈值

```text
语义分割精度: pixel accuracy >= 90%
实时语义分割延迟: avg_frame_latency_ms <= 200 ms
```

## 运行方式

一键运行标准两项指标 + 汇总：

```bash
cd /data1/user/Dense-Object-level-Mapping
bash dinov3/benchmark/run_metric_pipeline.sh
```

单独运行某个指标：

```bash
conda run -n dinov3 python dinov3/benchmark/metric_semantic_segmentation_accuracy.py
conda run -n dinov3 python dinov3/benchmark/metric_real_time_segmentation_latency.py
conda run -n dinov3 python dinov3/benchmark/metric_summary_report.py
```

延迟脚本默认在 GPU 0 和 GPU 2 上分别评测，需要这两张卡都对当前 shell 可见（不要 `export CUDA_VISIBLE_DEVICES=2` 屏蔽其他卡）。

## 输出规范

```text
dinov3/output/metric_semantic_segmentation_accuracy/semantic_segmentation_accuracy_result.json
dinov3/output/metric_real_time_segmentation_latency/real_time_segmentation_latency_result.json
dinov3/output/metric_summary/metric_summary_report.json
```

## 脚本说明

### `metric_semantic_segmentation_accuracy.py`

评估指标：机狗采集场景下学生模型的语义分割精度。以高精度教师 (DINOv3 ViT-7B + ADE20K M2F head) 推理结果为 GT，与学生蒸馏模型 (ViT-L + SFP + M2F head) 推理结果按像素比较。

输入：

```text
GT 根目录   (教师推理):    /data1/user/data/.../image/res_dinov3_whole
Pred 根目录 (学生推理):    /data1/user/data/.../image/res_robotdog_distill
文件命名:                  {stem}_mask_id.png (uint8, 0..149)
类别空间:                  ADE20K 150 类
```

评估逻辑：

1. 按文件 stem 一一配对教师/学生 mask。
2. 像素级累计 150x150 混淆矩阵，并行 16 线程。
3. 计算 pixel accuracy、mIoU、mAcc 与 frequency-weighted IoU（按 GT 像素占比加权）。
4. 输出 top-K 频繁类别的逐类 IoU，便于观察主要类别的一致性。
5. 当 `pixel_accuracy >= 90%` 时指标通过。

输出：

```text
dinov3/output/metric_semantic_segmentation_accuracy/metric_semantic_segmentation_accuracy_full_image_metrics.csv
dinov3/output/metric_semantic_segmentation_accuracy/semantic_segmentation_accuracy_result.json
```

### `metric_real_time_segmentation_latency.py`

评估指标：实时语义分割延迟，在 GPU 0 与 GPU 2 上分别评测。

输入：

```text
原图目录:   /data1/user/data/2026.05.10_机狗二次数据采集结果_标定/image
模型:       MODEL_KEY 决定 (默认 robotdog_vitl_ade20k_m2f_distill)
batch size: 1
warmup:     1
计时迭代:   5
```

评估逻辑：

1. 列出可见 CUDA 设备，定位目标卡（默认 `(0, 2)`）；若某张目标卡不可见则告警跳过。
2. 对每张目标卡：`torch.cuda.set_device(idx)` 切线程默认设备，避免 synchronize 落到错误卡。
3. 用 detectron2 在该卡上构建 M2F 模型（含包装层将 ImageNet-normalized RGB 转 0-255 BGR）。
4. warmup 1 次，正式计时 5 次（包含前向 + argmax），用 `torch.cuda.synchronize()` 保证时间准确。
5. 统计平均单帧延迟、p50、p95、FPS。
6. 评测完释放显存再进入下一张卡。
7. 同时记录卡的 `gpu_index` 与 `gpu_name`（例如 `RTX A6000`、`RTX PRO 6000 Blackwell Server Edition`），整合写入同一 JSON。
8. 当任一卡满足 `avg_frame_latency_ms <= 200` 时整体指标通过；JSON 中 `best_device` 标识延迟最低的卡。

输出：

```text
dinov3/output/metric_real_time_segmentation_latency/real_time_segmentation_latency_result.json
```

### `metric_summary_report.py`

功能：汇总语义分割精度与实时延迟两项指标。

输入：

```text
dinov3/output/metric_semantic_segmentation_accuracy/semantic_segmentation_accuracy_result.json
dinov3/output/metric_real_time_segmentation_latency/real_time_segmentation_latency_result.json
```

输出：

```text
dinov3/output/metric_summary/metric_summary_report.json
```

### `run_metric_pipeline.sh`

功能：清理旧结果并依次运行标准两项指标 + 汇总。

执行逻辑：

1. 删除旧的输出目录：

```text
dinov3/output/metric_semantic_segmentation_accuracy
dinov3/output/metric_real_time_segmentation_latency
dinov3/output/metric_summary
```

2. 使用 `conda run -n dinov3` 依次运行：

```text
metric_semantic_segmentation_accuracy.py
metric_real_time_segmentation_latency.py
metric_summary_report.py
```
