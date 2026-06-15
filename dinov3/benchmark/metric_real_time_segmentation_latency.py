#!/usr/bin/env python3
"""标准化实时语义分割延迟评测：在 GPU 0 与 GPU 2 上分别测一次。

要求启动时两张卡都可见 (默认情况下 CUDA_VISIBLE_DEVICES 不应屏蔽 0 / 2)。
结果同时记录卡的物理名称 (例如 'RTX A6000', 'RTX PRO 6000 Blackwell')。
"""

from __future__ import annotations

from metric_common import (
    BENCHMARK_ITERS,
    BENCHMARK_WARMUP,
    RAW_IMAGE_DIR,
    TARGET_LATENCY_MS,
    LATENCY_DIR,
    ensure_dir,
    build_m2f_runner,
    benchmark_runner,
    list_images,
    save_json,
)

RESULT_FILE = "real_time_segmentation_latency_result.json"
TARGET_GPU_IDS = (0, 2)


def evaluate_on_device(device: str, files) -> dict:
    import torch

    gpu_idx = int(device.split(":")[1]) if ":" in device else 0
    gpu_name = torch.cuda.get_device_name(gpu_idx)

    print(f"\n[INFO] === 评测 {device} ({gpu_name}) ===")
    # 把当前线程默认 device 切到目标卡, 让 torch.cuda.synchronize() 等无参 API 落到正确卡上
    torch.cuda.set_device(gpu_idx)
    with torch.cuda.device(gpu_idx):
        infer_once, model_info = build_m2f_runner(files, 1, device, torch)
        stats = benchmark_runner(infer_once, 1, BENCHMARK_WARMUP, BENCHMARK_ITERS, device)

    record = {
        "device": device,
        "gpu_index": gpu_idx,
        "gpu_name": gpu_name,
        "label": f"GPU {gpu_idx} ({gpu_name})",
        **model_info,
        **stats,
        "target_latency_ms": TARGET_LATENCY_MS,
        "pass": stats["avg_frame_latency_ms"] <= TARGET_LATENCY_MS,
    }

    # 释放显存, 给下一张卡的构建腾空间
    del infer_once
    torch.cuda.empty_cache()
    return record


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("本评测要求 CUDA")

    visible_count = torch.cuda.device_count()
    print(f"[INFO] 可见 CUDA 设备数: {visible_count}")
    for idx in range(visible_count):
        print(f"  cuda:{idx} -> {torch.cuda.get_device_name(idx)}")

    target_devices = [f"cuda:{idx}" for idx in TARGET_GPU_IDS if idx < visible_count]
    missing = [idx for idx in TARGET_GPU_IDS if idx >= visible_count]
    if missing:
        print(f"[WARN] 目标卡 {missing} 不可见 (CUDA_VISIBLE_DEVICES 屏蔽了?), 跳过")
    if not target_devices:
        raise RuntimeError(f"没有任何目标 GPU {TARGET_GPU_IDS} 可用")

    files = list_images(RAW_IMAGE_DIR)
    per_device_records = [evaluate_on_device(dev, files) for dev in target_devices]

    summary = {
        "metric": "real_time_segmentation_latency",
        "definition": "average per-frame latency from input to segmentation output, evaluated on multiple GPUs",
        "source_image_dir": str(RAW_IMAGE_DIR),
        "output_dir": str(LATENCY_DIR),
        "target_latency_ms": TARGET_LATENCY_MS,
        "evaluated_devices": [r["label"] for r in per_device_records],
        "per_device": per_device_records,
        "best_device": min(per_device_records, key=lambda r: r["avg_frame_latency_ms"])["label"],
        "pass": any(r["pass"] for r in per_device_records),
    }

    ensure_dir(LATENCY_DIR)
    out = LATENCY_DIR / RESULT_FILE
    save_json(out, summary)

    print()
    print(f"[INFO] result JSON: {out}")
    print()
    print(f"{'device':<40s} {'avg_frame_ms':>14s} {'p50_ms':>10s} {'p95_ms':>10s} {'fps':>8s}  pass")
    print("-" * 100)
    for r in per_device_records:
        print(
            f"{r['label']:<40s} {r['avg_frame_latency_ms']:>14.2f} "
            f"{r['p50_batch_latency_ms']:>10.2f} {r['p95_batch_latency_ms']:>10.2f} "
            f"{r['fps']:>8.2f}  {r['pass']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
