#!/usr/bin/env python3
"""Evaluate semantic segmentation on the scannet_evaluate evaluation set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import metric_semantic_segmentation_accuracy as base
from metric_common import (
    OUTPUT_ROOT,
    SCANNET_NUM_CLASSES,
    TARGET_ACCURACY,
    ensure_dir,
    find_scannet_pairs,
    load_class_meta,
    save_json,
)

DATASET_NAME = "scannet_evaluate"
DATASET_ROOT = Path("/data1/data/scannet/scannet_evaluate")
GT_ROOT = DATASET_ROOT / "frames"
PRED_ROOT = DATASET_ROOT / "pred"
OUT_DIR = OUTPUT_ROOT / "metric_semantic_segmentation_accuracy_scannet_evaluate"
PALETTE_FILE = Path("/data1/user/Dense-Object-level-Mapping/dinov3/yaml/scannet_nyu40.yaml")


def main() -> int:
    import numpy as np

    class_names, _ = load_class_meta(PALETTE_FILE, SCANNET_NUM_CLASSES)
    pairs = find_scannet_pairs(GT_ROOT, PRED_ROOT)
    if not pairs:
        raise FileNotFoundError(f"No matched {DATASET_NAME} pairs found in {GT_ROOT} and {PRED_ROOT}")

    ensure_dir(OUT_DIR)
    full_confusion = np.zeros((SCANNET_NUM_CLASSES, SCANNET_NUM_CLASSES), dtype=np.int64)
    rows = []
    for pair in pairs:
        row, conf = base.evaluate_record(pair)
        rows.append(row)
        full_confusion += conf

    full_summary = base.metrics_from_confusion(full_confusion)
    csv_path = OUT_DIR / "metric_semantic_segmentation_accuracy_scannet_evaluate_full_image_metrics.csv"
    base.write_csv(
        csv_path,
        rows,
        ["scene_name", "image_stem", "image_path", "gt_path", "pred_path", "valid_pixels", "correct_pixels", "pixel_accuracy", "mIoU"],
    )

    dataset_meta_path = DATASET_ROOT / "dataset_metadata.json"
    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8")) if dataset_meta_path.exists() else {}
    result = {
        "metric": "semantic_segmentation_accuracy_scannet_evaluate",
        "dataset_name": DATASET_NAME,
        "definition": "evaluate semantic segmentation against NYU40 ground truth on the scannet_evaluate evaluation set",
        "source_gt_root": str(GT_ROOT),
        "source_pred_root": str(PRED_ROOT),
        "total_images": len(pairs),
        "dataset_metadata": dataset_meta,
        "full_image_metric": {
            **full_summary,
            "target_pixel_accuracy": TARGET_ACCURACY,
            "pass": full_summary["pixel_accuracy"] is not None and full_summary["pixel_accuracy"] >= TARGET_ACCURACY,
        },
        "per_image_csv_full": str(csv_path),
        "class_names": class_names,
    }
    out = OUT_DIR / "semantic_segmentation_accuracy_scannet_evaluate_result.json"
    save_json(out, result)
    print(f"[INFO] result JSON: {out}")
    print(f"[INFO] full CSV: {csv_path}")
    print(f"[INFO] total images: {len(pairs)}")
    print(f"[INFO] pixel accuracy: {full_summary['pixel_accuracy_percent']:.4f}%")
    print(f"[INFO] mIoU: {full_summary['mIoU_percent']:.4f}%")
    print(f"[INFO] pass: {result['full_image_metric']['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
