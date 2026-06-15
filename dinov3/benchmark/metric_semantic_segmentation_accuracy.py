#!/usr/bin/env python3
"""
GT  = res_dinov3_whole/*_mask_id.png        
Pred = res_robotdog_distill/*_mask_id.png   

仅 robotdog 机狗采集的 2232 张图，按 stem 一一对应。
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from metric_common import (
    SEM_ACC_DIR,
    TARGET_ACCURACY,
    ensure_dir,
    save_json,
)

# ============ 配置 ============
GT_ROOT = Path("/data1/user/data/2026.05.10_机狗二次数据采集结果_标定/image/res_dinov3_whole")
PRED_ROOT = Path("/data1/user/data/2026.05.10_机狗二次数据采集结果_标定/image/res_robotdog_distill")
MASK_ID_SUFFIX = "_mask_id.png"

NUM_CLASSES = 150  # ADE20K
IGNORE_LABELS = (255,)  # 教师伪标签中无 255, 但保留兼容
EVAL_WORKERS = 16

PALETTE_FILE = Path("/data1/user/Dense-Object-level-Mapping/dinov3/yaml/ADE20k.yaml")
RESULT_FILE = "semantic_segmentation_accuracy_result.json"
FULL_CSV_FILE = "metric_semantic_segmentation_accuracy_full_image_metrics.csv"


def find_pairs() -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    if not GT_ROOT.exists():
        raise FileNotFoundError(f"GT root not found: {GT_ROOT}")
    if not PRED_ROOT.exists():
        raise FileNotFoundError(f"Pred root not found: {PRED_ROOT}")

    for gt_path in sorted(GT_ROOT.glob(f"*{MASK_ID_SUFFIX}")):
        stem = gt_path.name[: -len(MASK_ID_SUFFIX)]
        pred_path = PRED_ROOT / f"{stem}{MASK_ID_SUFFIX}"
        if pred_path.exists():
            pairs.append((stem, gt_path, pred_path))
    return pairs


def read_mask_fast(path: Path):
    import cv2
    import numpy as np

    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if array.ndim == 3:
        array = array[:, :, 0]
    return array.astype(np.int64)


def resize_mask_fast(mask, width: int, height: int):
    import cv2

    if mask.shape == (height, width):
        return mask
    return cv2.resize(mask.astype("int32"), (width, height), interpolation=cv2.INTER_NEAREST).astype("int64")


def confusion_matrix_for_pair(pred: Path, gt: Path):
    import numpy as np

    pred_mask = read_mask_fast(pred)
    gt_mask = read_mask_fast(gt)
    if pred_mask.shape != gt_mask.shape:
        pred_mask = resize_mask_fast(pred_mask, gt_mask.shape[1], gt_mask.shape[0])

    valid = np.ones_like(gt_mask, dtype=bool)
    for ignore_label in IGNORE_LABELS:
        valid &= gt_mask != ignore_label

    pred_valid = np.clip(pred_mask[valid].astype(np.int64), 0, NUM_CLASSES - 1)
    gt_valid = np.clip(gt_mask[valid].astype(np.int64), 0, NUM_CLASSES - 1)
    if pred_valid.size == 0:
        return np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64), 0, 0

    correct_pixels = int((pred_valid == gt_valid).sum())
    valid_pixels = int(gt_valid.size)
    conf = np.bincount(
        NUM_CLASSES * gt_valid + pred_valid,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    return conf.astype(np.int64), valid_pixels, correct_pixels


def metrics_from_confusion(confusion) -> dict:
    import numpy as np

    total_valid_pixels = int(confusion.sum())
    total_correct_pixels = int(np.trace(confusion))
    pixel_accuracy = total_correct_pixels / total_valid_pixels if total_valid_pixels > 0 else None

    true_positive = np.diag(confusion).astype(np.float64)
    pred_total = confusion.sum(axis=0).astype(np.float64)
    gt_total = confusion.sum(axis=1).astype(np.float64)
    union = pred_total + gt_total - true_positive
    iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan, dtype=np.float64), where=union > 0)
    acc = np.divide(true_positive, gt_total, out=np.full_like(true_positive, np.nan, dtype=np.float64), where=gt_total > 0)
    # ADE20K: 0..149 全部参与, 不像 NYU40 那样跳过 idx 0
    miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else None
    macc = float(np.nanmean(acc)) if np.any(~np.isnan(acc)) else None
    return {
        "pixel_accuracy": pixel_accuracy,
        "pixel_accuracy_percent": pixel_accuracy * 100 if pixel_accuracy is not None else None,
        "mIoU": miou,
        "mIoU_percent": miou * 100 if miou is not None else None,
        "mAcc": macc,
        "mAcc_percent": macc * 100 if macc is not None else None,
        "total_valid_pixels": total_valid_pixels,
        "total_correct_pixels": total_correct_pixels,
        "num_classes_present": int((gt_total > 0).sum()),
        "class_iou": iou.tolist(),
        "class_acc": acc.tolist(),
    }


def pair_metrics_from_confusion(confusion) -> dict:
    import numpy as np

    total_valid_pixels = int(confusion.sum())
    total_correct_pixels = int(np.trace(confusion))
    pixel_accuracy = total_correct_pixels / total_valid_pixels if total_valid_pixels > 0 else None
    true_positive = np.diag(confusion).astype(np.float64)
    pred_total = confusion.sum(axis=0).astype(np.float64)
    gt_total = confusion.sum(axis=1).astype(np.float64)
    union = pred_total + gt_total - true_positive
    iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan, dtype=np.float64), where=union > 0)
    miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else None
    return {
        "pixel_accuracy": pixel_accuracy,
        "mIoU": miou,
    }


def load_ade20k_class_names() -> list[str]:
    import sys as _sys
    dino_dir = "/data1/user/Dense-Object-level-Mapping/dinov3"
    if dino_dir not in _sys.path:
        _sys.path.insert(0, dino_dir)
    from utils.ADE20k_2_mydataset import ADE20KDataset  # noqa: E402
    return [c.strip() for c in ADE20KDataset.METAINFO["classes"]]


def evaluate_record(pair: tuple[str, Path, Path]) -> tuple[dict, "np.ndarray"]:
    stem, gt_path, pred_path = pair
    conf, valid_pixels, correct_pixels = confusion_matrix_for_pair(pred_path, gt_path)
    pair_metrics = pair_metrics_from_confusion(conf)
    row = {
        "image_stem": stem,
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "valid_pixels": valid_pixels,
        "correct_pixels": correct_pixels,
        "pixel_accuracy": pair_metrics["pixel_accuracy"],
        "mIoU": pair_metrics["mIoU"],
    }
    return row, conf


def main() -> int:
    import numpy as np

    class_names = load_ade20k_class_names()
    assert len(class_names) == NUM_CLASSES, f"ADE20K class count != {NUM_CLASSES}"

    pairs = find_pairs()
    if not pairs:
        raise FileNotFoundError(f"No matched GT/pred pairs found in {GT_ROOT} and {PRED_ROOT}")

    ensure_dir(SEM_ACC_DIR)
    full_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    full_csv = SEM_ACC_DIR / FULL_CSV_FILE
    fieldnames = ["image_stem", "gt_path", "pred_path", "valid_pixels", "correct_pixels", "pixel_accuracy", "mIoU"]

    print(f"[INFO] 配对成功 {len(pairs)} 张图, 开始评估 (workers={EVAL_WORKERS})...")
    with full_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as executor:
            for i, (row, conf) in enumerate(executor.map(evaluate_record, pairs, chunksize=16), start=1):
                writer.writerow(row)
                full_confusion += conf
                if i % 200 == 0:
                    print(f"  已处理 {i}/{len(pairs)}")

    full_summary = metrics_from_confusion(full_confusion)

    # 重要类别 (frequency-weighted) 一致性: 用 GT pixel 数加权 IoU
    gt_total = full_confusion.sum(axis=1).astype(np.float64)
    freq = gt_total / gt_total.sum() if gt_total.sum() > 0 else np.zeros_like(gt_total)
    class_iou = np.asarray(full_summary["class_iou"], dtype=np.float64)
    fwiou_mask = ~np.isnan(class_iou)
    fwiou = float((freq[fwiou_mask] * class_iou[fwiou_mask]).sum()) if fwiou_mask.any() else None

    # Top-K 频繁类别的 IoU 列表 (机狗场景实际出现的主要类别)
    top_idx = np.argsort(-gt_total)[:15]
    top_classes = [
        {
            "class_id": int(idx),
            "class_name": class_names[idx],
            "gt_pixel_share": float(freq[idx]),
            "IoU": (None if np.isnan(class_iou[idx]) else float(class_iou[idx])),
            "IoU_percent": (None if np.isnan(class_iou[idx]) else float(class_iou[idx]) * 100),
        }
        for idx in top_idx
        if gt_total[idx] > 0
    ]

    result = {
        "metric": "semantic_segmentation_accuracy_student_vs_teacher",
        "definition": "student (M2F + DINOv3 ViT-L distilled) consistency vs teacher (DINOv3 ViT-7B + ADE20K M2F) pseudo-labels",
        "gt_root": str(GT_ROOT),
        "pred_root": str(PRED_ROOT),
        "num_classes": NUM_CLASSES,
        "total_images": len(pairs),
        "full_image_metric": {
            **full_summary,
            "frequency_weighted_IoU": fwiou,
            "frequency_weighted_IoU_percent": (fwiou * 100 if fwiou is not None else None),
            "target_pixel_accuracy": TARGET_ACCURACY,
            "pass": full_summary["pixel_accuracy"] is not None and full_summary["pixel_accuracy"] >= TARGET_ACCURACY,
        },
        "top_classes_by_pixel_share": top_classes,
        "per_image_csv_full": str(full_csv),
        "class_names": class_names,
    }

    out = SEM_ACC_DIR / RESULT_FILE
    save_json(out, result)
    print()
    print(f"[INFO] result JSON: {out}")
    print(f"[INFO] full CSV: {full_csv}")
    print()
    print(f"=== 整体指标 ({len(pairs)} 张图) ===")
    print(f"  Pixel Accuracy : {full_summary['pixel_accuracy_percent']:.4f}%")
    print(f"  mIoU (150)     : {full_summary['mIoU_percent']:.4f}%")
    print(f"  mAcc (150)     : {full_summary['mAcc_percent']:.4f}%")
    print(f"  fwIoU          : {fwiou * 100:.4f}%" if fwiou is not None else "  fwIoU          : None")
    print(f"  实际出现类别数 : {full_summary['num_classes_present']} / {NUM_CLASSES}")
    print()
    print(f"=== Top {len(top_classes)} 频繁类别 IoU ===")
    for item in top_classes:
        iou_str = f"{item['IoU_percent']:.2f}%" if item["IoU_percent"] is not None else "  n/a"
        print(f"  {item['class_id']:3d} {item['class_name']:<18s}  GT占比 {item['gt_pixel_share']*100:6.2f}%  IoU {iou_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
