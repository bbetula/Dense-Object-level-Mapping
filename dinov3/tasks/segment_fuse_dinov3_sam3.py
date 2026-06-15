"""
DINOv3 + SAM3 融合语义分割

替换原 segment_images_batch_v2.py 的输出：
  对每帧图像，用 SAM3 的精细实例掩码边界修正 DINOv3 的语义标签，
  输出格式与原脚本完全一致（_mask_id.png / _mask.png / _original.png / _overlap.png），
  可直接接入后续 pointcloud_colorized_v2_normal.py 管线。

原理：
  DINOv3 (Mask2Former) 给出 ADE20K 150 类逐像素语义标签，但边界粗糙；
  SAM3 给出类无关的实例掩码，边界像素级精确。
  融合策略：在 SAM3 的每个实例区域内，用 DINOv3 多数投票确定语义类别，
  再用 SAM3 的边界覆盖 DINOv3 → 语义类别不变，边界大幅精化。
"""

import os
import sys
import json
import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPT_DIR.parent))
from fused_construction.class_statics_config import IMAGES_DIR, ADE20K_CATEGORIES

# ===== 配置区 =====
DINOV3_DIR = "/data1/user/data/fastlivo_output_qs2_03.17/image/res_dinov3_whole"
SAM3_DIR = "/data1/user/Dense-Object-level-Mapping/sam3/output"
OUTPUT_DIR = "/data1/user/data/fastlivo_output_qs2_03.17/image/res_fused_dinov3_sam3"

PALETTE_FILE = os.path.join(SCRIPT_DIR, "yaml", "ADE20k.yaml")

SCORE_THRESHOLD = 0.3
MIN_MASK_AREA = 100
PROGRESS_INTERVAL = 50

THING_CATEGORIES = {
    "car", "van", "truck", "bus", "bicycle", "minibike",
    "boat", "ship", "airplane",
    "person", "animal",
    "pole", "streetlight", "traffic light", "light",
    "signboard", "flag",
    "bench", "booth", "ashcan",
    "box", "barrel", "basket", "bottle", "pot", "vase",
    "sculpture", "fountain", "tent",
    "tree", "plant", "palm", "flower",
    "chair", "swivel chair", "armchair", "seat", "stool",
    "sofa", "ottoman", "cushion",
    "table", "coffee table", "pool table", "counter", "countertop", "desk",
    "bed", "pillow", "blanket", "cradle",
    "cabinet", "chest of drawers", "wardrobe", "bookcase", "shelf", "buffet",
    "television receiver", "monitor", "screen", "crt screen", "computer",
    "refrigerator", "oven", "stove", "microwave", "dishwasher", "washer",
    "sink", "bathtub", "toilet", "shower",
    "lamp", "chandelier",
    "mirror", "painting", "clock",
    "bag", "fan", "plaything", "towel", "tray",
}
# ==================================

ADE20K_NAMES = list(ADE20K_CATEGORIES.keys())
NUM_CLASSES = len(ADE20K_NAMES)
THING_IDS = {i for i, name in enumerate(ADE20K_NAMES) if name in THING_CATEGORIES}


def load_palette(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array(data["palette"], dtype=np.uint8)


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    indices = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[indices]


def fuse_single_frame(
    dinov3_mask_id: np.ndarray,
    sam3_instance_map: np.ndarray,
    sam3_meta: List[dict],
) -> np.ndarray:
    """
    用 SAM3 实例边界精化 DINOv3 语义标签。

    对每个 SAM3 实例区域：
      1. 统计区域内 DINOv3 标签的多数类别
      2. 若属于 thing 类别且置信度足够，用多数类别统一覆盖该区域
      → 边界由 SAM3 决定（精确），语义由 DINOv3 多数投票决定（稳定）
    """
    fused = dinov3_mask_id.copy().astype(np.int16)

    sorted_meta = sorted(sam3_meta, key=lambda x: x.get("score", 0), reverse=True)

    for det in sorted_meta:
        score = det.get("score", 0)
        if score < SCORE_THRESHOLD:
            continue

        inst_id = det["instance_id"]
        mask = sam3_instance_map == inst_id
        area = int(mask.sum())
        if area < MIN_MASK_AREA:
            continue

        labels_in_mask = dinov3_mask_id[mask]
        valid = labels_in_mask < NUM_CLASSES
        if not valid.any():
            continue

        counts = np.bincount(labels_in_mask[valid].astype(int), minlength=NUM_CLASSES)
        majority_id = int(np.argmax(counts))

        if majority_id not in THING_IDS:
            continue

        fused[mask] = majority_id

    return fused.astype(np.uint8)


def main() -> None:
    palette = load_palette(PALETTE_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dinov3_mask_files = sorted(Path(DINOV3_DIR).glob("*_mask_id.png"))
    if not dinov3_mask_files:
        print(f"在 {DINOV3_DIR} 中未找到 _mask_id.png 文件")
        return

    sam3_frames = set(os.listdir(SAM3_DIR)) if os.path.isdir(SAM3_DIR) else set()

    total = len(dinov3_mask_files)
    fused_count = 0
    passthrough_count = 0

    print(f"DINOv3 + SAM3 融合语义分割")
    print(f"  DINOv3 帧数: {total}")
    print(f"  SAM3  帧数: {len(sam3_frames)}")
    print(f"  输出目录:   {OUTPUT_DIR}")
    print()

    for idx, mask_id_path in enumerate(dinov3_mask_files):
        timestamp = mask_id_path.stem.replace("_mask_id", "")

        dinov3_mask_id = cv2.imread(str(mask_id_path), cv2.IMREAD_UNCHANGED)
        if dinov3_mask_id is None:
            continue
        if dinov3_mask_id.ndim == 3:
            dinov3_mask_id = dinov3_mask_id[:, :, 0]

        sam3_frame_dir = os.path.join(SAM3_DIR, timestamp)
        sam3_inst_path = os.path.join(sam3_frame_dir, "instance_map.png")
        sam3_json_path = os.path.join(sam3_frame_dir, "results.json")

        if (timestamp in sam3_frames
                and os.path.exists(sam3_inst_path)
                and os.path.exists(sam3_json_path)):
            sam3_map = np.array(
                cv2.imread(sam3_inst_path, cv2.IMREAD_UNCHANGED)
            ).astype(np.int32)
            if sam3_map.ndim == 3:
                sam3_map = sam3_map[:, :, 0]

            with open(sam3_json_path, "r", encoding="utf-8") as f:
                sam3_meta = json.load(f)

            if sam3_map.shape != dinov3_mask_id.shape:
                sam3_map = cv2.resize(
                    sam3_map.astype(np.uint16),
                    (dinov3_mask_id.shape[1], dinov3_mask_id.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)

            fused_mask = fuse_single_frame(dinov3_mask_id, sam3_map, sam3_meta)
            fused_count += 1
        else:
            fused_mask = dinov3_mask_id.copy()
            passthrough_count += 1

        # 输出 _mask_id.png
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{timestamp}_mask_id.png"), fused_mask)

        # 输出 _mask.png（彩色）
        colored_mask = colorize_mask(fused_mask, palette)
        colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{timestamp}_mask.png"), colored_mask_bgr)

        # 输出 _original.png
        orig_path = os.path.join(DINOV3_DIR, f"{timestamp}_original.png")
        if os.path.exists(orig_path):
            orig_img = cv2.imread(orig_path)
        else:
            img_path = os.path.join(IMAGES_DIR, f"{timestamp}.png")
            orig_img = cv2.imread(img_path) if os.path.exists(img_path) else None

        if orig_img is not None:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{timestamp}_original.png"), orig_img)
            overlay = cv2.addWeighted(orig_img, 1, colored_mask_bgr, 0.5, 0)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{timestamp}_overlap.png"), overlay)

        if (idx + 1) % PROGRESS_INTERVAL == 0 or idx == total - 1:
            print(f"  [{idx + 1}/{total}] 融合: {fused_count}, 直通: {passthrough_count}")

    print(f"\n完成！")
    print(f"  融合帧: {fused_count} (SAM3 精化边界)")
    print(f"  直通帧: {passthrough_count} (无 SAM3 结果，保留原 DINOv3)")
    print(f"  输出:   {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
