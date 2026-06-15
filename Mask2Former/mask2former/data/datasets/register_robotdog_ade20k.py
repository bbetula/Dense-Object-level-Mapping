"""注册机狗采集数据集 (使用 DINOv3 ViT-7B + ADE20K M2F 教师伪标签)。

数据布局:
  IMAGE_DIR/{stem}.png           -> 源图像
  IMAGE_DIR/res_dinov3_whole/{stem}_mask_id.png  -> 教师伪标签 (uint8, 0..149)

train/val 按种子 42 做 90/10 随机切分。"""

import os
import sys
from pathlib import Path
from typing import List, Dict

from detectron2.data import DatasetCatalog, MetadataCatalog

# 复用 ADE20K 150 类别名 (与教师推理使用的调色板顺序一致)
_DINO_DIR = "/data1/user/Dense-Object-level-Mapping/dinov3"
if _DINO_DIR not in sys.path:
    sys.path.insert(0, _DINO_DIR)
from utils.ADE20k_2_mydataset import ADE20KDataset  # noqa: E402

ROBOTDOG_IMAGE_DIR = Path("/data1/user/data/2026.05.10_机狗二次数据采集结果_标定/image")
ROBOTDOG_LABEL_DIR = ROBOTDOG_IMAGE_DIR / "res_dinov3_whole"

MASK_ID_SUFFIX = "_mask_id.png"
SPLIT_SEED = 42
VAL_RATIO = 0.10

# 训练时图像保持 640x512, 后续 crop 由 dataset mapper 处理
IMG_HEIGHT = 512
IMG_WIDTH = 640

ADE20K_150_CLASSES = [c.strip() for c in ADE20KDataset.METAINFO["classes"]]
assert len(ADE20K_150_CLASSES) == 150
ADE20K_150_PALETTE = ADE20KDataset.METAINFO["palette"]


def _build_dataset_dicts(split: str) -> List[Dict]:
    import random

    mask_files = sorted(p for p in ROBOTDOG_LABEL_DIR.glob(f"*{MASK_ID_SUFFIX}"))
    if not mask_files:
        raise FileNotFoundError(f"No *_mask_id.png in {ROBOTDOG_LABEL_DIR}")

    pairs = []
    for mask_path in mask_files:
        stem = mask_path.name[: -len(MASK_ID_SUFFIX)]
        image_path = ROBOTDOG_IMAGE_DIR / f"{stem}.png"
        if image_path.exists():
            pairs.append((str(image_path), str(mask_path)))

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_RATIO))
    if split == "train":
        selected = pairs[n_val:]
    elif split == "val":
        selected = pairs[:n_val]
    else:
        raise ValueError(f"Unknown split: {split}")

    return [
        {
            "file_name": img,
            "sem_seg_file_name": mask,
            "height": IMG_HEIGHT,
            "width": IMG_WIDTH,
        }
        for img, mask in selected
    ]


def register_all_robotdog_ade20k():
    for split in ("train", "val"):
        name = f"robotdog_ade20k_distill_{split}"
        if name in DatasetCatalog.list():
            continue
        DatasetCatalog.register(name, lambda s=split: _build_dataset_dicts(s))
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_150_CLASSES,
            stuff_colors=ADE20K_150_PALETTE,
            image_root=str(ROBOTDOG_IMAGE_DIR),
            sem_seg_root=str(ROBOTDOG_LABEL_DIR),
            evaluator_type="sem_seg",
            ignore_label=255,
        )


register_all_robotdog_ade20k()
