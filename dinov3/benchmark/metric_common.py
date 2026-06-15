#!/usr/bin/env python3
"""Shared utilities for standardized DINOv3 segmentation metric scripts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
DINO_DIR = BENCHMARK_DIR.parent
REPO_ROOT = DINO_DIR.parent
for path in (str(REPO_ROOT), str(DINO_DIR), str(BENCHMARK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

RAW_IMAGE_DIR = Path("/data1/user/data/2026.05.10_机狗二次数据采集结果_标定/image")
RAW_PRED_DIR = RAW_IMAGE_DIR / "res_dinov3_whole"

SCANNET_GT_ROOT = Path("/data1/data/scannet/scannet_frames_25k")
SCANNET_PRED_ROOT = Path("/data1/data/scannet/scannet_frames_25k_dinov3_seg")
SCANNET_IMAGE_SUFFIX = ".jpg"
SCANNET_LABEL_SUFFIX = ".png"
SCANNET_PRED_SUFFIX = "_mask_id.png"
SCANNET_SAMPLE_IMAGES = 30
SCANNET_SAMPLE_SEED = 20260511
SCANNET_NUM_CLASSES = 41
SCANNET_IGNORE_LABELS = (0, 255)
SCANNET_EVAL_WORKERS = 16

OUTPUT_ROOT = DINO_DIR / "output"
SEM_ACC_DIR = OUTPUT_ROOT / "metric_semantic_segmentation_accuracy"
LATENCY_DIR = OUTPUT_ROOT / "metric_real_time_segmentation_latency"
SUMMARY_DIR = OUTPUT_ROOT / "metric_summary"

M2F_DIR = REPO_ROOT / "Mask2Former"

# ============ M2F 模型注册表 ============
# 切换被评测模型只需修改 MODEL_KEY (或设置环境变量 BENCHMARK_MODEL)。
# 评测脚本统一从 metric_common 导入 CHECKPOINT_HEAD / CONFIG_FILE / MODEL_NAME / NUM_CLASSES。
MODEL_REGISTRY = {
    "city": {
        "head": REPO_ROOT / "Mask2Former/output/dinov3_vitl_cityscapes_m2f/model_final.pth",
        "config_file": REPO_ROOT / "Mask2Former/configs/cityscapes/semantic-segmentation/maskformer2_dinov3_vitl_bs16_90k.yaml",
        "model_name": "cityscapes_vitl_m2f",
        "num_classes": 19,
    },
    "robotdog": {
        "head": REPO_ROOT / "Mask2Former/output/dinov3_vitl_robotdog_ade20k_distill/model_final.pth",
        "config_file": REPO_ROOT / "Mask2Former/configs/robotdog/maskformer2_dinov3_vitl_robotdog_ade20k.yaml",
        "model_name": "robotdog_vitl_ade20k_m2f_distill",
        "num_classes": 150,
    },
}

# 默认切换 robotdog; 环境变量 BENCHMARK_MODEL 可在不改源码情况下覆盖。
import os as _os
MODEL_KEY = _os.environ.get("BENCHMARK_MODEL", "robotdog")
if MODEL_KEY not in MODEL_REGISTRY:
    raise ValueError(f"未知 MODEL_KEY={MODEL_KEY!r}, 可选: {list(MODEL_REGISTRY)}")

_cfg = MODEL_REGISTRY[MODEL_KEY]
CHECKPOINT_HEAD = _cfg["head"]
CONFIG_FILE = _cfg["config_file"]
MODEL_NAME = _cfg["model_name"]
NUM_CLASSES = _cfg["num_classes"]

# 历史别名 (向后兼容现有引用)
CITY_HEAD = MODEL_REGISTRY["city"]["head"]
CITY_CONFIG_FILE = MODEL_REGISTRY["city"]["config_file"]
CITY_MODEL_NAME = MODEL_REGISTRY["city"]["model_name"]
CITY_NUM_CLASSES = MODEL_REGISTRY["city"]["num_classes"]
SEM_ACC_SAMPLE_IMAGES = 30
SEM_ACC_POINTS_PER_IMAGE = 20
SEM_ACC_SEED = 20260511
BENCHMARK_WARMUP = 1
BENCHMARK_ITERS = 5
TARGET_ACCURACY = 0.90
TARGET_LATENCY_MS = 200.0
DEFAULT_SUFFIX = ".png"
DEFAULT_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_images(image_dir: Path, suffix: str = DEFAULT_SUFFIX) -> list[Path]:
    files = sorted(path for path in image_dir.iterdir() if path.name.lower().endswith(suffix.lower()))
    if not files:
        raise FileNotFoundError(f"No images ending with {suffix} in {image_dir}")
    return files


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def load_palette(path: Path):
    import numpy as np
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return np.asarray(data.get("palette", []), dtype=np.uint8)


def colorize(mask, palette):
    import numpy as np

    return palette[np.clip(mask.astype(np.int64), 0, len(palette) - 1)]


def build_m2f_model(device: str, torch, *, head=None, config_file=None):
    """通用 M2F 模型构建器。

    默认读取顶层 CHECKPOINT_HEAD / CONFIG_FILE (由 MODEL_KEY 决定),
    亦可通过参数显式覆盖以同时评测多个模型。

    输入: ImageNet-normalized RGB tensor (B, 3, H, W)。
    输出: (B, num_classes, H, W) logits。
    """
    import torch.nn as nn
    import sys as _sys

    head = Path(head) if head is not None else CHECKPOINT_HEAD
    config_file = Path(config_file) if config_file is not None else CONFIG_FILE

    if str(M2F_DIR) not in _sys.path:
        _sys.path.insert(0, str(M2F_DIR))

    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.WEIGHTS = str(head)
    cfg.MODEL.DEVICE = device
    cfg.freeze()

    m2f_model = build_model(cfg)
    DetectionCheckpointer(m2f_model).load(cfg.MODEL.WEIGHTS)
    m2f_model.eval()

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    class M2FWrapper(nn.Module):
        """适配 benchmark 调用约定: 输入 ImageNet-normalized RGB, 输出 logits。"""

        def __init__(self):
            super().__init__()
            self.model = m2f_model

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: ImageNet-normalized RGB, (B, 3, H, W). M2F 期望 0-255 BGR。
            rgb_255 = x * imagenet_std + imagenet_mean
            rgb_255 = rgb_255.clamp(0, 1) * 255.0
            bgr_255 = rgb_255.flip(dims=[1])

            batch = []
            for i in range(bgr_255.shape[0]):
                batch.append({
                    "image": bgr_255[i],
                    "height": bgr_255.shape[2],
                    "width": bgr_255.shape[3],
                })
            outputs = self.model(batch)
            logits = torch.stack([o["sem_seg"] for o in outputs], dim=0)
            return logits

    wrapper = M2FWrapper().to(device).eval()
    return wrapper


def make_city_batch(files: list[Path], batch_size: int, device: str, torch, width: int | None = None, height: int | None = None):
    from PIL import Image
    import torchvision.transforms.functional as TF

    tensors = []
    for path in files[:batch_size]:
        img = Image.open(path).convert("RGB")
        if width is not None and height is not None:
            img = img.resize((width, height), Image.BILINEAR)
        tensor = TF.normalize(TF.to_tensor(img), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        tensors.append(tensor)
    while len(tensors) < batch_size:
        tensors.append(tensors[-1].clone())
    return torch.stack(tensors).to(device)


def build_m2f_runner(
    files: list[Path],
    batch_size: int,
    device: str,
    torch,
    width: int | None = None,
    height: int | None = None,
    *,
    head=None,
    config_file=None,
    model_name: str | None = None,
    num_classes: int | None = None,
):
    """构建 (infer_once, model_info) 二元组供 latency / fps 等 benchmark 调用。

    默认走 MODEL_KEY 决定的当前模型, 可通过 kwargs 显式指定其他模型。
    """
    model = build_m2f_model(device, torch, head=head, config_file=config_file)
    batch = make_city_batch(files, batch_size, device, torch, width=width, height=height)

    def infer_once() -> None:
        with torch.inference_mode():
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = model(batch)
                _ = logits.argmax(dim=1)

    model_info = {
        "model": model_name if model_name is not None else MODEL_NAME,
        "backbone": "vit_large_patch16_dinov3.lvd1689m",
        "num_classes": num_classes if num_classes is not None else NUM_CLASSES,
        "input_shape": list(batch.shape),
    }
    return infer_once, model_info


# 历史别名 (向后兼容)
build_cityscapes_model = build_m2f_model
build_city_runner = build_m2f_runner


def benchmark_runner(infer_once, batch_size: int, warmup: int, iters: int, device: str) -> dict:
    import time

    for _ in range(warmup):
        infer_once()
    synchronize(device)

    latencies_ms: list[float] = []
    start = time.perf_counter()
    for _ in range(iters):
        iter_start = time.perf_counter()
        infer_once()
        synchronize(device)
        latencies_ms.append((time.perf_counter() - iter_start) * 1000.0)
    elapsed = time.perf_counter() - start
    frames = iters * batch_size
    fps = frames / elapsed if elapsed > 0 else 0.0
    avg_batch_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    return {
        "timed_frames": frames,
        "elapsed_sec": elapsed,
        "fps": fps,
        "avg_batch_latency_ms": avg_batch_latency_ms,
        "avg_frame_latency_ms": avg_batch_latency_ms / batch_size if batch_size else 0.0,
        "p50_batch_latency_ms": sorted(latencies_ms)[len(latencies_ms) // 2] if latencies_ms else 0.0,
        "p95_batch_latency_ms": sorted(latencies_ms)[max(0, min(len(latencies_ms) - 1, round(0.95 * (len(latencies_ms) - 1))))] if latencies_ms else 0.0,
    }


def load_class_meta(path: Path, num_classes: int):
    import numpy as np
    import yaml

    if not path.exists():
        return [f"class_{i}" for i in range(num_classes)], None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    classes = list(data.get("classes") or [f"class_{i}" for i in range(num_classes)])
    if len(classes) < num_classes:
        classes.extend(f"class_{i}" for i in range(len(classes), num_classes))
    palette = data.get("palette")
    palette_array = np.asarray(palette, dtype=np.uint8) if palette is not None else None
    return classes[:num_classes], palette_array


def find_image_for_stem(image_dir: Path, stem: str, forced_suffix: str = "") -> Path | None:
    if forced_suffix:
        path = image_dir / f"{stem}{forced_suffix}"
        return path if path.exists() else None
    for suffix in DEFAULT_IMAGE_SUFFIXES:
        path = image_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def find_pairs(image_dir: Path, pred_dir: Path, pred_suffix: str, image_suffix: str = "") -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for pred_path in sorted(pred_dir.glob(f"*{pred_suffix}")):
        stem = pred_path.name[: -len(pred_suffix)]
        image_path = find_image_for_stem(image_dir, stem, image_suffix)
        if image_path is not None:
            pairs.append((stem, image_path, pred_path))
    return pairs


def find_scannet_pairs(
    gt_root: Path,
    pred_root: Path,
    image_suffix: str = SCANNET_IMAGE_SUFFIX,
    label_suffix: str = SCANNET_LABEL_SUFFIX,
    pred_suffix: str = SCANNET_PRED_SUFFIX,
) -> list[tuple[str, str, Path, Path, Path]]:
    pairs: list[tuple[str, str, Path, Path, Path]] = []
    if not gt_root.exists():
        raise FileNotFoundError(f"ScanNet GT root not found: {gt_root}")
    if not pred_root.exists():
        raise FileNotFoundError(f"ScanNet prediction root not found: {pred_root}")

    for scene_dir in sorted(path for path in gt_root.iterdir() if path.is_dir()):
        color_dir = scene_dir / "color"
        label_dir = scene_dir / "label"
        pred_scene_dir = pred_root / scene_dir.name
        if not color_dir.exists() or not label_dir.exists() or not pred_scene_dir.exists():
            continue

        for image_path in sorted(color_dir.glob(f"*{image_suffix}")):
            stem = image_path.stem
            gt_path = label_dir / f"{stem}{label_suffix}"
            pred_path = pred_scene_dir / f"{stem}{pred_suffix}"
            if gt_path.exists() and pred_path.exists():
                pairs.append((scene_dir.name, stem, image_path, gt_path, pred_path))
    return pairs


def read_id_mask(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array.astype(np.int64)


def resize_mask_to(mask, width: int, height: int):
    import numpy as np
    from PIL import Image

    if mask.shape == (height, width):
        return mask
    resized = Image.fromarray(mask.astype(np.int32)).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized).astype(np.int64)


def marker_color(index: int) -> tuple[int, int, int]:
    colors = [
        (255, 0, 0),
        (0, 180, 255),
        (255, 180, 0),
        (0, 220, 80),
        (200, 0, 255),
        (255, 80, 160),
    ]
    return colors[index % len(colors)]


def draw_cross(draw, x: int, y: int, color: tuple[int, int, int]) -> None:
    r = 7
    draw.line((x - r, y, x + r, y), fill=color, width=3)
    draw.line((x, y - r, x, y + r), fill=color, width=3)
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)


def crop_box(width: int, height: int, x: int, y: int, crop_size: int) -> tuple[int, int, int, int]:
    half = crop_size // 2
    left = max(0, min(width - crop_size, x - half)) if width >= crop_size else 0
    top = max(0, min(height - crop_size, y - half)) if height >= crop_size else 0
    return left, top, min(width, left + crop_size), min(height, top + crop_size)
