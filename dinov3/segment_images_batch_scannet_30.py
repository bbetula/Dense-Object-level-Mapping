"""
DINOv3 语义分割 - ScanNet Val 30 场景批量推理

输入：/data1/data/scannet/scans/<scene>/color/*.jpg
      场景列表由 scannet_val.txt 指定（30 个场景）
输出：/data1/data/scannet/scans/<scene>/prediction/<frame>_mask.png
      prediction/ 与 color/ 同级，保存彩色 mask（NYU40 色盘）
"""

import sys
import os

REPO_DIR = "/data1/user/Dense-Object-level-Mapping/dinov3"
sys.path.append(REPO_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.v2 as v2
from functools import partial
from dinov3.eval.segmentation.inference import make_inference
from hubconf import dinov3_vit7b16_ms
import cv2
import numpy as np
import yaml

# ============================================================
# 配置区
# ============================================================
SCANS_ROOT = "/data1/data/scannet/scans"
VAL_SPLIT  = "/data1/data/scannet/experiment/splits/scannet_val.txt"

BACKBONE_DIR = "checkpoint/dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
HEAD_DIR     = "checkpoint/dinov3_pretrained/DINOv3 Adapters/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
N_OUTPUT_CHANNELS = 150

PALETTE_FILE = os.path.join(REPO_DIR, "yaml", "scannet_nyu40.yaml")

IMG_SIZE   = 1024
BATCH_SIZE = 8
FRAME_STEP = 1   # 每隔 N 帧处理一帧，1 = 全部帧
# ============================================================

# ADE20K (0-149) → NYU40 (0-40) 映射
ADE20K_TO_NYU40 = np.zeros(150, dtype=np.uint8)
_mapping = {
    0: 1,    3: 2,    5: 22,   7: 4,    8: 9,
    10: 3,   12: 31,  14: 8,   15: 7,   18: 16,
    19: 5,   22: 11,  23: 6,   24: 15,  27: 19,
    28: 20,  30: 5,   31: 5,   33: 14,  35: 3,
    36: 35,  37: 36,  38: 38,  39: 18,  41: 29,
    42: 38,  44: 17,  45: 12,  47: 34,  50: 24,
    53: 38,  55: 39,  56: 7,   57: 18,  58: 8,
    59: 38,  62: 10,  63: 13,  64: 7,   65: 33,
    67: 23,  70: 12,  75: 5,   81: 27,  82: 35,
    85: 35,  89: 25,  92: 21,  93: 38,  95: 38,
    100: 11, 107: 39, 110: 39, 115: 37, 118: 39,
    124: 39, 129: 39, 130: 25, 134: 35, 141: 25,
    143: 25, 144: 30, 145: 28, 146: 39,
}
for ade_id, nyu_id in _mapping.items():
    ADE20K_TO_NYU40[ade_id] = nyu_id


def make_transform(resize_size: int = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def map_ade20k_to_nyu40(ade20k_mask: np.ndarray) -> np.ndarray:
    return ADE20K_TO_NYU40[np.clip(ade20k_mask, 0, 149)]


def load_palette(palette_file: str) -> np.ndarray:
    with open(palette_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array(data["palette"], dtype=np.uint8)


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    indices = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[indices]


# ============================================================
# GPU 检查 & 模型加载
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("[WARN] CUDA not available, using CPU.")

print(f"[INFO] Loading model...")
segmentor = dinov3_vit7b16_ms(
    pretrained=True,
    weights=HEAD_DIR,
    backbone_weights=BACKBONE_DIR,
    check_hash=False,
)
segmentor.to(DEVICE)
segmentor.eval()

transform = make_transform(IMG_SIZE)
palette_array = load_palette(PALETTE_FILE)

# ============================================================
# 读取 val 场景列表
# ============================================================
with open(VAL_SPLIT, "r") as f:
    val_scenes = [line.strip() for line in f if line.strip()]
print(f"[INFO] Val scenes: {len(val_scenes)}")

total_images = 0
for scene_name in val_scenes:
    scene_color_dir = os.path.join(SCANS_ROOT, scene_name, "color")
    if not os.path.isdir(scene_color_dir):
        print(f"[WARN] {scene_name}: color/ 不存在，跳过")
        continue

    all_files = sorted(
        [f for f in os.listdir(scene_color_dir) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    if FRAME_STEP > 1:
        all_files = all_files[::FRAME_STEP]
    if not all_files:
        print(f"[WARN] {scene_name}: 无 jpg 文件，跳过")
        continue

    pred_dir = os.path.join(SCANS_ROOT, scene_name, "prediction")
    os.makedirs(pred_dir, exist_ok=True)

    print(f"\n[SCENE] {scene_name}: {len(all_files)} images -> {pred_dir}")
    total_images += len(all_files)

    for i in range(0, len(all_files), BATCH_SIZE):
        batch_files = all_files[i : i + BATCH_SIZE]
        pil_images = []
        orig_sizes = []

        for img_name in batch_files:
            img = Image.open(os.path.join(scene_color_dir, img_name)).convert("RGB")
            pil_images.append(img)
            orig_sizes.append(img.size)

        batch_img = torch.stack([transform(img) for img in pil_images]).to(DEVICE)

        with torch.inference_mode():
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_batch = make_inference(
                    batch_img, segmentor,
                    inference_mode="whole", decoder_head_type="m2f",
                    rescale_to=(IMG_SIZE, IMG_SIZE),
                    n_output_channels=N_OUTPUT_CHANNELS,
                    output_activation=partial(torch.nn.functional.softmax, dim=1),
                )
                seg_maps = pred_batch.argmax(dim=1, keepdim=True)

        for idx, img_name in enumerate(batch_files):
            base = os.path.splitext(img_name)[0]
            orig_w, orig_h = orig_sizes[idx]
            mask = seg_maps[idx, 0].cpu().numpy().astype(np.uint8)
            nyu40_mask = map_ade20k_to_nyu40(mask)
            nyu40_mask_resized = cv2.resize(nyu40_mask, (orig_w, orig_h),
                                            interpolation=cv2.INTER_NEAREST)
            colored_mask = colorize_mask(nyu40_mask_resized, palette_array)
            colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(pred_dir, f"{base}_mask.png"),
                        colored_mask_bgr)

        done = min(i + BATCH_SIZE, len(all_files))
        print(f"  [{done}/{len(all_files)}]", end="\r")

    print(f"  [{len(all_files)}/{len(all_files)}] done")

print(f"\n[INFO] 全部完成，共处理 {total_images} 张图片")
