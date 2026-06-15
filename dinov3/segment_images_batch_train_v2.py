import sys
import os
from pathlib import Path

os.environ['TORCH_HUB_DISABLE_DOWNLOAD'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

sys.path.append("/data1/user/Dense-Object-level-Mapping/Mask2Former")

from PIL import Image
import torch
import cv2
import numpy as np
import yaml

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from mask2former import add_maskformer2_config

# 导入路径
from fused_construction.class_statics_config import IMAGES_DIR

SAVE_MASK_ID = True  # 保存单通道类别ID图（_mask_id.png），供 3D 多数投票使用


def load_palette(palette_file):
    with open(palette_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array(data["palette"], dtype=np.uint8)


def colorize_mask(mask, palette):
    indices = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[indices]


PALETTE_FILE = os.path.join(SCRIPT_DIR, "yaml", "ADE20k.yaml")
palette_array = load_palette(PALETTE_FILE)

# ── model ──────────────────────────────────────────────────────────
CONFIG_FILE = "/data1/user/Dense-Object-level-Mapping/Mask2Former/configs/robotdog/maskformer2_dinov3_vitl_robotdog_ade20k.yaml"
CHECKPOINT  = "/data1/user/Dense-Object-level-Mapping/Mask2Former/output/dinov3_vitl_robotdog_ade20k_distill/model_final.pth"

print("构建 Mask2Former + DINOv3 ViT-L 学生模型...")
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file(CONFIG_FILE)
cfg.MODEL.WEIGHTS = CHECKPOINT
cfg.freeze()

model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()
print(f"已加载 checkpoint: {CHECKPOINT}")

# ── dataset ────────────────────────────────────────────────────────
IMAGE_ROOT = IMAGES_DIR
all_files = sorted([f for f in os.listdir(IMAGE_ROOT) if f.lower().endswith(".png")])
SAVE_DIR   = os.path.join(IMAGES_DIR, "res_robotdog_distill")
batch_size = 8

os.makedirs(SAVE_DIR, exist_ok=True)

for i in range(0, len(all_files), batch_size):
    batch_files = all_files[i : i + batch_size]
    pil_images, orig_sizes, inputs = [], [], []
    for img_name in batch_files:
        img = Image.open(os.path.join(IMAGE_ROOT, img_name)).convert("RGB")
        pil_images.append(img)
        orig_sizes.append(img.size)  # (W, H)

        img_array = np.array(img)
        h, w = img_array.shape[:2]
        img_bgr = img_array[:, :, ::-1].copy()
        inputs.append({
            "image": torch.as_tensor(img_bgr.transpose(2, 0, 1).astype("float32")),
            "height": h,
            "width": w,
        })

    with torch.inference_mode():
        outputs = model(inputs)

    for idx, img_name in enumerate(batch_files):
        base = img_name.split('.')[0]
        orig_w, orig_h = orig_sizes[idx]
        mask = outputs[idx]["sem_seg"].argmax(dim=0).cpu().numpy()

        colored_mask = colorize_mask(mask, palette_array)
        # 学生模型 sem_seg 输出已是原图分辨率，下面 resize 仅做保险
        if colored_mask.shape[:2] != (orig_h, orig_w):
            colored_mask = cv2.resize(colored_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        img_array = np.array(pil_images[idx])
        image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(image_bgr, 1, colored_mask_bgr, 0.5, 0)

        cv2.imwrite(os.path.join(SAVE_DIR, f'{base}_original.png'), image_bgr)
        cv2.imwrite(os.path.join(SAVE_DIR, f'{base}_mask.png'),     colored_mask_bgr)
        if SAVE_MASK_ID:
            mask_uint8 = mask.astype(np.uint8)
            if mask_uint8.shape != (orig_h, orig_w):
                mask_uint8 = cv2.resize(mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(SAVE_DIR, f'{base}_mask_id.png'), mask_uint8)
        cv2.imwrite(os.path.join(SAVE_DIR, f'{base}_overlap.png'),  overlay)
        print(f"保存到 {os.path.join(SAVE_DIR, base)}")

print(f"\n[INFO] 完成。结果保存至: {SAVE_DIR}")
