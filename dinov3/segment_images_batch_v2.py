import sys
REPO_DIR = "/data1/user/Dense-Object-level-Mapping/dinov3"
sys.path.append(REPO_DIR)

from PIL import Image
import torch
import torchvision.transforms.v2 as v2
from functools import partial
from dinov3.eval.segmentation.inference import make_inference
import os
import cv2
import numpy as np
import yaml
from hubconf import dinov3_vit7b16_ms

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

SAVE_MASK_ID = True  # 保存单通道类别ID图（_mask_id.png），供 3D 多数投票使用


def make_transform(resize_size=768):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_palette(palette_file):
    with open(palette_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array(data["palette"], dtype=np.uint8)


def colorize_mask(mask, palette):
    indices = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[indices]


PALETTE_FILE = os.path.join(REPO_DIR, "yaml", "ADE20k.yaml")
palette_array = load_palette(PALETTE_FILE)

# ── model ──────────────────────────────────────────────────────────
HEAD_DIR     = "checkpoint/dinov3_pretrained/DINOv3 Adapters/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
BACKBONE_DIR = "checkpoint/dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
segmentor = dinov3_vit7b16_ms(pretrained=True, weights=HEAD_DIR, backbone_weights=BACKBONE_DIR, check_hash=False)
segmentor.to("cuda").eval()

# ── dataset ────────────────────────────────────────────────────────
IMAGES_DIR = "/data1/user/data/fastlivo_output_qs2_03.17/image"
all_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".png")])

SINGLE_IMAGE_TEST = False
if SINGLE_IMAGE_TEST:
    all_files  = all_files[:1]
    SAVE_DIR   = "/data1/user/Dense-Object-level-Mapping/dinov3/output"
    batch_size = 1
else:
    SAVE_DIR   = os.path.join(IMAGES_DIR, "res_dinov3_whole")
    batch_size = 8

os.makedirs(SAVE_DIR, exist_ok=True)
img_size  = 1024
transform = make_transform(img_size)

for i in range(0, len(all_files), batch_size):
    batch_files = all_files[i : i + batch_size]
    pil_images, orig_sizes = [], []
    for img_name in batch_files:
        img = Image.open(os.path.join(IMAGES_DIR, img_name)).convert("RGB")
        pil_images.append(img)
        orig_sizes.append(img.size)

    batch_img = torch.stack([transform(img) for img in pil_images]).to("cuda")
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_batch = make_inference(
                batch_img, segmentor,
                inference_mode="whole", decoder_head_type="m2f",
                rescale_to=(img_size, img_size), n_output_channels=150,
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            )
        seg_maps = pred_batch.argmax(dim=1, keepdim=True)

    for idx, img_name in enumerate(batch_files):
        base = img_name.split('.')[0]
        orig_w, orig_h = orig_sizes[idx]
        mask = seg_maps[idx, 0].cpu().numpy()
        colored_mask = colorize_mask(mask, palette_array)
        colored_mask = cv2.resize(colored_mask, (orig_w, orig_h))
        img_array = np.array(pil_images[idx])
        image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(image_bgr, 1, colored_mask_bgr, 0.5, 0)
        cv2.imwrite(os.path.join(SAVE_DIR, f'{base}_original.png'), image_bgr)
        cv2.imwrite(os.path.join(SAVE_DIR, f'{base}_mask.png'),     colored_mask_bgr)
        if SAVE_MASK_ID:
            mask_full = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(SAVE_DIR, f'{base}_mask_id.png'), mask_full)
        cv2.imwrite(os.path.join(SAVE_DIR, f'{base}_overlap.png'),  overlay)
        print(f"保存到 {os.path.join(SAVE_DIR, base)}")

print(f"\n[INFO] 完成。结果保存至: {SAVE_DIR}")
