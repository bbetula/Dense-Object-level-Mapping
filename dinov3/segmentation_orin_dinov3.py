import sys
import os
from pathlib import Path
from functools import partial

import torch
import torchvision.transforms.v2 as v2
import numpy as np
import cv2
import time
import yaml
from PIL import Image

os.environ['TORCH_HUB_DISABLE_DOWNLOAD'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dinov3.eval.segmentation.inference import make_inference
from hubconf import dinov3_vit7b16_ms

# ============ 配置 ============
image_path = "/data1/user/data/2026.05.10_机狗二次数据采集结果_标定/image/1590192560109912832.png"

HEAD_DIR     = "checkpoint/dinov3_pretrained/DINOv3 Adapters/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
BACKBONE_DIR = "checkpoint/dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
PALETTE_FILE = os.path.join(SCRIPT_DIR, "yaml", "ADE20k.yaml")

output_path = "output/train/dinov3_vit7b_ade20k_orin/"

INFER_SIZE = 1024
NUM_CLASSES = 150

# ============ 工具函数 ============
def make_transform(resize_size=INFER_SIZE):
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


# ============ 构建 DINOv3 ViT-7B + ADE20K M2F 模型 ============
print("构建 DINOv3 ViT-7B + ADE20K M2F 模型...")
segmentor = dinov3_vit7b16_ms(
    pretrained=True,
    weights=HEAD_DIR,
    backbone_weights=BACKBONE_DIR,
    check_hash=False,
)
segmentor.to("cuda").eval()
print(f"已加载 head:     {HEAD_DIR}")
print(f"已加载 backbone: {BACKBONE_DIR}")

# ============ 推理 ============
img = Image.open(image_path).convert("RGB")
img_array = np.array(img)
h, w = img_array.shape[:2]

transform = make_transform(INFER_SIZE)
batch_img = transform(img)[None].to("cuda")

palette_array = load_palette(PALETTE_FILE)

with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        start_time = time.time()
        pred_batch = make_inference(
            batch_img, segmentor,
            inference_mode="whole", decoder_head_type="m2f",
            rescale_to=(INFER_SIZE, INFER_SIZE), n_output_channels=NUM_CLASSES,
            output_activation=partial(torch.nn.functional.softmax, dim=1),
        )
        elapsed = time.time() - start_time
        print(f"推理时间: {elapsed:.2f}s")

seg_map = pred_batch.argmax(dim=1, keepdim=True)  # (1, 1, H, W)
mask = seg_map[0, 0].cpu().numpy()

# ============ 可视化保存 ============
os.makedirs(output_path, exist_ok=True)
input_filename = os.path.splitext(os.path.basename(image_path))[0]

colored_mask = colorize_mask(mask, palette_array)
colored_mask = cv2.resize(colored_mask, (w, h), interpolation=cv2.INTER_NEAREST)

# 横向拼接：原图 | mask
combination = np.hstack([img_array, colored_mask])
combination_output = os.path.join(output_path, f"{input_filename}_result.png")
cv2.imwrite(combination_output, cv2.cvtColor(combination, cv2.COLOR_RGB2BGR))

# 叠加图：原图 + mask 半透明融合
overlay = cv2.addWeighted(img_array, 0.6, colored_mask, 0.4, 0)
overlay_output = os.path.join(output_path, f"{input_filename}_overlay.png")
cv2.imwrite(overlay_output, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f"Result saved to: {combination_output}")
print(f"Overlay saved to: {overlay_output}")
