import sys
import os
import torch
import numpy as np
import cv2
import time
import yaml
from PIL import Image

os.environ['TORCH_HUB_DISABLE_DOWNLOAD'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append("/data1/user/Dense-Object-level-Mapping/Mask2Former")

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from mask2former import add_maskformer2_config

# ============ 配置 ============
image_path = "/data1/user/data/2026.05.10_机狗二次数据采集结果_标定/image/1590192560109912832.png"

# CONFIG_FILE = "/data1/user/Dense-Object-level-Mapping/Mask2Former/configs/cityscapes/semantic-segmentation/maskformer2_dinov3_vitl_bs16_90k.yaml"
# CHECKPOINT = "/data1/user/Dense-Object-level-Mapping/Mask2Former/output/dinov3_vitl_cityscapes_m2f/model_final.pth"
# PALETTE_FILE = "/data1/user/Dense-Object-level-Mapping/dinov3/yaml/cityscape.yaml"

# 总推理时间: 2.89s
CONFIG_FILE = "/data1/user/Dense-Object-level-Mapping/Mask2Former/configs/robotdog/maskformer2_dinov3_vitl_robotdog_ade20k.yaml"
CHECKPOINT = "/data1/user/Dense-Object-level-Mapping/Mask2Former/output/dinov3_vitl_robotdog_ade20k_distill/model_final.pth"
PALETTE_FILE = "/data1/user/Dense-Object-level-Mapping/dinov3/yaml/ADE20k.yaml"  # 换成 ADE20K 调色板

output_path = "output/train/dinov3_vitl_robotdog_ade20k_distill/"

# ============ 工具函数 ============
def load_palette(palette_file):
    with open(palette_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array(data["palette"], dtype=np.uint8)

def colorize_mask(mask, palette):
    indices = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[indices]

# ============ 构建 M2F 模型 ============
print("构建 Mask2Former + DINOv3 模型...")
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

# ============ 推理 ============
img = Image.open(image_path).convert("RGB")
img_array = np.array(img)
h, w = img_array.shape[:2]

img_bgr = img_array[:, :, ::-1].copy()
inputs = [{
    "image": torch.as_tensor(img_bgr.transpose(2, 0, 1).astype("float32")),
    "height": h,
    "width": w,
}]

palette_array = load_palette(PALETTE_FILE)

with torch.inference_mode():
    start_time = time.time()
    outputs = model(inputs)
    elapsed = time.time() - start_time
    print(f"推理时间: {elapsed:.2f}s")

sem_seg = outputs[0]["sem_seg"]  # (NUM_CLASSES, H, W)
mask = sem_seg.argmax(dim=0).cpu().numpy()

# ============ 可视化保存 ============
os.makedirs(output_path, exist_ok=True)
input_filename = os.path.splitext(os.path.basename(image_path))[0]

colored_mask = colorize_mask(mask, palette_array)

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
