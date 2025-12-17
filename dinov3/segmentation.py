"from wanglinman, dinov3 segmentation example"
import sys
from PIL import Image
import os
os.environ['TORCH_HUB_DISABLE_DOWNLOAD'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from torchvision import transforms
from functools import partial
from dinov3.eval.segmentation.inference import make_inference
from hubconf import dinov3_vit7b16_ms
import numpy as np
import cv2
import time
import yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

REPO_DIR="/data1/user/Dense-Object-level-Mapping/dinov3"
sys.path.append(REPO_DIR)
# 7B ViT + Mask2Former 头
HEAD_DIR="checkpoint/dinov3_pretrained/DINOv3 Adapters/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
BACKBONE_DIR="checkpoint/dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
PALETTE_FILE = os.path.join(REPO_DIR, "yaml", "ADE20k.yaml")

# Large ViT + Mask2Former 头
# HEAD_DIR
# BACKBONE_DIR="checkpoint/dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# PALETTE_FILE = os.path.join(REPO_DIR, "yaml", "cityscape.yaml")

image_path="/data1/user/data/cityscape/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"
output_path="output/"


def make_transform(resize_size: int | list[int] = 768):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

def load_palette(palette_file: str) -> np.ndarray:
    with open(palette_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array(data["palette"], dtype=np.uint8)

palette_array = load_palette(PALETTE_FILE)
def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    indices = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[indices]

# segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", weights=HEAD_DIR, backbone_weights=BACKBONE_DIR)
segmentor = dinov3_vit7b16_ms(pretrained=True, weights=HEAD_DIR, backbone_weights=BACKBONE_DIR, check_hash=False)

img_size = 1024 #896
transform = make_transform(img_size)
img  = Image.open(image_path).convert("RGB")

with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        pred_vit7b = segmentor(batch_img)  # raw predictions  
        start_time= time.time()
        # actual segmentation map
        segmentation_map_vit7b = make_inference(
            batch_img,
            segmentor,
            inference_mode="whole", # slide: 分块处理; whole: 整图处理
            decoder_head_type="m2f", # Mask2Former
            rescale_to=(img.size[-1], img.size[-2]),
            n_output_channels=150, # ADE20K 150类别
            crop_size=(img_size, img_size), # 模型一次处理的图像块尺寸
            stride=(img_size, img_size),
            output_activation=partial(torch.nn.functional.softmax, dim=1),
        ).argmax(dim=1, keepdim=True) # 每个像素得到最可能的语义类别
        end_time = time.time()
        print(f"inference time: {end_time - start_time:.2f}s")

# 确保输出路径存在
os.makedirs(output_path, exist_ok=True)
input_filename = os.path.splitext(os.path.basename(image_path))[0] 

# 获取分割掩码
mask = segmentation_map_vit7b[0,0].cpu().numpy()

# 获取原图数组
img_array = np.array(img)
h, w = img_array.shape[:2]

# 创建彩色分割掩码
colored_mask = colorize_mask(mask, palette_array)
colored_mask_resized = cv2.resize(colored_mask, (w, h), interpolation=cv2.INTER_NEAREST)

# 创建组合图：原图（左）+ mask（右）
combination = np.hstack([img_array, colored_mask_resized])
combination_output = os.path.join(output_path, f"{input_filename}_result.png")
cv2.imwrite(combination_output, cv2.cvtColor(combination, cv2.COLOR_RGB2BGR))

print(f"Result saved to: {combination_output}")