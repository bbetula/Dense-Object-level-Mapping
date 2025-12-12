"from wanglinman, dinov3 segmentation example"
import sys
from PIL import Image
import os
os.environ['TORCH_HUB_DISABLE_DOWNLOAD'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from dinov3.eval.segmentation.inference import make_inference
from hubconf import dinov3_vit7b16_ms
import numpy as np
import cv2
import time


REPO_DIR="/mnt/DATA1/bbetula/dinov3"
sys.path.append(REPO_DIR)
HEAD_DIR="checkpoint/dinov3_pretrained/DINOv3 Adapters/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
BACKBONE_DIR="checkpoint/dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
# image_path="/mnt/DATA1/bbetula/GroundedSAM/Grounded-SAM-2-main/notebooks/selected_images/1590194895899498940.png"
image_path="/mnt/DATA1/bbetula/GroundedSAM/Grounded-SAM-2-main/datasets/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"
output_path="output/"

def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

def apply_colormap(mask, colormap_name='Spectral'):
    mask_normalized = (mask - mask.min()) / (mask.max() - mask.min()) if mask.max() > mask.min() else mask
    cmap = colormaps[colormap_name]
    colored_mask = cmap(mask_normalized)
    colored_mask_rgb = (colored_mask[:, :, :3] * 255).astype(np.uint8)
    return colored_mask_rgb

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
            inference_mode="slide",
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
colored_mask = apply_colormap(mask, 'Spectral')
colored_mask_resized = cv2.resize(colored_mask, (w, h))

# 创建组合图：原图（左）+ mask（右）
combination = np.hstack([img_array, colored_mask_resized])
combination_output = os.path.join(output_path, f"{input_filename}_result.png")
cv2.imwrite(combination_output, cv2.cvtColor(combination, cv2.COLOR_RGB2BGR))

print(f"Result saved to: {combination_output}")

# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(img)
# plt.axis("off")
# plt.subplot(122)
# plt.imshow(segmentation_map_vit7b[0,0].cpu(), cmap=colormaps["Spectral"])
# plt.axis("off")