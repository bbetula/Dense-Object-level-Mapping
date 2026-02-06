"caozhijin, our dataset segmentation example(sigle image, slide)"

import sys
REPO_DIR = "/home/czj/program/MVSSeg/dinov3" # Please add here the path to your DINOv3 repository
sys.path.append(REPO_DIR)

from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from dinov3.eval.segmentation.inference import make_inference
import torchvision.transforms.v2 as v2
import os
import cv2
import numpy as np
import yaml
from segment.config import *

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

def load_palette(palette_file: str) -> np.ndarray:
    with open(palette_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array(data["palette"], dtype=np.uint8)

def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    indices = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[indices]

def main_origin():
    palette_array = load_palette(PALETTE_FILE)

    # load model
    segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", 
                                weights=HEAD_WEIGHTS_URL, 
                                backbone_weights=BACKBONE_WEIGHTS_URL)

    # move model to GPU and set eval mode to avoid dtype/device mismatches
    if torch.cuda.is_available():
        segmentor.to("cuda")
    segmentor.eval()

    # load images
    images_dir = IMAGES_DIR
    all_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() == '.png']
    all_files.sort()

    save_dir = os.path.join(os.path.dirname(IMAGES_DIR), "image_seg")
    os.makedirs(save_dir, exist_ok=True)

    img_size = 896
    batch_size = 4
    for i in range(0, len(all_files), batch_size):
        time_start = cv2.getTickCount()

        batch_files = all_files[i : i + batch_size]
        pil_images = []
        orig_sizes = []  # list of (w, h)
        for img_name in batch_files:
            image_path = os.path.join(images_dir, img_name)
            img = Image.open(image_path).convert("RGB")
            pil_images.append(img)
            orig_sizes.append(img.size)

        # apply transforms and stack
        transform = make_transform(img_size)
        tensors = [transform(img) for img in pil_images]
        batch_img = torch.stack(tensors, dim=0).to("cuda")  # [B, C, H, W]

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_batch = make_inference(
                    batch_img,
                    segmentor,
                    inference_mode="whole",
                    decoder_head_type="m2f",
                    # rescale to model size; we'll resize per image for saving
                    rescale_to=(img_size, img_size),
                    n_output_channels=150,
                    output_activation=partial(torch.nn.functional.softmax, dim=1),
                )
                # pred_batch: [B, num_classes, H_pred, W_pred]
                segmentation_maps = pred_batch.argmax(dim=1, keepdim=True)  # [B, 1, H, W]

                # save each image's outputs individually
                for idx, img_name in enumerate(batch_files):
                    
                    base_name = img_name.split('.')[0]
                    img = pil_images[idx]
                    orig_w, orig_h = orig_sizes[idx]

                    mask = segmentation_maps[idx, 0].cpu().numpy()
                    colored_mask = colorize_mask(mask, palette_array)
                    colored_mask = cv2.resize(colored_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                    img_array = np.array(img)
                    image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
                    overlay = cv2.addWeighted(image_bgr, 1, colored_mask_bgr, 0.5, 0)

                    cv2.imwrite(os.path.join(save_dir, f'{base_name}_original.png'), image_bgr)
                    print(f"保存到 {os.path.join(save_dir, f'{base_name}_original.png')}")
                    cv2.imwrite(os.path.join(save_dir, f'{base_name}_mask.png'), colored_mask_bgr)
                    print(f"保存到 {os.path.join(save_dir, f'{base_name}_mask.png')}")
                    cv2.imwrite(os.path.join(save_dir, f'{base_name}_overlap.png'), overlay)
                    print(f"保存到 {os.path.join(save_dir, f'{base_name}_overlap.png')}")

        time_end = cv2.getTickCount()
        time_batch = (time_end - time_start) / cv2.getTickFrequency()
        print(f"spend {time_batch:.3f} seconds for segment {len(batch_files)} images")


def main():
    palette_array = load_palette(PALETTE_FILE)

    # load model
    segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", 
                                weights=HEAD_WEIGHTS_URL, 
                                backbone_weights=BACKBONE_WEIGHTS_URL)

    # move model to GPU and set eval mode to avoid dtype/device mismatches
    if torch.cuda.is_available():
        segmentor.to("cuda")
    segmentor.eval()

    # load images
    images_dir = IMAGES_DIR
    all_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() == '.png']
    all_files.sort()

    img_size = 896
    batch_size = 4
    for i in range(0, len(all_files), batch_size):
        time_start = cv2.getTickCount()

        batch_files = all_files[i : i + batch_size]
        pil_images = []
        orig_sizes = []  # list of (w, h)
        for img_name in batch_files:
            image_path = os.path.join(images_dir, img_name)
            img = Image.open(image_path).convert("RGB")
            pil_images.append(img)
            orig_sizes.append(img.size)

        # apply transforms and stack
        transform = make_transform(img_size)
        tensors = [transform(img) for img in pil_images]
        batch_img = torch.stack(tensors, dim=0).to("cuda")  # [B, C, H, W]

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):

                # pred_batch: [B, num_classes, H_pred, W_pred]
                pred_batch = make_inference(
                    batch_img,
                    segmentor,
                    inference_mode="whole",
                    decoder_head_type="m2f",
                    # rescale to model size; we'll resize per image for saving
                    rescale_to=(img_size, img_size),
                    n_output_channels=150,
                    output_activation=partial(torch.nn.functional.softmax, dim=1),
                )

                # save each image's outputs individually
                for idx, img_name in enumerate(batch_files):
                    
                    base_name = img_name.split('.')[0]
                    orig_w, orig_h = orig_sizes[idx]

                    # 1. 提取单张图概率图：[num_classes, img_size, img_size]（C, H, W）
                    pred_single = pred_batch[idx].cpu()
                    # 2. 转换为 NumPy 数组，并调整格式为 [H, W, C]（适配 cv2.resize）
                    pred_single_np = pred_single.numpy()  # [C, H, W]
                    pred_single_hwc = np.transpose(pred_single_np, (1, 2, 0))  # 转置：(C, H, W) → (H, W, C)
                    # 3. cv2.resize 会对 [H, W, C] 的每个通道（第3维）同时进行缩放
                    pred_resized_hwc = cv2.resize(pred_single_hwc, (orig_w, orig_h))  # 输出形状：[orig_h, orig_w, num_classes]

                    save_dir = os.path.join(os.path.dirname(IMAGES_DIR), "seg_npy")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{base_name}_pred_class.npy")
                    np.save(save_path, pred_resized_hwc)
                    print(f"保存 {base_name}_pred_class.npy 到 {save_path}")

        
        time_end = cv2.getTickCount()
        time_batch = (time_end - time_start) / cv2.getTickFrequency()
        print(f"spend {time_batch:.3f} seconds for segment {len(batch_files)} images")


if __name__ == '__main__':
    main()