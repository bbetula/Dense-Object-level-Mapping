"caozhijin, our dataset segmentation example(batch image, whole)"
import sys
REPO_DIR = "/home/czj/program/dinov3" # Please add here the path to your DINOv3 repository
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

PALETTE_FILE = os.path.join(REPO_DIR, "yaml", "ADE20k.yaml")
palette_array = load_palette(PALETTE_FILE)
def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    indices = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[indices]

# load model
backbone_weights_url = "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaGNiemFiOTNzZzMwOWhnbG9kaTd5aTdmIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjUyMDE0OTB9fX1dfQ__&Signature=NLF2B0QuchPcUM-U2t9k180wsIfMnYa6lDbjPn1gyZrl5YOL2WXp-JFcrYwjevzDbtHOMIwAnze0zDYez4jgcdWNrVXcRLFS4EXGmHF1xhrk02I8U1iuQx-qZO2DGmqN5wTRgYaBPF2ittQlDOziiuveaAl3PQhUHPG8ke4OPWMIxxANUU8jw1WT4QJgH5ZLsCry3-cozX54-LMdNdHK%7ECLZMPdkWSGeInmInG1xHHYcybKl9Jo0GQBSSeTfrvoTiiHZOjxksW7Z3yimrxsssSl8%7EraWFhrRGa2RSad0LBZF0QFR8E7YNU9y2CzO6EcPlbR%7EG8Wzf0RyK-880FlfcA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1176142748036319"
head_weights_url = "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaGNiemFiOTNzZzMwOWhnbG9kaTd5aTdmIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjUyMDE0OTB9fX1dfQ__&Signature=NLF2B0QuchPcUM-U2t9k180wsIfMnYa6lDbjPn1gyZrl5YOL2WXp-JFcrYwjevzDbtHOMIwAnze0zDYez4jgcdWNrVXcRLFS4EXGmHF1xhrk02I8U1iuQx-qZO2DGmqN5wTRgYaBPF2ittQlDOziiuveaAl3PQhUHPG8ke4OPWMIxxANUU8jw1WT4QJgH5ZLsCry3-cozX54-LMdNdHK%7ECLZMPdkWSGeInmInG1xHHYcybKl9Jo0GQBSSeTfrvoTiiHZOjxksW7Z3yimrxsssSl8%7EraWFhrRGa2RSad0LBZF0QFR8E7YNU9y2CzO6EcPlbR%7EG8Wzf0RyK-880FlfcA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1176142748036319"
segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", 
                            weights=head_weights_url, 
                            backbone_weights=backbone_weights_url)
# move model to GPU and set eval mode to avoid dtype/device mismatches
if torch.cuda.is_available():
    segmentor.to("cuda")
segmentor.eval()

# load images
# images_dir = f"/home/czj/datasets/fastlivo_output_outdoor_1s/image"
images_dir = f"/home/czj/datasets/fastlivo_output_indoor_107/image"
all_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() == '.png']
all_files.sort()

# save_dir = f"/home/czj/datasets/fastlivo_output_outdoor_1s/image/res_dinov3_whole"
save_dir = f"/home/czj/datasets/fastlivo_output_indoor_107/image/res_dinov3_whole"
os.makedirs(save_dir, exist_ok=True)

img_size = 896
batch_size = 4
for i in range(0, len(all_files), batch_size):
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
                colored_mask = cv2.resize(colored_mask, (orig_w, orig_h))

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
