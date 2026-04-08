import sys
from PIL import Image
import os
os.environ['TORCH_HUB_DISABLE_DOWNLOAD'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import time
import yaml

REPO_DIR = "/data1/user/Dense-Object-level-Mapping/dinov3"
TRAIN_DIR = "/data1/user/Dense-Object-level-Mapping/dinov3-train"
sys.path.append(REPO_DIR)

# ============ 配置 ============
HEAD_DIR = TRAIN_DIR + "/output/best_cityscapes_dino_head.pth"
BACKBONE_DIR = TRAIN_DIR + "/weights/vit_large_patch16_dinov3.lvd1689m.pth"
PALETTE_FILE = os.path.join(REPO_DIR, "yaml", "cityscape.yaml")
NUM_CLASSES = 19

image_path = "/data1/user/data/cityscape/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"
output_path = "output/"

# ============ 与训练一致的模型定义 ============
class DINOv3SegmentationModel(nn.Module):
    """与 segment-train-cityscape.py 中完全一致的模型结构"""
    def __init__(self, backbone_name: str = "vit_large_patch16_dinov3.lvd1689m", num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.patch_size = 16
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.feature_dim = self.backbone.num_features
        self.feature_projection = nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone_feats = self.backbone.forward_features(x)
        if isinstance(backbone_feats, dict):
            features = backbone_feats.get("x_norm_patchtokens", backbone_feats.get("x_norm"))
            if features is None:
                raise ValueError("无法在 forward_features 输出中找到 patch token 张量")
        else:
            features = backbone_feats

        if features.dim() == 2:
            features = features.unsqueeze(1)

        batch_size, num_tokens, feat_dim = features.shape
        h_tokens = x.shape[2] // self.patch_size
        w_tokens = x.shape[3] // self.patch_size
        expected_tokens = h_tokens * w_tokens

        if num_tokens >= expected_tokens + 1:
            patch_tokens = features[:, 1:1 + expected_tokens, :]
        else:
            raise ValueError(f"Patch数量不匹配，期望 >= {expected_tokens + 1}，收到 {num_tokens}")

        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, feat_dim, h_tokens, w_tokens)
        projected = self.feature_projection(feature_map)
        seg_logits = self.head(projected)
        seg_logits = F.interpolate(seg_logits, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        return seg_logits

# ============ 工具函数 ============
def load_palette(palette_file: str) -> np.ndarray:
    with open(palette_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array(data["palette"], dtype=np.uint8)

def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    indices = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[indices]

# ============ 构建模型并加载权重 ============
print("构建模型...")
model = DINOv3SegmentationModel()

# 加载 backbone 权重
backbone_state = torch.load(BACKBONE_DIR, map_location="cpu")
model.backbone.load_state_dict(backbone_state, strict=True)
print(f"已加载 backbone 权重: {BACKBONE_DIR}")

# 加载分割头权重
ckpt = torch.load(HEAD_DIR, map_location="cpu")
model.feature_projection.load_state_dict(ckpt["feature_projection"])
model.head.load_state_dict(ckpt["head"])
print(f"已加载分割头权重: {HEAD_DIR}")

model = model.cuda().eval()

# ============ 推理 ============
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

img = Image.open(image_path).convert("RGB")
# 使用与训练验证集一致的预处理（保持宽高比，不做正方形 resize）
img_tensor = TF.to_tensor(img)
img_tensor = TF.normalize(img_tensor, mean=MEAN, std=STD)
batch_img = img_tensor[None].cuda()

palette_array = load_palette(PALETTE_FILE)

with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        start_time = time.time()
        logits = model(batch_img)  # (1, 19, H, W)
        seg_map = logits.argmax(dim=1)  # (1, H, W)
        end_time = time.time()
        print(f"inference time: {end_time - start_time:.2f}s")

# ============ 可视化保存 ============
os.makedirs(output_path, exist_ok=True)
input_filename = os.path.splitext(os.path.basename(image_path))[0]

mask = seg_map[0].cpu().numpy()
img_array = np.array(img)
h, w = img_array.shape[:2]

colored_mask = colorize_mask(mask, palette_array)
colored_mask_resized = cv2.resize(colored_mask, (w, h), interpolation=cv2.INTER_NEAREST)

combination = np.hstack([img_array, colored_mask_resized])
combination_output = os.path.join(output_path, f"{input_filename}_result.png")
cv2.imwrite(combination_output, cv2.cvtColor(combination, cv2.COLOR_RGB2BGR))

print(f"Result saved to: {combination_output}")
