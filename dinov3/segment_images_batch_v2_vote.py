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
import shutil
from scipy.ndimage import gaussian_filter
from hubconf import dinov3_vit7b16_ms

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 2026/03/31 仍然存在错分 形状受损 不建议使用
# ─── 逐像素置信度门控混合（per-pixel confidence-gated blending）─────
#
# 核心思路：
#   当前帧对某像素置信度高（模型确定）→ α≈1，完全信任当前帧结果（保留细节/形状）
#   当前帧对某像素置信度低（模型不确定）→ α≈0，用时序窗口平均来纠正错分
#
#   blended = α · P_current + (1-α) · P_temporal_avg
#   α = clip((max_prob_current - CONF_LOW) / (CONF_HIGH - CONF_LOW), 0, 1)
#
# 相比纯时序平均的优势：
#   - 树干、细长物体等高置信区域不会被其他帧"稀释"（形状保留）
#   - 低置信度区域（模型困惑的帧）仍由邻帧投票纠正（减少错分）
# ─────────────────────────────────────────────────────────────────
TEMPORAL_VOTE   = True  # 启用置信度门控混合（两遍处理）
VOTE_WINDOW     = 15    # 时序窗口 ±N 帧（10fps → ±1.5s）
PROB_STORE_SIZE = 512   # 概率图存储分辨率，512²×150×float16 ≈ 75 MB/帧
SMOOTH_SIGMA    = 1.0   # Gaussian 平滑 σ（仅作用于时序平均部分，不影响当前帧）
CONF_LOW        = 0.5   # 低于此置信度：完全用时序平均（α=0）
CONF_HIGH       = 0.85  # 高于此置信度：完全用当前帧（α=1）
KEEP_PROBS      = False # 处理完后是否保留临时 .npy 概率文件
# ─────────────────────────────────────────────────────────────────


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
USE_URL = False
if USE_URL:
    backbone_weights_url = "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaGNiemFiOTNzZzMwOWhnbG9kaTd5aTdmIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjUyMDE0OTB9fX1dfQ__&Signature=NLF2B0QuchPcUM-U2t9k180wsIfMnYa6lDbjPn1gyZrl5YOL2WXp-JFcrYwjevzDbtHOMIwAnze0zDYez4jgcdWNrVXcRLFS4EXGmHF1xhrk02I8U1iuQx-qZO2DGmqN5wTRgYaBPF2ittQlDOziiuveaAl3PQhUHPG8ke4OPWMIxxANUU8jw1WT4QJgH5ZLsCry3-cozX54-LMdNdHK%7ECLZMPdkWSGeInmInG1xHHYcybKl9Jo0GQBSSeTfrvoTiiHZOjxksW7Z3yimrxsssSl8%7EraWFhrRGa2RSad0LBZF0QFR8E7YNU9y2CzO6EcPlbR%7EG8Wzf0RyK-880FlfcA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1176142748036319"
    head_weights_url = "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaGNiemFiOTNzZzMwOWhnbG9kaTd5aTdmIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjUyMDE0OTB9fX1dfQ__&Signature=NLF2B0QuchPcUM-U2t9k180wsIfMnYa6lDbjPn1gyZrl5YOL2WXp-JFcrYwjevzDbtHOMIwAnze0zDYez4jgcdWNrVXcRLFS4EXGmHF1xhrk02I8U1iuQx-qZO2DGmqN5wTRgYaBPF2ittQlDOziiuveaAl3PQhUHPG8ke4OPWMIxxANUU8jw1WT4QJgH5ZLsCry3-cozX54-LMdNdHK%7ECLZMPdkWSGeInmInG1xHHYcybKl9Jo0GQBSSeTfrvoTiiHZOjxksW7Z3yimrxsssSl8%7EraWFhrRGa2RSad0LBZF0QFR8E7YNU9y2CzO6EcPlbR%7EG8Wzf0RyK-880FlfcA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1176142748036319"
    segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local",
                                weights=head_weights_url,
                                backbone_weights=backbone_weights_url)
else:
    HEAD_DIR     = "checkpoint/dinov3_pretrained/DINOv3 Adapters/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
    BACKBONE_DIR = "checkpoint/dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
    segmentor = dinov3_vit7b16_ms(pretrained=True, weights=HEAD_DIR, backbone_weights=BACKBONE_DIR, check_hash=False)

segmentor.to("cuda").eval()

# ── dataset ────────────────────────────────────────────────────────
IMAGES_DIR = "/data1/user/data/fastlivo_output_qs2_03.17/image"
all_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".png")])

SINGLE_IMAGE_TEST = False
if SINGLE_IMAGE_TEST:
    all_files     = all_files[:1]
    SAVE_DIR      = "/data1/user/Dense-Object-level-Mapping/dinov3/output"
    batch_size    = 1
    TEMPORAL_VOTE = False  # 单帧无需投票
else:
    SAVE_DIR   = os.path.join(IMAGES_DIR, "res_dinov3_whole")
    batch_size = 8

os.makedirs(SAVE_DIR, exist_ok=True)
img_size  = 1024
transform = make_transform(img_size)
# 用于将推理输出从 img_size² 下采样到 PROB_STORE_SIZE² 以节省磁盘
prob_pool = torch.nn.AdaptiveAvgPool2d((PROB_STORE_SIZE, PROB_STORE_SIZE))


# ══════════════════════════════════════════════════════════════════
# Branch A — 时序软投票（两遍处理）
# ══════════════════════════════════════════════════════════════════
if TEMPORAL_VOTE:
    PROB_DIR = os.path.join(SAVE_DIR, "_probs_tmp")
    os.makedirs(PROB_DIR, exist_ok=True)
    frame_sizes = {}  # base_name → (orig_w, orig_h)

    # ── Pass 1：推理 → 保存 float16 概率图 ───────────────────────
    print(f"[INFO] Pass 1/2 — 推理中（{len(all_files)} 帧，概率图分辨率 {PROB_STORE_SIZE}²）")
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i : i + batch_size]
        pil_images = []
        for img_name in batch_files:
            img = Image.open(os.path.join(IMAGES_DIR, img_name)).convert("RGB")
            pil_images.append(img)
            frame_sizes[img_name.split('.')[0]] = img.size  # (w, h)

        batch_tensor = torch.stack([transform(img) for img in pil_images]).to("cuda")
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_batch = make_inference(
                    batch_tensor, segmentor,
                    inference_mode="whole", decoder_head_type="m2f",
                    rescale_to=(img_size, img_size), n_output_channels=150,
                    output_activation=partial(torch.nn.functional.softmax, dim=1),
                )  # [B, 150, img_size, img_size]
            # 下采样后转 float16 存盘，节省约 16× 存储
            pred_small = prob_pool(pred_batch.float()).cpu().numpy().astype(np.float16)
            # shape: [B, 150, PROB_STORE_SIZE, PROB_STORE_SIZE]

        for idx, img_name in enumerate(batch_files):
            base = img_name.split('.')[0]
            np.save(os.path.join(PROB_DIR, f"{base}.npy"), pred_small[idx])
            # 同时保存原图（供 Pass 2 叠加）
            img_bgr = cv2.cvtColor(np.array(pil_images[idx]), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(SAVE_DIR, f"{base}_original.png"), img_bgr)

        print(f"  [{min(i + batch_size, len(all_files))}/{len(all_files)}] pass 1 done")

    # ── Pass 2：置信度门控混合 → argmax → 保存 mask ─────────────
    print(f"\n[INFO] Pass 2/2 — 置信度门控混合（window=±{VOTE_WINDOW}, "
          f"conf=[{CONF_LOW},{CONF_HIGH}]）")
    base_names = [f.split('.')[0] for f in all_files]

    for i, base in enumerate(base_names):
        lo = max(0, i - VOTE_WINDOW)
        hi = min(len(base_names), i + VOTE_WINDOW + 1)

        # 当前帧概率（作为高置信区域的锚点）
        curr_prob = np.load(os.path.join(PROB_DIR, f"{base}.npy")).astype(np.float32)
        # [150, PS, PS]

        # 时序平均概率（邻帧投票，用于纠正低置信区域）
        acc = np.zeros((150, PROB_STORE_SIZE, PROB_STORE_SIZE), dtype=np.float32)
        for j in range(lo, hi):
            acc += np.load(os.path.join(PROB_DIR, f"{base_names[j]}.npy")).astype(np.float32)
        temporal_avg = acc / (hi - lo)  # [150, PS, PS]

        # 对时序平均部分做轻微 Gaussian 平滑（消散散点噪声）
        if SMOOTH_SIGMA > 0:
            temporal_avg = np.stack([gaussian_filter(temporal_avg[c], sigma=SMOOTH_SIGMA)
                                     for c in range(150)])

        # ── 逐像素置信度权重 α ─────────────────────────────────
        # α=1 → 完全用当前帧（高置信，如清晰的树干）
        # α=0 → 完全用时序平均（低置信，如靠后帧错分区域）
        curr_conf = curr_prob.max(axis=0)  # [PS, PS]，每像素最高类别概率
        alpha = np.clip(
            (curr_conf - CONF_LOW) / (CONF_HIGH - CONF_LOW + 1e-6),
            0.0, 1.0
        )  # [PS, PS]

        # 混合：在概率空间做加权融合，再 argmax（比对 label 做混合更精确）
        blended = alpha[np.newaxis] * curr_prob + (1.0 - alpha[np.newaxis]) * temporal_avg
        # [150, PS, PS]

        orig_w, orig_h = frame_sizes[base]
        # 双线性上采样到原图分辨率，再 argmax（避免锯齿）
        prob_t = torch.from_numpy(blended).unsqueeze(0)  # [1, 150, PS, PS]
        prob_full = torch.nn.functional.interpolate(
            prob_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )  # [1, 150, orig_h, orig_w]
        mask = prob_full[0].argmax(dim=0).numpy().astype(np.uint8)  # [orig_h, orig_w]

        colored_mask     = colorize_mask(mask, palette_array)
        colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
        orig_bgr         = cv2.imread(os.path.join(SAVE_DIR, f"{base}_original.png"))
        overlay          = cv2.addWeighted(orig_bgr, 1, colored_mask_bgr, 0.5, 0)

        cv2.imwrite(os.path.join(SAVE_DIR, f"{base}_mask.png"),    colored_mask_bgr)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{base}_overlap.png"), overlay)

        if (i + 1) % 50 == 0 or i == len(base_names) - 1:
            print(f"  [{i + 1}/{len(base_names)}] pass 2 done")

    if not KEEP_PROBS:
        shutil.rmtree(PROB_DIR)
        print(f"[INFO] 已删除临时概率目录: {PROB_DIR}")


# ══════════════════════════════════════════════════════════════════
# Branch B — 原始单遍处理（不投票）
# ══════════════════════════════════════════════════════════════════
else:
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
            cv2.imwrite(os.path.join(SAVE_DIR, f'{base}_overlap.png'),  overlay)
            print(f"保存到 {os.path.join(SAVE_DIR, base)}")


print(f"\n[INFO] 完成。结果保存至: {SAVE_DIR}")
