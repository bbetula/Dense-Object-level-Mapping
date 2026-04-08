"""
DINOv3 语义分割 - ScanNet 数据集批量处理 (v3)

适配 ScanNet scannet_frames_25k 数据集：
  - 输入：scene*/color/*.jpg
  - 输出：mask (NYU40 色盘) + overlay

Head/Backbone 选择策略:
  MODE = "ade20k_mapped"  → 用现有 ADE20K head (150类) + 后映射到 NYU40 (40类)，无需训练
  MODE = "scannet_native" → 用自训练的 ScanNet M2F head (41类)，需先训练
"""

import sys
REPO_DIR = "/data1/user/Dense-Object-level-Mapping/dinov3"
sys.path.append(REPO_DIR)

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.v2 as v2
from functools import partial
from dinov3.eval.segmentation.inference import make_inference
from hubconf import dinov3_vit7b16_ms
import os
import cv2
import numpy as np
import yaml
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# GPU 检查
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("[WARN] CUDA not available, using CPU. Inference will be very slow.")

# ─── Soft temporal voting ─────────────────────────────────────────
# 原理：每帧推理后保存 softmax 概率图，在同一场景内对时序窗口求概率均值，
# 再 argmax。这样对物体预测不稳定的帧，多帧共识的类别会"覆盖"噪声。
# 对 ScanNet 场景内连续帧（相机缓慢移动）尤其有效。
TEMPORAL_VOTE   = True  # 启用时序软投票（两遍处理）
VOTE_WINDOW     = 5     # ±N 帧，共 2N+1 帧参与平均
PROB_STORE_SIZE = 256   # 概率图存储分辨率，256²×C×float16 ≈ 18–28 MB/帧
KEEP_PROBS      = False # 处理完场景后是否保留临时 .npy 概率文件
# ─────────────────────────────────────────────────────────────────

# ============================================================
# 配置区
# ============================================================
MODE = "ade20k_mapped"  # "ade20k_mapped" | "scannet_native"

SCANNET_ROOT = "/data1/data/scannet/scannet_frames_25k"
SAVE_ROOT = "/data1/data/scannet/scannet_frames_25k_dinov3_seg"

# 模型配置
BACKBONE_DIR = "checkpoint/dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"

if MODE == "ade20k_mapped":
    HEAD_DIR = "checkpoint/dinov3_pretrained/DINOv3 Adapters/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
    N_OUTPUT_CHANNELS = 150
    PALETTE_FILE = os.path.join(REPO_DIR, "yaml", "scannet_nyu40.yaml")  # 输出用 NYU40 色盘
elif MODE == "scannet_native":
    # 训练好 ScanNet head 后填入路径
    HEAD_DIR = "checkpoint/dinov3_pretrained/DINOv3 Adapters/dinov3_vit7b16_scannet_nyu40_m2f_head.pth"
    N_OUTPUT_CHANNELS = 41
    PALETTE_FILE = os.path.join(REPO_DIR, "yaml", "scannet_nyu40.yaml")

IMG_SIZE = 1024
BATCH_SIZE = 8
# ============================================================
# ADE20K (0-149) → ScanNet NYU40 (0-40) 映射表
# ADE20K 模型输出 argmax 后的 class id → NYU40 id
# 未映射到的 ADE20K 类归为 0 (unlabeled)
# ============================================================
ADE20K_TO_NYU40 = np.zeros(150, dtype=np.uint8)  # 默认全部映射到 0 (unlabeled)
_mapping = {
    # ADE20K_id: NYU40_id
    0: 1,      # wall → wall
    3: 2,      # floor → floor
    5: 22,     # ceiling → ceiling
    7: 4,      # bed → bed
    8: 9,      # windowpane → window
    10: 3,     # cabinet → cabinet
    12: 31,    # person → person
    14: 8,     # door → door
    15: 7,     # table → table
    18: 16,    # curtain → curtain
    19: 5,     # chair → chair
    22: 11,    # painting → picture
    23: 6,     # sofa → sofa
    24: 15,    # shelf → shelves
    27: 19,    # mirror → mirror
    28: 20,    # rug → floor mat
    30: 5,     # armchair → chair
    31: 5,     # seat → chair
    33: 14,    # desk → desk
    35: 3,     # wardrobe → cabinet
    36: 35,    # lamp → lamp
    37: 36,    # bathtub → bathtub
    38: 38,    # railing → otherstructure
    39: 18,    # cushion → pillow
    41: 29,    # box → box
    42: 38,    # column → otherstructure
    44: 17,    # chest of drawers → dresser
    45: 12,    # counter → counter
    47: 34,    # sink → sink
    50: 24,    # refrigerator → refrigerator
    53: 38,    # stairs → otherstructure
    55: 39,    # case → otherfurniture
    56: 7,     # pool table → table
    57: 18,    # pillow → pillow
    58: 8,     # screen door → door
    59: 38,    # stairway → otherstructure
    62: 10,    # bookcase → bookshelf
    63: 13,    # blind → blinds
    64: 7,     # coffee table → table
    65: 33,    # toilet → toilet
    67: 23,    # book → books
    70: 12,    # countertop → counter
    75: 5,     # swivel chair → chair
    81: 27,    # towel → towel
    82: 35,    # light → lamp
    85: 35,    # chandelier → lamp
    89: 25,    # television receiver → television
    92: 21,    # apparel → clothes
    93: 38,    # pole → otherstructure
    95: 38,    # bannister → otherstructure
    100: 11,   # poster → picture
    107: 39,   # washer → otherfurniture
    110: 39,   # stool → otherfurniture
    115: 37,   # bag → bag
    118: 39,   # oven → otherfurniture
    124: 39,   # microwave → otherfurniture
    129: 39,   # dishwasher → otherfurniture
    130: 25,   # screen → television
    134: 35,   # sconce → lamp
    141: 25,   # crt screen → television
    143: 25,   # monitor → television
    144: 30,   # bulletin board → whiteboard
    145: 28,   # shower → shower curtain
    # 其余 ADE20K 类 → 0 (unlabeled)
}
for ade_id, nyu_id in _mapping.items():
    ADE20K_TO_NYU40[ade_id] = nyu_id

# 工具函数
def make_transform(resize_size: int = 768):
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


def map_ade20k_to_nyu40(ade20k_mask: np.ndarray) -> np.ndarray:
    """将 ADE20K class id (0-149) 映射为 NYU40 class id (0-40)"""
    return ADE20K_TO_NYU40[np.clip(ade20k_mask, 0, 149)]


def compute_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 41, ignore_label: int = 0):
    """计算 mIoU，忽略 unlabeled (0)"""
    ious = []
    for c in range(1, num_classes):  # 跳过 unlabeled
        pred_c = (pred == c)
        gt_c = (gt == c)
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


# 加载模型
print(f"[INFO] Mode: {MODE}")
print(f"[INFO] Loading backbone: {BACKBONE_DIR}")
print(f"[INFO] Loading head: {HEAD_DIR}")

segmentor = dinov3_vit7b16_ms(
    pretrained=True,
    weights=HEAD_DIR,
    backbone_weights=BACKBONE_DIR,
    check_hash=False,
)
segmentor.to(DEVICE)
segmentor.eval()

palette_array = load_palette(PALETTE_FILE)
transform = make_transform(IMG_SIZE)
# 用于下采样概率图以节省磁盘（仅在 TEMPORAL_VOTE=True 时使用）
prob_pool = torch.nn.AdaptiveAvgPool2d((PROB_STORE_SIZE, PROB_STORE_SIZE))

# 遍历 ScanNet 场景
scene_dirs = sorted([
    d for d in os.listdir(SCANNET_ROOT)
    if os.path.isdir(os.path.join(SCANNET_ROOT, d, "color"))
])
print(f"[INFO] Found {len(scene_dirs)} scenes in {SCANNET_ROOT}")

EVAL_WITH_GT = True  # 是否和 GT label 对比计算 mIoU
all_scene_mious = []  # [(scene_name, miou), ...]
all_per_image_mious = []  # [(scene_name, img_name, miou), ...]

for scene_name in scene_dirs:
    scene_color_dir = os.path.join(SCANNET_ROOT, scene_name, "color")
    scene_label_dir = os.path.join(SCANNET_ROOT, scene_name, "label")
    save_dir = os.path.join(SAVE_ROOT, scene_name)
    os.makedirs(save_dir, exist_ok=True)

    all_files = sorted([f for f in os.listdir(scene_color_dir) if f.endswith(".jpg")])
    if not all_files:
        continue

    scene_mious = []
    print(f"\n[SCENE] {scene_name}: {len(all_files)} images")

    # ══════════════════════════════════════════════════════════════
    # Branch A — 时序软投票（两遍处理）
    # ══════════════════════════════════════════════════════════════
    if TEMPORAL_VOTE:
        PROB_DIR = os.path.join(save_dir, "_probs_tmp")
        os.makedirs(PROB_DIR, exist_ok=True)
        scene_orig_sizes = {}  # base_name → (orig_w, orig_h)

        # ── Pass 1：推理 → 保存 float16 概率图 ───────────────────
        for i in range(0, len(all_files), BATCH_SIZE):
            batch_files = all_files[i : i + BATCH_SIZE]
            pil_images = []
            for img_name in batch_files:
                img = Image.open(os.path.join(scene_color_dir, img_name)).convert("RGB")
                pil_images.append(img)
                scene_orig_sizes[os.path.splitext(img_name)[0]] = img.size

            batch_img = torch.stack([transform(img) for img in pil_images]).to(DEVICE)
            with torch.inference_mode():
                with torch.autocast(DEVICE, dtype=torch.bfloat16):
                    pred_batch = make_inference(
                        batch_img, segmentor,
                        inference_mode="whole", decoder_head_type="m2f",
                        rescale_to=(IMG_SIZE, IMG_SIZE),
                        n_output_channels=N_OUTPUT_CHANNELS,
                        output_activation=partial(torch.nn.functional.softmax, dim=1),
                    )  # [B, C, IMG_SIZE, IMG_SIZE]
                pred_small = prob_pool(pred_batch.float()).cpu().numpy().astype(np.float16)
                # shape: [B, C, PROB_STORE_SIZE, PROB_STORE_SIZE]

            for idx, img_name in enumerate(batch_files):
                base = os.path.splitext(img_name)[0]
                np.save(os.path.join(PROB_DIR, f"{base}.npy"), pred_small[idx])

            print(f"  [{min(i + BATCH_SIZE, len(all_files))}/{len(all_files)}] pass 1")

        # ── Pass 2：时序窗口平均概率 → argmax → 保存 mask ─────────
        base_names = [os.path.splitext(f)[0] for f in all_files]
        for i, (img_name, base) in enumerate(zip(all_files, base_names)):
            lo = max(0, i - VOTE_WINDOW)
            hi = min(len(all_files), i + VOTE_WINDOW + 1)

            # 在原始类别空间（ADE20K 150维 或 NYU40 41维）内累加概率
            acc = np.zeros((N_OUTPUT_CHANNELS, PROB_STORE_SIZE, PROB_STORE_SIZE), dtype=np.float32)
            for j in range(lo, hi):
                acc += np.load(os.path.join(PROB_DIR, f"{base_names[j]}.npy")).astype(np.float32)
            avg_prob = acc / (hi - lo)  # [C, PS, PS]

            # 先在原始类别空间 argmax，再做 ADE20K→NYU40 映射（顺序很重要！）
            mask = avg_prob.argmax(axis=0).astype(np.uint8)
            if MODE == "ade20k_mapped":
                nyu40_mask = map_ade20k_to_nyu40(mask)
            else:
                nyu40_mask = mask

            orig_w, orig_h = scene_orig_sizes[base]
            nyu40_mask_resized = cv2.resize(nyu40_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            colored_mask = colorize_mask(nyu40_mask_resized, palette_array)
            colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
            img_array = np.array(Image.open(os.path.join(scene_color_dir, img_name)).convert("RGB"))
            image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(image_bgr, 0.6, colored_mask_bgr, 0.4, 0)

            cv2.imwrite(os.path.join(save_dir, f"{base}_mask_id.png"), nyu40_mask_resized)
            cv2.imwrite(os.path.join(save_dir, f"{base}_mask.png"),    colored_mask_bgr)
            cv2.imwrite(os.path.join(save_dir, f"{base}_overlay.png"), overlay)

            if EVAL_WITH_GT:
                gt_path = os.path.join(scene_label_dir, base + ".png")
                if os.path.exists(gt_path):
                    gt_label = np.array(Image.open(gt_path))
                    miou = compute_miou(nyu40_mask_resized, gt_label, num_classes=41)
                    scene_mious.append(miou)
                    all_per_image_mious.append((scene_name, base, miou))

        print(f"  [{len(all_files)}/{len(all_files)}] pass 2 done")

        if not KEEP_PROBS:
            shutil.rmtree(PROB_DIR)

    # ══════════════════════════════════════════════════════════════
    # Branch B — 原始单遍处理（不投票）
    # ══════════════════════════════════════════════════════════════
    else:
        for i in range(0, len(all_files), BATCH_SIZE):
            batch_files = all_files[i : i + BATCH_SIZE]
            pil_images = []
            orig_sizes = []

            for img_name in batch_files:
                img = Image.open(os.path.join(scene_color_dir, img_name)).convert("RGB")
                pil_images.append(img)
                orig_sizes.append(img.size)  # (w, h)

            tensors = [transform(img) for img in pil_images]
            batch_img = torch.stack(tensors, dim=0).to(DEVICE)

            with torch.inference_mode():
                with torch.autocast(DEVICE, dtype=torch.bfloat16):
                    pred_batch = make_inference(
                        batch_img, segmentor,
                        inference_mode="whole", decoder_head_type="m2f",
                        rescale_to=(IMG_SIZE, IMG_SIZE),
                        n_output_channels=N_OUTPUT_CHANNELS,
                        output_activation=partial(torch.nn.functional.softmax, dim=1),
                    )
                    seg_maps = pred_batch.argmax(dim=1, keepdim=True)  # [B, 1, H, W]

            for idx, img_name in enumerate(batch_files):
                base = os.path.splitext(img_name)[0]
                orig_w, orig_h = orig_sizes[idx]
                mask = seg_maps[idx, 0].cpu().numpy().astype(np.uint8)

                if MODE == "ade20k_mapped":
                    nyu40_mask = map_ade20k_to_nyu40(mask)
                else:
                    nyu40_mask = mask

                nyu40_mask_resized = cv2.resize(nyu40_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                colored_mask = colorize_mask(nyu40_mask_resized, palette_array)
                colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
                img_array = np.array(pil_images[idx])
                image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(image_bgr, 0.6, colored_mask_bgr, 0.4, 0)

                cv2.imwrite(os.path.join(save_dir, f"{base}_mask_id.png"), nyu40_mask_resized)
                cv2.imwrite(os.path.join(save_dir, f"{base}_mask.png"),    colored_mask_bgr)
                cv2.imwrite(os.path.join(save_dir, f"{base}_overlay.png"), overlay)

                if EVAL_WITH_GT:
                    gt_path = os.path.join(scene_label_dir, base + ".png")
                    if os.path.exists(gt_path):
                        gt_label = np.array(Image.open(gt_path))
                        miou = compute_miou(nyu40_mask_resized, gt_label, num_classes=41)
                        scene_mious.append(miou)
                        all_per_image_mious.append((scene_name, base, miou))

            print(f"  batch {i//BATCH_SIZE + 1}/{(len(all_files) + BATCH_SIZE - 1)//BATCH_SIZE} done")

    if scene_mious:
        scene_avg = np.mean(scene_mious)
        all_scene_mious.append((scene_name, scene_avg))
        print(f"  [EVAL] {scene_name} mIoU: {scene_avg:.4f}")

# 总结 & 写入 txt
miou_txt_path = os.path.join(SAVE_ROOT, "miou_results.txt")
with open(miou_txt_path, "w", encoding="utf-8") as f:
    f.write(f"DINOv3 ScanNet Segmentation - mIoU Results\n")
    f.write(f"Mode: {MODE}\n")
    f.write(f"Temporal voting: {'ON (window=±' + str(VOTE_WINDOW) + ')' if TEMPORAL_VOTE else 'OFF'}\n")
    f.write(f"Backbone: {BACKBONE_DIR}\n")
    f.write(f"Head: {HEAD_DIR}\n")
    f.write(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        f.write(f" ({torch.cuda.get_device_name(0)})")
    f.write(f"\nImg size: {IMG_SIZE}\n")
    f.write(f"{'='*60}\n\n")

    # 每个场景的 mIoU
    f.write(f"{'Scene':<25} {'mIoU':>10}\n")
    f.write(f"{'-'*35}\n")
    for scene_name, miou in all_scene_mious:
        f.write(f"{scene_name:<25} {miou:>10.4f}\n")

    # 总体
    if all_scene_mious:
        overall = np.mean([m for _, m in all_scene_mious])
        f.write(f"{'-'*35}\n")
        f.write(f"{'Overall':<25} {overall:>10.4f}\n")
        f.write(f"{'Num scenes':<25} {len(all_scene_mious):>10d}\n")

    # 每张图的明细
    f.write(f"\n{'='*60}\n")
    f.write(f"Per-image details\n")
    f.write(f"{'='*60}\n")
    f.write(f"{'Scene':<25} {'Image':<15} {'mIoU':>10}\n")
    f.write(f"{'-'*50}\n")
    for scene_name, img_name, miou in all_per_image_mious:
        f.write(f"{scene_name:<25} {img_name:<15} {miou:>10.4f}\n")

if all_scene_mious:
    overall = np.mean([m for _, m in all_scene_mious])
    print(f"\n{'='*50}")
    print(f"[RESULT] Overall mIoU across {len(all_scene_mious)} scenes: {overall:.4f}")
    print(f"{'='*50}")

print(f"\n[INFO] Results saved to: {SAVE_ROOT}")
print(f"[INFO] mIoU report: {miou_txt_path}")
