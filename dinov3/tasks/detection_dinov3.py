import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(SCRIPT_DIR.parent), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

OUTPUT_DIR = str(SCRIPT_DIR / "output" / "detection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DINOV3_LOCATION = str(SCRIPT_DIR)
HEAD_DIR = str(SCRIPT_DIR / "checkpoint" / "dinov3_pretrained" / "DINOv3 Adapters" / "dinov3_vit7b16_coco_detr_head-b0235ff7.pth")
BACKBONE_DIR = str(SCRIPT_DIR / "checkpoint" / "dinov3_pretrained" / "DINOv3 ViT LVD-1689M" / "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")

MODEL_NAME = "dinov3_vit7b16_de"

detector = torch.hub.load(
    repo_or_dir=DINOV3_LOCATION,
    model=MODEL_NAME,
    source="local",
    weights=HEAD_DIR,
    backbone_weights=BACKBONE_DIR,
)
detector.cuda().eval()

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def detector_img(img_filename, score_threshold=0.3):
    pil_img = Image.open(img_filename).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((768, 768)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    input_tensor = transform(pil_img).unsqueeze(0).cuda()

    with torch.inference_mode():
        detections = detector(input_tensor)[0]

    orig_w, orig_h = pil_img.size
    scale_x = orig_w / 768.0
    scale_y = orig_h / 768.0

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for score, label, box in zip(detections['scores'], detections['labels'], detections['boxes']):
        if score.item() > score_threshold:
            x0, y0, x1, y1 = box.cpu().numpy()
            x0, x1 = x0 * scale_x, x1 * scale_x
            y0, y1 = y0 * scale_y, y1 * scale_y
            class_idx = int(label.item())
            class_name = COCO_CLASSES[class_idx] if 0 <= class_idx < len(COCO_CLASSES) else str(class_idx)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                         fill=False, color='red', linewidth=2))
            ax.text(x0, y0, f'{class_name}: {score:.2f}', fontsize=12,
                    color='white', bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')

    out_name = Path(img_filename).stem + "_det.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"保存到 {out_path}")


if __name__ == "__main__":
    detector_img('/data1/user/data/fastlivo_output_qs2_03.17/image/res_dinov3_whole/1590195416609791744_original.png')
