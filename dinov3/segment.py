"dinov3 team, dinov3 segmentation example"
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


def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

backbone_weights_url = "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaGNiemFiOTNzZzMwOWhnbG9kaTd5aTdmIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjUyMDE0OTB9fX1dfQ__&Signature=NLF2B0QuchPcUM-U2t9k180wsIfMnYa6lDbjPn1gyZrl5YOL2WXp-JFcrYwjevzDbtHOMIwAnze0zDYez4jgcdWNrVXcRLFS4EXGmHF1xhrk02I8U1iuQx-qZO2DGmqN5wTRgYaBPF2ittQlDOziiuveaAl3PQhUHPG8ke4OPWMIxxANUU8jw1WT4QJgH5ZLsCry3-cozX54-LMdNdHK%7ECLZMPdkWSGeInmInG1xHHYcybKl9Jo0GQBSSeTfrvoTiiHZOjxksW7Z3yimrxsssSl8%7EraWFhrRGa2RSad0LBZF0QFR8E7YNU9y2CzO6EcPlbR%7EG8Wzf0RyK-880FlfcA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1176142748036319"
head_weights_url = "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaGNiemFiOTNzZzMwOWhnbG9kaTd5aTdmIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjUyMDE0OTB9fX1dfQ__&Signature=NLF2B0QuchPcUM-U2t9k180wsIfMnYa6lDbjPn1gyZrl5YOL2WXp-JFcrYwjevzDbtHOMIwAnze0zDYez4jgcdWNrVXcRLFS4EXGmHF1xhrk02I8U1iuQx-qZO2DGmqN5wTRgYaBPF2ittQlDOziiuveaAl3PQhUHPG8ke4OPWMIxxANUU8jw1WT4QJgH5ZLsCry3-cozX54-LMdNdHK%7ECLZMPdkWSGeInmInG1xHHYcybKl9Jo0GQBSSeTfrvoTiiHZOjxksW7Z3yimrxsssSl8%7EraWFhrRGa2RSad0LBZF0QFR8E7YNU9y2CzO6EcPlbR%7EG8Wzf0RyK-880FlfcA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1176142748036319"
segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", 
                            weights=head_weights_url, 
                            backbone_weights=backbone_weights_url)

save_dir = 'segment_results'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"result.png")

img_size = 896
img  = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        pred_vit7b = segmentor(batch_img)  # raw predictions  
        # actual segmentation map
        segmentation_map_vit7b = make_inference(
            batch_img,
            segmentor,
            inference_mode="slide",
            decoder_head_type="m2f",
            rescale_to=(img.size[-1], img.size[-2]),
            n_output_channels=150,
            crop_size=(img_size, img_size),
            stride=(img_size, img_size),
            output_activation=partial(torch.nn.functional.softmax, dim=1),
        ).argmax(dim=1, keepdim=True)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(segmentation_map_vit7b[0,0].cpu(), cmap=colormaps["Spectral"])
plt.axis("off")

plt.savefig(save_path)