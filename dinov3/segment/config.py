# config.py
"""项目配置文件：集中管理路径、URL、常量等配置"""
import os

# ===================== 路径配置 =====================
REPO_DIR = "/home/czj/program/MVSSeg/dinov3"
PALETTE_FILE = os.path.join(REPO_DIR, "segment", "dataset_yaml", "ADE20k.yaml")
ALL_IMAGES_DIR = {
    "indoor": "/home/czj/datasets/MVSSeg_dataset/fastlivo_output_indoor_107/image",
    "outdoor": "/home/czj/datasets/MVSSeg_dataset/fastlivo_output_outdoor_1s/image"
}
IMAGES_DIR = ALL_IMAGES_DIR["outdoor"]

# ===================== 模型权重URL配置 =====================
BACKBONE_WEIGHTS_URL = (
    "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
    "?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaGNiemFiOTNzZzMwOWhnbG9kaTd5aTdmIiwiUmVzb3VyY2Ui"
    "OiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjUyMDE0OTB9fX1dfQ__"
    "&Signature=NLF2B0QuchPcUM-U2t9k180wsIfMnYa6lDbjPn1gyZrl5YOL2WXp-JFcrYwjevzDbtHOMIwAnze0zDYez4jgcdWNrVXcRLFS4EXGmHF1xhrk02I8U1iuQx-qZO2DGmqN5wTRgYaBPF2ittQlDOziiuveaAl3PQhUHPG8ke4OPWMIxxANUU8jw1WT4QJgH5ZLsCry3-cozX54-LMdNdHK%7ECLZMPdkWSGeInmInG1xHHYcybKl9Jo0GQBSSeTfrvoTiiHZOjxksW7Z3yimrxsssSl8%7EraWFhrRGa2RSad0LBZF0QFR8E7YNU9y2CzO6EcPlbR%7EG8Wzf0RyK-880FlfcA__"
    "&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1176142748036319"
)

HEAD_WEIGHTS_URL = (
    "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
    "?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaGNiemFiOTNzZzMwOWhnbG9kaTd5aTdmIiwiUmVzb3VyY2Ui"
    "OiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjUyMDE0OTB9fX1dfQ__"
    "&Signature=NLF2B0QuchPcUM-U2t9k180wsIfMnYa6lDbjPn1gyZrl5YOL2WXp-JFcrYwjevzDbtHOMIwAnze0zDYez4jgcdWNrVXcRLFS4EXGmHF1xhrk02I8U1iuQx-qZO2DGmqN5wTRgYaBPF2ittQlDOziiuveaAl3PQhUHPG8ke4OPWMIxxANUU8jw1WT4QJgH5ZLsCry3-cozX54-LMdNdHK%7ECLZMPdkWSGeInmInG1xHHYcybKl9Jo0GQBSSeTfrvoTiiHZOjxksW7Z3yimrxsssSl8%7EraWFhrRGa2RSad0LBZF0QFR8E7YNU9y2CzO6EcPlbR%7EG8Wzf0RyK-880FlfcA__"
    "&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1176142748036319"
)