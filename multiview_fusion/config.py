import os

class Config:
    """配置类：集中管理所有路径和参数"""
    DATASET = "indoor"

    DEFAULT_PCD_PATHS = {
        "indoor": "/home/czj/datasets/MVSSeg_dataset/indoor.pcd",
        "outdoor": "/home/czj/datasets/MVSSeg_dataset/outdoor.pcd",
    }
    DEFAULT_IMG_PATHS = {
        "indoor": "/home/czj/datasets/MVSSeg_dataset/fastlivo_output_indoor_107/image",
        "outdoor": "/home/czj/datasets/MVSSeg_dataset/fastlivo_output_outdoor_1s/image",
    }
    DEFAULT_PCD_PATH = DEFAULT_PCD_PATHS[DATASET]
    DEFAULT_IMG_PATH = DEFAULT_IMG_PATHS[DATASET]

    DEFAULT_SEG_PATH = os.path.join(os.path.dirname(DEFAULT_IMG_PATH), "seg_npy")

    # YAML配置路径
    DEFAULT_YAML_PATH = "/home/czj/program/r3live_semantics_ws"
    EXT_YAML_PATH = os.path.join(DEFAULT_YAML_PATH, "avia.yaml")
    INTER_YAML_PATH = os.path.join(DEFAULT_YAML_PATH, "camera_pinhole.yaml")
    
    # 输入图像尺寸
    IMAGE_SHAPE = (512, 640)
    