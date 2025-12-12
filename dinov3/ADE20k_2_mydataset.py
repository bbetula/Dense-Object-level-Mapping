import torch
import numpy as np

class ADE20KDataset():
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                 'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                 'chandelier', 'awning', 'streetlight', 'booth',
                 'television receiver', 'airplane', 'dirt track', 'apparel',
                 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                 'conveyer belt', 'canopy', 'washer', 'plaything',
                 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                 'clock', 'flag'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]])

class CityscapesDataset():
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])


# 定义ADE20K与Cityscapes的类别映射
# 键：ADE20K的类别名称（对应ADE20KDataset.METAINFO['classes']）
# 值：Cityscapes的类别ID（对应CityscapesDataset.METAINFO['classes']的索引）
ade2city_mapping = {
    # ADE20K类别名: Cityscapes类别ID
    'road': 0,               # Cityscapes: road (0)
    'sidewalk': 1,           # Cityscapes: sidewalk (1)
    'building': 2,           # Cityscapes: building (2)
    'wall': 3,               # Cityscapes: wall (3)
    'fence': 4,              # Cityscapes: fence (4)
    'pole': 5,               # Cityscapes: pole (5)
    'traffic light': 6,      # Cityscapes: traffic light (6)
    'signboard': 7,          # Cityscapes: traffic sign (7)
    'tree': 8,               # Cityscapes: vegetation (8) → 合并ADE的tree/grass/plant等
    'grass': 8,              # 归为vegetation
    'plant': 8,              # 归为vegetation
    'earth': 9,              # Cityscapes: terrain (9)
    'hill': 9,               # 归为terrain
    'land': 9,               # 归为terrain
    'sky': 10,               # Cityscapes: sky (10)
    'person': 11,            # Cityscapes: person (11)
    'car': 13,               # Cityscapes: car (13)
    'truck': 14,             # Cityscapes: truck (14)
    'bus': 15,               # Cityscapes: bus (15)
    'motorcycle': 17,        # Cityscapes: motorcycle (17)
    'bicycle': 18,           # Cityscapes: bicycle (18)
    # 补充：ADE中无rider/train，暂不映射；可根据需求扩展
}

# 构造重定向的权重矩阵（150→19维）
def build_ade2city_redirect_matrix(ade_name2idx, ade2city_mapping, city_num_classes=19):
    ade_num_classes = len(ade_name2idx)  # 150
    # 初始化重定向矩阵：ADE维度→City维度，默认全0
    redirect_matrix = torch.zeros((city_num_classes, ade_num_classes), dtype=torch.float32)
    
    for ade_name, city_idx in ade2city_mapping.items():
        if ade_name in ade_name2idx:
            ade_idx = ade_name2idx[ade_name]
            redirect_matrix[city_idx, ade_idx] = 1.0  # 对应位置置1，其余为0
    return redirect_matrix

# 推理结果重定向与后处理
def redirect_ade2city(ade_seg_logits):
    """
    将ADE20K的150维推理结果重定向为Cityscapes的19维结果
    
    Args:
        ade_seg_logits (torch.Tensor): ADE推理输出，shape=[B, 150, H, W]（softmax后的概率）
        redirect_matrix (torch.Tensor): 重定向矩阵，shape=[19, 150]
    
    Returns:
        city_seg_label (torch.Tensor): Cityscapes格式的分割标签，shape=[B, 1, H, W]
    """
    # 获取ADE20K类别名→索引（ID）的映射
    ade_classes = ADE20KDataset.METAINFO['classes']
    ade_name2idx = {name.strip(): idx for idx, name in enumerate(ade_classes)}
    # 生成重定向矩阵
    redirect_matrix = build_ade2city_redirect_matrix(ade_name2idx, ade2city_mapping)

    # 1. 重定向：150维 → 19维（仅保留匹配类别，其余置0）
    city_seg_logits = torch.einsum("ca, bahw -> bchw", redirect_matrix, ade_seg_logits)  # [B, 19, H, W]
    # 2. 对每像素取最大概率的类别ID
    city_seg_label = city_seg_logits.argmax(dim=1, keepdim=True)  # [B, 1, H, W]
    
    return city_seg_label
