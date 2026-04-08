# 配置常量
COLOR_TOLERANCE = 7  # 每个RGB通道允许的误差 ADE20K:7
DEFAULT_MAP_PATHS = {
    "indoor1": "/data1/user/data/fastlivo_output_indoor_107/lidar/res",
    "outdoor1": "/data1/user/data/fastlivo_output_outdoor_1s/lidar/res",
    "outdoor2": "/data1/user/data/fastlivo_output_qs2_03.17/lidar/res",
}
DATA_CHOICE = "outdoor2"
LABEL_CHOICE = "ADE20K"  # "ADE20K" or "SCANNET_NYU40"
DEFAULT_MAP_PATH = DEFAULT_MAP_PATHS[DATA_CHOICE]

# 多类别颜色配置文件
# 格式：{类别名: RGB颜色列表}
ADE20K_CATEGORIES = {
    "wall": [120, 120, 120],
    "building": [180, 120, 120],
    "sky": [6, 230, 230],
    "floor": [80, 50, 50],
    "tree": [4, 200, 3],
    "ceiling": [120, 120, 80],
    "road": [140, 140, 140],
    "bed": [204, 5, 255],
    "windowpane": [230, 230, 230],
    "grass": [4, 250, 7],
    "cabinet": [224, 5, 255],
    "sidewalk": [235, 255, 7],
    "person": [150, 5, 61],
    "earth": [120, 120, 70],
    "door": [8, 255, 51],
    "table": [255, 6, 82],
    "mountain": [143, 255, 140],
    "plant": [204, 255, 4],
    "curtain": [255, 51, 7],
    "chair": [204, 70, 3],
    "car": [0, 102, 200],
    "water": [61, 230, 250],
    "painting": [255, 6, 51],
    "sofa": [11, 102, 255],
    "shelf": [255, 7, 71],
    "house": [255, 9, 224],
    "sea": [9, 7, 230],
    "mirror": [220, 220, 220],
    "rug": [255, 9, 92],
    "field": [112, 9, 255],
    "armchair": [8, 255, 214],
    "seat": [7, 255, 224],
    "fence": [255, 184, 6],
    "desk": [10, 255, 71],
    "rock": [255, 41, 10],
    "wardrobe": [7, 255, 255],
    "lamp": [224, 255, 8],
    "bathtub": [102, 8, 255],
    "railing": [255, 61, 6],
    "cushion": [255, 194, 7],
    "base": [255, 122, 8],
    "box": [0, 255, 20],
    "column": [255, 8, 41],
    "signboard": [255, 5, 153],
    "chest of drawers": [6, 51, 255],
    "counter": [235, 12, 255],
    "sand": [160, 150, 20],
    "sink": [0, 163, 255],
    "skyscraper": [140, 140, 140],
    "fireplace": [250, 10, 15],
    "refrigerator": [20, 255, 0],
    "grandstand": [31, 255, 0],
    "path": [255, 31, 0],
    "stairs": [255, 224, 0],
    "runway": [153, 255, 0],
    "case": [0, 0, 255],
    "pool table": [255, 71, 0],
    "pillow": [0, 235, 255],
    "screen door": [0, 173, 255],
    "stairway": [31, 0, 255],
    "river": [11, 200, 200],
    "bridge": [255, 82, 0],
    "bookcase": [0, 255, 245],
    "blind": [0, 61, 255],
    "coffee table": [0, 255, 112],
    "toilet": [0, 255, 133],
    "flower": [255, 0, 0],
    "book": [255, 163, 0],
    "hill": [255, 102, 0],
    "bench": [194, 255, 0],
    "countertop": [0, 143, 255],
    "stove": [51, 255, 0],
    "palm": [0, 82, 255],
    "kitchen island": [0, 255, 41],
    "computer": [0, 255, 173],
    "swivel chair": [10, 0, 255],
    "boat": [173, 255, 0],
    "bar": [0, 255, 153],
    "arcade machine": [255, 92, 0],
    "hovel": [255, 0, 255],
    "bus": [255, 0, 245],
    "towel": [255, 0, 102],
    "light": [255, 173, 0],
    "truck": [255, 0, 20],
    "tower": [255, 184, 184],
    "chandelier": [0, 31, 255],
    "awning": [0, 255, 61],
    "streetlight": [0, 71, 255],
    "booth": [255, 0, 204],
    "television receiver": [0, 255, 194],
    "airplane": [0, 255, 82],
    "dirt track": [0, 10, 255],
    "apparel": [0, 112, 255],
    "pole": [51, 0, 255],
    "land": [0, 194, 255],
    "bannister": [0, 122, 255],
    "escalator": [0, 255, 163],
    "ottoman": [255, 153, 0],
    "bottle": [0, 255, 10],
    "buffet": [255, 112, 0],
    "poster": [143, 255, 0],
    "stage": [82, 0, 255],
    "van": [163, 255, 0],
    "ship": [255, 235, 0],
    "fountain": [8, 184, 170],
    "conveyer belt": [133, 0, 255],
    "canopy": [0, 255, 92],
    "washer": [184, 0, 255],
    "plaything": [255, 0, 31],
    "swimming pool": [0, 184, 255],
    "stool": [0, 214, 255],
    "barrel": [255, 0, 112],
    "basket": [92, 255, 0],
    "waterfall": [0, 224, 255],
    "tent": [112, 224, 255],
    "bag": [70, 184, 160],
    "minibike": [163, 0, 255],
    "cradle": [153, 0, 255],
    "oven": [71, 255, 0],
    "ball": [255, 0, 163],
    "food": [255, 204, 0],
    "step": [255, 0, 143],
    "tank": [0, 255, 235],
    "trade name": [133, 255, 0],
    "microwave": [255, 0, 235],
    "pot": [245, 0, 255],
    "animal": [255, 0, 122],
    "bicycle": [255, 245, 0],
    "lake": [10, 190, 212],
    "dishwasher": [214, 255, 0],
    "screen": [0, 204, 255],
    "blanket": [20, 0, 255],
    "sculpture": [255, 255, 0],
    "hood": [0, 153, 255],
    "sconce": [0, 41, 255],
    "vase": [0, 255, 204],
    "traffic light": [41, 0, 255],
    "tray": [41, 255, 0],
    "ashcan": [173, 0, 255],
    "fan": [0, 245, 255],
    "pier": [71, 0, 255],
    "crt screen": [122, 0, 255],
    "plate": [0, 255, 184],
    "monitor": [0, 92, 255],
    "bulletin board": [184, 255, 0],
    "shower": [0, 133, 255],
    "radiator": [255, 214, 0],
    "glass": [25, 194, 194],
    "clock": [102, 255, 0],
    "flag": [92, 0, 255]
}

# ScanNet NYU40 类别颜色配置（labels.ply 色盘）
# nyu40id -> 类别名: RGB颜色
# ignore -> 室内后续处理忽略的类别
SCANNET_NYU40_CATEGORIES = {
    "unlabeled": [0, 0, 0],            # 0 ignore
    "wall": [174, 199, 232],           # 1 ignore
    "floor": [152, 223, 138],          # 2 ignore
    "cabinet": [31, 119, 180],         # 3
    "bed": [255, 187, 120],            # 4
    "chair": [188, 189, 34],           # 5
    "sofa": [140, 86, 75],             # 6
    "table": [255, 152, 150],          # 7
    "door": [214, 39, 40],             # 8
    "window": [197, 176, 213],         # 9
    "bookshelf": [148, 103, 189],      # 10
    "picture": [196, 156, 148],        # 11
    "counter": [23, 190, 207],         # 12
    "blinds": [178, 76, 76],           # 13
    "desk": [247, 182, 210],           # 14
    "shelves": [66, 188, 102],         # 15
    "curtain": [219, 219, 141],        # 16
    "dresser": [140, 57, 197],         # 17
    "pillow": [202, 185, 52],          # 18
    "mirror": [51, 176, 203],          # 19
    "floor mat": [200, 54, 131],       # 20
    "clothes": [92, 193, 61],          # 21
    "ceiling": [78, 71, 183],          # 22 ignore
    "books": [172, 114, 82],           # 23
    "refrigerator": [255, 127, 14],    # 24
    "television": [91, 163, 138],      # 25
    "paper": [153, 98, 156],           # 26
    "towel": [140, 153, 101],          # 27
    "shower curtain": [158, 218, 229], # 28
    "box": [100, 125, 154],            # 29
    "whiteboard": [178, 127, 135],     # 30
    "person": [120, 185, 128],         # 31
    "night stand": [146, 111, 194],    # 32
    "toilet": [44, 160, 44],           # 33
    "sink": [112, 128, 144],           # 34
    "lamp": [96, 207, 209],            # 35
    "bathtub": [227, 119, 194],        # 36
    "bag": [213, 92, 176],             # 37
    "otherstructure": [94, 106, 211],  # 38
    "otherfurniture": [82, 84, 163],   # 39
    "otherprop": [100, 85, 144],       # 40
}


