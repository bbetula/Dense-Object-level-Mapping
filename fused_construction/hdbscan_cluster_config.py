"""
HDBSCAN 分类别聚类参数配置

根据 ADE20K / ScanNet NYU40 类别的几何特征，为不同类别分配不同的 HDBSCAN 参数。
PCD 文件名经过 sanitize（空格→下划线），查找时自动做归一化匹配。
"""
import re
from dataclasses import dataclass


@dataclass
class ClusterProfile:
    """单个类别的 HDBSCAN 聚类参数"""
    min_cluster_size: int = 5
    min_samples: int = 3
    min_cluster_points: int = 10
    max_cluster_filter: int = 50
    noise_absorb_factor: float = 15.0
    epsilon_multiplier: float = 1.0
    epsilon_clip_min: float = 0.05
    epsilon_clip_max: float = 2.0
    cluster_selection_method: str = "eom"
    # SOR（Statistical Outlier Removal）预处理参数
    sor_nb_neighbors: int = 20
    sor_std_ratio: float = 2.0
    # bbox 后过滤参数
    bbox_max_volume: float = 30.0       # 最大 bbox 体积 (m³)
    bbox_max_extent: float = 8.0        # 单维最大尺寸 (m)
    bbox_min_density: float = 1.0       # 最小点密度 (点/m³)
    # PCA 平面度过滤（过滤墙壁/天花板投影形成的平面簇）
    planarity_threshold: float = 0.4    # PCA planarity > 此值视为平面
    planarity_min_volume: float = 0.5   # 仅过滤体积 > 此值的平面簇 (m³)


# =====================================================
#  聚类参数模板
# =====================================================
CLUSTER_PROFILES = {
    # ── 默认：中型紧凑物体（椅子、桌面电器等标准家具）──
    "default": ClusterProfile(),

    # ── 大型紧凑物体：车辆、大型家具 ──
    # 点密集、体积大，需要更高的最小簇阈值过滤碎片
    "compact_large": ClusterProfile(
        min_cluster_size=10,
        min_samples=5,
        min_cluster_points=20,
        max_cluster_filter=80,
        noise_absorb_factor=12.0,
        sor_nb_neighbors=30,
        sor_std_ratio=2.5,
        bbox_max_volume=150.0,
        bbox_max_extent=15.0,
        bbox_min_density=0.5,
        planarity_threshold=0.35,
        planarity_min_volume=2.0,
    ),

    # ── 细长结构：杆、灯柱、栏杆 ──
    # 每个实例点极少，需放宽最小簇；增大 epsilon 合并同一根杆的不同段
    "thin_structure": ClusterProfile(
        min_cluster_size=3,
        min_samples=2,
        min_cluster_points=5,
        max_cluster_filter=30,
        noise_absorb_factor=20.0,
        epsilon_multiplier=1.5,
        epsilon_clip_max=3.0,
        sor_nb_neighbors=10,
        sor_std_ratio=1.5,
        bbox_max_volume=5.0,
        bbox_max_extent=12.0,
        bbox_min_density=2.0,
        planarity_threshold=0.6,
        planarity_min_volume=0.3,
    ),

    # ── 小型移动目标：人、动物、自行车 ──
    # 点少且分散，需降低 epsilon 防止远距离目标被合并为一个簇
    "small_dynamic": ClusterProfile(
        min_cluster_size=5,
        min_samples=2,
        min_cluster_points=8,
        max_cluster_filter=30,
        noise_absorb_factor=8.0,
        epsilon_multiplier=0.6,
        epsilon_clip_max=1.0,
        sor_nb_neighbors=15,
        sor_std_ratio=1.0,
        bbox_max_volume=8.0,
        bbox_max_extent=3.0,
        bbox_min_density=3.0,
        planarity_threshold=0.4,
        planarity_min_volume=0.3,
    ),

    # ── 植被：树、灌木、棕榈 ──
    # 形状不规则、点云散乱，需高阈值过滤碎片
    "vegetation": ClusterProfile(
        min_cluster_size=15,
        min_samples=5,
        min_cluster_points=30,
        max_cluster_filter=100,
        noise_absorb_factor=10.0,
        epsilon_multiplier=1.2,
        sor_nb_neighbors=25,
        sor_std_ratio=2.0,
        bbox_max_volume=500.0,
        bbox_max_extent=20.0,
        bbox_min_density=0.3,
        planarity_threshold=0.35,
        planarity_min_volume=1.0,
    ),

    # ── 小型物品：瓶、罐、球、书等 ──
    # 体积小、点极少，需最宽松的最小簇
    "small_item": ClusterProfile(
        min_cluster_size=3,
        min_samples=2,
        min_cluster_points=5,
        max_cluster_filter=20,
        noise_absorb_factor=20.0,
        sor_nb_neighbors=10,
        sor_std_ratio=1.5,
        bbox_max_volume=2.0,
        bbox_max_extent=2.0,
        bbox_min_density=5.0,
        planarity_threshold=0.5,
        planarity_min_volume=0.1,
    ),

    # ── 中型稀疏室外物体：雕塑、喷泉、标牌 ──
    # 需要更大的 epsilon 防止同一物体被过度分割
    "medium_sparse": ClusterProfile(
        min_cluster_size=5,
        min_samples=2,
        min_cluster_points=8,
        max_cluster_filter=40,
        noise_absorb_factor=15.0,
        epsilon_multiplier=2.0,
        epsilon_clip_max=5.0,
        sor_nb_neighbors=15,
        sor_std_ratio=1.5,
        bbox_max_volume=80.0,
        bbox_max_extent=10.0,
        bbox_min_density=0.5,
        planarity_threshold=0.4,
        planarity_min_volume=0.5,
    ),
}


# =====================================================
#  ADE20K 类别 → 参数模板映射（150 类）
# =====================================================
_ADE20K_PROFILE_GROUPS = {
    "compact_large": [
        # 车辆
        "car", "van", "truck", "bus", "boat", "ship", "airplane", "tank",
        # 大型家具 / 电器
        "bed", "sofa", "table", "cabinet", "wardrobe", "bookcase",
        "refrigerator", "bathtub", "pool table", "kitchen island",
        "bar", "buffet", "swimming pool", "chest of drawers", "fireplace",
        # 大型建筑结构（通常不做检测目标，但以防万一）
        "building", "house", "skyscraper", "hovel", "tower", "bridge",
        "grandstand", "stage", "pier",
    ],
    "thin_structure": [
        "pole", "streetlight", "traffic light", "column",
        "bannister", "railing", "fence", "light",
    ],
    "small_dynamic": [
        "person", "animal", "bicycle", "minibike",
    ],
    "vegetation": [
        "tree", "plant", "palm",
    ],
    "small_item": [
        "box", "bottle", "pot", "vase", "basket", "barrel", "ashcan",
        "bag", "ball", "tray", "plate", "glass", "book", "flower",
        "food", "plaything", "clock", "fan", "radiator", "towel",
        "pillow", "cushion", "blanket", "case", "step", "apparel",
        "hood", "sconce", "poster", "trade name",
    ],
    "medium_sparse": [
        "sculpture", "fountain", "bench", "booth", "signboard",
        "flag", "tent", "canopy", "awning",
    ],
    # 未列出的类别（wall, floor, ceiling, road, sidewalk, chair, desk,
    # armchair, swivel chair, seat, stool, counter, countertop, coffee table,
    # toilet, sink, stove, oven, microwave, dishwasher, washer, shower,
    # ottoman, cradle, computer, television receiver, monitor, screen,
    # crt screen, arcade machine, conveyer belt, escalator, lamp, chandelier,
    # door, windowpane, screen door, stairs, stairway, base, shelf, curtain,
    # blind, mirror, painting, rug, bulletin board, whiteboard, ...
    # ）全部使用 "default"
}

# =====================================================
#  ScanNet NYU40 类别 → 参数模板映射（41 类）
# =====================================================
_SCANNET_PROFILE_GROUPS = {
    "compact_large": [
        "cabinet", "bed", "sofa", "table", "bookshelf", "counter",
        "dresser", "bathtub", "refrigerator",
    ],
    "small_dynamic": [
        "person",
    ],
    "small_item": [
        "pillow", "towel", "box", "books", "paper", "bag",
        "clothes", "floor mat",
    ],
    # 其余（chair, desk, shelves, lamp, television, ...）使用 "default"
}


# =====================================================
#  内部构建查找表
# =====================================================
def _sanitize(name: str) -> str:
    """与 generate_n_pcd_bbox.sanitize_category_name 保持一致（额外转小写）"""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip()).lower()


def _build_lookup(groups: dict) -> dict:
    lookup = {}
    for profile_name, categories in groups.items():
        for cat in categories:
            lookup[_sanitize(cat)] = profile_name
    return lookup


_ADE20K_LOOKUP = _build_lookup(_ADE20K_PROFILE_GROUPS)
_SCANNET_LOOKUP = _build_lookup(_SCANNET_PROFILE_GROUPS)


def get_cluster_profile(category_name: str, label_choice: str = "ADE20K") -> ClusterProfile:
    """
    根据类别名称和数据集返回对应的聚类参数。

    Args:
        category_name: 类别名称（支持原始名或 sanitized 名，如 "traffic light" / "traffic_light"）
        label_choice: "ADE20K" 或 "SCANNET_NYU40"

    Returns:
        ClusterProfile 实例
    """
    key = _sanitize(re.sub(r"_\d+$", "", category_name))

    if label_choice == "ADE20K":
        profile_name = _ADE20K_LOOKUP.get(key, "default")
    else:
        profile_name = _SCANNET_LOOKUP.get(key, "default")

    return CLUSTER_PROFILES[profile_name]


def get_profile_name(category_name: str, label_choice: str = "ADE20K") -> str:
    """返回类别对应的模板名称（用于日志打印）"""
    key = _sanitize(re.sub(r"_\d+$", "", category_name))
    if label_choice == "ADE20K":
        return _ADE20K_LOOKUP.get(key, "default")
    return _SCANNET_LOOKUP.get(key, "default")
