from pathlib import Path
import shutil
from class_statics_config import SCANNET_NYU40_CATEGORIES, ADE20K_CATEGORIES, LABEL_CHOICE

# ===== 配置区 =====
if LABEL_CHOICE == "SCANNET_NYU40":
    INPUT_BASE_DIR = Path("/data1/data/scannet/output/color_separated_scenes")  
    # INPUT_BASE_DIR = Path("/data1/data/scannet/output_single/color_separated_scenes")  
    OUTPUT_BASE_DIR = INPUT_BASE_DIR.parent / "detection_targets"               # 筛选结果根目录

    # 类别筛选模式：
    #   "exclude"  — 从 CATEGORIES 中排除指定类别，保留其余所有类别
    #   "include"  — 只保留手动指定的类别
    CATEGORY_MODE = "exclude"

    # CATEGORY_MODE = "exclude" 时：从 CATEGORIES 中去掉这些类别
    EXCLUDE_CATEGORIES = {
        "wall", "floor", "ceiling", "unlabeled"
    }

    # CATEGORY_MODE = "include" 时：只保留这些类别
    INCLUDE_CATEGORIES = {
        "chair", "sofa", "table", "desk", "bed", "cabinet", "bookshelf", "television", "lamp"
    }

    # 室外场景快捷开关：True 时强制使用室外类别（覆盖上面的设置）
    OUTSCENE = False
    OUTSCENE_CATEGORIES = {
        "car", "truck", "bus",
        "person", "pedestrian",
        "rider", "cyclist", "bicycle",
    }
    CATEGORIES = SCANNET_NYU40_CATEGORIES

elif LABEL_CHOICE == "ADE20K":
    INPUT_BASE_DIR = Path("/data1/user/data/fastlivo_output_qs2_03.17/lidar/res/color_separated_scenes")
    OUTPUT_BASE_DIR = INPUT_BASE_DIR.parent / "detection_targets"               # 筛选结果根目录

    # 类别筛选模式：
    #   "exclude"  — 从 CATEGORIES 中排除指定类别，保留其余所有类别
    #   "include"  — 只保留手动指定的类别
    CATEGORY_MODE = "exclude"

    # CATEGORY_MODE = "exclude" 时：从 CATEGORIES 中去掉这些类别（结构/背景/自然环境）
    EXCLUDE_CATEGORIES = {
        "wall", "floor", "ceiling", "fence"  
    }

    # CATEGORY_MODE = "include" 时：只保留这些类别（对应 ScanNet 的感兴趣目标）
    INCLUDE_CATEGORIES = {
        "chair", "swivel chair", "armchair", "seat", "stool", "bench",  # ScanNet: chair
        "sofa", "ottoman", "cushion",                                    # ScanNet: sofa
        "table", "coffee table", "pool table", "counter", "countertop", # ScanNet: table, counter
        "desk",                                                          # ScanNet: desk
        "bed", "pillow", "blanket", "cradle",                           # ScanNet: bed, pillow
        "cabinet", "chest of drawers", "wardrobe", "bookcase",          # ScanNet: cabinet, bookshelf
        "shelf", "buffet",                                               # ScanNet: shelves
        "television receiver", "monitor", "screen", "crt screen",       # ScanNet: television
        "lamp", "chandelier", "sconce", "light",                        # ScanNet: lamp
        "refrigerator", "oven", "stove", "microwave", "dishwasher",     # 厨电
        "sink", "bathtub", "toilet", "washer", "shower",                # ScanNet: sink, bathtub, toilet
        "mirror",                                                        # ScanNet: mirror
    }

    # 室外场景快捷开关：True 时强制使用室外类别（覆盖上面的设置）
    OUTSCENE = False
    OUTSCENE_CATEGORIES = {
        "car", "van", "truck", "bus", "boat", "ship", "airplane",      # 车辆/交通
        "person",                                                        # 行人
        "bicycle", "minibike",                                           # 骑行者
    }
    CATEGORIES = ADE20K_CATEGORIES
# ==================================

def _build_target_categories() -> set:
    if OUTSCENE:
        return OUTSCENE_CATEGORIES
    if CATEGORY_MODE == "exclude":
        return {k for k in CATEGORIES if k not in EXCLUDE_CATEGORIES}
    return INCLUDE_CATEGORIES

TARGET_CATEGORIES = _build_target_categories()

def is_target_file(stem: str) -> bool:
    stem_lower = stem.lower()
    return any(keyword in stem_lower for keyword in TARGET_CATEGORIES)

def main() -> None:
    scene_dirs = sorted(d for d in INPUT_BASE_DIR.iterdir() if d.is_dir())
    if not scene_dirs:
        print(f"在 {INPUT_BASE_DIR} 中未找到场景子文件夹。")
        return
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"共找到 {len(scene_dirs)} 个场景，输出根目录: {OUTPUT_BASE_DIR}")
    for scene_dir in scene_dirs:
        files = sorted(scene_dir.glob("*.pcd"))
        if not files:
            print(f"场景 {scene_dir.name}: 无 PCD 文件，跳过。")
            continue
        scene_out_dir = OUTPUT_BASE_DIR / scene_dir.name
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for pcd_path in files:
            if is_target_file(pcd_path.stem):
                shutil.copy2(pcd_path, scene_out_dir / pcd_path.name)
                kept += 1
                print(f"  保留: {pcd_path.name}")
        print(f"场景 {scene_dir.name}: 共复制 {kept} 个目标类别 PCD。")
    print("全部场景筛选完成。")

if __name__ == "__main__":
    main()
