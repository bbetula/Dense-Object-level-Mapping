from pathlib import Path
import shutil
from class_statics_config import SCANNET_NYU40_CATEGORIES

# ===== 配置区 =====
INPUT_BASE_DIR = Path("/data1/data/scannet/output_bbox/color_separated_scenes")  
# INPUT_BASE_DIR = Path("/data1/data/scannet/output_single_bbox/color_separated_scenes")  
OUTPUT_BASE_DIR = INPUT_BASE_DIR.parent / "detection_targets"               # 筛选结果根目录

# 类别筛选模式：
#   "exclude"  — 从 SCANNET_NYU40_CATEGORIES 中排除指定类别，保留其余所有类别
#   "include"  — 只保留手动指定的类别
CATEGORY_MODE = "exclude"

# CATEGORY_MODE = "exclude" 时：从 SCANNET_NYU40_CATEGORIES 中去掉这些类别
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
# ==================================

def _build_target_categories() -> set:
    if OUTSCENE:
        return OUTSCENE_CATEGORIES
    if CATEGORY_MODE == "exclude":
        return {k for k in SCANNET_NYU40_CATEGORIES if k not in EXCLUDE_CATEGORIES}
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
