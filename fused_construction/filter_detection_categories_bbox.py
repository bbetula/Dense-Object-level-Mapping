from pathlib import Path
import shutil
from class_statics_config import SCANNET_NYU40_CATEGORIES, ADE20K_CATEGORIES, LABEL_CHOICE, OUTSCENE
from generate_n_pcd_bbox import OUTPUT_BASE_DIR as GENERATED_COLOR_SEPARATED_DIR

# ===== 配置区 =====
INPUT_BASE_DIR = GENERATED_COLOR_SEPARATED_DIR
OUTPUT_BASE_DIR = INPUT_BASE_DIR.parent / "detection_targets"
if LABEL_CHOICE == "SCANNET_NYU40":

    CATEGORIES = SCANNET_NYU40_CATEGORIES

    CATEGORY_MODE = "exclude"

    EXCLUDE_CATEGORIES = {
        "wall", "floor", "ceiling", "unlabeled"
    }

    INCLUDE_CATEGORIES = {
        "chair", "sofa", "table", "desk", "bed", "cabinet", "bookshelf", "television", "lamp"
    }

    OUTSCENE_CATEGORIES = {
        "car", "truck", "bus",
        "person", "pedestrian",
        "rider", "cyclist", "bicycle",
    }


elif LABEL_CHOICE == "ADE20K":

    CATEGORIES = ADE20K_CATEGORIES

    EXCLUDE_CATEGORIES = {
        "wall", "floor", "ceiling", "fence"
    }

    # ── 室内场景：只检测独立物体，排除墙/天花板附着结构 ──
    INSCENE_CATEGORIES = {
        # 座椅
        "chair", "swivel chair", "armchair", "seat", "stool", "bench",
        "sofa", "ottoman", "cushion",
        # 桌面
        "table", "coffee table", "pool table", "counter", "countertop",
        "desk",
        # 床/寝具
        "bed", "pillow", "blanket", "cradle",
        # 柜类
        "cabinet", "chest of drawers", "wardrobe", "bookcase",
        "shelf", "buffet",
        # 电器/屏幕
        "television receiver", "monitor", "screen", "crt screen", "computer",
        "refrigerator", "oven", "stove", "microwave", "dishwasher",
        "washer",
        # 卫浴
        "sink", "bathtub", "toilet", "shower",
        # 照明（独立灯具）
        "lamp", "chandelier",
        # 镜/装饰
        "mirror", "painting", "clock",
        # 小型物品
        "box", "bottle", "pot", "vase", "basket", "bag", "book",
        "fan", "flower", "plaything", "towel", "tray",
        # 人
        "person",
        # 植物
        "plant", "tree",
    }

    # ── 室外场景：城市/街道环境下的独立物体 ──
    OUTSCENE_CATEGORIES = {
        # 交通工具
        "car", "van", "truck", "bus",
        "bicycle", "minibike",
        "boat", "ship", "airplane",
        # 人/动物
        "person", "animal",
        # 杆状/灯具
        "pole", "streetlight", "traffic light", "light",
        # 标识/指示
        "signboard", "flag",
        # 城市家具
        "bench", "booth", "ashcan",
        # 独立物体
        "box", "barrel", "basket", "bottle", "pot", "vase",
        "sculpture", "fountain", "tent",
    }
# ==================================

def _build_target_categories() -> set:
    if OUTSCENE:
        return OUTSCENE_CATEGORIES
    return INSCENE_CATEGORIES

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
