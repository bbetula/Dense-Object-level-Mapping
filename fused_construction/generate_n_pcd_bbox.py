from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import json
import re
import numpy as np
import open3d as o3d
from class_statics_config import SCANNET_NYU40_CATEGORIES, ADE20K_CATEGORIES, COLOR_TOLERANCE

# ===== 配置区 =====
# 输入：包含多个场景 PCD 文件的文件夹
INPUT_SCENES_DIR = Path("/data1/data/scannet/output_bbox/pcd_from_ply")
# INPUT_SCENES_DIR = Path("/data1/data/scannet/output_single_bbox/pcd_from_ply")
# 输出：大文件夹，每个场景在其中对应一个子文件夹
OUTPUT_BASE_DIR = INPUT_SCENES_DIR.parent / "color_separated_scenes"
# 过滤掉颜色对应点数过少的簇
MIN_POINTS_PER_COLOR = 10        
# 选择使用的类别列表              
CATEGORIES = SCANNET_NYU40_CATEGORIES
# ==================================

def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """读取点云并检查颜色。"""
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"无法读取点云或点云为空: {path}")
    if not pcd.has_colors():
        raise ValueError("点云缺少颜色信息，无法按颜色拆分。")
    return pcd


def color_to_key(rgb: np.ndarray) -> Tuple[int, int, int]:
    """将 0~1 范围 RGB 转换为整数键 (0~255)。"""
    return tuple(int(np.clip(round(val * 255), 0, 255)) for val in rgb)


def split_by_color(pcd: o3d.geometry.PointCloud, min_points: int) -> Dict[Tuple[int, int, int], o3d.geometry.PointCloud]:
    """按照颜色拆分点云，返回颜色到点云的映射。"""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    color_keys = [color_to_key(rgb) for rgb in colors]

    color_to_indices: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, key in enumerate(color_keys):
        color_to_indices.setdefault(key, []).append(idx)

    color_pcds: Dict[Tuple[int, int, int], o3d.geometry.PointCloud] = {}
    for key, idxs in color_to_indices.items():
        if len(idxs) < min_points:
            continue
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(points[idxs])
        sub_pcd.colors = o3d.utility.Vector3dVector(colors[idxs])
        color_pcds[key] = sub_pcd
    return color_pcds


def save_color_metadata(path: Path, categories: Sequence[str]) -> None:
    """记录颜色与输出文件的对应关系。"""
    payload = {"categories": sorted(set(categories))}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sanitize_category_name(category: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", category.strip()) or "unknown"


def match_category(color_key: Tuple[int, int, int], tolerance: int) -> Tuple[str, Optional[Tuple[int, int, int]]]:
    if not CATEGORIES:
        return "unknown", None
    best_name: Optional[str] = None
    best_rgb: Optional[Tuple[int, int, int]] = None
    best_distance = float("inf")
    for name, palette_rgb in CATEGORIES.items():
        channel_diffs = [abs(a - b) for a, b in zip(color_key, palette_rgb)]
        if any(diff > tolerance for diff in channel_diffs):
            continue
        distance = sum(channel_diffs)
        if distance < best_distance:
            best_distance = distance
            best_name = name
            best_rgb = palette_rgb
    if best_name is None:
        return "unknown", None
    return best_name, best_rgb


def build_unique_filename(category: str, counter: Dict[str, int]) -> str:
    counter[category] = counter.get(category, 0) + 1
    suffix = f"_{counter[category]}" if counter[category] > 1 else ""
    safe_name = sanitize_category_name(category)
    return f"{safe_name}{suffix}.pcd"


def process_single_scene(input_pcd: Path, scene_output_dir: Path) -> None:
    """处理单个场景 PCD 文件，将颜色拆分结果写入对应子文件夹。"""
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    pcd = load_point_cloud(input_pcd)
    color_pcds = split_by_color(pcd, MIN_POINTS_PER_COLOR)
    if not color_pcds:
        print(f"  {input_pcd.name}: 没有满足阈值的颜色簇，跳过。")
        return

    color_to_category: Dict[Tuple[int, int, int], str] = {}
    category_groups: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for color_key, sub_pcd in color_pcds.items():
        r, g, b = color_key
        category, _ = match_category(color_key, COLOR_TOLERANCE)
        if category == "unknown":
            category = f"unknown_{r:03d}_{g:03d}_{b:03d}"
        color_to_category[color_key] = category

        group = category_groups.setdefault(
            category,
            {"points": [], "colors": [], "color_keys": []}
        )
        group["points"].append(np.asarray(sub_pcd.points))
        group["colors"].append(np.asarray(sub_pcd.colors))
        group["color_keys"].append(color_key)

    category_counter: Dict[str, int] = {}
    for category, group in category_groups.items():
        fname = build_unique_filename(category, category_counter)
        out_path = scene_output_dir / fname

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(group["points"]))
        merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(group["colors"]))
        o3d.io.write_point_cloud(str(out_path), merged_pcd)

    category_names = list(category_groups.keys())
    save_color_metadata(scene_output_dir / "color_mapping.json", category_names)
    print(f"  {input_pcd.name}: 输出 {len(category_groups)} 个类别点云 → {scene_output_dir}")


def main() -> None:
    pcd_files = sorted(INPUT_SCENES_DIR.glob("*.pcd"))
    if not pcd_files:
        print(f"在 {INPUT_SCENES_DIR} 中未找到 PCD 文件。")
        return
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"共找到 {len(pcd_files)} 个场景 PCD 文件，输出根目录: {OUTPUT_BASE_DIR}")
    for pcd_path in pcd_files:
        scene_name = pcd_path.stem          # 用文件名（无扩展名）作为子文件夹名
        scene_output_dir = OUTPUT_BASE_DIR / scene_name
        print(f"处理场景: {scene_name}")
        process_single_scene(pcd_path, scene_output_dir)
    print("全部场景处理完成。")


if __name__ == "__main__":
    main()