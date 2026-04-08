from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
from hdbscan_single_label import write_ply_with_edges
from class_statics_config import LABEL_CHOICE
from filter_detection_categories import is_target_file

# ===== 配置区 =====
# True: 在聚类结果之外，还组合 color_separated_scenes 中未被聚类的背景点云（如 wall、floor 等）
# False: 仅组合 hdbscan_scenes 中的聚类结果
INCLUDE_BACKGROUND = True

if LABEL_CHOICE == "SCANNET_NYU40":
    INPUT_BASE_DIR = Path("/data1/data/scannet/output/hdbscan_scenes")
    # INPUT_BASE_DIR = Path("/data1/data/scannet/output_single/hdbscan_scenes")
    COLOR_SEPARATED_DIR = Path("/data1/data/scannet/output/color_separated_scenes")
    # COLOR_SEPARATED_DIR = Path("/data1/data/scannet/output_single/color_separated_scenes")
    OUTPUT_DIR = INPUT_BASE_DIR.parent / "combined_scenes"

elif LABEL_CHOICE == "ADE20K":
    INPUT_BASE_DIR = Path("/data1/user/data/fastlivo_output_qs2_03.17/lidar/res/hdbscan_scenes")
    COLOR_SEPARATED_DIR = Path("/data1/user/data/fastlivo_output_qs2_03.17/lidar/res/color_separated_scenes")
    OUTPUT_DIR = INPUT_BASE_DIR.parent / "combined_scenes"

FILE_SUFFIX = "_with_bboxes.ply"
# ==================================

def read_ply_with_edges(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取包含 vertex+edge 的 ASCII PLY，返回顶点/颜色/边/边颜色。"""
    with path.open("r", encoding="utf-8") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} header 不完整")
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        vertex_count = next(int(h.split()[2]) for h in header if h.startswith("element vertex"))
        edge_count = next(int(h.split()[2]) for h in header if h.startswith("element edge"))

        vertices = np.zeros((vertex_count, 3), dtype=np.float64)
        vcolors = np.zeros((vertex_count, 3), dtype=np.uint8)
        for i in range(vertex_count):
            x, y, z, r, g, b = f.readline().strip().split()
            vertices[i] = [float(x), float(y), float(z)]
            vcolors[i] = [int(r), int(g), int(b)]

        edges = np.zeros((edge_count, 2), dtype=np.int32)
        ecolors = np.zeros((edge_count, 3), dtype=np.uint8)
        for i in range(edge_count):
            v1, v2, r, g, b = f.readline().strip().split()
            edges[i] = [int(v1), int(v2)]
            ecolors[i] = [int(r), int(g), int(b)]

    return vertices, vcolors, edges, ecolors


def load_background_pcds(scene_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """从 color_separated_scenes 中加载未被 filter_detection_categories 选中的背景点云。"""
    bg_scene_dir = COLOR_SEPARATED_DIR / scene_name
    if not bg_scene_dir.is_dir():
        print(f"  背景目录不存在: {bg_scene_dir}")
        return np.zeros((0, 3)), np.zeros((0, 3))

    all_pts = []
    all_cols = []
    for pcd_path in sorted(bg_scene_dir.glob("*.pcd")):
        # 跳过被 filter_detection_categories 选中的目标类别（这些已在聚类结果中）
        if is_target_file(pcd_path.stem):
            continue
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        if pcd.is_empty():
            continue
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(pts), 3)) * 0.5
        all_pts.append(pts)
        all_cols.append(cols)
        print(f"  背景: {pcd_path.name} -> {len(pts)} 点")

    if not all_pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack(all_pts), np.vstack(all_cols)


def merge_scene(scene_dir: Path, output_ply: Path) -> None:
    """合并单个场景子文件夹内所有 *_with_bboxes.ply 为一个 PLY 文件。"""
    ply_files = sorted(p for p in scene_dir.glob("*.ply") if p.name.endswith(FILE_SUFFIX))
    if not ply_files and not INCLUDE_BACKGROUND:
        print(f"  {scene_dir.name}: 未找到 {FILE_SUFFIX} 文件，跳过。")
        return

    all_vertices = []
    all_vcolors = []
    all_edges = []
    all_ecolors = []
    offset = 0

    # 加载聚类结果 PLY
    for ply in ply_files:
        verts, cols, edges, edge_cols = read_ply_with_edges(ply)
        if not len(verts):
            print(f"  跳过空文件: {ply.name}")
            continue
        all_vertices.append(verts)
        all_vcolors.append(cols)
        if len(edges):
            all_edges.append(edges + offset)
            all_ecolors.append(edge_cols)
        offset += len(verts)
        print(f"  加载: {ply.name} -> 顶点 {len(verts)}, 线段 {len(edges)}")

    # 加载背景点云（未被聚类的类别）
    if INCLUDE_BACKGROUND:
        bg_pts, bg_cols = load_background_pcds(scene_dir.name)
        if len(bg_pts):
            # bg_cols 已经是 [0,1] 浮点范围
            bg_cols_uint8 = (np.clip(bg_cols, 0, 1) * 255).astype(np.uint8)
            all_vertices.append(bg_pts)
            all_vcolors.append(bg_cols_uint8)
            offset += len(bg_pts)
            print(f"  背景合计: {len(bg_pts)} 点")

    if not all_vertices:
        print(f"  {scene_dir.name}: 无有效数据。")
        return

    merged_vertices = np.vstack(all_vertices)
    merged_vcolors = np.vstack(all_vcolors) / 255.0
    merged_edges = np.vstack(all_edges) if all_edges else np.zeros((0, 2), dtype=np.int32)
    merged_ecolors = np.vstack(all_ecolors) / 255.0 if all_ecolors else np.zeros((0, 3))

    write_ply_with_edges(merged_vertices, merged_vcolors, merged_edges, merged_ecolors, output_ply)
    print(f"  合并完成 -> {output_ply}")


def main() -> None:
    scene_dirs = sorted(d for d in INPUT_BASE_DIR.iterdir() if d.is_dir())
    if not scene_dirs:
        print(f"在 {INPUT_BASE_DIR} 中未找到场景子文件夹。")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"共找到 {len(scene_dirs)} 个场景，输出目录: {OUTPUT_DIR}")
    for scene_dir in scene_dirs:
        output_ply = OUTPUT_DIR / f"{scene_dir.name}_merged.ply"
        print(f"场景 {scene_dir.name}:")
        merge_scene(scene_dir, output_ply)
    print("全部场景合并完成。")


if __name__ == "__main__":
    main()
