from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
from hdbscan_single_label_bbox_config import OUTPUT_BASE_DIR as HDBSCAN_OUTPUT_BASE_DIR, write_ply_with_faces
from class_statics_config import LABEL_CHOICE, OUTSCENE
from filter_detection_categories_bbox import is_target_file
from generate_n_pcd_bbox import OUTPUT_BASE_DIR as GENERATED_COLOR_SEPARATED_DIR

# ===== 配置区 =====
INCLUDE_BACKGROUND = True

# 室外模式：仅加载地面参考类别作为灰色底图
GROUND_REFERENCE_CATEGORIES = {"road", "sidewalk", "path", "earth"}
GROUND_COLOR = np.array([0.55, 0.55, 0.55])   # 中性灰
GROUND_VOXEL_SIZE = 0.05                        # 体素降采样尺寸 (m)

if LABEL_CHOICE == "SCANNET_NYU40":
    INPUT_BASE_DIR = HDBSCAN_OUTPUT_BASE_DIR
    COLOR_SEPARATED_DIR = GENERATED_COLOR_SEPARATED_DIR
    OUTPUT_DIR = INPUT_BASE_DIR.parent / "combined_scenes"

elif LABEL_CHOICE == "ADE20K":
    INPUT_BASE_DIR = HDBSCAN_OUTPUT_BASE_DIR
    COLOR_SEPARATED_DIR = GENERATED_COLOR_SEPARATED_DIR
    OUTPUT_DIR = INPUT_BASE_DIR.parent / "combined_scenes_bbox"

FILE_SUFFIX = "_with_bboxes.ply"
# ==================================

def read_ply_with_faces(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取包含 vertex+face 的 ASCII PLY，返回顶点/顶点颜色/面片索引/面片颜色。

    兼容旧的 edge 格式（自动检测）。
    """
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
        # 检测是 face 还是 edge 格式
        face_count = 0
        edge_count = 0
        for h in header:
            if h.startswith("element face"):
                face_count = int(h.split()[2])
            elif h.startswith("element edge"):
                edge_count = int(h.split()[2])

        vertices = np.zeros((vertex_count, 3), dtype=np.float64)
        vcolors = np.zeros((vertex_count, 3), dtype=np.uint8)
        for i in range(vertex_count):
            x, y, z, r, g, b = f.readline().strip().split()
            vertices[i] = [float(x), float(y), float(z)]
            vcolors[i] = [int(r), int(g), int(b)]

        if face_count > 0:
            # 新格式：face（三角面片）
            faces = np.zeros((face_count, 3), dtype=np.int32)
            fcolors = np.zeros((face_count, 3), dtype=np.uint8)
            for i in range(face_count):
                parts = f.readline().strip().split()
                # "3 v0 v1 v2 R G B"
                faces[i] = [int(parts[1]), int(parts[2]), int(parts[3])]
                fcolors[i] = [int(parts[4]), int(parts[5]), int(parts[6])]
            return vertices, vcolors, faces, fcolors
        elif edge_count > 0:
            # 旧格式：edge → 转为空 face 数组（write_ply_with_edges 会重新生成面片）
            edges = np.zeros((edge_count, 2), dtype=np.int32)
            ecolors = np.zeros((edge_count, 3), dtype=np.uint8)
            for i in range(edge_count):
                v1, v2, r, g, b = f.readline().strip().split()
                edges[i] = [int(v1), int(v2)]
                ecolors[i] = [int(r), int(g), int(b)]
            return vertices, vcolors, edges, ecolors
        else:
            return vertices, vcolors, np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8)


def load_background_pcds(scene_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """从 color_separated_scenes 中加载未被 filter_detection_categories 选中的背景点云。"""
    bg_scene_dir = COLOR_SEPARATED_DIR / scene_name
    if not bg_scene_dir.is_dir():
        print(f"  背景目录不存在: {bg_scene_dir}")
        return np.zeros((0, 3)), np.zeros((0, 3))

    all_pts = []
    all_cols = []
    for pcd_path in sorted(bg_scene_dir.glob("*.pcd")):
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


def load_ground_reference_pcds(scene_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """室外模式：仅加载 road/sidewalk 等地面类别，统一灰色 + 体素降采样。"""
    bg_scene_dir = COLOR_SEPARATED_DIR / scene_name
    if not bg_scene_dir.is_dir():
        print(f"  地面参考目录不存在: {bg_scene_dir}")
        return np.zeros((0, 3)), np.zeros((0, 3))

    ground_pcds = []
    for pcd_path in sorted(bg_scene_dir.glob("*.pcd")):
        stem_lower = pcd_path.stem.lower()
        if not any(cat in stem_lower for cat in GROUND_REFERENCE_CATEGORIES):
            continue
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        if pcd.is_empty():
            continue
        ground_pcds.append((pcd_path.name, pcd))
        print(f"  地面参考: {pcd_path.name} -> {len(pcd.points)} 点")

    if not ground_pcds:
        return np.zeros((0, 3)), np.zeros((0, 3))

    merged = o3d.geometry.PointCloud()
    for _, pcd in ground_pcds:
        merged += pcd

    raw_count = len(merged.points)
    if GROUND_VOXEL_SIZE > 0:
        merged = merged.voxel_down_sample(voxel_size=GROUND_VOXEL_SIZE)
    ds_count = len(merged.points)
    print(f"  地面降采样: {raw_count} → {ds_count} 点 (voxel={GROUND_VOXEL_SIZE}m)")

    pts = np.asarray(merged.points)
    cols = np.tile(GROUND_COLOR, (len(pts), 1))
    return pts, cols


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
        verts, cols, elems, elem_cols = read_ply_with_faces(ply)
        if not len(verts):
            print(f"  跳过空文件: {ply.name}")
            continue
        all_vertices.append(verts)
        all_vcolors.append(cols)
        if len(elems):
            all_edges.append(elems + offset)
            all_ecolors.append(elem_cols)
        offset += len(verts)
        print(f"  加载: {ply.name} -> 顶点 {len(verts)}, 面片/边 {len(elems)}")

    # 加载参考底图
    if OUTSCENE:
        ground_pts, ground_cols = load_ground_reference_pcds(scene_dir.name)
        if len(ground_pts):
            ground_cols_uint8 = (np.clip(ground_cols, 0, 1) * 255).astype(np.uint8)
            all_vertices.append(ground_pts)
            all_vcolors.append(ground_cols_uint8)
            offset += len(ground_pts)
            print(f"  地面参考合计: {len(ground_pts)} 点")
    elif INCLUDE_BACKGROUND:
        bg_pts, bg_cols = load_background_pcds(scene_dir.name)
        if len(bg_pts):
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

    write_ply_with_faces(merged_vertices, merged_vcolors, merged_edges, merged_ecolors, output_ply)
    print(f"  合并完成 -> {output_ply}")


def main() -> None:
    scene_dirs = sorted(d for d in INPUT_BASE_DIR.iterdir() if d.is_dir())
    if not scene_dirs:
        print(f"在 {INPUT_BASE_DIR} 中未找到场景子文件夹。")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"共找到 {len(scene_dirs)} 个场景，输出目录: {OUTPUT_DIR}")
    for scene_dir in scene_dirs:
        output_ply = OUTPUT_DIR / f"{scene_dir.name}_merged_bbox.ply"
        print(f"场景 {scene_dir.name}:")
        merge_scene(scene_dir, output_ply)
    print("全部场景合并完成。")


if __name__ == "__main__":
    main()
