from pathlib import Path
from typing import Tuple

import numpy as np
from hdbscan_single_label_bbox import write_ply_with_faces

# ===== 配置区 =====
INPUT_BASE_DIR = Path("/data1/data/scannet/output_bbox/hdbscan_scenes")  
# INPUT_BASE_DIR = Path("/data1/data/scannet/output_single_bbox/hdbscan_scenes")  
OUTPUT_DIR = INPUT_BASE_DIR.parent / "combined_scenes"              # 每个场景合并结果输出目录
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


def merge_scene(scene_dir: Path, output_ply: Path) -> None:
    """合并单个场景子文件夹内所有 *_with_bboxes.ply 为一个 PLY 文件。"""
    ply_files = sorted(p for p in scene_dir.glob("*.ply") if p.name.endswith(FILE_SUFFIX))
    if not ply_files:
        print(f"  {scene_dir.name}: 未找到 {FILE_SUFFIX} 文件，跳过。")
        return

    all_vertices = []
    all_vcolors = []
    all_edges = []
    all_ecolors = []
    offset = 0

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
