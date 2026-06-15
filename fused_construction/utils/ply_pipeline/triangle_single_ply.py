from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

# ===== 配置 =====
INPUT_PLY_DIR = Path("results/scannet/detection_targets")
OUTPUT_DIR = INPUT_PLY_DIR.parent / "hdbscan_meshes_triangles"
USE_ORIENTED_BBOX = True
LINE_HALF_WIDTH = 0.01
LINE_COLOR = np.array([1.0, 1.0, 1.0])  # 亮白色

# 自适应 min_triangles 的绝对下限（兜底值），估计结果不会低于此值
MIN_TRIANGLES = 5
# ==================================

LINE_CONNECTIONS = np.asarray(
    [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"网格为空: {path}")
    if not mesh.has_vertex_colors():
        raise ValueError(f"网格缺少颜色: {path}")
    return mesh


def ordered_box_corners(center: np.ndarray, extent: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    half = extent * 0.5
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local = signs * half
    rotated = (rotation @ local.T).T
    return rotated + center


def extract_bbox(instance: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.Geometry3D, np.ndarray]:
    if USE_ORIENTED_BBOX:
        bbox = instance.get_oriented_bounding_box()
        rot = np.asarray(bbox.R)
    else:
        bbox = instance.get_axis_aligned_bounding_box()
        rot = np.eye(3)
    return bbox, rot


    """从连通分量面片数分布自动估计最小面片阈值。

    核心思路：同一语义类别的 PLY 中，连通分量面片数在对数坐标下呈双峰：
      - 低端峰：网格重建产生的噪声碎片（1~20 个三角面）
      - 高端峰：真实物体实例（数十~数千个三角面）
    最大 log-gap 对应两类之间的自然边界，阈值设在此处，
    可在无需类别先验的情况下过滤噪声、保留所有真实实例。

    当分布无显著双峰（所有分量大小相近，log-gap < 1.0）时，
    仅使用 MIN_TRIANGLES 作为兜底，避免过度过滤。
    """

def estimate_min_triangles(mesh: o3d.geometry.TriangleMesh) -> int:
    if len(np.asarray(mesh.triangles)) == 0:
        return MIN_TRIANGLES

    tri_labels, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(cluster_n_triangles, dtype=np.int64)  # 每个连通分量的三角面数
    n_clusters = len(counts)

    if n_clusters <= 1:
        return MIN_TRIANGLES

    counts_sorted = np.sort(counts)
    log_counts = np.log1p(counts_sorted.astype(float))
    gaps = np.diff(log_counts)
    gap_idx = int(np.argmax(gaps))

    if gaps[gap_idx] >= 1.0:
        threshold = int(counts_sorted[gap_idx]) + 1
    else:
        threshold = MIN_TRIANGLES

    return max(MIN_TRIANGLES, threshold)


    """基于三角网格拓扑连通性聚类，返回各连通分量的包围盒角点列表。

    原理：cluster_connected_triangles() 在三角面邻接图上做连通分量分析。
    - 同一连续曲面的所有三角面 → 同一连通分量 → 1 个包围盒
    - 物理上断开的网格片段 → 各自独立的连通分量 → 各自的包围盒
    不依赖任何距离或密度参数，对连续大物体（沙发、床）不会误切割，
    对独立小物体（分散的椅子）不会错误合并。
    """
def cluster_by_connectivity(
    mesh: o3d.geometry.TriangleMesh,
    min_triangles: int,
) -> List[np.ndarray]:
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(triangles) == 0:
        return []

    tri_labels, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(tri_labels, dtype=np.int64)   # 每个三角面所属簇 id
    counts = np.asarray(cluster_n_triangles, dtype=np.int64)
    n_clusters = len(counts)

    bboxes: List[np.ndarray] = []
    for cid in range(n_clusters):
        # 用官方返回的每簇面数先过滤，更快更稳
        if counts[cid] < min_triangles:
            continue

        tri_mask = labels == cid
        if not np.any(tri_mask):
            continue

        vertex_idxs = np.unique(triangles[tri_mask])
        cluster_verts = vertices[vertex_idxs]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_verts)
        bbox, rot = extract_bbox(pcd)
        extent = bbox.extent if hasattr(bbox, "extent") else bbox.get_extent()
        corners = ordered_box_corners(
            np.asarray(bbox.get_center()), np.asarray(extent), rot
        )
        bboxes.append(corners)

    return bboxes


def build_mesh_arrays(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
    triangles = np.asarray(mesh.triangles)
    return vertices, colors, normals, triangles


def build_line_mesh_faces(bboxes: Sequence[np.ndarray], half_width: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not bboxes:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
        )

    verts_list: List[np.ndarray] = []
    faces_list: List[np.ndarray] = []
    offset = 0

    for corners in bboxes:
        for edge in LINE_CONNECTIONS:
            p0 = corners[edge[0]]
            p1 = corners[edge[1]]
            vec = p1 - p0
            length = np.linalg.norm(vec)
            if length < 1e-9:
                continue
            dir_unit = vec / length
            arbitrary = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(dir_unit, arbitrary)) > 0.99:
                arbitrary = np.array([0.0, 1.0, 0.0])
            perp = np.cross(dir_unit, arbitrary)
            norm_perp = np.linalg.norm(perp)
            if norm_perp < 1e-9:
                arbitrary = np.array([1.0, 0.0, 0.0])
                perp = np.cross(dir_unit, arbitrary)
                norm_perp = np.linalg.norm(perp)
                if norm_perp < 1e-9:
                    perp = np.array([0.0, 0.0, 1.0])
                    norm_perp = 1.0
            perp /= norm_perp
            offset_vec = perp * half_width

            v0 = p0 + offset_vec
            v1 = p0 - offset_vec
            v2 = p1 - offset_vec
            v3 = p1 + offset_vec

            verts_list.append(np.array([v0, v1, v2, v3]))
            faces_list.append(np.array([[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]], dtype=np.int32))
            offset += 4

    colors = np.tile(LINE_COLOR, (len(verts_list) * 4, 1))
    return (
        np.vstack(verts_list),
        colors,
        np.vstack(faces_list),
    )


def write_combined_ply(
    mesh_vertices: np.ndarray,
    mesh_colors: np.ndarray,
    mesh_normals: Optional[np.ndarray],
    mesh_triangles: np.ndarray,
    line_vertices: np.ndarray,
    line_colors: np.ndarray,
    line_faces: np.ndarray,
    output_path: Path,
) -> None:
    all_vertices = np.vstack([mesh_vertices, line_vertices])
    all_colors = np.vstack([mesh_colors, line_colors])
    all_normals = None
    if mesh_normals is not None:
        extra_normals = np.zeros((len(line_vertices), 3))
        all_normals = np.vstack([mesh_normals, extra_normals])

    line_faces_shifted = line_faces + len(mesh_vertices)
    total_faces = np.vstack([mesh_triangles, line_faces_shifted])

    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(all_vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if all_normals is not None:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write(f"element face {len(total_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for idx, v in enumerate(all_vertices):
            r, g, b = np.clip(all_colors[idx] * 255, 0, 255).astype(np.uint8)
            if all_normals is not None:
                nx, ny, nz = all_normals[idx]
                f.write(f"{v[0]} {v[1]} {v[2]} {r} {g} {b} {nx} {ny} {nz}\n")
            else:
                f.write(f"{v[0]} {v[1]} {v[2]} {r} {g} {b}\n")

        for face in total_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def process_mesh(path: Path) -> None:
    mesh = load_mesh(path)
    min_tri = estimate_min_triangles(mesh)
    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    print(f"{path.stem}: 顶点={n_verts}, 三角面={n_tris}, auto min_triangles={min_tri}")

    bboxes = cluster_by_connectivity(mesh, min_tri)

    mesh_vertices, mesh_colors, mesh_normals, mesh_triangles = build_mesh_arrays(mesh)
    line_vertices, line_colors, line_faces = build_line_mesh_faces(bboxes, LINE_HALF_WIDTH)

    out_path = OUTPUT_DIR / f"{path.stem}_mesh_with_bboxes.ply"
    write_combined_ply(
        mesh_vertices, mesh_colors, mesh_normals, mesh_triangles,
        line_vertices, line_colors, line_faces, out_path,
    )
    print(f"输出: {out_path} (簇 {len(bboxes)})")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ply_files = sorted(INPUT_PLY_DIR.glob("*.ply"))
    if not ply_files:
        print(f"目录 {INPUT_PLY_DIR} 中无 PLY 文件。")
        return
    for ply_path in ply_files:
        process_mesh(ply_path)
    print("全部网格处理完成。")


if __name__ == "__main__":
    main()
