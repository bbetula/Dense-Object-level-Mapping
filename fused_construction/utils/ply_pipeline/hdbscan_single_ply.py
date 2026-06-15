from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

# ===== 配置 =====
INPUT_PLY_DIR = Path("results/scannet/detection_targets")
OUTPUT_DIR = INPUT_PLY_DIR.parent / "hdbscan_meshes"
MIN_CLUSTER_SIZE = 400
MIN_SAMPLES = 80
CLUSTER_SELECTION_EPSILON = 0.3
AUTO_EPSILON = True
CLUSTER_SELECTION_METHOD = "eom"
USE_ORIENTED_BBOX = True
LINE_HALF_WIDTH = 0.01
LINE_COLOR = np.array([1.0, 1.0, 1.0])  # 亮白色
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


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    return pcd


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


def estimate_cluster_epsilon(points: np.ndarray, k: int = 5, sample_size: int = 5000) -> float:
    if len(points) == 0:
        return 0.0
    sample = points
    if len(points) > sample_size:
        idx = np.random.choice(len(points), size=sample_size, replace=False)
        sample = points[idx]
    if len(sample) <= 1:
        return 0.1

    n_neighbors = min(k + 1, len(sample))
    if n_neighbors <= 1:
        return 0.1

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(sample)
    distances, _ = nbrs.kneighbors(sample, return_distance=True)
    per_point = distances[:, -1]
    epsilon = float(np.percentile(per_point, 95))
    return float(np.clip(epsilon, 0.05, 2.0))


def cluster_mesh(
    mesh: o3d.geometry.TriangleMesh,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
    method: str,
) -> List[np.ndarray]:
    pcd = mesh_to_point_cloud(mesh)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return []

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=method,
        gen_min_span_tree=False,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(points)

    bboxes: List[np.ndarray] = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idxs = np.where(labels == cid)[0]
        inst = pcd.select_by_index(idxs)
        bbox, rot = extract_bbox(inst)
        extent = bbox.extent if hasattr(bbox, "extent") else bbox.get_extent()
        corners = ordered_box_corners(np.asarray(bbox.get_center()), np.asarray(extent), rot)
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
    pcd = mesh_to_point_cloud(mesh)
    points = np.asarray(pcd.points)
    epsilon = estimate_cluster_epsilon(points) if AUTO_EPSILON else CLUSTER_SELECTION_EPSILON
    print(
        f"{path.stem}: 点数={len(points)}, min_cluster_size={MIN_CLUSTER_SIZE}, "
        f"min_samples={MIN_SAMPLES}, eps={epsilon:.3f}"
    )

    bboxes = cluster_mesh(mesh, MIN_CLUSTER_SIZE, MIN_SAMPLES, epsilon, CLUSTER_SELECTION_METHOD)
    mesh_vertices, mesh_colors, mesh_normals, mesh_triangles = build_mesh_arrays(mesh)
    line_vertices, line_colors, line_faces = build_line_mesh_faces(bboxes, LINE_HALF_WIDTH)

    out_path = OUTPUT_DIR / f"{path.stem}_mesh_with_bboxes.ply"
    write_combined_ply(
        mesh_vertices,
        mesh_colors,
        mesh_normals,
        mesh_triangles,
        line_vertices,
        line_colors,
        line_faces,
        out_path,
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