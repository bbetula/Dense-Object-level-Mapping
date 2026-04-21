"""
分类别参数 HDBSCAN 聚类 + AABB 包围盒

与 hdbscan_single_label_bbox.py 的区别：
  - 每个类别从 hdbscan_cluster_config 查询独立的 HDBSCAN 参数
  - estimate_cluster_epsilon 返回原始值，由调用方根据 profile 做缩放和截断
"""
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Tuple

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

from class_statics_config import LABEL_CHOICE
from filter_detection_categories_bbox import OUTPUT_BASE_DIR as FILTERED_TARGETS_DIR
from hdbscan_cluster_config import ClusterProfile, get_cluster_profile, get_profile_name

# ===== 路径配置 =====
if LABEL_CHOICE == "SCANNET_NYU40":
    INPUT_BASE_DIR = FILTERED_TARGETS_DIR
    OUTPUT_BASE_DIR = INPUT_BASE_DIR.parent / "hdbscan_scenes"
elif LABEL_CHOICE == "ADE20K":
    INPUT_BASE_DIR = FILTERED_TARGETS_DIR
    OUTPUT_BASE_DIR = INPUT_BASE_DIR.parent / "hdbscan_scenes_bbox"

# ===== 全局显示参数（与聚类无关）=====
BBOX_USE_INSTANCE_COLOR = True
BBOX_LINE_WIDTH = 0.005
AUTO_EPSILON = True

LINE_CONNECTIONS = np.asarray(
    [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)


# =====================================================
#  基础工具函数（与原版一致）
# =====================================================
def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"无法读取点云或点云为空: {path}")
    return pcd


def ordered_box_corners(center: np.ndarray, extent: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    half = extent * 0.5
    signs = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local_corners = signs * half
    rotated = (rotation @ local_corners.T).T
    return rotated + center


def create_bbox_line_set(
    bbox: o3d.geometry.Geometry3D,
    color: Sequence[float],
    rotation: np.ndarray,
) -> o3d.geometry.LineSet:
    center = np.asarray(bbox.get_center())
    extent = np.asarray(bbox.extent if hasattr(bbox, "extent") else bbox.get_extent())
    corners = ordered_box_corners(center, extent, rotation)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(LINE_CONNECTIONS)
    line_colors = np.tile(np.asarray(color).reshape(1, 3), (LINE_CONNECTIONS.shape[0], 1))
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def extract_bbox(instance: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.Geometry3D, np.ndarray, List[List[float]]]:
    bbox = instance.get_axis_aligned_bounding_box()
    rotation = np.eye(3)
    return bbox, rotation, rotation.tolist()


def combine_line_sets(line_sets: Sequence[o3d.geometry.LineSet]) -> Optional[o3d.geometry.LineSet]:
    valid = [ls for ls in line_sets if ls is not None and len(ls.points)]
    if not valid:
        return None
    all_points: List[np.ndarray] = []
    all_lines: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    offset = 0
    for ls in valid:
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        cols = np.asarray(ls.colors) if len(ls.colors) else np.ones((len(lines), 3))
        all_points.append(pts)
        all_lines.append(lines + offset)
        all_colors.append(cols)
        offset += len(pts)
    merged = o3d.geometry.LineSet()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    return merged


# =====================================================
#  PLY 输出函数（与原版一致）
# =====================================================
def write_ply_with_edges(
    vertices: np.ndarray,
    vertex_colors: np.ndarray,
    edges: np.ndarray,
    edge_colors: np.ndarray,
    output_path: Path,
    line_width: float = BBOX_LINE_WIDTH,
) -> None:
    if len(vertices) == 0:
        raise ValueError("没有可写入的顶点。")

    def normalize_colors(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.astype(np.uint8)
        if arr.dtype == np.uint8 or arr.max() > 1.0:
            colors = arr.astype(np.float64) / 255.0
        else:
            colors = arr.astype(np.float64)
        colors = np.clip(colors, 0.0, 1.0)
        return (colors * 255).astype(np.uint8)

    vc_uint8 = normalize_colors(vertex_colors if vertex_colors.size else np.zeros((len(vertices), 3)))

    extra_verts = []
    extra_vcols = []
    faces = []
    face_colors = []
    base_offset = len(vertices)

    if len(edges):
        ec_uint8 = normalize_colors(edge_colors)
        for i, (v1_idx, v2_idx) in enumerate(edges):
            p1 = vertices[int(v1_idx)]
            p2 = vertices[int(v2_idx)]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-12:
                continue
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(direction / length, ref)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            offset_dir = np.cross(direction, ref)
            offset_dir = offset_dir / (np.linalg.norm(offset_dir) + 1e-12) * line_width * 0.5
            q0 = p1 - offset_dir
            q1 = p1 + offset_dir
            q2 = p2 + offset_dir
            q3 = p2 - offset_dir
            idx_base = base_offset + len(extra_verts)
            extra_verts.extend([q0, q1, q2, q3])
            col = ec_uint8[i]
            extra_vcols.extend([col, col, col, col])
            faces.append((idx_base, idx_base + 1, idx_base + 2))
            faces.append((idx_base, idx_base + 2, idx_base + 3))
            face_colors.append(col)
            face_colors.append(col)

    all_verts = np.vstack([vertices] + ([np.array(extra_verts)] if extra_verts else []))
    all_vcols = np.vstack([vc_uint8] + ([np.array(extra_vcols, dtype=np.uint8)] if extra_vcols else []))

    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(all_verts, all_vcols):
            f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for tri, fc in zip(faces, face_colors):
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]} {int(fc[0])} {int(fc[1])} {int(fc[2])}\n")


def write_ply_with_faces(
    vertices: np.ndarray,
    vertex_colors: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    output_path: Path,
) -> None:
    if len(vertices) == 0:
        raise ValueError("没有可写入的顶点。")

    def normalize_colors(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.astype(np.uint8)
        if arr.dtype == np.uint8 or arr.max() > 1.0:
            colors = arr.astype(np.float64) / 255.0
        else:
            colors = arr.astype(np.float64)
        colors = np.clip(colors, 0.0, 1.0)
        return (colors * 255).astype(np.uint8)

    vc_uint8 = normalize_colors(vertex_colors if vertex_colors.size else np.zeros((len(vertices), 3)))
    fc_uint8 = normalize_colors(face_colors) if len(faces) else np.zeros((0, 3), dtype=np.uint8)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(vertices, vc_uint8):
            f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for tri, fc in zip(faces, fc_uint8):
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])} {int(fc[0])} {int(fc[1])} {int(fc[2])}\n")


def export_point_cloud_with_bboxes(
    point_cloud: o3d.geometry.PointCloud,
    line_sets: Sequence[o3d.geometry.LineSet],
    output_path: Path,
) -> None:
    vertices: List[np.ndarray] = []
    vertex_colors: List[np.ndarray] = []
    edges: List[np.ndarray] = []
    edge_colors: List[np.ndarray] = []

    pts = np.asarray(point_cloud.points)
    cols = np.asarray(point_cloud.colors) if point_cloud.has_colors() else np.zeros((len(pts), 3))
    vertices.append(pts)
    vertex_colors.append(cols)
    offset = len(pts)

    for ls in line_sets:
        bbox_pts = np.asarray(ls.points)
        bbox_lines = np.asarray(ls.lines)
        if not len(bbox_pts) or not len(bbox_lines):
            continue
        line_cols = np.asarray(ls.colors) if len(ls.colors) else np.ones((len(bbox_lines), 3))
        vertex_color = line_cols[0] if len(line_cols) else np.ones(3)
        vertices.append(bbox_pts)
        vertex_colors.append(np.tile(vertex_color, (len(bbox_pts), 1)))
        edges.append(bbox_lines + offset)
        edge_colors.append(line_cols)
        offset += len(bbox_pts)

    merged_vertices = np.vstack(vertices)
    merged_vertex_colors = np.vstack(vertex_colors)
    merged_edges = np.vstack(edges) if edges else np.zeros((0, 2), dtype=np.int32)
    merged_edge_colors = np.vstack(edge_colors) if edge_colors else np.zeros((0, 3))

    write_ply_with_edges(merged_vertices, merged_vertex_colors, merged_edges, merged_edge_colors, output_path)


# =====================================================
#  自动 epsilon 估计（返回原始值，不做截断）
# =====================================================
def estimate_cluster_epsilon(points: np.ndarray, k: int = 5, sample_size: int = 5000) -> float:
    """用 k-NN 距离分布的最大 log-gap 自动估计 cluster_selection_epsilon。

    返回原始估计值（不截断），由调用方根据 ClusterProfile 做缩放和截断。
    """
    if len(points) == 0:
        return 0.0
    if len(points) > sample_size:
        indices = np.random.choice(len(points), size=sample_size, replace=False)
        sample = points[indices]
    else:
        sample = points
    n_neighbors = min(k + 1, len(sample))
    if n_neighbors <= 1:
        return 0.1
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", n_jobs=-1)
    nn.fit(sample)
    distances, _ = nn.kneighbors(sample)
    knn_dists = np.sort(distances[:, -1])
    log_dists = np.log1p(knn_dists)
    gaps = np.diff(log_dists)
    gap_idx = int(np.argmax(gaps))

    if gaps[gap_idx] >= 2.0:
        epsilon = float(knn_dists[gap_idx])
    else:
        epsilon = float(np.percentile(knn_dists, 50))
    return epsilon


# =====================================================
#  聚类后处理（参数化）
# =====================================================
def filter_clusters_by_loggap(
    cluster_labels: np.ndarray,
    min_points: int,
    max_filter: int,
) -> np.ndarray:
    """根据簇大小分布的最大 log-gap 过滤噪声碎片簇。"""
    unique, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
    if len(unique) <= 1:
        return cluster_labels

    counts_sorted_idx = np.argsort(counts)
    counts_sorted = counts[counts_sorted_idx]
    log_counts = np.log1p(counts_sorted.astype(float))
    gaps = np.diff(log_counts)
    gap_idx = int(np.argmax(gaps))

    if gaps[gap_idx] >= 1.0:
        threshold = int(counts_sorted[gap_idx]) + 1
    else:
        threshold = min_points
    threshold = max(min_points, min(threshold, max_filter))

    small_cluster_ids = unique[counts < threshold]
    filtered = cluster_labels.copy()
    for cid in small_cluster_ids:
        filtered[filtered == cid] = -1
    return filtered


def absorb_noise_points(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    epsilon: float,
    factor: float,
) -> np.ndarray:
    """将噪声点按最近邻重新归属到对应簇。"""
    absorb_radius = factor * epsilon
    noise_mask = cluster_labels == -1
    if not np.any(noise_mask):
        return cluster_labels
    cluster_mask = cluster_labels != -1
    if not np.any(cluster_mask):
        return cluster_labels

    cluster_pts = points[cluster_mask]
    cluster_ids = cluster_labels[cluster_mask]
    noise_pts = points[noise_mask]

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn.fit(cluster_pts)
    distances, indices = nn.kneighbors(noise_pts)
    distances = distances[:, 0]
    indices = indices[:, 0]

    updated = cluster_labels.copy()
    noise_indices = np.where(noise_mask)[0]
    absorb_mask = distances <= absorb_radius
    updated[noise_indices[absorb_mask]] = cluster_ids[indices[absorb_mask]]
    return updated


# =====================================================
#  PCA 平面度检测
# =====================================================
def compute_planarity(points: np.ndarray) -> float:
    """计算点集的 PCA planarity：(λ₂ - λ₁) / λ₃。
    高值 → 点近乎共面（墙壁/天花板）；低值 → 3D 物体或线性结构。
    """
    if len(points) < 4:
        return 0.0
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))
    if eigenvalues[2] < 1e-10:
        return 0.0
    return float((eigenvalues[1] - eigenvalues[0]) / eigenvalues[2])


# =====================================================
#  核心聚类函数（接收 ClusterProfile）
# =====================================================
def cluster_single_label_pcd(
    pcd: o3d.geometry.PointCloud,
    profile: ClusterProfile,
    epsilon: float,
) -> Dict[str, object]:
    """对单类点云执行 HDBSCAN 并生成包围盒。"""
    points = np.asarray(pcd.points)
    result = {
        "cluster_records": [],
        "line_sets": [],
        "bbox_point_cloud": None,
        "cluster_labels": np.array([], dtype=int),
        "combined_line_set": None,
    }
    if len(points) == 0:
        return result

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=profile.min_cluster_size,
        min_samples=profile.min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=profile.cluster_selection_method,
        gen_min_span_tree=False,
        core_dist_n_jobs=-1,
    )
    cluster_labels = clusterer.fit_predict(points)
    cluster_labels = filter_clusters_by_loggap(
        cluster_labels,
        min_points=profile.min_cluster_points,
        max_filter=profile.max_cluster_filter,
    )
    cluster_labels = absorb_noise_points(
        points, cluster_labels, epsilon,
        factor=profile.noise_absorb_factor,
    )

    result["cluster_labels"] = cluster_labels
    probs = getattr(clusterer, "probabilities_", None)
    unique_clusters = [cid for cid in np.unique(cluster_labels) if cid != -1]
    if not unique_clusters:
        return result

    cmap = plt.get_cmap("tab20")
    rejected_count = 0
    planarity_rejected = 0
    for cid in unique_clusters:
        idxs = np.where(cluster_labels == cid)[0]
        instance = pcd.select_by_index(idxs)
        bbox, rotation_np, rotation_list = extract_bbox(instance)
        extent_arr = np.asarray(bbox.extent if hasattr(bbox, "extent") else bbox.get_extent())

        volume = float(np.prod(np.maximum(extent_arr, 1e-6)))
        max_dim = float(extent_arr.max())
        density = len(idxs) / max(volume, 1e-6)

        if volume > profile.bbox_max_volume:
            rejected_count += 1
            continue
        if max_dim > profile.bbox_max_extent:
            rejected_count += 1
            continue
        if density < profile.bbox_min_density:
            rejected_count += 1
            continue

        if profile.planarity_threshold > 0 and volume > profile.planarity_min_volume:
            planarity = compute_planarity(points[idxs])
            if planarity > profile.planarity_threshold:
                planarity_rejected += 1
                continue

        if BBOX_USE_INSTANCE_COLOR:
            instance_colors = np.asarray(instance.colors)
            if len(instance_colors) > 0:
                color = instance_colors.mean(axis=0).tolist()
            else:
                color = (1.0, 1.0, 1.0)
        else:
            color = cmap(cid % 20)[:3]

        line_set = create_bbox_line_set(bbox, color, rotation_np)
        result["line_sets"].append(line_set)

        prob = float(probs[idxs].mean()) if probs is not None else None
        record = {
            "cluster_id": int(cid),
            "point_count": int(len(idxs)),
            "probability": prob,
            "bbox_center": bbox.get_center().tolist(),
            "bbox_extent": extent_arr.tolist(),
            "bbox_rotation": rotation_list,
        }
        result["cluster_records"].append(record)

    if rejected_count > 0:
        print(
            f"    [bbox-filter] 过滤 {rejected_count} 个不合理 bbox "
            f"(max_vol={profile.bbox_max_volume}, max_ext={profile.bbox_max_extent}, "
            f"min_den={profile.bbox_min_density})"
        )
    if planarity_rejected > 0:
        print(
            f"    [planarity-filter] 过滤 {planarity_rejected} 个平面簇 "
            f"(threshold={profile.planarity_threshold}, min_vol={profile.planarity_min_volume})"
        )

    result["combined_line_set"] = combine_line_sets(result["line_sets"])
    return result


# =====================================================
#  单文件处理入口
# =====================================================
def apply_sor(
    pcd: o3d.geometry.PointCloud,
    profile: ClusterProfile,
) -> o3d.geometry.PointCloud:
    """Statistical Outlier Removal 预处理，去除孤立噪声点。"""
    if len(pcd.points) < profile.sor_nb_neighbors + 1:
        return pcd
    pcd_clean, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=profile.sor_nb_neighbors,
        std_ratio=profile.sor_std_ratio,
    )
    return pcd_clean


def process_single_file(pcd_path: Path, output_dir: Path) -> None:
    single_pcd = load_point_cloud(pcd_path)
    point_count_raw = len(single_pcd.points)

    profile = get_cluster_profile(pcd_path.stem, LABEL_CHOICE)
    profile_name = get_profile_name(pcd_path.stem, LABEL_CHOICE)

    single_pcd = apply_sor(single_pcd, profile)
    point_count = len(single_pcd.points)
    removed = point_count_raw - point_count
    print(
        f"  [SOR] {pcd_path.stem}: {point_count_raw} → {point_count} "
        f"(去除 {removed} 个离群点, {removed / max(point_count_raw, 1) * 100:.1f}%), "
        f"nb={profile.sor_nb_neighbors}, std={profile.sor_std_ratio:.1f}"
    )

    if AUTO_EPSILON:
        points_np = np.asarray(single_pcd.points)
        raw_epsilon = estimate_cluster_epsilon(points_np)
        epsilon = raw_epsilon * profile.epsilon_multiplier
        epsilon = float(np.clip(epsilon, profile.epsilon_clip_min, profile.epsilon_clip_max))
        print(
            f"  [auto-epsilon] {pcd_path.stem}: raw={raw_epsilon:.4f}, "
            f"×{profile.epsilon_multiplier:.1f} → {epsilon:.4f} m "
            f"(clip [{profile.epsilon_clip_min}, {profile.epsilon_clip_max}])"
        )
    else:
        epsilon = float(np.clip(
            0.1 * profile.epsilon_multiplier,
            profile.epsilon_clip_min,
            profile.epsilon_clip_max,
        ))

    print(
        f"  {pcd_path.stem}: 点数={point_count}, profile={profile_name}, "
        f"min_cluster_size={profile.min_cluster_size}, min_samples={profile.min_samples}, "
        f"epsilon={epsilon:.4f}, method={profile.cluster_selection_method}"
    )

    result = cluster_single_label_pcd(single_pcd, profile, epsilon)

    num_clusters = len(result["cluster_records"])
    stem = pcd_path.stem
    output_path = output_dir / f"{stem}_with_bboxes.ply"

    if result["line_sets"]:
        export_point_cloud_with_bboxes(single_pcd, result["line_sets"], output_path)
        print(f"  点云+包围盒写入: {output_path} ({num_clusters} 个实例)")
    else:
        export_point_cloud_with_bboxes(single_pcd, [], output_path)
        print(f"  {stem}: 未检测到有效簇，仅写出原始点云: {output_path}")


# =====================================================
#  主函数
# =====================================================
def main() -> None:
    scene_dirs = sorted([d for d in INPUT_BASE_DIR.iterdir() if d.is_dir()])
    if not scene_dirs:
        print(f"在 {INPUT_BASE_DIR} 中未找到场景子文件夹。")
        return
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"共找到 {len(scene_dirs)} 个场景，输出根目录: {OUTPUT_BASE_DIR}")
    print(f"使用分类别聚类参数 (hdbscan_cluster_config)")
    for scene_dir in scene_dirs:
        pcd_files = sorted(scene_dir.glob("*.pcd"))
        if not pcd_files:
            print(f"场景 {scene_dir.name}: 无 PCD 文件，跳过。")
            continue
        scene_out_dir = OUTPUT_BASE_DIR / scene_dir.name
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"场景 {scene_dir.name}: {len(pcd_files)} 个类别点云")
        for pcd_path in pcd_files:
            process_single_file(pcd_path, scene_out_dir)
    print("全部场景处理完成。")


if __name__ == "__main__":
    main()
