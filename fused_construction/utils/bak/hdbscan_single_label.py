from pathlib import Path
from typing import Dict, List, Sequence, Optional, Tuple

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from class_statics_config import LABEL_CHOICE

if LABEL_CHOICE == "SCANNET_NYU40":
    INPUT_BASE_DIR = Path("/data1/data/scannet/output/detection_targets")
    # INPUT_BASE_DIR = Path("/data1/data/scannet/output_single/detection_targets")
    OUTPUT_BASE_DIR = INPUT_BASE_DIR.parent / "hdbscan_scenes"        # 各场景 HDBSCAN 结果根目录

elif LABEL_CHOICE == "ADE20K":
    INPUT_BASE_DIR = Path("/data1/user/data/fastlivo_output_qs2_03.17/lidar/res/detection_targets")
    OUTPUT_BASE_DIR = INPUT_BASE_DIR.parent / "hdbscan_scenes"

MIN_CLUSTER_SIZE = 5             # 第一阶段极宽松，让 HDBSCAN 发现所有可能的簇
MIN_SAMPLES = 3                  # 配合宽松的 min_cluster_size
MIN_CLUSTER_POINTS = 10          # 二阶段自适应过滤的绝对下限（兜底）
MAX_CLUSTER_FILTER = 50          # 自适应过滤阈值上限：防止 log-gap 误删合法的小物体簇
                                 # 点数 >= MAX_CLUSTER_FILTER 的簇永远不会被过滤掉
NOISE_ABSORB_FACTOR = 15.0       # 噪声点吸收半径 = NOISE_ABSORB_FACTOR × epsilon
                                 # 增大：bbox 覆盖更广；减小：仅吸收紧邻外围点
USE_ORIENTED_BBOX = False

#   cluster_selection_epsilon 控制簇间最小距离：
#   距离小于此阈值的子簇会被合并，保证不同簇之间有明显的距离间隔。
CLUSTER_SELECTION_EPSILON = 0.0  # 单位：米；设 0.0 可禁用
AUTO_EPSILON = True              # True 时从 k-NN 距离分布自动估计 epsilon
CLUSTER_SELECTION_METHOD = "eom" # "eom"（推荐）或 "leaf"（注意：leaf 会忽略 epsilon）


LINE_CONNECTIONS = np.asarray(
    [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)

def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """读取点云文件。"""
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"无法读取点云或点云为空: {path}")
    return pcd

def ordered_box_corners(center: np.ndarray, extent: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # 返回立方体 8 个顶点的世界坐标
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
    local_corners = signs * half
    rotated = (rotation @ local_corners.T).T
    return rotated + center

def create_bbox_line_set(
    bbox: o3d.geometry.Geometry3D,
    color: Sequence[float],
    rotation: np.ndarray,
) -> o3d.geometry.LineSet:
    """根据包围盒生成彩色线框。"""
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
    """依据开关返回包围盒、旋转矩阵"""
    if USE_ORIENTED_BBOX:
        bbox = instance.get_oriented_bounding_box()
        rotation = np.asarray(bbox.R)
    else:
        bbox = instance.get_axis_aligned_bounding_box()
        rotation = np.eye(3)
    return bbox, rotation, rotation.tolist()

def combine_line_sets(line_sets: Sequence[o3d.geometry.LineSet]) -> Optional[o3d.geometry.LineSet]:
    """合并多个线框以便整体写入 PLY。"""
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


def write_ply_with_edges(
    vertices: np.ndarray,
    vertex_colors: np.ndarray,
    edges: np.ndarray,
    edge_colors: np.ndarray,
    output_path: Path,
) -> None:
    """以 ASCII PLY 写出同时包含点云与线框的数据。"""
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

    vertex_colors_uint8 = normalize_colors(vertex_colors if vertex_colors.size else np.zeros((len(vertices), 3)))
    edge_colors_uint8 = normalize_colors(edge_colors) if len(edges) else np.zeros((0, 3), dtype=np.uint8)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(vertices, vertex_colors_uint8):
            f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for (v1, v2), col in zip(edges, edge_colors_uint8):
            f.write(f"{int(v1)} {int(v2)} {int(col[0])} {int(col[1])} {int(col[2])}\n")


def export_point_cloud_with_bboxes(
    point_cloud: o3d.geometry.PointCloud,
    line_sets: Sequence[o3d.geometry.LineSet],
    output_path: Path,
) -> None:
    """将点云与包围盒一并写入单个 PLY（包含顶点与边）。"""
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


def estimate_cluster_epsilon(points: np.ndarray, k: int = 5, sample_size: int = 5000) -> float:
    """用 k-NN 距离分布的最大 log-gap 自动估计 cluster_selection_epsilon。

    在多实例场景下，k-NN 距离分布在对数坐标下呈双峰：
    低端峰 = 同一实例内部的点间距离（密集核心区）；
    高端峰 = 跨实例的最近邻距离（实例间空隙）。
    最大 log-gap 对应这两个尺度之间的自然边界——
    将 epsilon 设为 gap 左端的距离（最大簇内近邻距离），
    可以确保 HDBSCAN 在实例内部合并子簇，同时不跨越实例间隔。
    单实例或分布均匀时（gap < 1.0）取中位数，偏保守。
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
    knn_dists = np.sort(distances[:, -1])   # 第 k 近邻距离，升序
    log_dists = np.log1p(knn_dists)
    gaps = np.diff(log_dists)
    gap_idx = int(np.argmax(gaps))
    
    # 20260322 增大 gap 阈值以适应更宽松的合并，减少过碎簇；同时增大 epsilon 下限以覆盖更广泛的实例间隔
    if gaps[gap_idx] >= 2.0: # 1.0--》2.0
        # 显著双峰：gap 左端 = 最大的簇内近邻距离
        epsilon = float(knn_dists[gap_idx])
    else:
        # 无显著双峰（单实例或分布均匀）：取中位数，偏保守
        epsilon = float(np.percentile(knn_dists, 50))
    return float(np.clip(epsilon, 0.05, 2.0)) # 0.02--》0.05


def filter_clusters_by_loggap(
    cluster_labels: np.ndarray,
    min_points: int = MIN_CLUSTER_POINTS,
) -> np.ndarray:
    """根据簇大小分布的最大 log-gap 过滤噪声碎片簇。

    逻辑同 estimate_min_triangles：若簇大小在对数坐标下存在显著双峰
    （最大 log-gap ≥ 1.0），则将阈值设在 gap 处，只保留高端（真实实例）簇；
    否则只过滤低于绝对下限 min_points 的极小簇。
    单簇情况下直接返回，不过滤。
    """
    unique, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
    if len(unique) == 0:
        return cluster_labels
    if len(unique) == 1:
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
    threshold = max(min_points, min(threshold, MAX_CLUSTER_FILTER))

    small_cluster_ids = unique[counts < threshold]
    filtered = cluster_labels.copy()
    for cid in small_cluster_ids:
        filtered[filtered == cid] = -1
    return filtered


def absorb_noise_points(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    epsilon: float,
    factor: float = NOISE_ABSORB_FACTOR,
) -> np.ndarray:
    """将噪声点按最近邻重新归属到对应簇，以扩展包围盒覆盖稀疏外围点。

    对每个 label=-1 的噪声点，找距其最近的已聚类点：
    - 若距离 ≤ factor × epsilon，归入该簇（外围稀疏点，属于同一物体）
    - 否则保持 -1（真正的孤立噪声，与任何簇都相距甚远）
    """
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


def cluster_single_label_pcd(
    pcd: o3d.geometry.PointCloud,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
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
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        gen_min_span_tree=False,
        core_dist_n_jobs=-1,
    )
    cluster_labels = clusterer.fit_predict(points)
    # 二阶段：自适应过滤噪声碎片簇
    cluster_labels = filter_clusters_by_loggap(cluster_labels)
    # 三阶段：噪声点吸收，扩展包围盒覆盖稀疏外围点
    cluster_labels = absorb_noise_points(points, cluster_labels, cluster_selection_epsilon)
    
    result["cluster_labels"] = cluster_labels
    probs = getattr(clusterer, "probabilities_", None)
    unique_clusters = [cid for cid in np.unique(cluster_labels) if cid != -1]
    if not unique_clusters:
        return result
    cmap = plt.get_cmap("tab20")
    for cid in unique_clusters:
        idxs = np.where(cluster_labels == cid)[0]

        instance = pcd.select_by_index(idxs)
        bbox, rotation_np, rotation_list = extract_bbox(instance)
        color = cmap(cid % 20)[:3]
        line_set = create_bbox_line_set(bbox, color, rotation_np)
        result["line_sets"].append(line_set)
        prob = float(probs[idxs].mean()) if probs is not None else None
        extent_arr = bbox.extent if hasattr(bbox, "extent") else bbox.get_extent()
        record = {
            "cluster_id": int(cid),
            "point_count": int(len(idxs)),
            "probability": prob,
            "bbox_center": bbox.get_center().tolist(),
            "bbox_extent": extent_arr.tolist(),
            "bbox_rotation": rotation_list,
        }

        result["cluster_records"].append(record)
    result["combined_line_set"] = combine_line_sets(result["line_sets"])
    return result


def process_single_file(pcd_path: Path, output_dir: Path) -> None:
    """处理单个点云文件并输出所有结果。"""
    single_pcd = load_point_cloud(pcd_path)
    point_count = len(single_pcd.points)

    if AUTO_EPSILON:
        points_np = np.asarray(single_pcd.points)
        epsilon = estimate_cluster_epsilon(points_np)
        print(f"  [auto-epsilon] {pcd_path.stem}: epsilon = {epsilon:.4f} m")
    else:
        epsilon = CLUSTER_SELECTION_EPSILON

    print(
        f"  {pcd_path.stem}: 点数={point_count}, min_cluster_size={MIN_CLUSTER_SIZE}, "
        f"min_samples={MIN_SAMPLES}, epsilon={epsilon:.4f}, method={CLUSTER_SELECTION_METHOD}"
    )
    result = cluster_single_label_pcd(
        single_pcd,
        MIN_CLUSTER_SIZE,
        MIN_SAMPLES,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
    )

    stem = pcd_path.stem
    output_path = output_dir / f"{stem}_with_bboxes.ply"
    if result["line_sets"]:
        export_point_cloud_with_bboxes(single_pcd, result["line_sets"], output_path)
        print(f"  点云+包围盒写入: {output_path}")
    else:
        export_point_cloud_with_bboxes(single_pcd, [], output_path)
        print(f"  {stem}: 未检测到有效簇，仅写出原始点云: {output_path}")


def main() -> None:
    scene_dirs = sorted([d for d in INPUT_BASE_DIR.iterdir() if d.is_dir()])
    if not scene_dirs:
        print(f"在 {INPUT_BASE_DIR} 中未找到场景子文件夹。")
        return
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"共找到 {len(scene_dirs)} 个场景，输出根目录: {OUTPUT_BASE_DIR}")
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
