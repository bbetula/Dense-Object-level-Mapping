from pathlib import Path
from typing import Dict, List, Sequence, Optional, Tuple

import hdbscan
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

INPUT_BASE_DIR = Path("/data1/data/scannet/output_single_hull/detection_targets")
OUTPUT_BASE_DIR = INPUT_BASE_DIR.parent / "hdbscan_scenes"        # 各场景 HDBSCAN 结果根目录

MIN_CLUSTER_SIZE = 5             # 第一阶段极宽松，让 HDBSCAN 发现所有可能的簇
MIN_SAMPLES = 3                  # 配合宽松的 min_cluster_size
MIN_CLUSTER_POINTS = 10          # 二阶段自适应过滤的绝对下限（兜底）
NOISE_ABSORB_FACTOR = 15.0       # 噪声点吸收半径 = NOISE_ABSORB_FACTOR × epsilon
                                 # 增大：凸包覆盖更广；减小：仅吸收紧邻外围点

# False: 凸包统一白色；True: 凸包使用对应实例的平均颜色
HULL_INSTANCE_COLOR = True
HULL_ALPHA = 0.6                 # 凸包面片透明度（仅供参考，PLY 不支持透明度）

#   cluster_selection_epsilon 控制簇间最小距离：
#   距离小于此阈值的子簇会被合并，保证不同簇之间有明显的距离间隔。
CLUSTER_SELECTION_EPSILON = 0.0  # 单位：米；设 0.0 可禁用
AUTO_EPSILON = True              # True 时从 k-NN 距离分布自动估计 epsilon
CLUSTER_SELECTION_METHOD = "eom" # "eom"（推荐）或 "leaf"（注意：leaf 会忽略 epsilon）

# 凸包最少点数：少于此值的簇无法计算凸包，跳过
MIN_HULL_POINTS = 4
MAX_HULL_FACES = 30              # 凸包最大面片数：超过此值时使用 quadric decimation 简化


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """读取点云文件。"""
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"无法读取点云或点云为空: {path}")
    return pcd


def compute_convex_hull(
    points: np.ndarray,
    color: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算点云凸包，返回 (顶点, 面片索引, 面片颜色)。

    points: (N, 3) 点云坐标
    color: 凸包面片 RGB 颜色 (0~1)
    返回:
        hull_vertices: (V, 3)
        hull_faces: (F, 3) 三角面片顶点索引
        hull_face_colors: (F, 3) 面片 RGB 颜色 (0~1)
    """
    if len(points) < MIN_HULL_POINTS:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3))
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3))

    hull_vertices = points[hull.vertices]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
    hull_faces = np.array(
        [[old_to_new[v] for v in simplex] for simplex in hull.simplices],
        dtype=np.int32,
    )

    # 面片过多时用 quadric decimation 简化
    if len(hull_faces) > MAX_HULL_FACES:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(hull_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(hull_faces)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=MAX_HULL_FACES)
        hull_vertices = np.asarray(mesh.vertices)
        hull_faces = np.asarray(mesh.triangles).astype(np.int32)

    hull_face_colors = np.tile(np.array(color).reshape(1, 3), (len(hull_faces), 1))
    return hull_vertices, hull_faces, hull_face_colors


def write_ply_with_faces(
    vertices: np.ndarray,
    vertex_colors: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    output_path: Path,
) -> None:
    """以 ASCII PLY 写出点云 + 凸包三角面片。"""
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


def export_point_cloud_with_hulls(
    point_cloud: o3d.geometry.PointCloud,
    hull_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """将点云与凸包面片一并写入单个 PLY。

    hull_list: [(hull_vertices, hull_faces, hull_face_colors), ...]
    """
    all_vertices: List[np.ndarray] = []
    all_vcolors: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    all_fcolors: List[np.ndarray] = []

    pts = np.asarray(point_cloud.points)
    cols = np.asarray(point_cloud.colors) if point_cloud.has_colors() else np.zeros((len(pts), 3))
    all_vertices.append(pts)
    all_vcolors.append(cols)
    offset = len(pts)

    for hull_verts, hull_faces, hull_fcols in hull_list:
        if len(hull_verts) == 0 or len(hull_faces) == 0:
            continue
        all_vertices.append(hull_verts)
        # 凸包顶点颜色取面片颜色的第一个（同一实例颜色一致）
        hull_vcol = hull_fcols[0] if len(hull_fcols) else np.ones(3)
        all_vcolors.append(np.tile(hull_vcol, (len(hull_verts), 1)))
        all_faces.append(hull_faces + offset)
        all_fcolors.append(hull_fcols)
        offset += len(hull_verts)

    merged_vertices = np.vstack(all_vertices)
    merged_vcolors = np.vstack(all_vcolors)
    merged_faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int32)
    merged_fcolors = np.vstack(all_fcolors) if all_fcolors else np.zeros((0, 3))

    write_ply_with_faces(merged_vertices, merged_vcolors, merged_faces, merged_fcolors, output_path)


def estimate_cluster_epsilon(points: np.ndarray, k: int = 5, sample_size: int = 5000) -> float:
    """用 k-NN 距离分布的最大 log-gap 自动估计 cluster_selection_epsilon。"""
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
    return float(np.clip(epsilon, 0.05, 2.0))


def filter_clusters_by_loggap(
    cluster_labels: np.ndarray,
    min_points: int = MIN_CLUSTER_POINTS,
) -> np.ndarray:
    """根据簇大小分布的最大 log-gap 过滤噪声碎片簇。"""
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
    threshold = max(min_points, threshold)

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
    """将噪声点按最近邻重新归属到对应簇，以扩展凸包覆盖稀疏外围点。"""
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
    """对单类点云执行 HDBSCAN 并生成凸包。"""
    points = np.asarray(pcd.points)
    result = {
        "cluster_records": [],
        "hull_list": [],          # [(hull_vertices, hull_faces, hull_face_colors), ...]
        "cluster_labels": np.array([], dtype=int),
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
    # 三阶段：噪声点吸收，扩展凸包覆盖稀疏外围点
    cluster_labels = absorb_noise_points(points, cluster_labels, cluster_selection_epsilon)

    result["cluster_labels"] = cluster_labels
    probs = getattr(clusterer, "probabilities_", None)
    unique_clusters = [cid for cid in np.unique(cluster_labels) if cid != -1]
    if not unique_clusters:
        return result
    for cid in unique_clusters:
        idxs = np.where(cluster_labels == cid)[0]

        instance = pcd.select_by_index(idxs)
        instance_points = np.asarray(instance.points)

        # 确定凸包颜色
        if HULL_INSTANCE_COLOR:
            instance_colors = np.asarray(instance.colors)
            if len(instance_colors) > 0:
                color = tuple(instance_colors.mean(axis=0).tolist())
            else:
                color = (1.0, 1.0, 1.0)
        else:
            color = (1.0, 1.0, 1.0)

        hull_verts, hull_faces, hull_fcols = compute_convex_hull(instance_points, color)
        result["hull_list"].append((hull_verts, hull_faces, hull_fcols))

        prob = float(probs[idxs].mean()) if probs is not None else None
        record = {
            "cluster_id": int(cid),
            "point_count": int(len(idxs)),
            "probability": prob,
            "hull_vertex_count": int(len(hull_verts)),
            "hull_face_count": int(len(hull_faces)),
        }
        result["cluster_records"].append(record)
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
    output_path = output_dir / f"{stem}_with_hulls.ply"
    if result["hull_list"]:
        export_point_cloud_with_hulls(single_pcd, result["hull_list"], output_path)
        hull_count = sum(1 for h in result["hull_list"] if len(h[0]) > 0)
        print(f"  点云+凸包写入: {output_path} ({hull_count} 个凸包)")
    else:
        export_point_cloud_with_hulls(single_pcd, [], output_path)
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
