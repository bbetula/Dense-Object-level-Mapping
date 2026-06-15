"""
几何路径 A：全局语义点云 → 地面移除 → HDBSCAN → 粗糙几何簇

参照 OpenBox (NeurIPS 2025) 的 LiDAR 分支设计：
  - 地面移除后直接 HDBSCAN，不做 SOR / bbox 过滤
  - 几何簇 R_k 是粗糙的空间提案，精确度由 Path B（图像实例）提供
  - 后续 context-aware refinement 融合 R_k 与图像实例 F_i

输入：全局语义点云（pointcloud_colorized_v2.py 的输出 *_color.pcd）
步骤：
  1. RANSAC 地面拟合与移除（地面点保留为灰色底图）
  2. HDBSCAN 聚类（全体非地面点一起聚类）
  3. (可选) 噪声点吸收 — 扩展簇边界
输出：
  - {stem}_geometric.ply  ：彩色实例点云 + bbox 线框 + 灰色地面（可视化）
  - {stem}_geometric.json ：每个几何簇 R_k 的元信息（供 context-aware fusion 使用）

用法：
  python geometric_path_a.py [语义点云.pcd]
  不指定路径时自动在 DEFAULT_MAP_PATH 下查找 *_color.pcd
"""

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import hdbscan
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

from class_statics_config import (
    DEFAULT_MAP_PATH,
    ADE20K_CATEGORIES,
    SCANNET_NYU40_CATEGORIES,
    LABEL_CHOICE,
    COLOR_TOLERANCE,
)

from hdbscan_single_label_bbox import (
    estimate_cluster_epsilon,
    filter_clusters_by_loggap,
    absorb_noise_points,
    create_bbox_line_set,
    extract_bbox,
    export_point_cloud_with_bboxes,
)


# =====================================================
#  配置
# =====================================================
@dataclass
class PathAConfig:
    """几何路径 A 的全部参数（对齐 OpenBox LiDAR branch）"""

    # ── RANSAC 地面分割 ──
    ransac_distance_threshold: float = 0.2    # 平面内点距离阈值 (m)
    ransac_n: int = 3
    ransac_num_iterations: int = 3000
    ground_normal_threshold: float = 0.8      # 法向量与 z 轴点积 > 此值视为地面
    ransac_max_attempts: int = 3              # 最多跳过几个非地面平面

    # ── 地面显示 ──
    ground_voxel_size: float = 0.05
    ground_color: Tuple[float, float, float] = (0.55, 0.55, 0.55)

    # ── 聚类前体素降采样（加速 HDBSCAN，0 = 不降采样）──
    cluster_voxel_size: float = 0.0

    # ── HDBSCAN（OpenBox 使用默认 HDBSCAN，参数宽松）──
    min_cluster_size: int = 10
    min_samples: int = 5
    cluster_selection_method: str = "eom"
    epsilon_clip_min: float = 0.05
    epsilon_clip_max: float = 3.0

    # ── 语义结构过滤 ──
    structural_color_tolerance: int = COLOR_TOLERANCE

    # ── 法向量过滤（移除地面/天花板溅射点）──
    normal_filter_enabled: bool = True
    normal_z_threshold: float = 0.85
    normal_knn: int = 30

    # ── 聚类后处理 ──
    min_cluster_points: int = 10              # log-gap 过滤下限
    max_cluster_filter: int = 80
    noise_absorb_factor: float = 12.0         # 噪声吸收半径 = factor × epsilon
    max_bbox_extent: float = 8.0              # 单维度 > 此值的簇视为结构残留


CFG = PathAConfig()
INPUT_PCD_DIR = Path(DEFAULT_MAP_PATH)
OUTPUT_DIR = INPUT_PCD_DIR.parent / "geometric_clusters"

# ── 结构类别（背景/建筑/地形，不参与物体聚类）──
STRUCTURAL_CATEGORIES_ADE20K = {
    "wall", "building", "ceiling", "floor", "road", "sidewalk",
    "sky", "earth", "grass", "path", "house", "skyscraper",
    "mountain", "hill", "field", "land", "sea", "water", "river",
    "lake", "dirt track", "sand", "rock",
    "windowpane", "door", "column", "stairs", "stairway",
    "fence", "railing", "bannister",
}
STRUCTURAL_CATEGORIES_NYU40 = {
    "wall", "floor", "ceiling", "unlabeled", "otherstructure",
    "door", "window",
}


def _build_structural_colors() -> np.ndarray:
    """根据 LABEL_CHOICE 构建结构类别的 RGB 查找表。"""
    if LABEL_CHOICE == "ADE20K":
        cat_map, cat_set = ADE20K_CATEGORIES, STRUCTURAL_CATEGORIES_ADE20K
    elif LABEL_CHOICE == "SCANNET_NYU40":
        cat_map, cat_set = SCANNET_NYU40_CATEGORIES, STRUCTURAL_CATEGORIES_NYU40
    else:
        return np.zeros((0, 3), dtype=np.uint8)
    colors = [rgb for name, rgb in cat_map.items() if name in cat_set]
    return np.array(colors, dtype=np.uint8) if colors else np.zeros((0, 3), dtype=np.uint8)


STRUCTURAL_COLORS = _build_structural_colors()


# =====================================================
#  0. 语义结构过滤（移除墙/天花板/道路等）
# =====================================================
def remove_structural_points(
    pcd: o3d.geometry.PointCloud,
    tolerance: int = COLOR_TOLERANCE,
) -> Tuple[o3d.geometry.PointCloud, int]:
    """按语义颜色移除结构类别点（wall/ceiling/building/road 等）。"""
    if len(STRUCTURAL_COLORS) == 0:
        return pcd, 0
    colors_uint8 = (np.asarray(pcd.colors) * 255).round().astype(np.int16)
    is_structural = np.zeros(len(colors_uint8), dtype=bool)
    for sc in STRUCTURAL_COLORS:
        diff = np.abs(colors_uint8 - sc.astype(np.int16))
        is_structural |= np.all(diff <= tolerance, axis=1)
    keep_indices = np.where(~is_structural)[0].tolist()
    filtered_pcd = pcd.select_by_index(keep_indices)
    return filtered_pcd, int(is_structural.sum())


# =====================================================
#  0.5 法向量过滤（移除地面/天花板溅射点）
# =====================================================
def filter_ground_splash_by_normal(
    pcd: o3d.geometry.PointCloud,
    cfg: PathAConfig,
) -> Tuple[o3d.geometry.PointCloud, int]:
    """移除法向量接近竖直的点（地面/天花板误投影）。

    参照 pointcloud_colorized_v2_normal：非地面类别点若法向量 |nz|>阈值，
    说明该点位于水平表面上，是投影溅射，应移除。
    结构点已在前一步按颜色滤除，此处剩余的水平面点即为溅射。
    """
    if not cfg.normal_filter_enabled or pcd.is_empty():
        return pcd, 0
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=cfg.normal_knn)
    )
    normals = np.asarray(pcd.normals)
    is_vertical = np.abs(normals[:, 2]) > cfg.normal_z_threshold
    keep_indices = np.where(~is_vertical)[0].tolist()
    removed = int(is_vertical.sum())
    return pcd.select_by_index(keep_indices), removed


# =====================================================
#  0.6 类别检测（从语义颜色投票确定簇的主类别）
# =====================================================
def detect_majority_category(
    colors_uint8: np.ndarray,
    tolerance: int = COLOR_TOLERANCE,
) -> Tuple[str, np.ndarray]:
    """从簇内点的语义颜色中投票确定主类别，返回 (类别名, ADE20K RGB)。"""
    best_cat = "unknown"
    best_count = 0
    best_rgb = np.array([153, 153, 153], dtype=np.uint8)
    for name, rgb in ADE20K_CATEGORIES.items():
        if name in STRUCTURAL_CATEGORIES_ADE20K:
            continue
        ref = np.array(rgb, dtype=np.int16)
        diff = np.abs(colors_uint8.astype(np.int16) - ref)
        count = int(np.all(diff <= tolerance, axis=1).sum())
        if count > best_count:
            best_count = count
            best_cat = name
            best_rgb = np.array(rgb, dtype=np.uint8)
    return best_cat, best_rgb


# =====================================================
#  1. RANSAC 地面分割
# =====================================================
def ransac_ground_removal(
    pcd: o3d.geometry.PointCloud,
    cfg: PathAConfig,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """RANSAC 迭代拟合地平面，返回 (非地面点云, 地面点云)。

    策略：每次找到最大平面后检查法向量是否近似水平。
    若命中墙壁/结构面，将其从候选集中剔除后继续搜索。
    """
    points_full = np.asarray(pcd.points)
    n_total = len(points_full)
    is_ground = np.zeros(n_total, dtype=bool)
    remaining_mask = np.ones(n_total, dtype=bool)

    for attempt in range(cfg.ransac_max_attempts):
        remaining_indices = np.where(remaining_mask)[0]
        if len(remaining_indices) < 100:
            break

        remaining_pcd = pcd.select_by_index(remaining_indices.tolist())
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=cfg.ransac_distance_threshold,
            ransac_n=cfg.ransac_n,
            num_iterations=cfg.ransac_num_iterations,
        )

        normal = np.array(plane_model[:3])
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        z_alignment = abs(normal[2])

        original_inlier_indices = remaining_indices[inliers]

        print(
            f"  [RANSAC #{attempt + 1}] "
            f"法向量=({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}), "
            f"z 对齐={z_alignment:.3f}, 内点={len(inliers)}/{len(remaining_indices)}"
        )

        if z_alignment >= cfg.ground_normal_threshold:
            is_ground[original_inlier_indices] = True
            print(f"    -> 地面平面，标记 {len(inliers)} 个地面点")
            break
        else:
            remaining_mask[original_inlier_indices] = False
            print(f"    -> 非地面（墙壁/结构），跳过")

    ground_count = int(is_ground.sum())
    if ground_count == 0:
        print("  [RANSAC] 未检测到地面平面，全部点保留为非地面")
        return pcd, o3d.geometry.PointCloud()

    ground_indices = np.where(is_ground)[0].tolist()
    non_ground_indices = np.where(~is_ground)[0].tolist()

    ground_pcd = pcd.select_by_index(ground_indices)
    non_ground_pcd = pcd.select_by_index(non_ground_indices)

    print(
        f"  [RANSAC] 地面: {ground_count} 点 ({ground_count / n_total * 100:.1f}%), "
        f"非地面: {n_total - ground_count} 点"
    )
    return non_ground_pcd, ground_pcd


# =====================================================
#  2. HDBSCAN 聚类（对齐 OpenBox: 不做 SOR / bbox 过滤）
# =====================================================
def cluster_non_ground(
    pcd: o3d.geometry.PointCloud,
    cfg: PathAConfig,
) -> Tuple[List[Dict], List[o3d.geometry.LineSet], np.ndarray]:
    """对非地面点云执行 HDBSCAN 聚类。

    Returns:
        cluster_records: 每个簇的元信息列表
        line_sets:       bbox 线框列表（仅用于可视化）
        labels:          与输入 pcd 等长的聚类标签数组（-1 = 噪声）
        cluster_colors:  {cluster_id: (r,g,b)} ADE20K 类别色（0-1 归一化）
    """
    points = np.asarray(pcd.points)
    empty_labels = np.full(len(points), -1, dtype=np.int32)
    if len(points) < cfg.min_cluster_size:
        print("  [HDBSCAN] 点数不足，跳过聚类")
        return [], [], empty_labels, {}

    # ── 可选体素降采样（大规模点云加速）──
    use_downsample = cfg.cluster_voxel_size > 0
    if use_downsample:
        pcd_ds = pcd.voxel_down_sample(cfg.cluster_voxel_size)
        points_ds = np.asarray(pcd_ds.points)
        print(
            f"  [降采样] {len(points)} -> {len(points_ds)} 点 "
            f"(voxel={cfg.cluster_voxel_size}m)"
        )
    else:
        points_ds = points

    # ── 自动 epsilon ──
    raw_eps = estimate_cluster_epsilon(points_ds)
    epsilon = float(np.clip(raw_eps, cfg.epsilon_clip_min, cfg.epsilon_clip_max))
    print(f"  [auto-epsilon] raw={raw_eps:.4f} -> clip -> {epsilon:.4f} m")

    # ── HDBSCAN ──
    t0 = time.perf_counter()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=cfg.cluster_selection_method,
        gen_min_span_tree=False,
        core_dist_n_jobs=-1,
    )
    labels_ds = clusterer.fit_predict(points_ds)
    t1 = time.perf_counter()
    n_clusters_raw = len(set(labels_ds)) - (1 if -1 in labels_ds else 0)
    n_noise = int((labels_ds == -1).sum())
    print(
        f"  [HDBSCAN] {n_clusters_raw} 个初始簇, {n_noise} 噪声点, "
        f"耗时 {t1 - t0:.1f}s"
    )

    # ── log-gap 过滤碎片簇（保留有意义的簇）──
    labels_ds = filter_clusters_by_loggap(labels_ds, cfg.min_cluster_points)

    # ── 噪声点吸收（扩展簇边界，提高覆盖率）──
    labels_ds = absorb_noise_points(
        points_ds, labels_ds, epsilon, cfg.noise_absorb_factor,
    )

    # ── 投影回原始分辨率 ──
    if use_downsample and len(points_ds) < len(points):
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
        nn.fit(points_ds)
        distances, indices = nn.kneighbors(points)
        labels = labels_ds[indices[:, 0]]
        labels[distances[:, 0] > cfg.cluster_voxel_size * 2] = -1
    else:
        labels = labels_ds

    # ── 提取 bbox + 类别检测 + 尺寸过滤 ──
    unique_clusters = sorted(c for c in np.unique(labels) if c != -1)
    has_colors = pcd.has_colors()
    all_colors_uint8 = (
        (np.asarray(pcd.colors) * 255).round().astype(np.uint8) if has_colors else None
    )
    cluster_records: List[Dict] = []
    cluster_colors: Dict[int, Tuple[float, float, float]] = {}
    line_sets: List[o3d.geometry.LineSet] = []
    n_oversized = 0
    n_unknown = 0

    for cid in unique_clusters:
        idxs = np.where(labels == cid)[0]
        instance = pcd.select_by_index(idxs.tolist())
        bbox, rotation_np, rotation_list = extract_bbox(instance)
        extent_arr = np.asarray(
            bbox.extent if hasattr(bbox, "extent") else bbox.get_extent()
        )

        if cfg.max_bbox_extent > 0 and max(extent_arr) > cfg.max_bbox_extent:
            labels[idxs] = -1
            n_oversized += 1
            continue

        cat_name = "unknown"
        color = (0.6, 0.6, 0.6)
        if all_colors_uint8 is not None:
            cat_name, cat_rgb = detect_majority_category(
                all_colors_uint8[idxs], cfg.structural_color_tolerance,
            )
            color = tuple(cat_rgb / 255.0)

        if cat_name == "unknown":
            labels[idxs] = -1
            n_unknown += 1
            continue

        cluster_colors[cid] = color
        line_set = create_bbox_line_set(bbox, color, rotation_np)
        line_sets.append(line_set)

        record = {
            "cluster_id": int(cid),
            "category": cat_name,
            "point_count": int(len(idxs)),
            "bbox_center": bbox.get_center().tolist(),
            "bbox_extent": extent_arr.tolist(),
            "bbox_rotation": rotation_list,
        }
        cluster_records.append(record)

    if n_oversized:
        print(f"  [bbox过滤] 移除 {n_oversized} 个超大簇 (>{cfg.max_bbox_extent}m)")
    if n_unknown:
        print(f"  [类别过滤] 跳过 {n_unknown} 个无法识别类别的簇")
    print(f"  最终输出: {len(cluster_records)} 个几何簇 R_k")
    return cluster_records, line_sets, labels, cluster_colors


# =====================================================
#  可视化辅助
# =====================================================
def colorize_by_cluster(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    cluster_colors: Dict[int, Tuple[float, float, float]],
) -> o3d.geometry.PointCloud:
    """按聚类 ID 使用 ADE20K 类别色上色，噪声点为深灰。"""
    points = np.asarray(pcd.points)
    colors = np.full((len(points), 3), 0.3)
    for cid, color in cluster_colors.items():
        mask = labels == cid
        colors[mask] = color
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points)
    result.colors = o3d.utility.Vector3dVector(colors)
    return result


def prepare_ground_display(
    ground_pcd: o3d.geometry.PointCloud,
    cfg: PathAConfig,
) -> o3d.geometry.PointCloud:
    """将地面点云降采样并统一染为灰色。"""
    if ground_pcd.is_empty():
        return ground_pcd
    if cfg.ground_voxel_size > 0:
        raw_count = len(ground_pcd.points)
        ground_pcd = ground_pcd.voxel_down_sample(cfg.ground_voxel_size)
        print(
            f"  [地面] 降采样: {raw_count} -> {len(ground_pcd.points)} 点 "
            f"(voxel={cfg.ground_voxel_size}m)"
        )
    pts = np.asarray(ground_pcd.points)
    gray = np.tile(np.array(cfg.ground_color), (len(pts), 1))
    ground_pcd.colors = o3d.utility.Vector3dVector(gray)
    return ground_pcd


# =====================================================
#  单文件处理入口
# =====================================================
def process_pcd(pcd_path: Path, output_dir: Path, cfg: PathAConfig) -> None:
    """处理单个语义点云文件的完整 Path A 流水线。"""
    print(f"\n{'=' * 60}")
    print(f"处理: {pcd_path.name}")
    print(f"{'=' * 60}")

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        print("  错误: 空点云，跳过")
        return
    n_total = len(pcd.points)
    print(f"  加载点云: {n_total} 点")

    t_start = time.perf_counter()

    # ── Step 1: RANSAC 地面移除 ──
    non_ground, ground = ransac_ground_removal(pcd, cfg)

    # ── Step 2: 语义结构过滤（移除 wall/ceiling/building/road 等）──
    non_ground_filtered, n_structural = remove_structural_points(
        non_ground, cfg.structural_color_tolerance,
    )
    print(
        f"  [语义过滤] 移除结构点: {n_structural} "
        f"({n_structural / n_total * 100:.1f}%), "
        f"剩余物体候选点: {len(non_ground_filtered.points)}"
    )

    # ── Step 2.5: 法向量过滤（移除地面/天花板溅射点）──
    non_ground_filtered, n_splash = filter_ground_splash_by_normal(
        non_ground_filtered, cfg,
    )
    if n_splash > 0:
        print(
            f"  [法向量过滤] 移除溅射点: {n_splash}, "
            f"剩余: {len(non_ground_filtered.points)}"
        )

    # ── Step 3: HDBSCAN 聚类 ──
    records, line_sets, labels, cluster_colors = cluster_non_ground(non_ground_filtered, cfg)

    t_end = time.perf_counter()
    print(f"  总耗时: {t_end - t_start:.1f}s")

    # ── Step 4: 输出 ──
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pcd_path.stem.replace("_color", "")

    # 3a. 实例着色点云
    instance_pcd = colorize_by_cluster(non_ground_filtered, labels, cluster_colors)

    # 3b. 灰色地面底图
    ground_display = prepare_ground_display(ground, cfg)

    # 3c. 合并：实例点云 + 灰色地面 → 导出 PLY（含 bbox 线框）
    merged_pcd = instance_pcd + ground_display
    output_ply = output_dir / f"{stem}_geometric.ply"
    export_point_cloud_with_bboxes(merged_pcd, line_sets, output_ply)
    print(f"  输出 PLY: {output_ply}")

    # 3d. JSON 元信息（供 context-aware fusion 使用）
    category_counts = dict(
        sorted(Counter(r["category"] for r in records).items(), key=lambda x: -x[1])
    )
    meta = {
        "source": str(pcd_path),
        "数据处理说明": "全部点云 → RANSAC 地面移除 → 语义结构过滤 → 法向量过滤 → HDBSCAN 聚类 → bbox 提取",
        "total_points(全部点云)": n_total,
        "ground_points(RANSAC 地面移除)": len(ground.points),
        "non_ground_points": len(non_ground.points),
        "structural_removed(语义结构过滤)": n_structural,
        "splash_removed(法向量过滤)": n_splash,
        "object_candidate_points(HDBSCAN 聚类)": len(non_ground_filtered.points),
        "num_clusters(最终有效几何簇数量)": len(records),
        "num_bboxes(bbox 总数)": len(records),
        "category_counts": category_counts,
        "clusters": records,
    }
    output_json = output_dir / f"{stem}_geometric.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  输出 JSON: {output_json}")


# =====================================================
#  主函数
# =====================================================
def main() -> None:
    if len(sys.argv) >= 2:
        pcd_path = Path(sys.argv[1])
        if not pcd_path.exists():
            print(f"错误: 文件不存在 {pcd_path}")
            return
        pcd_files = [pcd_path]
    else:
        pcd_files = sorted(INPUT_PCD_DIR.glob("*_color.pcd"))
        if not pcd_files:
            print(f"错误: 在 {INPUT_PCD_DIR} 中未找到 *_color.pcd 文件")
            return

    print("=" * 60)
    print("几何路径 A — 粗糙几何聚类（对齐 OpenBox LiDAR branch）")
    print("=" * 60)
    print(f"输入: {len(pcd_files)} 个语义点云")
    print(f"输出目录: {OUTPUT_DIR}")
    print(
        f"参数: RANSAC_dist={CFG.ransac_distance_threshold}m, "
        f"HDBSCAN_min_cluster={CFG.min_cluster_size}, "
        f"min_samples={CFG.min_samples}"
    )

    for pcd_path in pcd_files:
        process_pcd(pcd_path, OUTPUT_DIR, CFG)

    print(f"\n全部处理完成。输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
