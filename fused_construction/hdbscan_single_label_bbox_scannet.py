"""
ScanNet 语义点云直接聚类 + AABB 包围盒

独立脚本，不依赖 class_statics_config / filter_detection_categories 等管线。
直接读取 pointcloud_colorized_v2_scannet.py 输出的语义 PCD，
按 NYU40 调色板颜色拆分类别 → HDBSCAN 聚类 → 导出 bbox PLY。

输入: scans/{scene}/res/{scene}_semantic.pcd
输出: scans/{scene}/res/hdbscan/{category}_with_bboxes.ply
      scans/{scene}/res/hdbscan/{scene}_merged_bbox.ply
"""
import os
import sys
import re
import time
import yaml
import json
import numpy as np
import open3d as o3d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import hdbscan as hdbscan_lib
from sklearn.neighbors import NearestNeighbors

from hdbscan_cluster_config import (
    ClusterProfile, CLUSTER_PROFILES,
    _SCANNET_PROFILE_GROUPS, _build_lookup, _sanitize,
)

# ======================== 配置 ========================
SCANNET_BASE = "/data1/data/scannet/scans"
VAL_SPLIT = "/data1/data/scannet/experiment/splits/scannet_val.txt"
PALETTE_YAML = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "dinov3", "yaml", "scannet_nyu40.yaml")
)

VOXEL_SIZE = 0.02
COLOR_TOLERANCE = 15
AUTO_EPSILON = True
BBOX_LINE_WIDTH = 0.005

# 聚类后过滤：点数少于该类别最大簇的此比例 → 视为碎片丢弃
FRAGMENT_RATIO = 0.05

# FM-Fusion 论文 Table I 的 18 个评估类别（标准 ScanNet benchmark）
EVAL_CLASSES = {
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    14: "desk",
    16: "curtain",
    24: "refrigerator",
    28: "shower curtain",
    33: "toilet",
    34: "sink",
    36: "bathtub",
    39: "otherfurniture",
}

# ======================== 调色板 ========================
def load_nyu40_palette():
    with open(PALETTE_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    palette = np.array(cfg["palette"], dtype=np.uint8)
    classes = [str(c) for c in cfg["classes"]]
    return palette, classes


def color_to_class_vectorized(colors_float, palette, tolerance=COLOR_TOLERANCE):
    """将 float64 (0-1) 颜色批量反查为 class ID。"""
    colors_uint8 = np.clip(np.round(colors_float * 255), 0, 255).astype(np.uint8)
    n_pts = len(colors_uint8)
    n_cls = len(palette)
    labels = np.full(n_pts, -1, dtype=np.int16)

    for cid in range(n_cls):
        diff = np.abs(colors_uint8.astype(np.int16) - palette[cid].astype(np.int16))
        match = np.all(diff <= tolerance, axis=1)
        labels[match] = cid
    return labels


# ======================== HDBSCAN 聚类核心 ========================
SCANNET_LOOKUP = _build_lookup(_SCANNET_PROFILE_GROUPS)

LINE_CONNECTIONS = np.asarray([
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int32)


def get_scannet_profile(class_name):
    key = _sanitize(re.sub(r"_\d+$", "", class_name))
    profile_name = SCANNET_LOOKUP.get(key, "default")
    return CLUSTER_PROFILES[profile_name], profile_name


def estimate_cluster_epsilon(points, k=5, sample_size=5000):
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
        return float(knn_dists[gap_idx])
    return float(np.percentile(knn_dists, 50))


def filter_clusters_by_loggap(cluster_labels, min_points, max_filter):
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
    small_ids = unique[counts < threshold]
    filtered = cluster_labels.copy()
    for cid in small_ids:
        filtered[filtered == cid] = -1
    return filtered


def absorb_noise_points(points, cluster_labels, epsilon, factor):
    absorb_radius = factor * epsilon
    noise_mask = cluster_labels == -1
    if not np.any(noise_mask):
        return cluster_labels
    cluster_mask = cluster_labels != -1
    if not np.any(cluster_mask):
        return cluster_labels
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn.fit(points[cluster_mask])
    distances, indices = nn.kneighbors(points[noise_mask])
    updated = cluster_labels.copy()
    noise_indices = np.where(noise_mask)[0]
    absorb_mask = distances[:, 0] <= absorb_radius
    updated[noise_indices[absorb_mask]] = cluster_labels[cluster_mask][indices[absorb_mask, 0]]
    return updated


def compute_planarity(points):
    if len(points) < 4:
        return 0.0
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))
    if eigenvalues[2] < 1e-10:
        return 0.0
    return float((eigenvalues[1] - eigenvalues[0]) / eigenvalues[2])


def cluster_category(points, colors, profile):
    """对单类别点云执行 SOR + HDBSCAN。"""
    result = {"records": [], "total_points": len(points), "kept_points": 0}
    if len(points) < profile.min_cluster_size:
        return result

    # SOR（温和参数：nb=30, std=3.0）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    SOR_NB = 30
    SOR_STD = 3.0
    if len(points) > SOR_NB + 1:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=SOR_NB, std_ratio=SOR_STD,
        )
    pts = np.asarray(pcd.points)
    result["kept_points"] = len(pts)
    if len(pts) < profile.min_cluster_size:
        return result

    # epsilon
    if AUTO_EPSILON:
        raw_eps = estimate_cluster_epsilon(pts)
        epsilon = float(np.clip(
            raw_eps * profile.epsilon_multiplier,
            profile.epsilon_clip_min, profile.epsilon_clip_max,
        ))
    else:
        epsilon = float(np.clip(
            0.1 * profile.epsilon_multiplier,
            profile.epsilon_clip_min, profile.epsilon_clip_max,
        ))

    # HDBSCAN
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=profile.min_cluster_size,
        min_samples=profile.min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=profile.cluster_selection_method,
        gen_min_span_tree=False,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(pts)
    labels = filter_clusters_by_loggap(labels, profile.min_cluster_points, profile.max_cluster_filter)
    labels = absorb_noise_points(pts, labels, epsilon, profile.noise_absorb_factor)

    probs = getattr(clusterer, "probabilities_", None)
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    if not unique_clusters:
        return result

    # 计算各簇点数，用于碎片过滤
    cluster_sizes = {c: int((labels == c).sum()) for c in unique_clusters}
    max_size = max(cluster_sizes.values())
    min_keep = max(int(max_size * FRAGMENT_RATIO), 50)

    for cid in unique_clusters:
        idxs = np.where(labels == cid)[0]
        if len(idxs) < min_keep:
            continue

        cluster_pts = pts[idxs]
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(cluster_pts)
        )
        extent = np.asarray(bbox.get_extent())

        prob = float(probs[idxs].mean()) if probs is not None else 0.5
        result["records"].append({
            "cluster_id": int(cid),
            "point_count": int(len(idxs)),
            "confidence": prob,
            "bbox_center": bbox.get_center().tolist(),
            "bbox_extent": extent.tolist(),
        })

    return result


# ======================== PLY 导出 ========================
def ordered_box_corners(center, extent):
    half = extent * 0.5
    signs = np.array([
        [-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],
        [-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1],
    ], dtype=np.float64)
    return signs * half + center


def write_combined_ply(
    all_points, all_colors, all_bboxes, output_path,
    line_width=BBOX_LINE_WIDTH,
):
    """写入点云 + bbox 面片的 PLY。"""
    if len(all_points) == 0:
        return

    vc = np.clip(all_colors, 0, 1)
    vc_uint8 = (vc * 255).astype(np.uint8)

    extra_verts = []
    extra_cols = []
    faces = []
    face_colors = []
    base = len(all_points)

    for bbox_info in all_bboxes:
        center = np.array(bbox_info["bbox_center"])
        extent = np.array(bbox_info["bbox_extent"])
        color = np.array(bbox_info.get("color", [0, 255, 0]), dtype=np.uint8)
        corners = ordered_box_corners(center, extent)

        for li in LINE_CONNECTIONS:
            p1, p2 = corners[li[0]], corners[li[1]]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-12:
                continue
            ref = np.array([0., 0., 1.])
            if abs(np.dot(direction / length, ref)) > 0.9:
                ref = np.array([0., 1., 0.])
            offset_dir = np.cross(direction, ref)
            offset_dir = offset_dir / (np.linalg.norm(offset_dir) + 1e-12) * line_width * 0.5
            q0, q1, q2, q3 = p1 - offset_dir, p1 + offset_dir, p2 + offset_dir, p2 - offset_dir
            idx_base = base + len(extra_verts)
            extra_verts.extend([q0, q1, q2, q3])
            extra_cols.extend([color] * 4)
            faces.append((idx_base, idx_base + 1, idx_base + 2))
            faces.append((idx_base, idx_base + 2, idx_base + 3))
            face_colors.extend([color, color])

    verts = np.vstack([all_points] + ([np.array(extra_verts)] if extra_verts else []))
    vcols = np.vstack([vc_uint8] + ([np.array(extra_cols, dtype=np.uint8)] if extra_cols else []))

    with open(str(output_path), "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(verts, vcols):
            f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for tri, fc in zip(faces, face_colors):
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]} {int(fc[0])} {int(fc[1])} {int(fc[2])}\n")


# ======================== 单场景处理 ========================
def process_scene(scene_name, palette, class_names):
    scene_dir = os.path.join(SCANNET_BASE, scene_name)
    pcd_path = os.path.join(scene_dir, "res", f"{scene_name}_semantic.pcd")
    if not os.path.exists(pcd_path):
        print(f"  跳过: PCD 不存在 {pcd_path}")
        return

    output_dir = os.path.join(scene_dir, "res", "hdbscan")
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    print(f"  加载 PCD ...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    raw_count = len(pcd.points)
    print(f"  原始点数: {raw_count:,}")

    # voxel 降采样
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    ds_count = len(pcd.points)
    print(f"  降采样后: {ds_count:,} (voxel={VOXEL_SIZE}m)")

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # 颜色→class ID
    labels = color_to_class_vectorized(colors, palette)
    print(f"  类别识别: {(labels >= 0).sum():,} / {len(labels):,} 点有效")

    # 按类别拆分并聚类
    all_bboxes = []
    scene_summary = {"scene": scene_name, "categories": {}}

    for cid in range(len(class_names)):
        if cid not in EVAL_CLASSES:
            continue
        mask = labels == cid
        n_pts = mask.sum()
        if n_pts < 10:
            continue

        cname = class_names[cid]
        profile, profile_name = get_scannet_profile(cname)

        cat_pts = points[mask]
        cat_cols = colors[mask]

        result = cluster_category(cat_pts, cat_cols, profile)
        n_instances = len(result["records"])

        if n_instances > 0:
            # 为 bbox 分配类别平均颜色
            mean_color = (np.mean(cat_cols, axis=0) * 255).astype(np.uint8).tolist()
            for rec in result["records"]:
                rec["color"] = mean_color
                rec["class_name"] = cname
                rec["class_id"] = cid
            all_bboxes.extend(result["records"])
            print(
                f"    {cname:20s} (id={cid:2d}): {n_pts:7,} 点 → "
                f"{result['kept_points']:7,} (SOR) → {n_instances} 实例 "
                f"[{profile_name}]"
            )

        scene_summary["categories"][cname] = {
            "class_id": cid,
            "total_points": int(n_pts),
            "kept_points": result["kept_points"],
            "instances": n_instances,
            "records": result["records"],
        }

    # 导出合并 PLY
    merged_path = os.path.join(output_dir, f"{scene_name}_merged_bbox.ply")

    write_combined_ply(points, colors, all_bboxes, merged_path)
    print(
        f"  合计: {len(all_bboxes)} 个 bbox → {merged_path}"
    )

    # 保存 JSON 摘要
    summary_path = os.path.join(output_dir, f"{scene_name}_cluster_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(scene_summary, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s\n")


# ======================== 主函数 ========================
def main():
    if not os.path.exists(VAL_SPLIT):
        raise FileNotFoundError(f"验证集列表不存在: {VAL_SPLIT}")
    with open(VAL_SPLIT, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    palette, class_names = load_nyu40_palette()
    print(f"ScanNet HDBSCAN 聚类: {len(scenes)} 个场景")
    print(f"调色板: {len(palette)} 类, voxel={VOXEL_SIZE}m")
    print(f"评估类别 ({len(EVAL_CLASSES)}): {list(EVAL_CLASSES.values())}\n")

    t_total = time.time()
    for i, scene in enumerate(scenes):
        print(f"[{i + 1}/{len(scenes)}] {scene}")
        process_scene(scene, palette, class_names)

    print(f"全部完成, 总耗时 {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
