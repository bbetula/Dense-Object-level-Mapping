"""
Context-aware 双向重叠投票融合

对齐 OpenBox (NeurIPS 2025) Eq.2 的融合策略：
  - 输入 A: 几何路径的粗糙簇 R_k（geometric_path_a.py 输出）
  - 输入 B: 图像实例点云 F_i（fuse_dinov3_sam3_path_b.py 输出）
  - 双向计算 R_k 与 F_i 的 3D 点重叠率
  - 匹配的 R_k 继承 F_i 的语义标签和实例 ID
  - 未匹配的 R_k 保留为"仅几何"降级检测
  - 未匹配的 F_i 直接作为独立实例输出

输出：
  fusion_output/
  ├── fused_instances/       # 融合后的实例点云 + bbox
  ├── unmatched_geometric/   # 未匹配的几何簇（降级检测）
  ├── fusion_result.json     # 完整融合报告
  └── visualization.ply      # 可视化（所有实例 + 地面）
"""

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

from class_statics_config import ADE20K_CATEGORIES, COLOR_TOLERANCE, DEFAULT_MAP_PATH, OUTSCENE

from hdbscan_single_label_bbox import (
    create_bbox_line_set,
    extract_bbox,
    export_point_cloud_with_bboxes,
)


# =====================================================
#  配置
# =====================================================
@dataclass
class FusionConfig:
    overlap_threshold: float = 0.3
    spatial_tolerance: float = 0.1
    min_instance_points: int = 30
    min_geometric_points: int = 20
    ground_color: Tuple[float, float, float] = (0.55, 0.55, 0.55)
    ground_voxel_size: float = 0.05
    # ── 法向量过滤（仅对竖直物体类别启用，避免误杀桌面等水平面）──
    normal_filter_enabled: bool = True
    normal_z_threshold: float = 0.85
    normal_knn: int = 30
    # ── 输出控制 ──
    show_unmatched_geometric: bool = True
    max_bbox_extent: float = 0.0              # 0 = 不过滤（已有结构类别跳过）


CFG = FusionConfig()

# ── 结构类别（不生成 bbox）──
STRUCTURAL_CATEGORIES = {
    "wall", "building", "ceiling", "floor", "road", "sidewalk",
    "sky", "earth", "grass", "path", "house", "skyscraper",
    "mountain", "hill", "field", "land", "sea", "water", "river",
    "windowpane", "door", "column", "stairs", "stairway",
    "fence", "railing", "bannister",
}


# ── 竖直物体类别（适用法向量地面溅射过滤，水平面物体如 table/bed 跳过）──
TALL_OBJECT_CATEGORIES = {
    "person", "tree", "plant", "pole", "streetlight", "traffic light",
    "lamp", "chandelier", "signboard", "flag", "light",
    "sculpture", "fountain", "tent", "animal",
}


def get_category_color(category: str) -> Tuple[float, float, float]:
    """从 ADE20K 色盘获取类别颜色 (0-1 归一化)。"""
    cat_lower = category.lower().strip()
    for name, rgb in ADE20K_CATEGORIES.items():
        if name == cat_lower:
            return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    return (0.6, 0.6, 0.6)


def is_structural_category(category: str) -> bool:
    return category.lower().strip() in STRUCTURAL_CATEGORIES


def filter_ground_splash(
    points: np.ndarray,
    normal_z_threshold: float = 0.85,
    normal_knn: int = 30,
) -> np.ndarray:
    """返回非溅射点的布尔掩码（True=保留）。"""
    if len(points) < normal_knn:
        return np.ones(len(points), dtype=bool)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn)
    )
    normals = np.asarray(pcd.normals)
    return np.abs(normals[:, 2]) <= normal_z_threshold


PATH_A_DIR = Path(DEFAULT_MAP_PATH).parent / "geometric_clusters"
PATH_B_DIR = Path("/data1/user/data/fastlivo_output_qs2_03.17/lidar/path_b_output")
OUTPUT_DIR = Path(DEFAULT_MAP_PATH).parent / "fusion_output"


# =====================================================
#  数据结构
# =====================================================
@dataclass
class GeometricCluster:
    cluster_id: int
    points: np.ndarray
    bbox_center: np.ndarray
    bbox_extent: np.ndarray
    category: str = "unknown"
    source_file: str = ""

    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass
class ImageInstance:
    instance_id: int
    category: str
    points: np.ndarray
    bbox_center: np.ndarray
    bbox_extent: np.ndarray
    source_file: str = ""

    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass
class FusedInstance:
    fused_id: int
    category: str
    source: str
    points: np.ndarray
    geometric_ids: List[int] = field(default_factory=list)
    image_ids: List[int] = field(default_factory=list)
    confidence: float = 0.0
    bbox_center: Optional[np.ndarray] = None
    bbox_extent: Optional[np.ndarray] = None


# =====================================================
#  加载 Path A 几何簇
# =====================================================
def load_geometric_clusters(path_a_dir: Path) -> Tuple[List[GeometricCluster], Optional[o3d.geometry.PointCloud]]:
    """从 geometric_path_a.py 的 JSON + PLY 输出加载几何簇。"""
    json_files = sorted(path_a_dir.glob("*_geometric.json"))
    if not json_files:
        print(f"  错误: {path_a_dir} 中未找到 *_geometric.json")
        return [], None

    all_clusters: List[GeometricCluster] = []
    ground_pcd = None

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        ply_path = json_path.with_suffix(".ply")
        if not ply_path.exists():
            print(f"  警告: 缺少 PLY 文件 {ply_path}")
            continue

        pcd = o3d.io.read_point_cloud(str(ply_path))
        all_points = np.asarray(pcd.points)
        all_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        source_pcd_path = meta.get("source", "")
        source_pcd = o3d.io.read_point_cloud(source_pcd_path) if source_pcd_path and Path(source_pcd_path).exists() else None

        n_non_ground = meta.get("non_ground_points", 0)
        n_ground = meta.get("ground_points", 0)

        if n_ground > 0 and len(all_points) > n_non_ground:
            ground_pts = all_points[n_non_ground:]
            ground_cols = all_colors[n_non_ground:] if all_colors is not None else None
            ground_pcd = o3d.geometry.PointCloud()
            ground_pcd.points = o3d.utility.Vector3dVector(ground_pts)
            if ground_cols is not None:
                ground_pcd.colors = o3d.utility.Vector3dVector(ground_cols)
            non_ground_points = all_points[:n_non_ground]
        else:
            non_ground_points = all_points

        for rec in meta.get("clusters", []):
            cid = rec["cluster_id"]
            center = np.array(rec["bbox_center"])
            extent = np.array(rec["bbox_extent"])
            half = extent / 2.0

            if source_pcd is not None:
                src_pts = np.asarray(source_pcd.points)
                dists = np.abs(src_pts - center)
                margin = half + CFG.spatial_tolerance
                mask = np.all(dists < margin, axis=1)
                cluster_pts = src_pts[mask]
            else:
                dists = np.abs(non_ground_points - center)
                margin = half + CFG.spatial_tolerance
                mask = np.all(dists < margin, axis=1)
                cluster_pts = non_ground_points[mask]

            if len(cluster_pts) < CFG.min_geometric_points:
                continue

            all_clusters.append(GeometricCluster(
                cluster_id=cid,
                points=cluster_pts,
                bbox_center=center,
                bbox_extent=extent,
                category=rec.get("category", "unknown"),
                source_file=str(json_path.name),
            ))

    print(f"  加载 {len(all_clusters)} 个几何簇 R_k")
    return all_clusters, ground_pcd


# =====================================================
#  加载 Path B 图像实例
# =====================================================
def load_image_instances(path_b_dir: Path) -> List[ImageInstance]:
    """从 fuse_dinov3_sam3_path_b.py 的 instance_pcds/ 输出加载实例。"""
    inst_dir = path_b_dir / "instance_pcds"
    if not inst_dir.is_dir():
        print(f"  错误: {inst_dir} 不存在")
        return []

    summary_path = path_b_dir / "instance_summary.json"
    meta_lookup: Dict[str, dict] = {}
    if summary_path.exists():
        with summary_path.open("r") as f:
            for item in json.load(f):
                meta_lookup[item["file"]] = item

    instances: List[ImageInstance] = []
    for pcd_path in sorted(inst_dir.glob("*.pcd")):
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points)
        if len(pts) < CFG.min_instance_points:
            continue

        meta = meta_lookup.get(pcd_path.name, {})
        cat = meta.get("category", pcd_path.stem.rsplit("_inst_", 1)[0].replace("_", " "))
        inst_num = meta.get("instance_num", len(instances) + 1)

        bbox = pcd.get_axis_aligned_bounding_box()
        instances.append(ImageInstance(
            instance_id=inst_num,
            category=cat,
            points=pts,
            bbox_center=np.array(bbox.get_center()),
            bbox_extent=np.array(bbox.get_extent()),
            source_file=pcd_path.name,
        ))

    print(f"  加载 {len(instances)} 个图像实例 F_i")
    return instances


# =====================================================
#  双向重叠计算（OpenBox Eq.2）
# =====================================================
def compute_overlap_matrix(
    geo_clusters: List[GeometricCluster],
    img_instances: List[ImageInstance],
    tolerance: float,
) -> np.ndarray:
    """
    计算 K×I 的双向重叠矩阵。

    overlap(R_k, F_i) = |R_k ∩ F_i| / min(|R_k|, |F_i|)

    用 KD-Tree 近邻搜索近似点集交集：
    若 R_k 中一个点到 F_i 最近点的距离 < tolerance，视为交集。
    """
    K = len(geo_clusters)
    I = len(img_instances)
    if K == 0 or I == 0:
        return np.zeros((K, I))

    overlap = np.zeros((K, I), dtype=np.float64)

    for i, fi in enumerate(img_instances):
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        nn.fit(fi.points)

        for k, rk in enumerate(geo_clusters):
            dists, _ = nn.kneighbors(rk.points)
            n_close = int((dists[:, 0] < tolerance).sum())
            denom = min(rk.num_points, fi.num_points)
            overlap[k, i] = n_close / max(denom, 1)

    return overlap


def compute_reverse_overlap(
    geo_clusters: List[GeometricCluster],
    img_instances: List[ImageInstance],
    tolerance: float,
) -> np.ndarray:
    """反向：F_i 中的点到 R_k 的近邻比例。"""
    K = len(geo_clusters)
    I = len(img_instances)
    if K == 0 or I == 0:
        return np.zeros((K, I))

    reverse = np.zeros((K, I), dtype=np.float64)

    for k, rk in enumerate(geo_clusters):
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        nn.fit(rk.points)

        for i, fi in enumerate(img_instances):
            dists, _ = nn.kneighbors(fi.points)
            n_close = int((dists[:, 0] < tolerance).sum())
            denom = min(rk.num_points, fi.num_points)
            reverse[k, i] = n_close / max(denom, 1)

    return reverse


# =====================================================
#  匹配与融合
# =====================================================
def match_and_fuse(
    geo_clusters: List[GeometricCluster],
    img_instances: List[ImageInstance],
    cfg: FusionConfig,
) -> Tuple[List[FusedInstance], List[GeometricCluster], List[ImageInstance]]:
    """
    Context-aware 双向重叠投票（OpenBox Eq.2）

    策略：
      1. 计算正向 + 反向重叠矩阵
      2. 双向重叠 = max(forward, reverse)
      3. 贪心匹配：每次取最大重叠 > threshold 的 (k, i) 对
      4. 匹配后合并点云，继承 F_i 的语义标签

    Returns:
        fused:              成功融合的实例列表
        unmatched_geo:      未匹配的几何簇
        unmatched_img:      未匹配的图像实例
    """
    K = len(geo_clusters)
    I = len(img_instances)

    if K == 0 and I == 0:
        return [], [], []

    print(f"\n  计算 {K}×{I} 重叠矩阵 (tolerance={cfg.spatial_tolerance}m)...")
    t0 = time.perf_counter()

    forward = compute_overlap_matrix(geo_clusters, img_instances, cfg.spatial_tolerance)
    reverse = compute_reverse_overlap(geo_clusters, img_instances, cfg.spatial_tolerance)
    bidir = np.maximum(forward, reverse)

    t1 = time.perf_counter()
    print(f"  重叠矩阵计算完成: {t1 - t0:.1f}s")

    # ── 贪心匹配 ──
    matched_k = set()
    matched_i = set()
    fused: List[FusedInstance] = []
    fused_id_counter = 0

    score_matrix = bidir.copy()

    while True:
        if score_matrix.size == 0:
            break
        best_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        best_k, best_i = int(best_idx[0]), int(best_idx[1])
        best_score = float(score_matrix[best_k, best_i])

        if best_score < cfg.overlap_threshold:
            break

        rk = geo_clusters[best_k]
        fi = img_instances[best_i]

        merged_pts = np.vstack([fi.points, rk.points])
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_pts)
        merged_pcd = merged_pcd.voxel_down_sample(0.02)
        merged_pts = np.asarray(merged_pcd.points)

        bbox = merged_pcd.get_axis_aligned_bounding_box()

        fused_id_counter += 1
        fused.append(FusedInstance(
            fused_id=fused_id_counter,
            category=fi.category,
            source="fused",
            points=merged_pts,
            geometric_ids=[rk.cluster_id],
            image_ids=[fi.instance_id],
            confidence=best_score,
            bbox_center=np.array(bbox.get_center()),
            bbox_extent=np.array(bbox.get_extent()),
        ))

        matched_k.add(best_k)
        matched_i.add(best_i)

        score_matrix[best_k, :] = -1
        score_matrix[:, best_i] = -1

        print(
            f"    匹配: R_{rk.cluster_id} <-> F_{fi.instance_id}({fi.category}) "
            f"overlap={best_score:.3f}, 合并点数={len(merged_pts)}"
        )

    # ── 处理一对多：一个 F_i 匹配多个 R_k ──
    for i in range(I):
        if i in matched_i:
            continue
        fi = img_instances[i]
        related_k = []
        for k in range(K):
            if k in matched_k:
                continue
            if bidir[k, i] >= cfg.overlap_threshold * 0.5:
                related_k.append(k)

        if related_k:
            all_pts = [fi.points]
            geo_ids = []
            for k in related_k:
                all_pts.append(geo_clusters[k].points)
                geo_ids.append(geo_clusters[k].cluster_id)
                matched_k.add(k)

            merged_pts = np.vstack(all_pts)
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(merged_pts)
            merged_pcd = merged_pcd.voxel_down_sample(0.02)
            merged_pts = np.asarray(merged_pcd.points)
            bbox = merged_pcd.get_axis_aligned_bounding_box()

            fused_id_counter += 1
            fused.append(FusedInstance(
                fused_id=fused_id_counter,
                category=fi.category,
                source="fused_multi_geo",
                points=merged_pts,
                geometric_ids=geo_ids,
                image_ids=[fi.instance_id],
                confidence=float(np.mean([bidir[k, i] for k in related_k])),
                bbox_center=np.array(bbox.get_center()),
                bbox_extent=np.array(bbox.get_extent()),
            ))
            matched_i.add(i)
            print(
                f"    多簇匹配: R_{geo_ids} <-> F_{fi.instance_id}({fi.category}), "
                f"合并点数={len(merged_pts)}"
            )

    # ── 收集未匹配 ──
    unmatched_geo = [geo_clusters[k] for k in range(K) if k not in matched_k]
    unmatched_img = [img_instances[i] for i in range(I) if i not in matched_i]

    print(f"\n  融合结果:")
    print(f"    成功融合:       {len(fused)} 个实例")
    print(f"    未匹配几何簇:   {len(unmatched_geo)} (降级为无语义检测)")
    print(f"    未匹配图像实例: {len(unmatched_img)} (直接输出)")

    return fused, unmatched_geo, unmatched_img


# =====================================================
#  输出
# =====================================================
def save_results(
    fused: List[FusedInstance],
    unmatched_geo: List[GeometricCluster],
    unmatched_img: List[ImageInstance],
    ground_pcd: Optional[o3d.geometry.PointCloud],
    output_dir: Path,
    cfg: FusionConfig,
) -> None:
    """保存融合结果：实例点云、bbox、JSON 报告、可视化 PLY。

    颜色策略：同类别 → 同颜色（ADE20K 色盘），bbox 颜色 = 实例颜色。
    过滤策略：跳过结构类别 / 法向量溅射 / 超大 bbox。
    """
    fused_dir = output_dir / "fused_instances"
    unmatch_dir = output_dir / "unmatched_geometric"
    for d in [fused_dir, unmatch_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_pcds = []
    all_linesets = []
    report_instances = []

    def _sanitize(name: str) -> str:
        import re
        return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_")

    def _make_instance_pcd(
        points: np.ndarray,
        category: str,
        prefix: str,
        inst_id: int,
        source: str,
        save_dir: Path,
        geo_ids: List[int] = None,
        img_ids: List[int] = None,
        confidence: float = 0.0,
    ) -> Optional[dict]:
        """处理单个实例：结构跳过 → 法向量过滤 → ADE20K 上色 → bbox 生成。"""
        if is_structural_category(category):
            return None

        if (cfg.normal_filter_enabled
                and category.lower().strip() in TALL_OBJECT_CATEGORIES
                and len(points) >= cfg.normal_knn):
            keep = filter_ground_splash(points, cfg.normal_z_threshold, cfg.normal_knn)
            points = points[keep]

        if len(points) < cfg.min_instance_points:
            return None

        color = get_category_color(category)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))

        bbox, rotation_np, _ = extract_bbox(pcd)
        extent = np.asarray(
            bbox.extent if hasattr(bbox, "extent") else bbox.get_extent()
        )
        if cfg.max_bbox_extent > 0 and max(extent) > cfg.max_bbox_extent:
            return None

        filename = f"{_sanitize(category)}_{prefix}_{inst_id:03d}.pcd"
        o3d.io.write_point_cloud(str(save_dir / filename), pcd)

        ls = create_bbox_line_set(bbox, color, rotation_np)
        all_pcds.append(pcd)
        all_linesets.append(ls)

        return {
            "fused_id": inst_id if source.startswith("fused") else -1,
            "category": category,
            "source": source,
            "num_points": len(points),
            "confidence": round(confidence, 4),
            "geometric_cluster_ids": geo_ids or [],
            "image_instance_ids": img_ids or [],
            "bbox_center": [round(float(x), 3) for x in bbox.get_center()],
            "bbox_extent": [round(float(x), 3) for x in extent],
            "file": filename,
        }

    # ── 融合实例 ──
    for inst in fused:
        rec = _make_instance_pcd(
            inst.points, inst.category, "fused", inst.fused_id, inst.source,
            fused_dir, geo_ids=inst.geometric_ids, img_ids=inst.image_ids,
            confidence=inst.confidence,
        )
        if rec:
            report_instances.append(rec)

    # ── 未匹配图像实例 ──
    for inst in unmatched_img:
        rec = _make_instance_pcd(
            inst.points, inst.category, "imgonly", inst.instance_id, "image_only",
            fused_dir, img_ids=[inst.instance_id],
        )
        if rec:
            report_instances.append(rec)

    # ── 未匹配几何簇（利用 Path A 检测的类别信息，跳过结构类别）──
    n_geo_shown = 0
    for rk in unmatched_geo:
        cat = rk.category
        if is_structural_category(cat) or cat == "unknown":
            continue

        pts = rk.points
        if (cfg.normal_filter_enabled
                and cat.lower().strip() in TALL_OBJECT_CATEGORIES
                and len(pts) >= cfg.normal_knn):
            keep = filter_ground_splash(pts, cfg.normal_z_threshold, cfg.normal_knn)
            pts = pts[keep]

        color = get_category_color(cat)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))

        if len(pts) < cfg.min_instance_points:
            continue

        bbox, rotation_np, _ = extract_bbox(pcd)
        extent = np.asarray(
            bbox.extent if hasattr(bbox, "extent") else bbox.get_extent()
        )
        if cfg.max_bbox_extent > 0 and max(extent) > cfg.max_bbox_extent:
            continue

        filename = f"{_sanitize(cat)}_geo_{rk.cluster_id:03d}.pcd"
        o3d.io.write_point_cloud(str(unmatch_dir / filename), pcd)

        if cfg.show_unmatched_geometric:
            ls = create_bbox_line_set(bbox, color, rotation_np)
            all_pcds.append(pcd)
            all_linesets.append(ls)
            n_geo_shown += 1

        report_instances.append({
            "fused_id": -1,
            "category": cat,
            "source": "geometric_only",
            "num_points": len(rk.points),
            "confidence": 0.0,
            "geometric_cluster_ids": [rk.cluster_id],
            "image_instance_ids": [],
            "bbox_center": [round(float(x), 3) for x in bbox.get_center()],
            "bbox_extent": [round(float(x), 3) for x in extent],
            "file": filename,
        })
    if n_geo_shown:
        print(f"  未匹配几何簇（有效类别）: {n_geo_shown} 个加入可视化")

    # ── 地面底图 ──
    if ground_pcd is not None and not ground_pcd.is_empty():
        if cfg.ground_voxel_size > 0:
            ground_pcd = ground_pcd.voxel_down_sample(cfg.ground_voxel_size)
        pts = np.asarray(ground_pcd.points)
        ground_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(cfg.ground_color, (len(pts), 1))
        )
        all_pcds.append(ground_pcd)

    # ── 合并可视化 PLY ──
    merged = o3d.geometry.PointCloud()
    for p in all_pcds:
        merged += p
    vis_path = output_dir / "visualization.ply"
    export_point_cloud_with_bboxes(merged, all_linesets, vis_path)
    print(f"\n  可视化: {vis_path}")

    # ── JSON 报告 ──
    n_fused = sum(1 for r in report_instances if r["source"].startswith("fused"))
    n_img_only = sum(1 for r in report_instances if r["source"] == "image_only")
    n_geo_only = sum(1 for r in report_instances if r["source"] == "geometric_only")

    report = {
        "summary": {
            "total_instances": len(report_instances),
            "fused": n_fused,
            "image_only": n_img_only,
            "geometric_only": n_geo_only,
        },
        "config": {
            "overlap_threshold": cfg.overlap_threshold,
            "spatial_tolerance": cfg.spatial_tolerance,
            "min_instance_points": cfg.min_instance_points,
            "min_geometric_points": cfg.min_geometric_points,
        },
        "instances": report_instances,
    }
    report_path = output_dir / "fusion_result.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  报告: {report_path}")


# =====================================================
#  主函数
# =====================================================
def main() -> None:
    print("=" * 60)
    print("Context-aware 双向重叠投票融合")
    print("=" * 60)
    print(f"  Path A (几何簇):  {PATH_A_DIR}")
    print(f"  Path B (图像实例): {PATH_B_DIR}")
    print(f"  输出目录:          {OUTPUT_DIR}")
    print(f"  重叠阈值:          {CFG.overlap_threshold}")
    print(f"  空间容差:          {CFG.spatial_tolerance}m")

    t_start = time.perf_counter()

    # ── 加载两条路径 ──
    print("\n── 加载 Path A ──")
    geo_clusters, ground_pcd = load_geometric_clusters(PATH_A_DIR)

    print("\n── 加载 Path B ──")
    img_instances = load_image_instances(PATH_B_DIR)

    if not geo_clusters and not img_instances:
        print("\n错误: 两条路径均无有效数据，退出。")
        return

    # ── 双向重叠投票 ──
    print("\n── Context-aware 融合 ──")
    fused, unmatched_geo, unmatched_img = match_and_fuse(
        geo_clusters, img_instances, CFG,
    )

    # ── 保存结果 ──
    print("\n── 保存结果 ──")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_results(fused, unmatched_geo, unmatched_img, ground_pcd, OUTPUT_DIR, CFG)

    t_end = time.perf_counter()
    print(f"\n{'=' * 60}")
    print(f"融合完成！总耗时: {t_end - t_start:.1f}s")
    print(f"  融合实例:     {len(fused)}")
    print(f"  仅图像实例:   {len(unmatched_img)}")
    print(f"  仅几何簇:     {len(unmatched_geo)}")
    print(f"  总计:         {len(fused) + len(unmatched_img) + len(unmatched_geo)} 个检测")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
