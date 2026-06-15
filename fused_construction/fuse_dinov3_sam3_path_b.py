"""
Path B: DINOv3 + SAM3 融合投影

功能：
  1. 逐帧融合 DINOv3 语义分割 + SAM3 实例分割 → 精化语义边界
  2. 全局点云投影：语义多帧投票 + 实例跨帧关联（Union-Find）+ 地面法向量过滤
  3. 输出：
     - stuff 语义点云（road.pcd, building.pcd, ...）
     - thing 实例点云（tree_inst_001.pcd, car_inst_002.pcd, ...）
     - 融合语义标签图（可替代原 DINOv3 mask_id 用于现有流水线）
"""

import gc
import os
import sys
import json
import time
import re
import numpy as np
import cv2
import open3d as o3d
from collections import defaultdict
from PIL import Image
from typing import Dict, Tuple, Optional, List

sys.path.insert(0, os.path.dirname(__file__))
from pointcloud_colorized_v2 import (
    load_camera_params, load_palette_and_lookup,
    load_pose_files, build_pose_records,
    build_camera_model, build_extrinsic_transform, get_expected_label_size,
    transform_global_points_to_camera, project_points_to_image,
    zbuffer_filter,
)
from class_statics_config import (
    GLOBAL_PCD_PATH, DEFAULT_IMG_PATH, ADE20K_CATEGORIES,
    OUTSCENE,
)

# ===== 配置区 =====
DINOV3_DIR = "/data1/user/data/fastlivo_output_qs2_03.17/image/res_dinov3_whole"
SAM3_DIR = "/data1/user/Dense-Object-level-Mapping/sam3/output"
OUTPUT_DIR = "/data1/user/data/fastlivo_output_qs2_03.17/lidar/path_b_output"
SAVE_FUSED_MAPS = False
MIN_VOTES = 5
PROGRESS_INTERVAL = 50
MAX_INST_PER_FRAME = 1000
MIN_INSTANCE_POINTS = 30
NORMAL_Z_THRESHOLD = 0.85
NORMAL_KNN = 30
MAX_BBOX_EXTENT = 8.0
MAX_MERGE_DIST = 3.0
MASK_EROSION_RATIO = 0.05

INSCENE_THING_CATEGORIES = {
    "chair", "swivel chair", "armchair", "seat", "stool", "bench",
    "sofa", "ottoman", "cushion",
    "table", "coffee table", "pool table", "counter", "countertop", "desk",
    "bed", "pillow", "blanket", "cradle",
    "cabinet", "chest of drawers", "wardrobe", "bookcase", "shelf", "buffet",
    "television receiver", "monitor", "screen", "crt screen", "computer",
    "refrigerator", "oven", "stove", "microwave", "dishwasher", "washer",
    "sink", "bathtub", "toilet", "shower",
    "lamp", "chandelier",
    "mirror", "painting", "clock",
    "box", "bottle", "pot", "vase", "basket", "bag", "book",
    "fan", "flower", "plaything", "towel", "tray",
    "person",
    "plant", "tree",
}

OUTSCENE_THING_CATEGORIES = {
    "car", "van", "truck", "bus", "bicycle", "minibike",
    "boat", "ship", "airplane",
    "person", "animal",
    "pole", "streetlight", "traffic light", "light",
    "signboard", "flag",
    "bench", "booth", "ashcan",
    "box", "barrel", "basket", "bottle", "pot", "vase",
    "sculpture", "fountain", "tent",
}

THING_CATEGORIES = OUTSCENE_THING_CATEGORIES if OUTSCENE else INSCENE_THING_CATEGORIES
# ==================================


class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        while self.parent.get(x, x) != x:
            self.parent[x] = self.parent.get(self.parent[x], self.parent[x])
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _release_memory():
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def build_ade20k_name_to_id() -> Dict[str, int]:
    """ADE20K 类别名 → 类别 ID 映射"""
    return {name: idx for idx, name in enumerate(ADE20K_CATEGORIES.keys())}


def fuse_frame(
    dinov3_mask_id: np.ndarray,
    sam3_instance_map: np.ndarray,
    sam3_meta: List[dict],
    ade20k_names: List[str],
    name_to_id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    融合单帧 DINOv3 语义 + SAM3 实例

    Returns:
        fused_semantic: H×W int16, 精化后的语义标签
        fused_instance: H×W int32, 实例 ID (0=无实例)
        instance_info:  {local_inst_id: {"category": str, "ade20k_id": int, "score": float}}
    """
    fused_semantic = dinov3_mask_id.copy().astype(np.int16)
    fused_instance = np.zeros(dinov3_mask_id.shape, dtype=np.int32)
    instance_info = {}

    sorted_meta = sorted(sam3_meta, key=lambda x: x["score"], reverse=True)

    for det in sorted_meta:
        inst_id = det["instance_id"]
        mask = sam3_instance_map == inst_id
        if not mask.any():
            continue

        labels_in_mask = dinov3_mask_id[mask]
        valid = labels_in_mask >= 0
        if not valid.any():
            continue

        counts = np.bincount(labels_in_mask[valid].astype(int), minlength=len(ade20k_names))
        majority_id = int(np.argmax(counts))
        majority_name = ade20k_names[majority_id]

        if majority_name not in THING_CATEGORIES:
            continue

        fused_semantic[mask] = majority_id
        fused_instance[mask] = inst_id
        instance_info[inst_id] = {
            "category": majority_name,
            "ade20k_id": majority_id,
            "score": det["score"],
            "sam3_category": det["category"],
        }

    return fused_semantic, fused_instance, instance_info


def interpolate_int32(label_map: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """取整插值获取 int32 标签（用于 instance map）"""
    h, w = label_map.shape[:2]
    xi = np.round(points_2d[:, 0]).astype(int)
    yi = np.round(points_2d[:, 1]).astype(int)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    labels = np.full(len(points_2d), -1, dtype=np.int32)
    labels[valid] = label_map[yi[valid], xi[valid]].astype(np.int32)
    return labels


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_")


def main() -> None:
    print("=" * 60)
    print("Path B: DINOv3 + SAM3 融合投影")
    print("=" * 60)

    # ── 加载基础设施 ──
    ext_params, inter_params = load_camera_params()
    palette_rgb, _ = load_palette_and_lookup()
    camera_matrix, dist_coeffs = build_camera_model(inter_params)
    ext_pos, ext_rot_mat = build_extrinsic_transform(ext_params)
    expected_h, expected_w = get_expected_label_size(inter_params)

    ade20k_names = list(ADE20K_CATEGORIES.keys())
    name_to_id = build_ade20k_name_to_id()
    num_classes = len(ade20k_names)

    # ── 加载全局点云 ──
    print(f"加载全局点云: {GLOBAL_PCD_PATH}")
    cloud = o3d.io.read_point_cloud(GLOBAL_PCD_PATH)
    points = np.asarray(cloud.points)
    N = len(points)
    print(f"  点数: {N:,}")

    # ── 加载位姿 ──
    sorted_pose_files, _ = load_pose_files(DEFAULT_IMG_PATH)
    pose_records = build_pose_records(sorted_pose_files)
    print(f"  帧数: {len(pose_records)}")

    # ── 输出目录 ──
    stuff_dir = os.path.join(OUTPUT_DIR, "stuff_pcds")
    instance_dir = os.path.join(OUTPUT_DIR, "instance_pcds")
    fused_map_dir = os.path.join(OUTPUT_DIR, "fused_maps")
    for d in [stuff_dir, instance_dir]:
        os.makedirs(d, exist_ok=True)
    if SAVE_FUSED_MAPS:
        os.makedirs(fused_map_dir, exist_ok=True)

    t0 = time.perf_counter()

    # ── 检查阶段 1 缓存 ──
    stage1_cache_path = os.path.join(OUTPUT_DIR, "stage1_cache.npz")
    _cached = False
    if os.path.exists(stage1_cache_path):
        try:
            _c = np.load(stage1_cache_path, allow_pickle=False)
            if "winner_labels" in _c and len(_c["winner_labels"]) == N:
                winner_labels = _c["winner_labels"]
                inst_pt_all = _c["inst_pt_all"]
                inst_gid_all = _c["inst_gid_all"]
                total_inst_entries = len(inst_pt_all)
                _cached = True
                print(f"\n── 加载阶段 1 缓存, 跳过逐帧投影 ──")
                print(f"  缓存文件: {stage1_cache_path}")
                print(f"  有效语义点: {int((winner_labels >= 0).sum()):,}/{N:,}")
                print(f"  实例观测: {total_inst_entries:,}")
        except Exception as e:
            print(f"  缓存无效 ({e}), 重新计算")

    if not _cached:
        # ══════════════════════════════════════════════════════
        #  阶段 1: 逐帧投影 + 语义投票 + 实例收集
        #  语义: 单遍 N×150 密集矩阵（3.6GB，但比缓存 35B 条目更省）
        #  实例: numpy 扁平数组（替代 Python dict，省 ~6GB）
        # ══════════════════════════════════════════════════════
        vote_counts = np.zeros((N, num_classes), dtype=np.uint16)
        observation_counts = np.zeros(N, dtype=np.uint16)
        inst_pt_list: List[np.ndarray] = []
        inst_gid_list: List[np.ndarray] = []

        print(f"\n── 阶段 1: 逐帧融合 + 投影 ──")
        print(f"  vote_counts: {vote_counts.nbytes / 1e9:.2f} GB")
        used_frames = 0
        total_inst_entries = 0

        for frame_idx, (pose_ts, pose_pos, pose_quat) in enumerate(pose_records):
            dinov3_path = os.path.join(DINOV3_DIR, f"{pose_ts}_mask_id.png")
            if not os.path.exists(dinov3_path):
                continue
            dinov3_map = cv2.imread(dinov3_path, cv2.IMREAD_UNCHANGED)
            if dinov3_map is None:
                continue
            if dinov3_map.ndim == 3:
                dinov3_map = dinov3_map[:, :, 0]

            sam3_frame_dir = os.path.join(SAM3_DIR, str(pose_ts))
            sam3_inst_path = os.path.join(sam3_frame_dir, "instance_map.png")
            sam3_json_path = os.path.join(sam3_frame_dir, "results.json")

            if os.path.exists(sam3_inst_path) and os.path.exists(sam3_json_path):
                sam3_map = np.array(Image.open(sam3_inst_path)).astype(np.int32)
                with open(sam3_json_path) as f:
                    sam3_meta = json.load(f)
                fused_sem, fused_inst, inst_info = fuse_frame(
                    dinov3_map, sam3_map, sam3_meta, ade20k_names, name_to_id,
                )
            else:
                fused_sem = dinov3_map.astype(np.int16)
                fused_inst = np.zeros(dinov3_map.shape, dtype=np.int32)
                inst_info = {}

            if fused_sem.shape[0] != expected_h or fused_sem.shape[1] != expected_w:
                continue

            if SAVE_FUSED_MAPS:
                cv2.imwrite(
                    os.path.join(fused_map_dir, f"{pose_ts}_fused_id.png"),
                    fused_sem.astype(np.uint8),
                )

            pts_cam = transform_global_points_to_camera(
                points, pose_pos, pose_quat, ext_pos, ext_rot_mat,
            )
            pts_2d, valid_idx, depths = project_points_to_image(
                pts_cam, camera_matrix, dist_coeffs, fused_sem.shape,
            )
            if len(pts_2d) == 0:
                continue

            visible = zbuffer_filter(pts_2d, depths, fused_sem.shape)
            pts_2d = pts_2d[visible]
            valid_idx = valid_idx[visible]

            sem_labels = interpolate_int32(fused_sem.astype(np.int32), pts_2d)
            sem_valid = (sem_labels >= 0) & (sem_labels < num_classes)
            vi = valid_idx[sem_valid]
            vl = sem_labels[sem_valid].astype(np.int64)

            np.add.at(vote_counts, (vi, vl), 1)
            observation_counts[vi] += 1

            # 按实例大小自适应腐蚀 SAM3 mask 边界（OpenBox Sec 3.1）
            if MASK_EROSION_RATIO > 0:
                eroded_inst = np.zeros_like(fused_inst)
                for iid in np.unique(fused_inst):
                    if iid == 0:
                        continue
                    binary = (fused_inst == iid).astype(np.uint8)
                    area = int(binary.sum())
                    if area < 20:
                        eroded_inst[binary > 0] = iid
                        continue
                    ks = max(1, int(np.sqrt(area) * MASK_EROSION_RATIO))
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (ks * 2 + 1, ks * 2 + 1))
                    eroded = cv2.erode(binary, kernel)
                    if eroded.sum() > 0:
                        eroded_inst[eroded > 0] = iid
                    else:
                        eroded_inst[binary > 0] = iid
                fused_inst_for_proj = eroded_inst
            else:
                fused_inst_for_proj = fused_inst

            inst_labels = interpolate_int32(fused_inst_for_proj, pts_2d)
            inst_at_valid = inst_labels[sem_valid]
            thing_mask = inst_at_valid > 0
            if thing_mask.any():
                thing_pt_idx = vi[thing_mask].astype(np.int32)
                thing_local_id = inst_at_valid[thing_mask].astype(np.int32)
                global_ids = np.int32(frame_idx) * MAX_INST_PER_FRAME + thing_local_id
                inst_pt_list.append(thing_pt_idx)
                inst_gid_list.append(global_ids)
                total_inst_entries += len(thing_pt_idx)

            used_frames += 1
            if (frame_idx + 1) % PROGRESS_INTERVAL == 0 or frame_idx == len(pose_records) - 1:
                elapsed = time.perf_counter() - t0
                inst_mb = total_inst_entries * 8 / 1e6
                print(
                    f"  [{frame_idx + 1}/{len(pose_records)}] "
                    f"已用帧: {used_frames}, "
                    f"实例缓存: {inst_mb:.0f}MB, "
                    f"耗时: {elapsed:.1f}s"
                )

        print(f"\n阶段 1 完成: {used_frames} 帧, 实例条目: {total_inst_entries:,}")

        # ── 语义投票 ──
        print("\n── 语义投票 ──")
        has_votes = observation_counts >= MIN_VOTES
        winner_labels = np.argmax(vote_counts, axis=1).astype(np.int16)
        winner_labels[~has_votes] = -1
        valid_count = int(has_votes.sum())
        print(f"  有效点: {valid_count:,}/{N:,}")

        del vote_counts, observation_counts
        _release_memory()
        print("  已释放 vote_counts")

        # ══════════════════════════════════════════════════════
        #  法向量地面过滤：移除 thing 类别的地面溅射点
        # ══════════════════════════════════════════════════════
        print("\n── 法向量地面过滤 ──")
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_KNN),
        )
        normals = np.asarray(cloud.normals)
        is_ground_splash = np.abs(normals[:, 2]) > NORMAL_Z_THRESHOLD
        del normals

        thing_label_mask = np.zeros(num_classes, dtype=bool)
        for name in THING_CATEGORIES:
            if name in name_to_id:
                thing_label_mask[name_to_id[name]] = True
        safe_labels = np.clip(winner_labels, 0, num_classes - 1)
        is_thing_point = thing_label_mask[safe_labels] & (winner_labels >= 0)
        splash_thing = is_ground_splash & is_thing_point
        n_splash = int(splash_thing.sum())
        winner_labels[splash_thing] = -1
        print(f"  移除 thing类别 地面溅射点: {n_splash:,}")

        if total_inst_entries > 0:
            inst_pt_all = np.concatenate(inst_pt_list)
            inst_gid_all = np.concatenate(inst_gid_list)
        else:
            inst_pt_all = np.array([], dtype=np.int32)
            inst_gid_all = np.array([], dtype=np.int32)

        del inst_pt_list, inst_gid_list
        _release_memory()

        # 过滤掉地面溅射点的实例观测
        if len(inst_pt_all) > 0:
            keep = ~is_ground_splash[inst_pt_all]
            inst_pt_all = inst_pt_all[keep]
            inst_gid_all = inst_gid_all[keep]
            print(f"  过滤溅射后实例观测: {int(keep.sum()):,} / {len(keep):,}")
        del is_ground_splash

        # 保存阶段 1 缓存
        np.savez(stage1_cache_path,
                 winner_labels=winner_labels,
                 inst_pt_all=inst_pt_all,
                 inst_gid_all=inst_gid_all)
        print(f"  已保存阶段 1 缓存: {stage1_cache_path}")

    # ══════════════════════════════════════════════════════
    #  阶段 2: Union-Find 跨帧实例关联
    # ══════════════════════════════════════════════════════
    print("\n── 阶段 2: Union-Find 跨帧实例关联 ──")
    t2 = time.perf_counter()

    # 排序 + 分段
    uf = UnionFind()
    instance_3d: List[dict] = []

    if len(inst_pt_all) > 0:
        print(f"  排序 {len(inst_pt_all):,} 条实例观测...")
        order = inst_pt_all.argsort()
        inst_pt_sorted = inst_pt_all[order]
        del inst_pt_all
        inst_gid_sorted = inst_gid_all[order]
        del inst_gid_all, order
        _release_memory()

        unique_pts, starts = np.unique(inst_pt_sorted, return_index=True)
        ends = np.append(starts[1:], len(inst_pt_sorted))

        # Pass 1: 计算每个 gid 的 3D 质心（用于限制跨实体链式合并）
        print(f"  计算 global_instance_id 质心...")
        gid_sum: Dict[int, np.ndarray] = {}
        gid_cnt: Dict[int, int] = {}
        for i in range(len(unique_pts)):
            pt_i = int(unique_pts[i])
            xyz = points[pt_i]
            for g in np.unique(inst_gid_sorted[starts[i]:ends[i]]):
                g_int = int(g)
                if g_int in gid_sum:
                    gid_sum[g_int] += xyz
                    gid_cnt[g_int] += 1
                else:
                    gid_sum[g_int] = xyz.copy().astype(np.float64)
                    gid_cnt[g_int] = 1
        gid_centroid = {g: gid_sum[g] / gid_cnt[g] for g in gid_sum}
        print(f"  global_instance_id 总数: {len(gid_centroid):,}")
        del gid_sum, gid_cnt

        # Pass 2: 质心距离受限的 Union-Find（阻断跨实体链式合并）
        max_dist_sq = MAX_MERGE_DIST ** 2
        print(f"  Union-Find 合并（质心距离 < {MAX_MERGE_DIST}m）...")

        obs_counts = ends - starts
        multi_indices = np.where(obs_counts > 1)[0]
        print(f"  多观测点: {len(multi_indices):,} / {len(unique_pts):,}")

        co_pairs = set()
        n_multi = len(multi_indices)
        for idx, i in enumerate(multi_indices):
            if idx % 500_000 == 0 and idx > 0:
                print(f"    扫描进度: {idx:,}/{n_multi:,}, 已发现 {len(co_pairs):,} 共现对")
            gids = np.unique(inst_gid_sorted[starts[i]:ends[i]])
            if len(gids) > 1:
                gids_t = gids.tolist()
                for j in range(len(gids_t)):
                    for k in range(j + 1, len(gids_t)):
                        co_pairs.add((gids_t[j], gids_t[k]))
        print(f"  共现 global_instance_id 对数: {len(co_pairs):,}")

        n_merged, n_blocked = 0, 0
        for ga, gb in co_pairs:
            d = gid_centroid[ga] - gid_centroid[gb]
            if d[0]*d[0] + d[1]*d[1] + d[2]*d[2] < max_dist_sq:
                uf.union(ga, gb)
                n_merged += 1
            else:
                n_blocked += 1
        del gid_centroid
        print(f"  合并: {n_merged:,} 对, 距离阻断: {n_blocked:,} 对")

        # 每个点 → 主导 gid → root（仅 thing 类别参与实例）
        print("  分组实例点...")
        thing_mask_lut = np.zeros(num_classes, dtype=bool)
        for name in THING_CATEGORIES:
            if name in name_to_id:
                thing_mask_lut[name_to_id[name]] = True
        root_to_points: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(unique_pts)):
            pt_i = int(unique_pts[i])
            lbl = winner_labels[pt_i]
            if lbl < 0 or not thing_mask_lut[lbl]:
                continue
            gids = inst_gid_sorted[starts[i]:ends[i]]
            vals, counts = np.unique(gids, return_counts=True)
            dominant_gid = int(vals[np.argmax(counts)])
            root = uf.find(dominant_gid)
            root_to_points[root].append(pt_i)

        del inst_pt_sorted, inst_gid_sorted, unique_pts, starts, ends, uf
        _release_memory()

        # 确定每个实例的语义类别
        n_too_small, n_not_thing, n_oversized = 0, 0, 0
        for root, pt_list in root_to_points.items():
            pt_arr = np.array(pt_list)
            if len(pt_arr) < MIN_INSTANCE_POINTS:
                n_too_small += 1
                continue
            labels = winner_labels[pt_arr]
            valid = labels >= 0
            if not valid.any():
                n_too_small += 1
                continue
            sem_id = int(np.bincount(
                labels[valid].astype(int), minlength=num_classes,
            ).argmax())
            sem_name = ade20k_names[sem_id]
            if sem_name not in THING_CATEGORIES:
                n_not_thing += 1
                continue

            # bbox 尺寸过滤
            pts_xyz = points[pt_arr]
            extent = pts_xyz.max(axis=0) - pts_xyz.min(axis=0)
            if MAX_BBOX_EXTENT > 0 and max(extent) > MAX_BBOX_EXTENT:
                n_oversized += 1
                print(f"    [bounding_box过滤] {sem_name}: {len(pt_arr):,} 点, "
                      f"extent={extent[0]:.1f}x{extent[1]:.1f}x{extent[2]:.1f}m")
                continue

            instance_3d.append({
                "point_indices": pt_arr,
                "semantic_id": sem_id,
                "semantic_name": sem_name,
                "num_points": len(pt_arr),
            })
        del root_to_points
        print(f"  过滤: 点数不足={n_too_small}, 非thing类别={n_not_thing}, "
              f"bounding_box超限={n_oversized}")

    instance_3d.sort(key=lambda x: x["num_points"], reverse=True)
    print(f"  3D 实例总数: {len(instance_3d)}, 耗时: {time.perf_counter() - t2:.1f}s")

    # ══════════════════════════════════════════════════════
    #  阶段 3: 输出点云
    # ══════════════════════════════════════════════════════
    print("\n── 阶段 3: 输出 stuff 语义点云 ──")

    is_thing = np.zeros(N, dtype=bool)
    if instance_3d:
        thing_indices = np.concatenate([inst["point_indices"] for inst in instance_3d])
        is_thing[thing_indices] = True
        del thing_indices

    stuff_count = 0
    for class_id, class_name in enumerate(ade20k_names):
        if class_name in THING_CATEGORIES:
            continue
        mask = (winner_labels == class_id) & ~is_thing
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[idx])
        color = np.array(ADE20K_CATEGORIES[class_name], dtype=np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(idx), 1)))
        out_path = os.path.join(stuff_dir, f"{sanitize_name(class_name)}.pcd")
        o3d.io.write_point_cloud(out_path, pcd)
        stuff_count += 1
        print(f"  {class_name}: {len(idx):,} 点")

    print(f"  stuff 类别数: {stuff_count}")

    print("\n── 输出 thing 实例点云 ──")
    cat_counters: Dict[str, int] = defaultdict(int)
    instance_summary = []

    for inst in sorted(instance_3d, key=lambda x: x["num_points"], reverse=True):
        cat = inst["semantic_name"]
        cat_counters[cat] += 1
        inst_num = cat_counters[cat]
        idx = inst["point_indices"]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[idx])
        color = np.array(ADE20K_CATEGORIES[cat], dtype=np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(idx), 1)))

        filename = f"{sanitize_name(cat)}_inst_{inst_num:03d}.pcd"
        out_path = os.path.join(instance_dir, filename)
        o3d.io.write_point_cloud(out_path, pcd)

        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()

        instance_summary.append({
            "file": filename,
            "category": cat,
            "instance_num": inst_num,
            "num_points": int(len(idx)),
            "bbox_extent": [round(float(e), 3) for e in extent],
        })
        print(f"  {filename}: {len(idx):,} 点, bounding_box={extent[0]:.2f}x{extent[1]:.2f}x{extent[2]:.2f}m")

    summary_path = os.path.join(OUTPUT_DIR, "instance_summary.json")
    with open(summary_path, "w") as f:
        json.dump(instance_summary, f, indent=2, ensure_ascii=False)

    total_time = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"完成！总耗时: {total_time:.1f}s")
    print(f"  stuff 点云:  {stuff_dir}/ ({stuff_count} 类)")
    print(f"  实例点云:    {instance_dir}/ ({len(instance_3d)} 个实例)")
    if SAVE_FUSED_MAPS:
        print(f"  融合标签图:  {fused_map_dir}/")
    print(f"  实例汇总:    {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
