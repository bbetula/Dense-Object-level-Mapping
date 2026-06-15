"""
ScanNet 场景语义投影脚本

将 DINOv3 分割结果（彩色 mask，NYU40 色盘）通过 depth + pose 反投影
到 3D 空间，生成每个场景的语义着色点云。

输入（每个场景文件夹内）:
  - scans/{scene}/color/                原始 RGB 图像
  - scans/{scene}/depth/                深度图 (uint16, mm)
  - scans/{scene}/pose/                 相机位姿 (4×4 camera-to-world)
  - scans/{scene}/intrinsic/            相机内参
  - scans/{scene}/prediction/           DINOv3 彩色 mask（{frame}_mask.png）
输出:
  - scans/{scene}/res/{scene}_semantic.pcd
"""
import os
import re
import time
import glob
import yaml
import numpy as np
import cv2
import open3d as o3d
from typing import Dict, List, Optional, Tuple


# ======================== 配置 ========================
SCANNET_BASE = "/data1/data/scannet/scans"
VAL_SPLIT = "/data1/data/scannet/experiment/splits/scannet_val.txt"
PREDICTION_SUBDIR = "prediction"       # 与 color/ 同级的子目录名
MASK_SUFFIX = "_mask.png"              # 彩色 mask 文件后缀

PALETTE_YAML_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "dinov3", "yaml", "scannet_nyu40.yaml")
)

DEPTH_SCALE = 1000.0                   # ScanNet 深度 uint16 mm → m
MAX_DEPTH = 10.0                       # 最大有效深度 (m)
PROGRESS_INTERVAL = 100

# ── 多帧投票参数 ──
VOTE_ENABLED = True
VOTE_VOXEL_SIZE = 0.02                 # 投票体素大小 (m)
VOTE_MIN_OBSERVATIONS = 3             # 最少被观测帧数


# ======================== 调色板加载 ========================
def load_palette(yaml_path: str) -> Tuple[np.ndarray, Dict[int, int], List[str]]:
    """加载 NYU40 色盘。

    Returns:
        palette_rgb: (N_classes, 3) uint8
        color_to_class: {rgb_packed_int: class_id}  用于从彩色 mask 反查 class ID
        class_names: 类别名称列表
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    palette = np.array(cfg["palette"], dtype=np.uint8)
    classes = [str(c) for c in cfg["classes"]]
    color_to_class: Dict[int, int] = {}
    for class_id, rgb in enumerate(palette):
        key = (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])
        color_to_class[key] = class_id
    return palette, color_to_class, classes


def color_mask_to_label_map(
    mask_bgr: np.ndarray, color_to_class: Dict[int, int]
) -> np.ndarray:
    """BGR 彩色 mask → class ID 图 (0-indexed, -1 = 未识别)"""
    rgb = mask_bgr[:, :, ::-1].astype(np.uint32)
    keys = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
    labels = np.full(keys.shape, -1, dtype=np.int16)
    for key in np.unique(keys):
        cid = color_to_class.get(int(key), -1)
        if cid >= 0:
            labels[keys == key] = cid
    return labels


# ======================== 数据加载 ========================
def load_intrinsic(path: str) -> np.ndarray:
    """读取 4×4 内参文件，返回 3×3 矩阵"""
    return np.loadtxt(path)[:3, :3]


def load_pose(path: str) -> Optional[np.ndarray]:
    """读取 4×4 camera-to-world 位姿，无效帧返回 None"""
    pose = np.loadtxt(path)
    if pose.shape != (4, 4):
        return None
    if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
        return None
    return pose


def load_depth(path: str) -> Optional[np.ndarray]:
    """读取 uint16 深度图 → float32 (m)"""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    return depth.astype(np.float32) / DEPTH_SCALE


def load_label_map(
    frame_id: str,
    pred_dir: str,
    color_to_class: Dict[int, int],
    target_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """加载彩色 mask，转为 class ID 图，缩放到 depth 分辨率。"""
    mask_path = os.path.join(pred_dir, f"{frame_id}{MASK_SUFFIX}")
    if not os.path.exists(mask_path):
        return None

    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_bgr is None:
        return None

    label_map = color_mask_to_label_map(mask_bgr, color_to_class)

    th, tw = target_shape
    if label_map.shape[0] != th or label_map.shape[1] != tw:
        label_map = cv2.resize(label_map, (tw, th), interpolation=cv2.INTER_NEAREST)
    return label_map


# ======================== 核心反投影 ========================
def backproject_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """深度图 → 相机坐标系 3D 点。

    Returns:
        points_cam: (N, 3)
        pixel_indices: (N,) 展平像素索引
    """
    h, w = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                        np.arange(h, dtype=np.float32))

    valid = (depth > 0) & (depth < MAX_DEPTH)
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=-1)
    pixel_indices = np.where(valid.ravel())[0]
    return points_cam, pixel_indices


# ======================== 多帧体素投票 ========================
def voxel_majority_vote(
    points: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    voxel_size: float = VOTE_VOXEL_SIZE,
    min_obs: int = VOTE_MIN_OBSERVATIONS,
) -> Tuple[np.ndarray, np.ndarray]:
    """对所有帧的投影点做体素化多帧投票。

    Returns:
        voted_points: (M, 3) 体素中心坐标
        voted_labels: (M,)   投票胜出的类别 ID
    """
    voxel_ijk = np.floor(points / voxel_size).astype(np.int32)
    mins = voxel_ijk.min(axis=0)
    shifted = (voxel_ijk - mins).astype(np.int64)
    keys = (shifted[:, 0] << 40) | (shifted[:, 1] << 20) | shifted[:, 2]

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n_voxels = len(unique_keys)

    vote_counts = np.zeros((n_voxels, n_classes), dtype=np.int32)
    np.add.at(vote_counts, (inverse, labels), 1)

    total_obs = vote_counts.sum(axis=1)
    winner = np.argmax(vote_counts, axis=1)

    valid = total_obs >= min_obs

    voxel_sum = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(voxel_sum, inverse, points)
    voxel_count = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
    voxel_centers = voxel_sum / np.maximum(voxel_count[:, None], 1)

    return voxel_centers[valid], winner[valid]


# ======================== 单场景处理 ========================
def process_scene(
    scene_name: str,
    palette: np.ndarray,
    color_to_class: Dict[int, int],
) -> None:
    scene_dir = os.path.join(SCANNET_BASE, scene_name)
    pred_dir = os.path.join(scene_dir, PREDICTION_SUBDIR)

    if not os.path.isdir(scene_dir):
        print(f"  跳过: 场景目录不存在 {scene_dir}")
        return
    if not os.path.isdir(pred_dir):
        print(f"  跳过: prediction/ 不存在 {pred_dir}")
        return

    # 深度相机内参
    intrinsic = load_intrinsic(
        os.path.join(scene_dir, "intrinsic", "intrinsic_depth.txt")
    )

    # 枚举有 mask 的帧
    mask_files = sorted(glob.glob(os.path.join(pred_dir, f"*{MASK_SUFFIX}")))
    if not mask_files:
        print(f"  跳过: prediction/ 下无 mask 文件")
        return

    frame_ids = []
    for mf in mask_files:
        m = re.match(r"(\d+)", os.path.basename(mf))
        if m:
            frame_ids.append(m.group(1))
    frame_ids = sorted(set(frame_ids), key=int)

    all_points: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    valid_count = 0
    t0 = time.time()

    for i, fid in enumerate(frame_ids):
        depth_path = os.path.join(scene_dir, "depth", f"{fid}.png")
        pose_path = os.path.join(scene_dir, "pose", f"{fid}.txt")

        if not os.path.exists(depth_path) or not os.path.exists(pose_path):
            continue

        depth = load_depth(depth_path)
        if depth is None:
            continue
        pose = load_pose(pose_path)
        if pose is None:
            continue

        label_map = load_label_map(fid, pred_dir, color_to_class, depth.shape)
        if label_map is None:
            continue

        points_cam, pix_idx = backproject_depth(depth, intrinsic)
        if len(points_cam) == 0:
            continue

        point_labels = label_map.ravel()[pix_idx]
        valid_mask = point_labels >= 0
        points_cam = points_cam[valid_mask]
        point_labels = point_labels[valid_mask]
        if len(points_cam) == 0:
            continue

        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = points_cam @ R.T + t

        all_points.append(points_world)
        all_labels.append(point_labels)
        valid_count += 1

        if (i + 1) % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - t0
            print(f"    [{i + 1}/{len(frame_ids)}] {valid_count} 有效帧, {elapsed:.1f}s")

    if not all_points:
        print(f"  {scene_name}: 无有效帧")
        return

    points_all = np.vstack(all_points)
    labels_all = np.concatenate(all_labels).astype(np.int32)
    print(f"  投影完成: {len(points_all):,} 点, {valid_count} 帧")

    if VOTE_ENABLED:
        n_classes = len(palette)
        points_all, labels_all = voxel_majority_vote(
            points_all, labels_all, n_classes,
            VOTE_VOXEL_SIZE, VOTE_MIN_OBSERVATIONS,
        )
        print(
            f"  投票后: {len(points_all):,} 体素 "
            f"(voxel={VOTE_VOXEL_SIZE}m, min_obs={VOTE_MIN_OBSERVATIONS})"
        )

    colors = np.zeros((len(labels_all), 3), dtype=np.float64)
    valid = (labels_all >= 0) & (labels_all < len(palette))
    colors[valid] = palette[labels_all[valid]] / 255.0

    output_dir = os.path.join(scene_dir, "res")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scene_name}_semantic.pcd")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_all)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)

    elapsed = time.time() - t0
    print(
        f"  {scene_name}: {valid_count} 帧, "
        f"{len(points_all):,} 点 → {output_path} ({elapsed:.1f}s)"
    )


# ======================== 主函数 ========================
def main() -> None:
    if not os.path.exists(VAL_SPLIT):
        raise FileNotFoundError(f"验证集列表不存在: {VAL_SPLIT}")

    with open(VAL_SPLIT, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    palette, color_to_class, class_names = load_palette(PALETTE_YAML_PATH)
    print(f"共 {len(scenes)} 个场景, 调色板: {len(palette)} 类 ({PALETTE_YAML_PATH})")
    print(f"prediction 子目录: {PREDICTION_SUBDIR}/")
    print(f"输出: scans/{{scene}}/res/{{scene}}_semantic.pcd\n")

    t_total = time.time()
    for i, scene in enumerate(scenes):
        print(f"[{i + 1}/{len(scenes)}] {scene}")
        process_scene(scene, palette, color_to_class)

    print(f"\n全部完成, 总耗时 {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
