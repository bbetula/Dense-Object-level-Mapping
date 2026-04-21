"""
全局点云全帧投票上色 + 3D法向量过滤地面误投影

与 pointcloud_colorized_v2.py 的区别：
  - 投票完成后，对非地面类别的点做法向量检测
  - 法向量接近竖直向上的点 → 判定为地面误投影 → 不赋色
  - 解决"悬垂遮挡投影"问题（腐蚀无法解决的类型）
"""
import os
import sys
import glob
import time
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from typing import Tuple, List, Dict, Optional, Union, Set
from collections import OrderedDict
from class_statics_config import GLOBAL_PCD_PATH, DEFAULT_IMG_PATH, DEFAULT_YAML_PATH


# ======================== 配置常量 ========================
class Config:
    SEG_SUBDIR_NAME = "res_dinov3_whole"
    MASK_ID_SUFFIX = "_mask_id.png"
    MASK_COLOR_SUFFIX = "_mask.png"
    PALETTE_YAML_PATH = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "dinov3", "yaml", "ADE20k.yaml")
    )

    EXT_YAML_PATH = os.path.join(DEFAULT_YAML_PATH, "avia.yaml")
    INTER_YAML_PATH = os.path.join(DEFAULT_YAML_PATH, "camera_pinhole.yaml")
    CONFIG_TXT_PATH = os.path.join(DEFAULT_YAML_PATH, "config.txt")
    RES_SUBDIR = "res"

    MIN_VOTES_PER_POINT = 5
    LABEL_CACHE_SIZE = 32
    PROGRESS_INTERVAL = 50

    # ── 法向量过滤参数 ──
    NORMAL_FILTER_ENABLED = True
    NORMAL_Z_THRESHOLD = 0.85       # |normal_z| > 此值视为竖直（地面/天花板方向）
    NORMAL_KNN = 30                 # 估计法向量的 KNN 邻域
    GROUND_CLASS_NAMES = {
        "floor", "road", "sidewalk", "grass", "earth", "sand",
        "field", "path", "runway", "dirt track", "land",
        "rug", "floor mat", "sea", "water", "river", "lake",
        "ceiling",
    }


# ======================== 工具函数 ========================
def rotation_matrix_to_quaternion(rotation_matrix: Union[List[float], np.ndarray]) -> np.ndarray:
    rot_mat = np.array(rotation_matrix).reshape(3, 3) if len(rotation_matrix) == 9 else np.array(rotation_matrix)
    rotation = R.from_matrix(rot_mat)
    return rotation.as_quat()


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    rotation = R.from_quat(quaternion)
    return rotation.as_matrix()


def transform_point_cloud(points: np.ndarray, position: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    return np.dot(points, rotation_matrix.T) + position


# ======================== 配置加载 ========================
def load_yaml_config(yaml_path: str) -> Dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML文件不存在: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_camera_params() -> Tuple[Dict, Dict]:
    ext_params_yaml = load_yaml_config(Config.EXT_YAML_PATH)
    inter_params_yaml = load_yaml_config(Config.INTER_YAML_PATH)

    pcl = np.array(ext_params_yaml['extrin_calib']['Pcl'])
    rcl = np.array(ext_params_yaml['extrin_calib']['Rcl']).reshape(3, 3)
    T_cl = np.eye(4)
    T_cl[:3, :3] = rcl
    T_cl[:3, 3] = pcl
    T_lc = np.linalg.inv(T_cl)
    rcl_inv = T_lc[:3, :3]
    pcl_inv = T_lc[:3, 3]

    ext_quat = rotation_matrix_to_quaternion(rcl_inv)
    ext_params = {
        'ext_pos_x': pcl_inv[0], 'ext_pos_y': pcl_inv[1], 'ext_pos_z': pcl_inv[2],
        'ext_q_x': ext_quat[0], 'ext_q_y': ext_quat[1],
        'ext_q_z': ext_quat[2], 'ext_q_w': ext_quat[3],
    }

    scale = inter_params_yaml['scale']
    inter_params = {
        'scale': scale,
        'image_width': int(round(inter_params_yaml['cam_width'] * scale)),
        'image_height': int(round(inter_params_yaml['cam_height'] * scale)),
        'fx': inter_params_yaml['cam_fx'] * scale,
        'fy': inter_params_yaml['cam_fy'] * scale,
        'cx': inter_params_yaml['cam_cx'] * scale,
        'cy': inter_params_yaml['cam_cy'] * scale,
        'k1': inter_params_yaml['cam_d0'], 'k2': inter_params_yaml['cam_d1'],
        'p1': inter_params_yaml['cam_d2'], 'p2': inter_params_yaml['cam_d3'],
        'k3': 0.0,
    }

    save_camera_params_to_txt(ext_params, inter_params)
    return ext_params, inter_params


def save_camera_params_to_txt(ext_params: Dict, inter_params: Dict):
    os.makedirs(os.path.dirname(Config.CONFIG_TXT_PATH), exist_ok=True)
    with open(Config.CONFIG_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write("相机内参:\n")
        f.write(f"  image_width: {inter_params['image_width']}, image_height: {inter_params['image_height']}\n")
        f.write(f"  scale:{inter_params['scale']}, fx: {inter_params['fx']}, fy: {inter_params['fy']}, cx: {inter_params['cx']}, cy: {inter_params['cy']}\n")
        f.write(f"  畸变参数: k1: {inter_params['k1']}, k2: {inter_params['k2']}, p1: {inter_params['p1']}, p2: {inter_params['p2']}, k3: {inter_params['k3']}\n")
        f.write("相机外参:\n")
        for key, value in ext_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"内参/外参已保存到: {Config.CONFIG_TXT_PATH}")


def load_palette_and_lookup() -> Tuple[np.ndarray, Dict[int, int], List[str]]:
    """加载色盘，返回 (palette_rgb, color_to_class, class_names)"""
    palette_yaml = load_yaml_config(Config.PALETTE_YAML_PATH)
    palette = np.array(palette_yaml.get("palette", []), dtype=np.uint8)
    classes = palette_yaml.get("classes", [])
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"色盘格式错误: {Config.PALETTE_YAML_PATH}")

    color_to_class = {}
    for class_id, rgb in enumerate(palette):
        key = (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])
        color_to_class[key] = class_id

    class_names = [str(c) for c in classes]
    return palette, color_to_class, class_names


def build_ground_class_ids(class_names: List[str]) -> Set[int]:
    """从类别名构建地面类别 ID 集合"""
    ground_ids = set()
    for idx, name in enumerate(class_names):
        if name.lower() in Config.GROUND_CLASS_NAMES:
            ground_ids.add(idx)
    return ground_ids


# ======================== Pose 加载 ========================
def load_pose_files(img_path: str) -> Tuple[List[str], List[int]]:
    pose_files = glob.glob(os.path.join(img_path, "*.txt"))
    if not pose_files:
        print(f"警告: 在 {img_path} 中未找到pose文件")
        return [], []
    pose_data = []
    for pose_file in pose_files:
        try:
            with open(pose_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    continue
                timestamp = int(float(first_line.split()[0]))
                pose_data.append((timestamp, pose_file))
        except Exception as e:
            print(f"警告: 无法读取pose文件 {pose_file}: {e}，跳过")
            continue
    pose_data.sort(key=lambda x: x[0])
    return [d[1] for d in pose_data], [d[0] for d in pose_data]


def load_pose_from_file(pose_file: str) -> Tuple[int, np.ndarray, np.ndarray]:
    with open(pose_file, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f"pose文件为空: {pose_file}")
        data = line.split()
        if len(data) < 8:
            raise ValueError(f"pose文件格式错误: {pose_file}")
        timestamp = int(float(data[0]))
        position = np.array([float(data[1]), float(data[2]), float(data[3])])
        quaternion = np.array([float(data[4]), float(data[5]), float(data[6]), float(data[7])])
        return timestamp, position, quaternion


def build_pose_records(sorted_pose_files: List[str]) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    pose_records = []
    for pose_file in sorted_pose_files:
        try:
            pose_records.append(load_pose_from_file(pose_file))
        except Exception as e:
            print(f"警告: 加载pose文件失败 {pose_file}: {e}")
    return pose_records


# ======================== 分割标签 ========================
def find_corresponding_seg_files(pose_timestamp: int, img_path: str) -> Tuple[Optional[str], Optional[str]]:
    seg_dir = os.path.join(img_path, Config.SEG_SUBDIR_NAME)
    mask_id_file = os.path.join(seg_dir, f"{pose_timestamp}{Config.MASK_ID_SUFFIX}")
    mask_color_file = os.path.join(seg_dir, f"{pose_timestamp}{Config.MASK_COLOR_SUFFIX}")
    has_id = os.path.exists(mask_id_file)
    has_color = os.path.exists(mask_color_file)
    if not has_id and not has_color:
        return None, None
    return (mask_id_file if has_id else None), (mask_color_file if has_color else None)


def resolve_output_root(global_pcd_path: str) -> str:
    pcd_dir = os.path.dirname(global_pcd_path)
    if os.path.basename(os.path.normpath(pcd_dir)) == "lidar":
        return pcd_dir
    return os.path.join(pcd_dir, "lidar")


def build_camera_model(inter_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array([
        [inter_params['fx'], 0, inter_params['cx']],
        [0, inter_params['fy'], inter_params['cy']],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array([
        inter_params['k1'], inter_params['k2'],
        inter_params['p1'], inter_params['p2'],
        inter_params['k3']
    ], dtype=np.float64)
    return camera_matrix, dist_coeffs


def get_expected_label_size(inter_params: Dict) -> Tuple[int, int]:
    width = int(inter_params['image_width'])
    height = int(inter_params['image_height'])
    if width <= 0 or height <= 0:
        raise ValueError(f"无效的图像尺寸: width={width}, height={height}")
    return height, width


def build_extrinsic_transform(ext_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    ext_pos = np.array(
        [ext_params['ext_pos_x'], ext_params['ext_pos_y'], ext_params['ext_pos_z']],
        dtype=np.float64,
    )
    ext_quat = np.array(
        [ext_params['ext_q_x'], ext_params['ext_q_y'], ext_params['ext_q_z'], ext_params['ext_q_w']],
        dtype=np.float64,
    )
    return ext_pos, quaternion_to_rotation_matrix(ext_quat)


def transform_global_points_to_camera(
    points: np.ndarray,
    pose_position: np.ndarray,
    pose_quaternion: np.ndarray,
    ext_pos: np.ndarray,
    ext_rot_mat: np.ndarray,
) -> np.ndarray:
    pose_rot_mat = quaternion_to_rotation_matrix(pose_quaternion)
    final_rot_mat = np.dot(pose_rot_mat, ext_rot_mat)
    final_pos = pose_position + np.dot(pose_rot_mat, ext_pos)
    final_rot_mat_inv = final_rot_mat.T
    final_pos_inv = -np.dot(final_rot_mat_inv, final_pos)
    return transform_point_cloud(points, final_pos_inv, final_rot_mat_inv)


def make_colored_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    return colored_pcd


# ======================== 投影 ========================
def project_points_to_image(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_indices = points_3d[:, 2] > 0
    valid_points = points_3d[valid_indices]
    if len(valid_points) == 0:
        return np.array([]), np.array([], dtype=np.intp), np.array([])

    points_2d, _ = cv2.projectPoints(
        valid_points, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
    )
    points_2d = points_2d.reshape(-1, 2)
    depths = valid_points[:, 2]

    h, w = image_shape[:2]
    in_image = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    )
    original_indices = np.where(valid_indices)[0][in_image]
    return points_2d[in_image], original_indices, depths[in_image]


def zbuffer_filter(
    points_2d: np.ndarray,
    depths: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    h, w = image_shape[:2]
    xi = np.clip(np.round(points_2d[:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(points_2d[:, 1]).astype(np.int32), 0, h - 1)
    depth_buffer = np.full((h, w), np.inf, dtype=np.float32)
    np.minimum.at(depth_buffer, (yi, xi), depths.astype(np.float32))
    tolerance = 0.15
    keep = depths <= depth_buffer[yi, xi] + tolerance
    return keep


def round_interpolation_labels(label_map: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    h, w = label_map.shape[:2]
    xi = np.round(points_2d[:, 0]).astype(int)
    yi = np.round(points_2d[:, 1]).astype(int)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    labels = np.full(len(points_2d), -1, dtype=np.int16)
    labels[valid] = label_map[yi[valid], xi[valid]].astype(np.int16)
    return labels


def convert_color_mask_to_label_map(mask_bgr: np.ndarray, color_to_class: Dict[int, int]) -> np.ndarray:
    rgb = mask_bgr[:, :, ::-1].astype(np.uint32)
    keys = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
    labels = np.full(keys.shape, -1, dtype=np.int16)
    for key in np.unique(keys):
        class_id = color_to_class.get(int(key), -1)
        if class_id >= 0:
            labels[keys == key] = class_id
    return labels


def load_label_map_for_pose(
    pose_timestamp: int,
    img_path: str,
    color_to_class: Dict[int, int],
    label_cache: "OrderedDict[int, np.ndarray]",
    expected_image_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    if pose_timestamp in label_cache:
        label_cache.move_to_end(pose_timestamp)
        return label_cache[pose_timestamp]

    mask_id_file, mask_color_file = find_corresponding_seg_files(pose_timestamp, img_path)
    label_map = None

    if mask_id_file is not None:
        label_img = cv2.imread(mask_id_file, cv2.IMREAD_UNCHANGED)
        if label_img is not None:
            if label_img.ndim == 3:
                label_img = label_img[:, :, 0]
            label_map = label_img.astype(np.int16)

    if label_map is None and mask_color_file is not None:
        mask_bgr = cv2.imread(mask_color_file, cv2.IMREAD_COLOR)
        if mask_bgr is not None:
            label_map = convert_color_mask_to_label_map(mask_bgr, color_to_class)

    if label_map is None:
        return None

    target_h, target_w = expected_image_size
    if label_map.shape[0] != target_h or label_map.shape[1] != target_w:
        raise ValueError(
            f"标签图尺寸与投影内参不一致: timestamp={pose_timestamp}, "
            f"label_map={label_map.shape[1]}x{label_map.shape[0]}, "
            f"expected={target_w}x{target_h}. "
            f"请先确保 mask_id 已正确映回相机分辨率。"
        )

    label_cache[pose_timestamp] = label_map
    if len(label_cache) > Config.LABEL_CACHE_SIZE:
        label_cache.popitem(last=False)
    return label_map


# =====================================================
#  法向量过滤：移除地面误投影点
# =====================================================
def filter_ground_projection_by_normal(
    points: np.ndarray,
    winner_labels: np.ndarray,
    has_votes: np.ndarray,
    ground_class_ids: Set[int],
    class_names: List[str],
    normal_z_threshold: float = 0.85,
    normal_knn: int = 30,
) -> np.ndarray:
    """
    对非地面类别的已赋色点，根据法向量过滤地面误投影。

    原理：地面误投影点位于地面表面，其法向量几乎竖直向上(|nz|→1)。
    而真实物体（树、人、家具等）表面法向量以水平/倾斜为主。
    对非地面类别中法向量接近竖直的点，判定为地面误投影并移除赋色。

    Args:
        points: 全局点云坐标 (N, 3)
        winner_labels: 每点的投票获胜类别 (N,)
        has_votes: 每点是否有有效赋色 (N,) bool
        ground_class_ids: 地面类别 ID 集合（这些类别免于过滤）
        class_names: 类别名称列表
        normal_z_threshold: |normal_z| 超过此值视为竖直
        normal_knn: 估计法向量的邻域大小

    Returns:
        更新后的 has_votes 数组（被过滤的点设为 False）
    """
    print("  [法向量过滤] 估计点云法向量...")
    t0 = time.perf_counter()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn)
    )
    normals = np.asarray(pcd.normals)

    elapsed = time.perf_counter() - t0
    print(f"  [法向量过滤] 法向量估计完成: {elapsed:.2f}s")

    # 统一法向量方向：z 分量为正（朝上）
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] *= -1

    # 找出非地面类别 + 法向量接近竖直的点
    is_colored = has_votes.copy()
    is_non_ground = np.array([
        (is_colored[i] and winner_labels[i] not in ground_class_ids)
        for i in range(len(points))
    ], dtype=bool)

    is_vertical = np.abs(normals[:, 2]) > normal_z_threshold
    to_filter = is_non_ground & is_vertical

    filtered_count = int(np.sum(to_filter))
    updated_has_votes = has_votes.copy()
    updated_has_votes[to_filter] = False

    # 统计每个类别被过滤了多少点
    if filtered_count > 0:
        filtered_labels = winner_labels[to_filter]
        unique_labels, counts = np.unique(filtered_labels, return_counts=True)
        print(f"  [法向量过滤] 共过滤 {filtered_count} 个地面误投影点:")
        for label_id, count in sorted(zip(unique_labels, counts), key=lambda x: -x[1]):
            name = class_names[label_id] if label_id < len(class_names) else f"id_{label_id}"
            print(f"    {name}: {count}")
    else:
        print("  [法向量过滤] 未检测到地面误投影点")

    return updated_has_votes


# =====================================================
#  全帧投票上色
# =====================================================
def color_point_cloud_with_majority_vote(
    pcd_file: str,
    pose_records: List[Tuple[int, np.ndarray, np.ndarray]],
    img_path: str,
    ext_params: Dict,
    inter_params: Dict,
    palette_rgb: np.ndarray,
    color_to_class: Dict[int, int],
    label_cache: "OrderedDict[int, np.ndarray]",
    class_names: List[str],
    ground_class_ids: Set[int],
) -> Optional[o3d.geometry.PointCloud]:
    """全局点云全帧投票上色 + 法向量过滤"""
    try:
        cloud = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(cloud.points)
    except Exception as e:
        print(f"错误: 加载点云文件 {pcd_file} 失败: {e}")
        return None

    if len(points) == 0:
        print(f"警告: 空点云文件 {pcd_file}")
        return None

    num_points = len(points)
    print(f"  点云加载完成: {num_points} 个点")

    camera_matrix, dist_coeffs = build_camera_model(inter_params)
    ext_pos, ext_rot_mat = build_extrinsic_transform(ext_params)
    expected_image_size = get_expected_label_size(inter_params)

    num_classes = int(palette_rgb.shape[0])
    vote_counts = np.zeros((num_points, num_classes), dtype=np.uint16)
    observation_counts = np.zeros(num_points, dtype=np.uint16)

    total_frames = len(pose_records)
    used_frames = 0
    stage_start_time = time.perf_counter()
    chunk_start_time = stage_start_time
    last_report_frame = 0

    for pose_idx in range(total_frames):
        pose_timestamp, pose_position, pose_quaternion = pose_records[pose_idx]
        label_map = load_label_map_for_pose(
            pose_timestamp, img_path, color_to_class,
            label_cache, expected_image_size,
        )
        if label_map is None:
            continue

        points_3d = transform_global_points_to_camera(
            points, pose_position, pose_quaternion, ext_pos, ext_rot_mat
        )
        points_2d, valid_indices, depths = project_points_to_image(
            points_3d, camera_matrix, dist_coeffs, label_map.shape
        )
        if len(points_2d) == 0:
            continue

        visible = zbuffer_filter(points_2d, depths, label_map.shape)
        points_2d = points_2d[visible]
        valid_indices = valid_indices[visible]

        labels = round_interpolation_labels(label_map, points_2d)
        valid_label_mask = (labels >= 0) & (labels < num_classes)
        if not np.any(valid_label_mask):
            continue

        valid_point_indices = valid_indices[valid_label_mask]
        valid_labels = labels[valid_label_mask].astype(np.int64)
        np.add.at(vote_counts, (valid_point_indices, valid_labels), 1)
        observation_counts[valid_point_indices] += 1
        used_frames += 1

        if (pose_idx + 1) % Config.PROGRESS_INTERVAL == 0 or pose_idx == total_frames - 1:
            now = time.perf_counter()
            chunk_frames = (pose_idx + 1) - last_report_frame
            chunk_elapsed = now - chunk_start_time
            total_elapsed = now - stage_start_time
            print(
                f"  投票进度: {pose_idx + 1}/{total_frames} 帧，已用帧: {used_frames}，"
                f"最近{chunk_frames}帧耗时: {chunk_elapsed:.2f}s，"
                f"平均: {chunk_elapsed / max(chunk_frames, 1):.3f}s/帧，"
                f"累计: {total_elapsed:.2f}s"
            )
            chunk_start_time = now
            last_report_frame = pose_idx + 1

    has_votes = observation_counts >= Config.MIN_VOTES_PER_POINT
    voted_points_before = int(np.sum(has_votes))
    if voted_points_before == 0:
        print("  警告: 没有可用类别观测")
        return None

    winner_labels = np.argmax(vote_counts, axis=1)

    avg_votes = float(observation_counts[has_votes].mean()) if voted_points_before > 0 else 0.0
    print(
        f"  全帧投票完成: 有效帧 {used_frames}/{total_frames}，"
        f"赋色点 {voted_points_before}/{num_points}，平均票数 {avg_votes:.1f}"
    )

    # ── 法向量过滤 ──
    if Config.NORMAL_FILTER_ENABLED:
        has_votes = filter_ground_projection_by_normal(
            points, winner_labels, has_votes, ground_class_ids, class_names,
            normal_z_threshold=Config.NORMAL_Z_THRESHOLD,
            normal_knn=Config.NORMAL_KNN,
        )
        voted_points_after = int(np.sum(has_votes))
        print(
            f"  法向量过滤后: 赋色点 {voted_points_after}/{num_points} "
            f"(过滤 {voted_points_before - voted_points_after} 个)"
        )

    all_colors = np.zeros((num_points, 3), dtype=np.float32)
    all_colors[has_votes] = palette_rgb[winner_labels[has_votes]] / 255.0
    colored_pcd = make_colored_point_cloud(points, all_colors)
    return colored_pcd


# =====================================================
#  主函数
# =====================================================
def main(global_pcd_path: str, img_path: str) -> None:
    try:
        ext_params, inter_params = load_camera_params()
        palette_rgb, color_to_class, class_names = load_palette_and_lookup()
    except Exception as e:
        print(f"错误: 加载必要配置失败: {e}")
        return

    ground_class_ids = build_ground_class_ids(class_names)
    ground_names = [class_names[i] for i in sorted(ground_class_ids) if i < len(class_names)]
    print(f"地面类别（免于法向量过滤）: {ground_names}")
    print(f"法向量过滤: {'启用' if Config.NORMAL_FILTER_ENABLED else '禁用'}, "
          f"阈值={Config.NORMAL_Z_THRESHOLD}, KNN={Config.NORMAL_KNN}")

    sorted_pose_files, _ = load_pose_files(img_path)
    pose_records = build_pose_records(sorted_pose_files)
    if not pose_records:
        print("错误: 无有效pose文件，退出")
        return

    print(f"全局点云: {global_pcd_path}")
    print(f"pose帧数: {len(pose_records)}")
    print(f"最少票数: {Config.MIN_VOTES_PER_POINT}，类别数: {len(palette_rgb)}")
    print(f"投影标签尺寸: {inter_params['image_width']}x{inter_params['image_height']}")

    output_root = resolve_output_root(global_pcd_path)
    res_dir = os.path.join(output_root, Config.RES_SUBDIR)
    os.makedirs(res_dir, exist_ok=True)
    print(f"输出目录: res={res_dir}")

    label_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    try:
        colored_pcd = color_point_cloud_with_majority_vote(
            pcd_file=global_pcd_path,
            pose_records=pose_records,
            img_path=img_path,
            ext_params=ext_params,
            inter_params=inter_params,
            palette_rgb=palette_rgb,
            color_to_class=color_to_class,
            label_cache=label_cache,
            class_names=class_names,
            ground_class_ids=ground_class_ids,
        )
    except ValueError as e:
        print(f"错误: {e}")
        return
    if colored_pcd is None:
        print("错误: 上色失败")
        return

    pcd_stem = os.path.splitext(os.path.basename(global_pcd_path))[0]
    output_file = os.path.join(res_dir, f"{pcd_stem}_color_normal.pcd")
    o3d.io.write_point_cloud(output_file, colored_pcd)
    print(f"\n成功保存彩色全局点云: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        global_pcd_path = sys.argv[1]
        img_path = sys.argv[2]
    else:
        global_pcd_path = GLOBAL_PCD_PATH
        img_path = DEFAULT_IMG_PATH

    main(global_pcd_path, img_path)
