import os
import sys
import glob
import time
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from typing import Tuple, List, Dict, Optional, Union
from collections import OrderedDict
from class_statics_config import DATA_CHOICE


# ======================== 配置常量（集中管理，便于修改）========================
class Config:
    """配置类：集中管理所有路径和参数"""
    # 全局融合点云路径
    GLOBAL_PCD_PATHS = {
        "indoor1": "",
        "outdoor1": "",
        "outdoor2": "/data1/user/data/fastlivo_output_qs2_03.17/all_raw_qs2_03.17.pcd",
    }
    GLOBAL_PCD_PATH = GLOBAL_PCD_PATHS[DATA_CHOICE]

    DEFAULT_IMG_PATHS = {
        "indoor1": "/data1/user/data/fastlivo_output_indoor_107/image",
        "outdoor1": "/data1/user/data/fastlivo_output_outdoor_1s/image",
        "outdoor2": "/data1/user/data/fastlivo_output_qs2_03.17/image", #1.3h
    }
    DEFAULT_IMG_PATH = DEFAULT_IMG_PATHS[DATA_CHOICE]
    SEG_SUBDIR_NAME = "res_dinov3_whole"  # 分割图像子目录名称
    MASK_ID_SUFFIX = "_mask_id.png"
    MASK_COLOR_SUFFIX = "_mask.png"
    PALETTE_YAML_PATH = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "dinov3", "yaml", "ADE20k.yaml")
    )

    # YAML配置路径
    # 无人机配置文件：Drone_newcam_config
    # 手持设备：Handheld_device_config
    DEFAULT_YAML_PATH = "Drone_newcam_config"
    # DEFAULT_YAML_PATH = "Handheld_device_config"
    EXT_YAML_PATH = os.path.join(DEFAULT_YAML_PATH, "avia.yaml")
    INTER_YAML_PATH = os.path.join(DEFAULT_YAML_PATH, "camera_pinhole.yaml")

    # 输出配置（在DEFAULT_YAML_PATH下的config.txt）
    CONFIG_TXT_PATH = os.path.join(DEFAULT_YAML_PATH, "config.txt")
    RES_SUBDIR = "res"

    # 相机图像分辨率（内参对应的分辨率 = cam_width*scale × cam_height*scale）
    CAM_WIDTH = 640
    CAM_HEIGHT = 512

    MIN_VOTES_PER_POINT = 1       # 点被至少观测多少次才赋类别
    LABEL_CACHE_SIZE = 32         # 滑动缓存标签图
    PROGRESS_INTERVAL = 50        # 每隔多少帧打印一次耗时

# ======================== 工具函数（坐标变换相关）========================
def rotation_matrix_to_quaternion(rotation_matrix: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    将3x3旋转矩阵转换为四元数 [x, y, z, w]
    
    Args:
        rotation_matrix: 3x3旋转矩阵或9元素列表
    
    Returns:
        np.ndarray: 四元数 [x, y, z, w]
    """
    # 重塑为3x3矩阵
    rot_mat = np.array(rotation_matrix).reshape(3, 3) if len(rotation_matrix) == 9 else np.array(rotation_matrix)
    # 转换为四元数
    rotation = R.from_matrix(rot_mat)
    return rotation.as_quat()


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    将四元数转换为3x3旋转矩阵
    
    Args:
        quaternion: 四元数 [x, y, z, w]
    
    Returns:
        np.ndarray: 3x3旋转矩阵
    """
    rotation = R.from_quat(quaternion)
    return rotation.as_matrix()


def transform_point_cloud(points: np.ndarray, position: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    将点云变换到目标坐标系
    
    Args:
        points: 点云坐标 (N, 3)
        position: 平移向量 (3,)
        rotation_matrix: 旋转矩阵 (3, 3)
    
    Returns:
        np.ndarray: 变换后的点云 (N, 3)
    """
    return np.dot(points, rotation_matrix.T) + position


# ======================== 配置加载函数 ========================
def load_yaml_config(yaml_path: str) -> Dict:
    """
    加载YAML配置文件
    
    Args:
        yaml_path: YAML文件路径
    
    Returns:
        Dict: YAML配置字典
    
    Raises:
        FileNotFoundError: 文件不存在时抛出
        yaml.YAMLError: YAML解析失败时抛出
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML文件不存在: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_camera_params() -> Tuple[Dict, Dict]:
    """
    加载相机内参和外参，并保存到config.txt
    
    Returns:
        Tuple[Dict, Dict]: 外参字典, 内参字典
    """
    # 加载外参
    ext_params_yaml = load_yaml_config(Config.EXT_YAML_PATH)
    inter_params_yaml = load_yaml_config(Config.INTER_YAML_PATH)
    
    # 解析外参
    pcl = np.array(ext_params_yaml['extrin_calib']['Pcl'])
    rcl = np.array(ext_params_yaml['extrin_calib']['Rcl']).reshape(3, 3)
    
    # 构建雷达到相机的齐次变换矩阵 T_cl
    T_cl = np.eye(4)
    T_cl[:3, :3] = rcl
    T_cl[:3, 3] = pcl
    
    # 计算相机到雷达的逆变换 T_lc
    T_lc = np.linalg.inv(T_cl)
    rcl_inv = T_lc[:3, :3]
    pcl_inv = T_lc[:3, 3]
    
    # 转换为四元数
    ext_quat = rotation_matrix_to_quaternion(rcl_inv)
    ext_params = {
        'ext_pos_x': pcl_inv[0],
        'ext_pos_y': pcl_inv[1],
        'ext_pos_z': pcl_inv[2],
        'ext_q_x': ext_quat[0],
        'ext_q_y': ext_quat[1],
        'ext_q_z': ext_quat[2],
        'ext_q_w': ext_quat[3]
    }
    
    # 解析内参
    scale = inter_params_yaml['scale']
    inter_params = {
        'scale': scale,
        'fx': inter_params_yaml['cam_fx'] * scale,
        'fy': inter_params_yaml['cam_fy'] * scale,
        'cx': inter_params_yaml['cam_cx'] * scale,
        'cy': inter_params_yaml['cam_cy'] * scale,
        'k1': inter_params_yaml['cam_d0'],
        'k2': inter_params_yaml['cam_d1'],
        'p1': inter_params_yaml['cam_d2'],
        'p2': inter_params_yaml['cam_d3'],
        'k3': 0.0
    }
    
    # 保存到config.txt
    save_camera_params_to_txt(ext_params, inter_params)
    
    return ext_params, inter_params


def save_camera_params_to_txt(ext_params: Dict, inter_params: Dict):
    """
    将相机内参和外参保存到config.txt
    
    Args:
        ext_params: 外参字典
        inter_params: 内参字典
    """
    os.makedirs(os.path.dirname(Config.CONFIG_TXT_PATH), exist_ok=True)
    
    with open(Config.CONFIG_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write("相机内参:\n")
        f.write(f"  scale:{inter_params['scale']}, fx: {inter_params['fx']}, fy: {inter_params['fy']}, cx: {inter_params['cx']}, cy: {inter_params['cy']}\n")
        f.write(f"  畸变参数: k1: {inter_params['k1']}, k2: {inter_params['k2']}, p1: {inter_params['p1']}, p2: {inter_params['p2']}, k3: {inter_params['k3']}\n")
        f.write("相机外参:\n")
        for key, value in ext_params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"内参/外参已保存到: {Config.CONFIG_TXT_PATH}")


def load_palette_and_lookup() -> Tuple[np.ndarray, Dict[int, int]]:
    """
    加载色盘并建立 RGB->类别ID 映射表

    Returns:
        Tuple[np.ndarray, Dict[int, int]]: 色盘 (C,3) RGB, 映射表(key=24bit RGB整数, value=class_id)
    """
    palette_yaml = load_yaml_config(Config.PALETTE_YAML_PATH)
    palette = np.array(palette_yaml.get("palette", []), dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"色盘格式错误: {Config.PALETTE_YAML_PATH}")

    color_to_class = {}
    for class_id, rgb in enumerate(palette):
        key = (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])
        color_to_class[key] = class_id
    return palette, color_to_class


def load_pose_files(img_path: str) -> Tuple[List[str], List[int]]:
    """
    加载并排序所有pose文件（按时间戳）
    
    Args:
        img_path: pose文件所在目录（与图像同目录）
    
    Returns:
        Tuple[List[str], List[int]]: 排序后的pose文件路径, 对应的时间戳
    """
    pose_files = glob.glob(os.path.join(img_path, "*.txt"))
    if not pose_files:
        print(f"警告: 在 {img_path} 中未找到pose文件")
        return [], []
    
    # 提取时间戳
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
    
    # 按时间戳排序
    pose_data.sort(key=lambda x: x[0])
    sorted_pose_files = [d[1] for d in pose_data]
    sorted_pose_timestamps = [d[0] for d in pose_data]
    
    return sorted_pose_files, sorted_pose_timestamps


def load_pose_from_file(pose_file: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    从pose文件加载时间戳、位置、四元数
    
    Args:
        pose_file: pose文件路径
    
    Returns:
        Tuple[int, np.ndarray, np.ndarray]: 时间戳, 位置(3,), 四元数(4,)
    
    Raises:
        ValueError: 解析失败时抛出
    """
    with open(pose_file, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f"pose文件为空: {pose_file}")
        
        data = line.split()
        if len(data) < 8:
            raise ValueError(f"pose文件格式错误，至少需要8个字段: {pose_file}")
        
        timestamp = int(float(data[0]))
        position = np.array([float(data[1]), float(data[2]), float(data[3])])
        quaternion = np.array([float(data[4]), float(data[5]), float(data[6]), float(data[7])])
        
        return timestamp, position, quaternion


def build_pose_records(sorted_pose_files: List[str]) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    预加载pose文件，避免在滑窗投票中重复读盘

    Args:
        sorted_pose_files: 排序后的pose文件路径

    Returns:
        List[Tuple[int, np.ndarray, np.ndarray]]: (timestamp, position, quaternion)
    """
    pose_records = []
    for pose_file in sorted_pose_files:
        try:
            pose_records.append(load_pose_from_file(pose_file))
        except Exception as e:
            print(f"警告: 加载pose文件失败 {pose_file}: {e}")
    return pose_records


def find_corresponding_seg_files(pose_timestamp: int, img_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    查找与pose时间戳对应的分割结果文件（优先mask_id，兼容mask彩色图）

    Args:
        pose_timestamp: pose时间戳
        img_path: 图像目录

    Returns:
        Tuple[Optional[str], Optional[str]]: (mask_id_path, mask_color_path)
    """
    seg_dir = os.path.join(img_path, Config.SEG_SUBDIR_NAME)
    mask_id_file = os.path.join(seg_dir, f"{pose_timestamp}{Config.MASK_ID_SUFFIX}")
    mask_color_file = os.path.join(seg_dir, f"{pose_timestamp}{Config.MASK_COLOR_SUFFIX}")

    has_id = os.path.exists(mask_id_file)
    has_color = os.path.exists(mask_color_file)
    if not has_id and not has_color:
        return None, None
    return (mask_id_file if has_id else None), (mask_color_file if has_color else None)

def resolve_output_root(global_pcd_path: str) -> str:
    """
    输出统一落在 lidar 目录下
    """
    pcd_dir = os.path.dirname(global_pcd_path)
    if os.path.basename(os.path.normpath(pcd_dir)) == "lidar":
        return pcd_dir
    return os.path.join(pcd_dir, "lidar")


def build_camera_model(inter_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建相机内参与畸变参数
    """
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


def build_extrinsic_transform(ext_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    解析相机外参
    """
    ext_pos = np.array(
        [ext_params['ext_pos_x'], ext_params['ext_pos_y'], ext_params['ext_pos_z']],
        dtype=np.float64
    )
    ext_quat = np.array(
        [ext_params['ext_q_x'], ext_params['ext_q_y'], ext_params['ext_q_z'], ext_params['ext_q_w']],
        dtype=np.float64
    )
    return ext_pos, quaternion_to_rotation_matrix(ext_quat)


def transform_global_points_to_camera(
    points: np.ndarray,
    pose_position: np.ndarray,
    pose_quaternion: np.ndarray,
    ext_pos: np.ndarray,
    ext_rot_mat: np.ndarray,
) -> np.ndarray:
    """
    将全局点云变换到当前相机坐标系
    """
    pose_rot_mat = quaternion_to_rotation_matrix(pose_quaternion)
    final_rot_mat = np.dot(pose_rot_mat, ext_rot_mat)
    final_pos = pose_position + np.dot(pose_rot_mat, ext_pos)
    final_rot_mat_inv = final_rot_mat.T
    final_pos_inv = -np.dot(final_rot_mat_inv, final_pos)
    return transform_point_cloud(points, final_pos_inv, final_rot_mat_inv)


def make_colored_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """
    构建带颜色点云对象
    """
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    return colored_pcd


# ======================== 投影和插值函数 ========================
def project_points_to_image(points_3d: np.ndarray,
                           camera_matrix: np.ndarray,
                           dist_coeffs: np.ndarray,
                           image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将3D点云投影到2D图像平面

    Returns:
        Tuple: 2D投影点 (M, 2), 有效点的原始索引 (M,), 相机坐标系下深度 (M,)
    """
    # 只保留相机前方的点（z>0）
    valid_indices = points_3d[:, 2] > 0
    valid_points = points_3d[valid_indices]

    if len(valid_points) == 0:
        return np.array([]), np.array([], dtype=np.intp), np.array([])

    # 投影到图像平面
    points_2d, _ = cv2.projectPoints(
        valid_points, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
    )
    points_2d = points_2d.reshape(-1, 2)
    depths = valid_points[:, 2]  # 相机z轴深度

    # 过滤图像范围内的点
    h, w = image_shape[:2]
    in_image = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    )

    original_indices = np.where(valid_indices)[0][in_image]
    return points_2d[in_image], original_indices, depths[in_image]


def zbuffer_filter(points_2d: np.ndarray, depths: np.ndarray,
                   image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Z-buffer 遮挡过滤：对每个像素只保留深度最小（最近）的点

    Args:
        points_2d: 投影2D坐标 (M, 2)
        depths: 相机坐标系下的深度 (M,)
        image_shape: (h, w)

    Returns:
        np.ndarray: 布尔掩码 (M,)，True 表示该点是其像素上最近的点
    """
    h, w = image_shape[:2]
    xi = np.clip(np.round(points_2d[:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(points_2d[:, 1]).astype(np.int32), 0, h - 1)

    # 初始化深度缓冲为无穷大
    depth_buffer = np.full((h, w), np.inf, dtype=np.float32)

    # 向量化更新深度缓冲：np.minimum.at 取每个像素的最小深度
    np.minimum.at(depth_buffer, (yi, xi), depths.astype(np.float32))

    # 保留深度在容差范围内的点（容差应对同一表面多个点的微小深度差异）
    tolerance = 0.15  # 15cm
    keep = depths <= depth_buffer[yi, xi] + tolerance

    return keep


def round_interpolation_labels(label_map: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """
    取整插值获取标签ID（向量化）

    Args:
        label_map: 标签图 (h, w)，值为类别ID
        points_2d: 2D投影点 (M, 2)

    Returns:
        np.ndarray: 每个投影点的标签ID (M,)，无效点为-1
    """
    h, w = label_map.shape[:2]
    xi = np.round(points_2d[:, 0]).astype(int)
    yi = np.round(points_2d[:, 1]).astype(int)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    labels = np.full(len(points_2d), -1, dtype=np.int16)
    labels[valid] = label_map[yi[valid], xi[valid]].astype(np.int16)
    return labels


def convert_color_mask_to_label_map(mask_bgr: np.ndarray, color_to_class: Dict[int, int]) -> np.ndarray:
    """
    将彩色mask(BGR)转换为类别ID标签图

    Args:
        mask_bgr: 彩色mask图 (h, w, 3)，BGR
        color_to_class: RGB颜色到类别ID映射

    Returns:
        np.ndarray: 标签图 (h, w)，无匹配像素为-1
    """
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
    label_cache: "OrderedDict[int, np.ndarray]"
) -> Optional[np.ndarray]:
    """
    读取并缓存pose对应标签图
    """
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

    # 确保标签图分辨率与相机内参一致，避免投影坐标错位
    target_h, target_w = Config.CAM_HEIGHT, Config.CAM_WIDTH
    if label_map.shape[0] != target_h or label_map.shape[1] != target_w:
        label_map = cv2.resize(label_map.astype(np.uint8), (target_w, target_h),
                               interpolation=cv2.INTER_NEAREST).astype(np.int16)

    label_cache[pose_timestamp] = label_map
    if len(label_cache) > Config.LABEL_CACHE_SIZE:
        label_cache.popitem(last=False)
    return label_map

def color_point_cloud_with_majority_vote(
    pcd_file: str,
    pose_records: List[Tuple[int, np.ndarray, np.ndarray]],
    img_path: str,
    ext_params: Dict,
    inter_params: Dict,
    palette_rgb: np.ndarray,
    color_to_class: Dict[int, int],
    label_cache: "OrderedDict[int, np.ndarray]",
) -> Optional[o3d.geometry.PointCloud]:
    """
    给全局点云上色：对每个3D点统计所有帧的类别投票，取出现次数最多的类别
    """
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
        label_map = load_label_map_for_pose(pose_timestamp, img_path, color_to_class, label_cache)
        if label_map is None:
            continue

        points_3d = transform_global_points_to_camera(
            points, pose_position, pose_quaternion, ext_pos, ext_rot_mat
        )
        points_2d, valid_indices, depths = project_points_to_image(points_3d, camera_matrix, dist_coeffs, label_map.shape)
        if len(points_2d) == 0:
            continue

        # Z-buffer 遮挡过滤：只保留每个像素上最近的点，防止背景点偷到前景类别
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
    voted_points = int(np.sum(has_votes))
    if voted_points == 0:
        print("  警告: 没有可用类别观测")
        return None

    winner_labels = np.argmax(vote_counts, axis=1)
    all_colors = np.zeros((num_points, 3), dtype=np.float32)
    all_colors[has_votes] = palette_rgb[winner_labels[has_votes]] / 255.0

    colored_pcd = make_colored_point_cloud(points, all_colors)

    avg_votes = float(observation_counts[has_votes].mean()) if voted_points > 0 else 0.0
    print(f"  全帧投票完成: 有效帧 {used_frames}/{total_frames}，赋色点 {voted_points}/{num_points}，平均票数 {avg_votes:.1f}")

    return colored_pcd


def main(global_pcd_path: str, img_path: str) -> None:
    """
    主函数：加载全局融合点云，遍历所有帧投票上色，保存结果
    """
    # 加载相机参数与类别色盘
    try:
        ext_params, inter_params = load_camera_params()
        palette_rgb, color_to_class = load_palette_and_lookup()
    except Exception as e:
        print(f"错误: 加载必要配置失败: {e}")
        return

    # 加载pose
    sorted_pose_files, _ = load_pose_files(img_path)
    pose_records = build_pose_records(sorted_pose_files)
    if not pose_records:
        print("错误: 无有效pose文件，退出")
        return

    print(f"全局点云: {global_pcd_path}")
    print(f"pose帧数: {len(pose_records)}")
    print(f"最少票数: {Config.MIN_VOTES_PER_POINT}，类别数: {len(palette_rgb)}")

    output_root = resolve_output_root(global_pcd_path)
    res_dir = os.path.join(output_root, Config.RES_SUBDIR)
    os.makedirs(res_dir, exist_ok=True)
    print(f"输出目录: res={res_dir}")

    label_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    colored_pcd = color_point_cloud_with_majority_vote(
        pcd_file=global_pcd_path,
        pose_records=pose_records,
        img_path=img_path,
        ext_params=ext_params,
        inter_params=inter_params,
        palette_rgb=palette_rgb,
        color_to_class=color_to_class,
        label_cache=label_cache,
    )
    if colored_pcd is None:
        print("错误: 上色失败")
        return

    pcd_stem = os.path.splitext(os.path.basename(global_pcd_path))[0]
    output_file = os.path.join(res_dir, f"{pcd_stem}_color.pcd")
    o3d.io.write_point_cloud(output_file, colored_pcd)
    print(f"\n成功保存彩色全局点云: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        global_pcd_path = sys.argv[1]
        img_path = sys.argv[2]
    else:
        global_pcd_path = Config.GLOBAL_PCD_PATH
        img_path = Config.DEFAULT_IMG_PATH

    main(global_pcd_path, img_path)
