import os
import sys
import glob
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from typing import Tuple, List, Dict, Optional, Union


# ======================== 配置常量（集中管理，便于修改）========================
class Config:
    """配置类：集中管理所有路径和参数"""
    DEFAULT_PCD_PATHS = {
        "indoor": "/home/czj/datasets/fastlivo_output_indoor_107/lidar",
        "outdoor": "/home/czj/datasets/fastlivo_output_outdoor_1s/lidar",
    }
    DEFAULT_IMG_PATHS = {
        "indoor": "/home/czj/datasets/fastlivo_output_indoor_107/image",
        "outdoor": "/home/czj/datasets/fastlivo_output_outdoor_1s/image",
    }
    DEFAULT_PCD_PATH = DEFAULT_PCD_PATHS["indoor"]
    DEFAULT_IMG_PATH = DEFAULT_IMG_PATHS["indoor"]
    SEG_SUBDIR_NAME = "res_dinov3_whole"  # 分割图像子目录名称
    
    # YAML配置路径
    DEFAULT_YAML_PATH = "/home/czj/program/r3live_semantics_ws"
    EXT_YAML_PATH = os.path.join(DEFAULT_YAML_PATH, "avia.yaml")
    INTER_YAML_PATH = os.path.join(DEFAULT_YAML_PATH, "camera_pinhole.yaml")
    
    # 输出配置
    CONFIG_TXT_PATH = "/home/czj/program/r3live_semantics_ws/src/cityscape_segmantation-main/fused_reconstruction/config.txt"
    VIS_SUBDIR = "visualization"
    COLORED_SUBDIR = "colored"
    SEG_SUBDIR = "seg"
    
    # 功能开关
    IF_USE_SEG = True  # 是否使用分割图像
    CONVERT_TO_BGR = True  # 是否将点云颜色转为BGR
    VISUALIZE_PROJECTION = True  # 是否可视化投影结果
    
    # 时间阈值（700ms = 700000000纳秒）
    MAX_TIME_DIFF_NS = 700 * 1000 * 1000
    
    # 插值方式
    INTERP_METHOD = {
        "seg": "round",
        "normal": "bilinear"
    }
    
    # 可视化参数
    PROJECTION_POINT_COLOR = (0, 255, 0)  # 绿色
    PROJECTION_POINT_SIZE = 2
    PROJECTION_POINT_THICKNESS = -1  # 填充圆


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


# ======================== 文件加载函数 ========================
def load_pcd_files(pcd_path: str) -> Tuple[List[str], List[int]]:
    """
    加载并排序所有PCD文件（按时间戳）
    
    Args:
        pcd_path: PCD文件目录
    
    Returns:
        Tuple[List[str], List[int]]: 排序后的PCD文件路径, 对应的时间戳
    """
    pcd_files = glob.glob(os.path.join(pcd_path, "*.pcd"))
    if not pcd_files:
        print(f"警告: 在 {pcd_path} 中未找到PCD文件")
        return [], []
    
    # 提取时间戳并排序
    pcd_timestamps = []
    valid_pcd_files = []
    for f in pcd_files:
        try:
            timestamp = int(os.path.basename(f).split('.')[0])
            pcd_timestamps.append(timestamp)
            valid_pcd_files.append(f)
        except ValueError:
            print(f"警告: 无法解析PCD文件时间戳 {f}，跳过")
            continue
    
    # 按时间戳排序
    sorted_indices = np.argsort(pcd_timestamps)
    sorted_pcd_files = [valid_pcd_files[i] for i in sorted_indices]
    sorted_pcd_timestamps = [pcd_timestamps[i] for i in sorted_indices]
    
    return sorted_pcd_files, sorted_pcd_timestamps


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


def find_closest_pose(pcd_timestamp: int, 
                      sorted_pose_files: List[str], 
                      sorted_pose_timestamps: List[int]) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    查找与PCD时间戳最接近的pose
    
    Args:
        pcd_timestamp: PCD文件时间戳
        sorted_pose_files: 排序后的pose文件路径
        sorted_pose_timestamps: 排序后的pose时间戳
    
    Returns:
        Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray], float]: 
            pose时间戳(None表示无匹配), 位置, 四元数, 时间差(纳秒)
    """
    if not sorted_pose_timestamps:
        return None, None, None, float('inf')
    
    # 计算最小时间差
    pose_timestamps_np = np.array(sorted_pose_timestamps)
    closest_idx = np.argmin(np.abs(pose_timestamps_np - pcd_timestamp))
    time_diff = abs(pose_timestamps_np[closest_idx] - pcd_timestamp)
    
    # 超过阈值则返回None
    if time_diff > Config.MAX_TIME_DIFF_NS:
        return None, None, None, time_diff
    
    # 加载pose数据
    try:
        pose_timestamp, position, quaternion = load_pose_from_file(sorted_pose_files[closest_idx])
        # 打印匹配信息
        print(f"匹配PCD时间戳: {pcd_timestamp} 与pose时间戳: {pose_timestamp}, 差值: {time_diff}纳秒")
        print("----" if time_diff == 0 else "++++")
        return pose_timestamp, position, quaternion, time_diff
    except Exception as e:
        print(f"警告: 加载pose文件 {sorted_pose_files[closest_idx]} 失败: {e}")
        return None, None, None, time_diff


def find_corresponding_img(pose_timestamp: int, img_path: str) -> Optional[str]:
    """
    查找与pose时间戳对应的图像文件
    
    Args:
        pose_timestamp: pose时间戳
        img_path: 图像目录
    
    Returns:
        Optional[str]: 图像文件路径（None表示未找到）
    """
    img_extensions = ['.png', '.jpg', '.jpeg']
    
    # 优先查找分割图像
    if Config.IF_USE_SEG:
        seg_file = os.path.join(img_path, Config.SEG_SUBDIR_NAME, f"{pose_timestamp}_mask.jpg")
        if os.path.exists(seg_file):
            print(f"找到分割图像: {seg_file}")
            return seg_file
    
    # 查找原始图像（按时间戳）
    for ext in img_extensions:
        img_file = os.path.join(img_path, f"{pose_timestamp}{ext}")
        if os.path.exists(img_file):
            return img_file
    
    # 按pose文件名查找
    pose_files = glob.glob(os.path.join(img_path, "*.txt"))
    for pose_file in pose_files:
        try:
            base_name = os.path.splitext(os.path.basename(pose_file))[0]
            for ext in img_extensions:
                img_file = os.path.join(img_path, f"{base_name}{ext}")
                if os.path.exists(img_file):
                    return img_file
        except Exception:
            continue
    
    return None


# ======================== 投影和插值函数 ========================
def project_points_to_image(points_3d: np.ndarray, 
                           camera_matrix: np.ndarray, 
                           dist_coeffs: np.ndarray, 
                           image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将3D点云投影到2D图像平面
    
    Args:
        points_3d: 3D点云 (N, 3)
        camera_matrix: 相机内参矩阵 (3, 3)
        dist_coeffs: 畸变参数 (5,)
        image_shape: 图像形状 (h, w)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 2D投影点 (M, 2), 有效点索引 (M,)
    """
    # 只保留相机前方的点（z>0）
    valid_indices = points_3d[:, 2] > 0
    valid_points = points_3d[valid_indices]
    
    if len(valid_points) == 0:
        return np.array([]), np.array([])
    
    # 投影到图像平面
    points_2d, _ = cv2.projectPoints(
        valid_points, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
    )
    points_2d = points_2d.reshape(-1, 2)
    
    # 过滤图像范围内的点
    h, w = image_shape[:2]
    in_image = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    )
    
    return points_2d[in_image], np.where(valid_indices)[0][in_image]


def bilinear_interpolation(image: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """
    双线性插值获取图像颜色
    
    Args:
        image: 输入图像 (h, w, 3)
        points_2d: 2D投影点 (M, 2)
    
    Returns:
        np.ndarray: 插值后的颜色 (M, 3)
    """
    h, w = image.shape[:2]
    colors = np.zeros((points_2d.shape[0], 3), dtype=np.uint8)
    
    for i, (x, y) in enumerate(points_2d):
        if 0 <= x < w-1 and 0 <= y < h-1:
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + 1, y0 + 1
            
            # 插值权重
            wx = x - x0
            wy = y - y0
            
            # 四个邻近像素颜色
            c00 = image[y0, x0]
            c01 = image[y0, x1]
            c10 = image[y1, x0]
            c11 = image[y1, x1]
            
            # 双线性插值
            color = (1-wx)*(1-wy)*c00 + wx*(1-wy)*c01 + (1-wx)*wy*c10 + wx*wy*c11
            colors[i] = color.astype(np.uint8)
    
    return colors


def round_interpolation(image: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """
    取整插值获取图像颜色（适用于分割图像）
    
    Args:
        image: 输入图像 (h, w, 3)
        points_2d: 2D投影点 (M, 2)
    
    Returns:
        np.ndarray: 插值后的颜色 (M, 3)
    """
    h, w = image.shape[:2]
    colors = np.zeros((points_2d.shape[0], 3), dtype=np.uint8)
    
    for i, (x, y) in enumerate(points_2d):
        x_round = round(x)
        y_round = round(y)
        if 0 <= x_round < w and 0 <= y_round < h:
            colors[i] = image[y_round, x_round]
    
    return colors


def get_image_colors(image: np.ndarray, points_2d: np.ndarray, use_seg: bool) -> np.ndarray:
    """
    统一的颜色插值接口
    
    Args:
        image: 输入图像 (h, w, 3)
        points_2d: 2D投影点 (M, 2)
        use_seg: 是否使用分割插值
    
    Returns:
        np.ndarray: 插值后的颜色 (M, 3)
    """
    if use_seg:
        return round_interpolation(image, points_2d)
    else:
        return bilinear_interpolation(image, points_2d)


# ======================== 点云上色和可视化 ========================
def visualize_projection(image: np.ndarray, 
                         points_2d: np.ndarray, 
                         pcd_timestamp: int, 
                         pcd_dir: str) -> None:
    """
    可视化点云投影结果并保存
    
    Args:
        image: 输入图像 (h, w, 3)
        points_2d: 2D投影点 (M, 2)
        pcd_timestamp: PCD时间戳
        pcd_dir: PCD文件目录
    """
    # 创建可视化目录
    vis_dir = os.path.join(pcd_dir, Config.VIS_SUBDIR)
    os.makedirs(vis_dir, exist_ok=True)
    
    # 绘制投影点
    vis_image = image.copy()
    for pt in points_2d:
        cv2.circle(
            vis_image, 
            (int(pt[0]), int(pt[1])),
            Config.PROJECTION_POINT_SIZE,
            Config.PROJECTION_POINT_COLOR,
            Config.PROJECTION_POINT_THICKNESS
        )
    
    # 保存基础投影图
    vis_file = os.path.join(vis_dir, f"{pcd_timestamp}_projection.png")
    cv2.imwrite(vis_file, vis_image)
    print(f"  投影可视化已保存到: {vis_file}")
    
    # 彩色投影和密度图（可选，默认关闭）
    # if False:
    #     _save_colored_projection(image, points_2d, pcd_timestamp, vis_dir)
    #     _save_density_projection(image, points_2d, pcd_timestamp, vis_dir)


def _save_colored_projection(image: np.ndarray, 
                             points_2d: np.ndarray, 
                             timestamp: int, 
                             vis_dir: str) -> None:
    """保存彩色投影图（内部函数）"""
    colored_vis_image = image.copy()
    colors = get_image_colors(image, points_2d, Config.IF_USE_SEG)
    
    for i, pt in enumerate(points_2d):
        bgr_color = (int(colors[i][2]), int(colors[i][1]), int(colors[i][0]))
        cv2.circle(
            colored_vis_image,
            (int(pt[0]), int(pt[1])),
            Config.PROJECTION_POINT_SIZE,
            bgr_color,
            Config.PROJECTION_POINT_THICKNESS
        )
    
    colored_vis_file = os.path.join(vis_dir, f"{timestamp}_colored_projection.png")
    cv2.imwrite(colored_vis_file, colored_vis_image)
    print(f"  彩色投影可视化已保存到: {colored_vis_file}")


def _save_density_projection(image: np.ndarray, 
                             points_2d: np.ndarray, 
                             timestamp: int, 
                             vis_dir: str) -> None:
    """保存密度投影图（内部函数）"""
    density_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for pt in points_2d:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < density_map.shape[1] and 0 <= y < density_map.shape[0]:
            density_map[y, x] += 1
    
    if density_map.max() > 0:
        density_map = (density_map / density_map.max() * 255).astype(np.uint8)
        density_heatmap = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
        density_overlay = cv2.addWeighted(image, 0.7, density_heatmap, 0.3, 0)
        
        density_vis_file = os.path.join(vis_dir, f"{timestamp}_density_projection.png")
        cv2.imwrite(density_vis_file, density_overlay)
        print(f"  点密度投影可视化已保存到: {density_vis_file}")


def color_point_cloud(pcd_file: str, 
                      img_file: str, 
                      pose_position: np.ndarray, 
                      pose_quaternion: np.ndarray, 
                      ext_params: Dict, 
                      inter_params: Dict) -> Optional[o3d.geometry.PointCloud]:
    """
    给点云上色
    
    Args:
        pcd_file: PCD文件路径
        img_file: 图像文件路径
        pose_position: pose位置 (3,)
        pose_quaternion: pose四元数 (4,)
        ext_params: 外参字典
        inter_params: 内参字典
    
    Returns:
        Optional[o3d.geometry.PointCloud]: 上色后的点云（None表示失败）
    """
    # 加载点云
    try:
        cloud = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(cloud.points)
    except Exception as e:
        print(f"错误: 加载点云文件 {pcd_file} 失败: {e}")
        return None
    
    # 构建相机内参矩阵和畸变参数
    camera_matrix = np.array([
        [inter_params['fx'], 0, inter_params['cx']],
        [0, inter_params['fy'], inter_params['cy']],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([
        inter_params['k1'], inter_params['k2'],
        inter_params['p1'], inter_params['p2'],
        inter_params['k3']
    ])
    
    # 解析外参
    ext_pos = np.array([ext_params['ext_pos_x'], ext_params['ext_pos_y'], ext_params['ext_pos_z']])
    ext_quat = np.array([ext_params['ext_q_x'], ext_params['ext_q_y'], ext_params['ext_q_z'], ext_params['ext_q_w']])
    
    # 计算外参旋转矩阵及其逆
    ext_rot_mat = quaternion_to_rotation_matrix(ext_quat)
    ext_rot_mat_inv = ext_rot_mat.T
    ext_pos_inv = -np.dot(ext_rot_mat_inv, ext_pos)
    
    # 计算pose旋转矩阵
    pose_rot_mat = quaternion_to_rotation_matrix(pose_quaternion)
    
    # 最终变换矩阵（逆变换）
    final_rot_mat = np.dot(pose_rot_mat, ext_rot_mat)
    final_pos = pose_position + np.dot(pose_rot_mat, ext_pos)
    final_rot_mat_inv = final_rot_mat.T
    final_pos_inv = -np.dot(final_rot_mat_inv, final_pos)
    
    # 转换点云到相机坐标系
    points_3d = transform_point_cloud(points, final_pos_inv, final_rot_mat_inv)
    
    # 加载图像
    image = cv2.imread(img_file)
    if image is None:
        print(f"错误: 加载图像 {img_file} 失败")
        return None
    
    # 投影点云到图像
    points_2d, valid_indices = project_points_to_image(points_3d, camera_matrix, dist_coeffs, image.shape)
    if len(points_2d) == 0:
        print(f"警告: 没有点能投影到图像 {img_file} 上")
        return None
    
    # 插值获取颜色
    colors = get_image_colors(image, points_2d, Config.IF_USE_SEG)
    
    # 构建彩色点云
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    
    # 初始化颜色为黑色，给有效点赋值
    all_colors = np.zeros((len(points), 3), dtype=np.float32)
    all_colors[valid_indices] = colors / 255.0  # 归一化到[0,1]
    colored_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # 可视化投影结果
    if Config.VISUALIZE_PROJECTION:
        visualize_projection(image, points_2d, int(os.path.basename(pcd_file).split('.')[0]), os.path.dirname(pcd_file))
    
    # 转换为BGR格式
    if Config.CONVERT_TO_BGR:
        colors_np = np.asarray(colored_pcd.colors)
        bgr_colors = colors_np.copy()
        bgr_colors[:, 0], bgr_colors[:, 2] = colors_np[:, 2], colors_np[:, 0]  # RGB <-> BGR
        bgr_colors = np.clip(bgr_colors, 0.0, 1.0)
        colored_pcd.colors = o3d.utility.Vector3dVector(bgr_colors)
    
    return colored_pcd


def main(pcd_path: str, img_path: str) -> None:
    """
    主函数：处理所有PCD文件，给点云上色并保存
    
    Args:
        pcd_path: PCD文件目录
        img_path: 图像/pose文件目录
    """
    # 加载相机参数
    try:
        ext_params, inter_params = load_camera_params()
    except Exception as e:
        print(f"错误: 加载相机参数失败: {e}")
        return
    
    # 加载并排序文件
    sorted_pcd_files, sorted_pcd_timestamps = load_pcd_files(pcd_path)
    sorted_pose_files, sorted_pose_timestamps = load_pose_files(img_path)
    
    # 检查文件数量
    if not sorted_pcd_files:
        print("错误: 无有效PCD文件，退出")
        return
    if not sorted_pose_files:
        print("错误: 无有效pose文件，退出")
        return
    
    print(f"找到 {len(sorted_pcd_files)} 个PCD文件和 {len(sorted_pose_files)} 个pose文件")
    
    # 打印第一个pose文件预览
    try:
        with open(sorted_pose_files[0], 'r', encoding='utf-8') as f:
            print(f"第一个pose文件预览 ({sorted_pose_files[0]}): {f.readline().strip()}")
    except Exception as e:
        print(f"警告: 无法读取第一个pose文件: {e}")
    
    # 确定输出目录
    output_dir = os.path.join(pcd_path, Config.SEG_SUBDIR if Config.IF_USE_SEG else Config.COLORED_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个PCD文件
    for i, (pcd_file, pcd_timestamp) in enumerate(zip(sorted_pcd_files, sorted_pcd_timestamps)):
        print(f"\n处理点云 {i+1}/{len(sorted_pcd_files)}: {pcd_file}")
        
        # 查找匹配的pose
        pose_timestamp, pose_position, pose_quaternion, time_diff = find_closest_pose(
            pcd_timestamp, sorted_pose_files, sorted_pose_timestamps
        )
        
        if pose_timestamp is None:
            print(f"  跳过: 与最近pose的时间差过大 ({time_diff/1e9:.2f}秒 > {Config.MAX_TIME_DIFF_NS/1e9:.2f}秒)")
            continue
        
        # 查找匹配的图像
        img_file = find_corresponding_img(pose_timestamp, img_path)
        if img_file is None:
            print(f"  跳过: 未找到与pose {pose_timestamp} 对应的图像")
            continue
        
        print(f"  使用图像: {img_file} (时间差: {time_diff/1e9:.2f}秒)")
        
        # 给点云上色并保存
        try:
            colored_pcd = color_point_cloud(pcd_file, img_file, pose_position, pose_quaternion, ext_params, inter_params)
            if colored_pcd is None:
                continue
            
            # 保存彩色点云
            output_file = os.path.join(output_dir, f"{pcd_timestamp}_color.pcd")
            o3d.io.write_point_cloud(output_file, colored_pcd)
            print(f"  成功保存彩色点云到: {output_file}")
        
        except Exception as e:
            print(f"  处理点云失败: {e}")


if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) >= 3:
        pcd_path = sys.argv[1]
        img_path = sys.argv[2]
    else:
        pcd_path = Config.DEFAULT_PCD_PATH
        img_path = Config.DEFAULT_IMG_PATH
    
    # 运行主函数
    main(pcd_path, img_path)