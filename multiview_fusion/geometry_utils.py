import os
import sys
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, List, Dict, Optional, Union
from multiview_fusion.config import Config

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
    
    return ext_params, inter_params


def load_camera_inter_params():
    """
    加载并解析相机内参，构建相机内参矩阵和畸变参数数组

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - camera_matrix: 相机内参矩阵 (3, 3)，格式[[fx,0,cx],[0,fy,cy],[0,0,1]]
            - dist_coeffs: 相机畸变参数数组 (5,)，格式[k1, k2, p1, p2, k3]
    """
    _, inter_params = load_camera_params()

    fx = inter_params['fx']
    fy = inter_params['fy']
    cx = inter_params['cx']
    cy = inter_params['cy']
    k1 = inter_params.get('k1', 0.0)
    k2 = inter_params.get('k2', 0.0)
    p1 = inter_params.get('p1', 0.0)
    p2 = inter_params.get('p2', 0.0)
    k3 = inter_params.get('k3', 0.0)
    
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    return camera_matrix, dist_coeffs


def lidar2camera_point_cloud_transform(points, pose_position, pose_quaternion, ext_params):
    """
    将激光雷达坐标系下的点云，结合雷达位姿，转换到相机坐标系
    
    Args:
        points: 激光雷达坐标系下的原始点云，形状为 (N, 3)，N为点云的点数量
        pose_position: 雷达在世界坐标系下的平移位姿向量，形状为 (3,)，对应[x, y, z]
        pose_quaternion: 雷达在世界坐标系下的姿态四元数，形状为 (4,)，对应[x, y, z, w]
        ext_params: 雷达-相机外参字典，包含以下关键字段：
    
    Returns:
        np.ndarray: 相机坐标系下的点云，形状为 (N, 3)，与输入原始点云的点数量一致
    """

    # 外参
    ext_pos = np.array([
        ext_params['ext_pos_x'],
        ext_params['ext_pos_y'],
        ext_params['ext_pos_z']
    ])
    ext_quat = np.array([
        ext_params['ext_q_x'],
        ext_params['ext_q_y'],
        ext_params['ext_q_z'],
        ext_params['ext_q_w']
    ])
    
    # 计算外参的旋转矩阵
    ext_rot_mat = quaternion_to_rotation_matrix(ext_quat)
    
    # 外参取逆
    ext_rot_mat_inv = ext_rot_mat.T  # 旋转矩阵的逆等于其转置
    ext_pos_inv = -np.dot(ext_rot_mat_inv, ext_pos)  # 平移向量的逆
    
    # 计算pose的旋转矩阵及其逆
    pose_rot_mat = quaternion_to_rotation_matrix(pose_quaternion)
    
    # 计算最终的旋转矩阵和平移向量 (使用pose的逆和外参的逆)
    final_rot_mat = np.dot(pose_rot_mat, ext_rot_mat)
    final_pos = pose_position + np.dot(pose_rot_mat, ext_pos)
    
    # 对final_rot_mat和final_pos取逆
    final_rot_mat_inv = final_rot_mat.T  # 旋转矩阵的逆是其转置
    final_pos_inv = -np.dot(final_rot_mat_inv, final_pos)

    # 转换点云到相机坐标系
    points_3d = transform_point_cloud(points, final_pos_inv, final_rot_mat_inv)

    return points_3d