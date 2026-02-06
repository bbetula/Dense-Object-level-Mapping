import glob
import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from multiview_fusion.config import Config


def load_img_files(img_path: str) -> Tuple[List[str], List[int]]:
    """加载并排序所有img文件"""
    img_files = glob.glob(os.path.join(img_path, "*.png"))
    img_timestamps = [int(os.path.basename(f).split('.')[0]) for f in img_files]
    
    # 按时间戳排序
    sorted_indices = np.argsort(img_timestamps)
    sorted_img_files = [img_files[i] for i in sorted_indices]
    sorted_img_timestamps = [img_timestamps[i] for i in sorted_indices]
    
    return sorted_img_files, sorted_img_timestamps


def load_pose_files(img_path):
    """加载并排序所有pose文件"""
    pose_files = glob.glob(os.path.join(img_path, "*.txt"))
    
    # 存储时间戳和文件路径
    pose_data = []
    
    for pose_file in pose_files:
        try:
            # 读取第一行获取时间戳
            with open(pose_file, 'r') as f:
                first_line = f.readline().strip()
                tokens = first_line.split()
                if tokens:
                    # 先转换为float处理科学计数法，再转为int
                    timestamp = int(float(tokens[0]))
                    pose_data.append((timestamp, pose_file))
        except Exception as e:
            print(f"警告：无法读取时间戳从 {pose_file}: {e}")
    
    # 按时间戳排序
    pose_data.sort(key=lambda x: x[0])
    
    sorted_pose_timestamps = [data[0] for data in pose_data]
    sorted_pose_files = [data[1] for data in pose_data]
    
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


def load_point_cloud(pcd_path: str) -> np.ndarray:
	"""
	加载单个PCD文件，返回点云 (N,3)
	"""
	pcd = o3d.io.read_point_cloud(pcd_path)
	points = np.asarray(pcd.points)
	return points