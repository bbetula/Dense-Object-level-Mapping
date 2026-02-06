import os
import sys
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from typing import Tuple, List, Dict, Optional, Union
from multiview_fusion.config import Config

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


def project_points_to_image_nearest(points_3d: np.ndarray, 
                           camera_matrix: np.ndarray, 
                           dist_coeffs: np.ndarray, 
                           image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将3D点云投影到2D图像平面，并筛选每个2D像素对应的最近3D点（解决遮挡问题）
    核心逻辑：按2D像素坐标分组，保留每组内3D点云z值最小的点（相机坐标系下z越小表示距离相机越近）

    Args:
        points_3d: 3D点云 (N, 3)，相机坐标系下的坐标（x,y,z），z轴指向相机前方
        camera_matrix: 相机内参矩阵 (3, 3)
        dist_coeffs: 畸变参数 (5,)
        image_shape: 图像形状 (h, w)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            1. 去重后的2D投影点 (K, 2)，每个像素仅保留一个点
            2. 对应最近3D点的原始索引 (K,)，指向输入points_3d的下标
            3. 对应最近3D点的z值 (K,)，可用于后续深度相关处理
    """
    # 步骤1：筛选相机前方的有效点（z>0，排除相机后方的点）
    valid_indices = points_3d[:, 2] > 0
    valid_points_3d = points_3d[valid_indices]
    valid_original_indices = np.where(valid_indices)[0]  # 记录有效点的原始索引

    if len(valid_points_3d) == 0:
        return np.array([]), np.array([]), np.array([])

    # 步骤2：3D点投影到2D图像平面
    points_2d, _ = cv2.projectPoints(
        valid_points_3d, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
    )
    points_2d = points_2d.reshape(-1, 2)  # 重塑为(N_valid, 2)

    # 步骤3：过滤图像范围内的点
    h, w = image_shape[:2]
    in_image_mask = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    )
    # 提取图像内的2D点、3D点及对应原始索引
    in_image_points_2d = points_2d[in_image_mask]
    in_image_points_3d = valid_points_3d[in_image_mask]
    in_image_original_indices = valid_original_indices[in_image_mask]

    if len(in_image_points_2d) == 0:
        return np.array([]), np.array([]), np.array([])

    # 步骤4：关键！按2D像素分组，筛选每个像素最近的3D点（z最小=距离相机最近）
    # 4.1 将2D浮点物理坐标转为整数像素索引（先+0.5平移基准，再截断小数，符合像素覆盖范围）
    pixel_coords = (in_image_points_2d + 0.5).astype(np.int32)  # (M, 2)，格式为(u, v)（像素索引）

    # 4.2 构造唯一的像素标识（将(x,y)转为单个整数，方便分组，避免多维分组的繁琐）
    # 公式：pixel_id = y * w + x，确保每个像素对应唯一id
    pixel_ids = pixel_coords[:, 1] * w + pixel_coords[:, 0]

    # 4.3 按pixel_ids分组，找到每组内z值最小的点（最近点）
    # 先获取唯一的pixel_id，及对应的分组索引
    unique_pixel_ids, group_inverse = np.unique(pixel_ids, return_inverse=True)

    # 初始化数组，存储每组的最小z值索引
    min_z_indices_in_group = np.zeros(len(unique_pixel_ids), dtype=np.int32)
    min_z_values = np.full(len(unique_pixel_ids), np.inf)

    # 遍历每个点，更新对应分组的最小z值索引
    for idx in range(len(in_image_points_3d)):
        group_id = group_inverse[idx]  # 当前点所属的分组id
        current_z = in_image_points_3d[idx, 2]  # 当前点的z值

        # 若当前z值更小，更新该分组的最小z值及对应索引
        if current_z < min_z_values[group_id]:
            min_z_values[group_id] = current_z
            min_z_indices_in_group[group_id] = idx

    # 4.4 提取每个分组的最近点数据
    final_points_2d = in_image_points_2d[min_z_indices_in_group]  # 最终2D像素点
    final_original_indices = in_image_original_indices[min_z_indices_in_group]  # 原始3D点索引
    final_z_values = min_z_values  # 最终点的z值

    return final_points_2d, final_original_indices, final_z_values


def project_points_to_image_near(points_3d: np.ndarray, 
                           camera_matrix: np.ndarray, 
                           dist_coeffs: np.ndarray, 
                           image_shape: Tuple[int, int],
                           z_filter_threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将3D点云投影到2D图像平面，并对同一像素内的点按z值筛选，保留z最小值到z_min+阈值范围内的所有点
    核心逻辑：1. 按2D像素分组；2. 批量计算每个分组的z最小值；3. 向量化筛选范围内的点（无循环，大幅提速）
    优化点：移除逐分组遍历，改用NumPy向量化运算，解决原代码计算耗时过长的问题

    Args:
        points_3d: 3D点云 (N, 3)，相机坐标系下的坐标（x,y,z），z轴指向相机前方
        camera_matrix: 相机内参矩阵 (3, 3)
        dist_coeffs: 畸变参数 (5,)
        image_shape: 图像形状 (h, w)
        z_filter_threshold: z轴深度剔除阈值（单位：与点云一致，如米），保留z∈[z_min, z_min+该值]的点，默认0.1

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            1. 筛选后的2D投影点 (K, 2)，同一像素保留z最小值附近指定范围内的所有点
            2. 对应点的原始索引 (K,)，指向输入points_3d的下标
            3. 对应点的z值 (K,)，便于后续深度相关处理
    """
    # 步骤1：筛选相机前方的有效点（z>0，排除相机后方的点）
    valid_indices = points_3d[:, 2] > 0
    valid_points_3d = points_3d[valid_indices]
    valid_original_indices = np.where(valid_indices)[0]  # 记录有效点的原始索引

    if len(valid_points_3d) == 0:
        return np.array([]), np.array([]), np.array([])

    # 步骤2：3D点投影到2D图像平面
    points_2d, _ = cv2.projectPoints(
        valid_points_3d, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
    )
    points_2d = points_2d.reshape(-1, 2)  # 重塑为(N_valid, 2)

    # 步骤3：过滤图像范围内的点
    h, w = image_shape[:2]
    in_image_mask = (
        (points_2d[:, 0] >= -0.5) & (points_2d[:, 0] < (w - 0.5)) &  # x∈[-0.5, w-0.5)
        (points_2d[:, 1] >= -0.5) & (points_2d[:, 1] < (h - 0.5))    # y∈[-0.5, h-0.5)
    )
    # 提取图像内的2D点、3D点及对应原始索引
    in_image_points_2d = points_2d[in_image_mask]
    in_image_original_indices = valid_original_indices[in_image_mask]
    in_image_z_values = valid_points_3d[in_image_mask, 2]  # 提取图像内点的z值

    if len(in_image_points_2d) == 0:
        return np.array([]), np.array([]), np.array([])

    # 步骤4：按2D像素分组（先转整数像素坐标，与原逻辑一致）
    # 4.1 将2D浮点物理坐标转为整数像素索引
    pixel_coords = (in_image_points_2d + 0.5).astype(np.int32)  # (M, 2)，格式为(u, v)
    # 4.2 构造唯一的像素标识（将(x,y)转为单个整数，方便分组）
    pixel_ids = pixel_coords[:, 1] * w + pixel_coords[:, 0]
    # 4.3 获取唯一像素ID及对应分组（为向量化运算做准备）
    unique_pixel_ids, group_inverse, group_counts = np.unique(
        pixel_ids, return_inverse=True, return_counts=True
    )

    # 步骤5：向量化计算筛选
    # 5.1 批量计算每个像素分组的z最小值（替代逐分组遍历找z_min）
    # 创建与每个点对应的「所属分组z最小值」数组
    group_z_min = np.zeros_like(in_image_z_values)
    for i, unique_id in enumerate(unique_pixel_ids):
        # 找到当前分组的所有点的掩码
        group_mask = (group_inverse == i)
        # 获取该分组的z最小值，并赋值给分组内所有点对应的位置
        z_min = np.min(in_image_z_values[group_mask])
        group_z_min[group_mask] = z_min

    # 5.2 向量化筛选：保留z∈[z_min, z_min + z_filter_threshold]的点（无循环）
    z_upper = group_z_min + z_filter_threshold
    valid_z_mask = (in_image_z_values >= group_z_min) & (in_image_z_values <= z_upper)

    # 5.3 提取筛选后的结果（批量操作，替代逐分组添加列表）
    final_points_2d = in_image_points_2d[valid_z_mask]
    final_original_indices = in_image_original_indices[valid_z_mask]
    final_z_values = in_image_z_values[valid_z_mask]

    return final_points_2d, final_original_indices, final_z_values