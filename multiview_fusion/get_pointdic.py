import os
from typing import Dict, List, Tuple
import cv2
import numpy as np
import open3d as o3d

from multiview_fusion.config import Config
from multiview_fusion import point2image
from multiview_fusion import geometry_utils
from multiview_fusion import data_loader


def compute_point_dic(points: np.ndarray,
					  pose_files, pose_timestamps) -> Dict[int, List[Tuple[int, Tuple[float, float]]]]:
	"""
	对于给定点云和图像位姿目录，遍历所有相机位姿文件并将点云投影到每帧图片上，
	返回字典: point_id -> [(timestamp, (x,y)), ...]

	参数:
		points: 点云 (N, 3)
		pose_files: 位姿文件列表
		pose_timestamps: 位姿时间戳列表
	返回:
		Dict[int, List[ (timestamp, (x,y)) ]]
		key: 3D 点云的点 ID（point_id），对应点云转 NumPy 数组后的索引（下标）
		     范围是 0 ~ (点云总数-1)，每个键唯一对应一个 3D 点
		value: [(时间戳1,(x1,y1)), (时间戳2,(x2,y2)), ...]
		       (x1,y1)表示该3D点在时间戳1图像上的2D像素坐标(x, y)
	"""
	image_shape = Config.IMAGE_SHAPE

	# 加载相机内参/外参（主要取内参用于投影、畸变）
	ext_params, _ = geometry_utils.load_camera_params()
	camera_matrix, dist_coeffs = geometry_utils.load_camera_inter_params()

	point_dic: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {i: [] for i in range(points.shape[0])}

	# 遍历每个pose文件
	for pose_file in pose_files:

		# 解析当前pose
		pose_timestamp, pose_position, pose_quaternion = data_loader.load_pose_from_file(pose_file)

		# 将点云从世界坐标系转换到相机坐标系
		pts_cam = geometry_utils.lidar2camera_point_cloud_transform(points, pose_position, pose_quaternion, ext_params)

		# 使用 project_points_to_image_near 进行投影
		# z_filter_threshold = 0.1
		# pts_2d, orig_indices, _ = point2image.project_points_to_image_near(
		# 	pts_cam, camera_matrix, dist_coeffs, image_shape, z_filter_threshold
		# )

		# start_time = cv2.getTickCount()
		pts_2d, orig_indices, _ = point2image.project_points_to_image_nearest(
			pts_cam, camera_matrix, dist_coeffs, image_shape
		)
		# end_time = cv2.getTickCount()
		# time1 = (end_time - start_time) / cv2.getTickFrequency()
		# print(time1)

		if pts_2d.size == 0 or orig_indices.size == 0:
			continue

		# 记录每个投影点到字典中
		for p2d, idx in zip(pts_2d, orig_indices):
			x, y = float(p2d[0]), float(p2d[1])
			point_dic[int(idx)].append((int(pose_timestamp), (x, y)))

		print(f"已处理位姿文件: {os.path.basename(pose_file)}，投影点数: {pts_2d.shape[0]}")

	return point_dic


def compute_image_proj_dic(points: np.ndarray, selected_pose_files: List[str]) -> List[dict]:
	"""
	计算多张图片投影信息，返回每张图的投影条目列表。

	参数:
		points: (N,3) 点云
		selected_pose_files: 本 batch 的位姿文件路径列表（与图像一一对应）

	返回:
		projections: list of dict，每个pose_file对应一个dict
			每个 dict 包含字段：
			'time_idx': int 时间戳（从 pose 文件解析得到），
			'point_indices': np.ndarray M, 被投影点在点云中的索引，
			'coords': np.ndarray Mx2，对应像素坐标 (x,y),coordinates

	说明：此函数为 `compute_point_dic` 的批量按图返回版本，使用相同的相机参数与投影函数。
	"""

	image_shape = Config.IMAGE_SHAPE

	# 加载相机内参/外参（主要取内参用于投影、畸变）
	ext_params, _ = geometry_utils.load_camera_params()
	camera_matrix, dist_coeffs = geometry_utils.load_camera_inter_params()

	projections = []
	for pose_file in selected_pose_files:
		pose_timestamp, pose_position, pose_quaternion = data_loader.load_pose_from_file(pose_file)

		# 将点云从世界坐标系转换到相机坐标系
		pts_cam = geometry_utils.lidar2camera_point_cloud_transform(points, pose_position, pose_quaternion, ext_params)

		pts_2d, orig_indices, _ = point2image.project_points_to_image_nearest(
			pts_cam, camera_matrix, dist_coeffs, image_shape
		)

		if pts_2d.size == 0 or orig_indices.size == 0:
			projections.append({"time_idx": int(pose_timestamp), 
								"point_indices": np.array([], dtype=int), 
								"coords": np.zeros((0, 2), dtype=float)})
			continue

		projections.append({
			"time_idx": int(pose_timestamp),
			"point_indices": orig_indices.astype(np.int64),
			"coords": pts_2d.astype(np.float32),
		})

		print(f"compute_image_proj_dic: processed {os.path.basename(pose_file)}, projected {pts_2d.shape[0]} points")

	return projections
