import os
from typing import Dict, List, Tuple

import sys
REPO_DIR = "/home/czj/program/MVSSeg" # Please add here the path to your MVSSeg repository
sys.path.append(REPO_DIR)

import yaml
import numpy as np
import open3d as o3d

from multiview_fusion.config import Config
from multiview_fusion import point2image
from multiview_fusion import geometry_utils
from multiview_fusion import data_loader
from multiview_fusion.get_pointdic import compute_point_dic


def load_ade20k_palette(yaml_path: str) -> np.ndarray:
	data = geometry_utils.load_yaml_config(yaml_path)
	palette = data.get('palette', None)
	if palette is None:
		raise ValueError(f"在 {yaml_path} 中未找到 'palette' 字段")
	return np.array(palette, dtype=np.uint8)


def fuse_seg(output_pcd_path: str = None) -> str:
	"""
	主函数：根据多视角分割概率Numpy文件对点云进行语义融合并保存带颜色的点云。

	返回：输出的 pcd 文件路径
	"""
	pcd_path = Config.DEFAULT_PCD_PATH
	seg_dir = Config.DEFAULT_SEG_PATH
	img_path = Config.DEFAULT_IMG_PATH

	# 读取位姿文件列表（按时间戳排序）
	pose_files, pose_timestamps = data_loader.load_pose_files(img_path)

	# 读取点云
	points = data_loader.load_point_cloud(pcd_path)

	# 新建输出点云对象，保留原始点坐标
	out_pcd = o3d.geometry.PointCloud()
	out_pcd.points = o3d.utility.Vector3dVector(points)

	# 计算 point_dic
	point_dic = compute_point_dic(points, pose_files, pose_timestamps)
	print(f"完成计算 point_dic，包含 {len(point_dic)} 个点的投影信息")

	# 加载 ADE20K 调色板
	palette = load_ade20k_palette(os.path.join(os.path.dirname(__file__), 'ADE20k.yaml'))

	# 缓存已加载的 seg npy 文件
	seg_cache: Dict[int, np.ndarray] = {}

	colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
	point_classes = np.zeros((points.shape[0],), dtype=np.uint8)

	for pid in range(points.shape[0]):
		entries = point_dic.get(pid, [])
		print(f"处理点ID {pid}，共有 {len(entries)} 个投影条目")
		if not entries:
			colors[pid] = np.array([0, 0, 0], dtype=np.uint8)
			continue

		vecs = []
		for ts, (x, y) in entries:
			ts = int(ts)
			if ts not in seg_cache:
				seg_file = os.path.join(seg_dir, f"{ts}_pred_class.npy")
				seg_cache[ts] = np.load(seg_file)

			seg = seg_cache[ts]
			h, w = seg.shape[0], seg.shape[1]
			xi = int(round(x)); yi = int(round(y))

			if xi < 0 or xi >= w or yi < 0 or yi >= h:
				continue

			vec = seg[yi, xi, :]
			vecs.append(vec)

		if not vecs:
			colors[pid] = np.array([0, 0, 0], dtype=np.uint8)
			continue

		mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
		cls = int(np.argmax(mean_vec))
		cls = np.clip(cls, 0, len(palette) - 1)	
		point_classes[pid] = cls
		colors[pid] = palette[cls]

		print(f"完成计算点ID {pid}: 类别 {cls}, 颜色 {colors[pid].tolist()}")

	# 将颜色归一化到 [0,1]
	out_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

	# 输出路径
	base = os.path.basename(pcd_path)
	name = os.path.splitext(base)[0]
	out_dir = os.path.dirname(pcd_path)
	out_fname = f"{name}_seg.pcd"
	out_path = output_pcd_path or os.path.join(out_dir, out_fname)
	o3d.io.write_point_cloud(out_path, out_pcd)

	return out_path, point_classes


def fuse_seg_without_blackpoint(output_pcd_path: str = None) -> Tuple[str, np.ndarray]:
	"""
	主函数：根据多视角分割概率Numpy文件对点云进行语义融合并保存带颜色的点云（过滤所有黑色点不显示）。

	返回：输出的 pcd 文件路径、过滤后的点类别数组
	"""
	pcd_path = Config.DEFAULT_PCD_PATH
	seg_dir = Config.DEFAULT_SEG_PATH
	img_path = Config.DEFAULT_IMG_PATH

	# 读取位姿文件列表（按时间戳排序）
	pose_files, pose_timestamps = data_loader.load_pose_files(img_path)

	# 读取点云
	points = data_loader.load_point_cloud(pcd_path)

	# 新建输出点云对象，保留原始点坐标
	out_pcd = o3d.geometry.PointCloud()
	out_pcd.points = o3d.utility.Vector3dVector(points)

	# 计算 point_dic
	point_dic = compute_point_dic(points, pose_files, pose_timestamps)
	print(f"完成计算 point_dic，包含 {len(point_dic)} 个点的投影信息")

	# 加载 ADE20K 调色板
	palette = load_ade20k_palette(os.path.join(os.path.dirname(__file__), 'ADE20k.yaml'))

	# 缓存已加载的 seg npy 文件
	seg_cache: Dict[int, np.ndarray] = {}

	colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
	point_classes = np.zeros((points.shape[0],), dtype=np.uint8)

	for pid in range(points.shape[0]):
		entries = point_dic.get(pid, [])
		print(f"处理点ID {pid}，共有 {len(entries)} 个投影条目")
		if not entries:
			colors[pid] = np.array([0, 0, 0], dtype=np.uint8)
			continue

		vecs = []
		for ts, (x, y) in entries:
			ts = int(ts)
			if ts not in seg_cache:
				seg_file = os.path.join(seg_dir, f"{ts}_pred_class.npy")
				seg_cache[ts] = np.load(seg_file)

			seg = seg_cache[ts]
			h, w = seg.shape[0], seg.shape[1]
			xi = int(round(x)); yi = int(round(y))

			if xi < 0 or xi >= w or yi < 0 or yi >= h:
				continue

			vec = seg[yi, xi, :]
			vecs.append(vec)

		if not vecs:
			colors[pid] = np.array([0, 0, 0], dtype=np.uint8)
			continue

		mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
		cls = int(np.argmax(mean_vec))
		cls = np.clip(cls, 0, len(palette) - 1)	
		point_classes[pid] = cls
		colors[pid] = palette[cls]

		print(f"完成计算点ID {pid}: 类别 {cls}, 颜色 {colors[pid].tolist()}")

	# 过滤黑色点（RGB=[0,0,0]）
	# 1. 筛选非黑色点的索引（判断每个点的RGB是否全为0）
	# np.all(colors == [0,0,0], axis=1) 得到每个点是否为黑色的布尔数组，~ 取反得到非黑点索引
	non_black_indices = ~np.all(colors == [0, 0, 0], axis=1)
	print(f"原始点云共 {points.shape[0]} 个点，过滤掉 {np.sum(np.all(colors == [0,0,0], axis=1))} 个黑色点，剩余 {np.sum(non_black_indices)} 个非黑色点")

	# 2. 过滤点云：仅保留非黑色点（更新坐标、颜色、类别）
	filtered_points = points[non_black_indices]
	filtered_colors = colors[non_black_indices]
	filtered_point_classes = point_classes[non_black_indices]

	# 3. 更新输出点云的坐标和颜色
	out_pcd.points = o3d.utility.Vector3dVector(filtered_points)
	out_pcd.colors = o3d.utility.Vector3dVector(filtered_colors.astype(np.float64) / 255.0)

	# 输出路径
	base = os.path.basename(pcd_path)
	name = os.path.splitext(base)[0]
	out_dir = os.path.dirname(pcd_path)
	out_fname = f"{name}_seg.pcd"
	out_path = output_pcd_path or os.path.join(out_dir, out_fname)
	o3d.io.write_point_cloud(out_path, out_pcd)

	return out_path, filtered_point_classes


if __name__ == '__main__':
	out_path, point_classes = fuse_seg_without_blackpoint()
	print(f"保存融合后点云到: {out_path}")
