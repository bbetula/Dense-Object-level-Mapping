#!/usr/bin/env python3
"""
合并目录中原始的 .pcd 文件为一个整体点云。
输出：<pcd_dir_parent>/res/all_original.pcd

用法示例：
  python3 fused_origin_pcd.py /path/to/lidar_dir
  python3 fused_origin_pcd.py -o /custom/output.pcd

默认目录（如果未提供参数）：
  /home/czj/datasets/fastlivo_output_indoor_107/lidar
"""
import os
import sys
import glob
import argparse
import numpy as np
import open3d as o3d

# 配置常量
# DEFAULT_PCD_DIR = "/home/czj/datasets/fastlivo_output_indoor_107/lidar"
DEFAULT_PCD_DIR = "/home/czj/datasets/fastlivo_output_outdoor_1s/lidar"
DEFAULT_OUTPUT_FILENAME = "all_original.pcd"


def get_sorted_pcd_files(pcd_dir):
    """获取按时间戳/文件名排序的PCD文件列表"""
    files = glob.glob(os.path.join(pcd_dir, "*.pcd"))
    if not files:
        return []
    # 按文件名前缀整数排序，失败则按文件名排序
    try:
        files.sort(key=lambda f: int(os.path.basename(f).split('.')[0]))
    except Exception:
        files.sort()
    return files


def merge_pcds(pcd_files, out_path):
    """合并PCD文件并保存到指定路径"""
    if not pcd_files:
        raise ValueError("没有找到任何 .pcd 文件进行合并")

    points_list = []
    colors_list = []
    total_points = 0

    for f in pcd_files:
        try:
            pcd = o3d.io.read_point_cloud(f)
        except Exception as e:
            print(f"警告：无法读取 {f}: {e}")
            continue

        pts = np.asarray(pcd.points)
        if pts.size == 0:
            continue
        points_list.append(pts)
        total_points += pts.shape[0]

        if pcd.has_colors():
            colors = np.asarray(pcd.colors).astype(np.float64)
            # 归一化颜色到0-1范围
            if colors.max() > 1.0:
                colors = np.clip(colors / 255.0, 0.0, 1.0)
            colors_list.append(colors)
        else:
            colors_list.append(None)

    if total_points == 0:
        raise ValueError("合并后没有点")

    # 合并点云
    merged_points = np.vstack(points_list)

    # 处理颜色（无颜色填充黑色）
    merged_colors = None
    if any(c is not None for c in colors_list):
        merged_colors_parts = []
        for idx, c in enumerate(colors_list):
            if c is None:
                merged_colors_parts.append(np.zeros((points_list[idx].shape[0], 3), dtype=np.float64))
            else:
                merged_colors_parts.append(c)
        merged_colors = np.vstack(merged_colors_parts)

    # 保存点云
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(merged_points)
    if merged_colors is not None:
        out_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    success = o3d.io.write_point_cloud(out_path, out_pcd)
    if not success:
        raise RuntimeError(f"写出点云失败: {out_path}")
    return total_points


def main():
    parser = argparse.ArgumentParser(description="合并目录中原始的 .pcd 为一个整体点云")
    parser.add_argument('pcd_dir', nargs='?', default=DEFAULT_PCD_DIR,
                        help=f'输入点云目录（默认：{DEFAULT_PCD_DIR}）')
    parser.add_argument('--out', '-o', default=None,
                        help=f'输出文件路径（默认：<输入目录父级>/{DEFAULT_OUTPUT_FILENAME}）')
    args = parser.parse_args()

    # 校验输入目录
    pcd_dir = args.pcd_dir
    if not os.path.isdir(pcd_dir):
        print(f"错误：目录不存在: {pcd_dir}")
        sys.exit(2)

    # 获取排序后的PCD文件
    pcd_files = get_sorted_pcd_files(pcd_dir)
    if not pcd_files:
        print(f"目录中未找到 .pcd 文件: {pcd_dir}")
        sys.exit(0)

    # 确定输出路径（显式展示）
    out_path = args.out
    if out_path is None:
        parent = os.path.dirname(pcd_dir)
        out_path = os.path.join(parent, DEFAULT_OUTPUT_FILENAME)
    
    # 显式打印输入输出路径
    print(f"输入路径：{pcd_dir}")
    print(f"输出路径：{out_path}")
    print(f"找到 {len(pcd_files)} 个 PCD 文件，开始合并...")

    # 执行合并
    try:
        total = merge_pcds(pcd_files, out_path)
        print(f"合并完成，点数总计: {total}，保存到: {out_path}")
    except Exception as e:
        print(f"合并失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()