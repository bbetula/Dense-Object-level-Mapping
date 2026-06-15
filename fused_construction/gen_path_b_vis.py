"""
从 Path B 输出的 stuff_pcds/ 和 instance_pcds/ 生成可视化 PLY：
  - all_vis.ply:            stuff（灰色） + instances（随机彩色） + bbox 线框
  - instances_only_vis.ply: 仅 instances（随机彩色） + bbox 线框
"""

import os
import re
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from class_statics_config import ADE20K_CATEGORIES
from hdbscan_single_label_bbox import (
    create_bbox_line_set,
    extract_bbox,
    export_point_cloud_with_bboxes,
)

OUTPUT_DIR = "/data1/user/data/fastlivo_output_qs2_03.17/lidar/path_b_output"
STUFF_DIR  = os.path.join(OUTPUT_DIR, "stuff_pcds")
INST_DIR   = os.path.join(OUTPUT_DIR, "instance_pcds")

GROUND_GRAY = np.array([0.6, 0.6, 0.6])


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_")


_SANITIZED_TO_COLOR = {
    _sanitize_name(name): np.array(rgb, dtype=np.float64) / 255.0
    for name, rgb in ADE20K_CATEGORIES.items()
}


def get_instance_color(filename: str) -> np.ndarray:
    match = re.match(r"(.+)_inst_\d+\.pcd$", filename)
    if match and match.group(1) in _SANITIZED_TO_COLOR:
        return _SANITIZED_TO_COLOR[match.group(1)]
    return GROUND_GRAY


def load_pcds(directory, ext=".pcd"):
    pcds = []
    for f in sorted(os.listdir(directory)):
        if f.endswith(ext):
            path = os.path.join(directory, f)
            pcd = o3d.io.read_point_cloud(path)
            if len(pcd.points) > 0:
                pcds.append((f, pcd))
    return pcds


def main():
    print("加载 instance PCDs...")
    inst_pcds = load_pcds(INST_DIR)
    print(f"  {len(inst_pcds)} 个实例")

    inst_combined = o3d.geometry.PointCloud()
    line_sets = []

    for fname, pcd in inst_pcds:
        n = len(pcd.points)
        color = get_instance_color(fname)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
        inst_combined += pcd

        bbox, rotation_np, _ = extract_bbox(pcd)
        ls = create_bbox_line_set(bbox, color.tolist(), rotation_np)
        line_sets.append(ls)

        extent = bbox.get_extent()
        print(f"  {fname}: {n:,} 点, bbox={extent[0]:.2f}x{extent[1]:.2f}x{extent[2]:.2f}m")

    inst_path = os.path.join(OUTPUT_DIR, "instances_only_vis.ply")
    export_point_cloud_with_bboxes(inst_combined, line_sets, Path(inst_path))
    print(f"\n=> instances_only_vis.ply: {len(inst_combined.points):,} 点, {len(line_sets)} bbox")

    print("\n加载 stuff PCDs...")
    stuff_pcds = load_pcds(STUFF_DIR)
    print(f"  {len(stuff_pcds)} 个 stuff 类别")

    stuff_combined = o3d.geometry.PointCloud()
    for fname, pcd in stuff_pcds:
        n = len(pcd.points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(GROUND_GRAY, (n, 1)))
        stuff_combined += pcd

    all_combined = stuff_combined + inst_combined
    all_path = os.path.join(OUTPUT_DIR, "all_vis.ply")
    export_point_cloud_with_bboxes(all_combined, line_sets, Path(all_path))
    print(f"=> all_vis.ply: {len(all_combined.points):,} 点 (stuff={len(stuff_combined.points):,} + inst={len(inst_combined.points):,})")

    print("\n完成！")


if __name__ == "__main__":
    main()
