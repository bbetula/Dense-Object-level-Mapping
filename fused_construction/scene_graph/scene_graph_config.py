#!/usr/bin/env python3
"""Default scene graph task configuration."""

from pathlib import Path


REPO_ROOT = Path("/data1/user/Dense-Object-level-Mapping")

# SEMANTIC_PCD_PATH = Path(
#     "/data1/user/data/fastlivo_output_qs2_03.17/lidar/res/all_raw_qs2_03.17_color_normal.pcd"
# )
SEMANTIC_PCD_PATH = Path(
    "/data1/user/data/2026.05.28_655/点云图片文件/fastlivo_output_2026.05.29_2slow/lidar/res/all_raw_points_color_normal_no_vote.pcd"
)


# SCENE_GRAPH_OUT_DIR = Path(
#     "/data1/user/data/fastlivo_output_qs2_03.17/lidar/scene_graph/all_raw_qs2_03.17_color_normal_strict"
# )
SCENE_GRAPH_OUT_DIR = Path(
    "/data1/user/data/2026.05.28_655/点云图片文件/fastlivo_output_2026.05.29_2slow/lidar/scene_graph/all_raw_points_color_normal_strict"
)

VISUALIZATION_TITLE = "飞场语义场景图"

BUILD_PARAMS = {
    "voxel_size": 0.18,
    "min_category_points": 1,
    "min_instance_points": 15,
    "dbscan_eps": 0.55,
    "dbscan_min_points": 5,
    "max_instances_per_class": 5000,
    "relation_confidence": 0.65,
    "min_relation_node_points": 15,
}

TARGET_COMPLETENESS = 0.90
TARGET_STRUCTURED_ACCURACY = 0.85
