#!/usr/bin/env python3
"""Audit whether every semantic class in the input map appears as scene graph nodes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FUSED_DIR = SCRIPT_DIR.parent
REPO_ROOT = FUSED_DIR.parent
for path in (str(SCRIPT_DIR), str(FUSED_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_scene_graph import (  # noqa: E402
    STRUCTURAL_CATEGORIES,
    categories,
    classify_colors,
    cluster_points,
    downsample_labeled_points,
    load_raw_semantic_cloud,
)
from class_statics_config import COLOR_TOLERANCE  # noqa: E402
from scene_graph_config import BUILD_PARAMS, SCENE_GRAPH_OUT_DIR, SEMANTIC_PCD_PATH  # noqa: E402


def main() -> int:
    pcd_path = SEMANTIC_PCD_PATH
    graph_path = SCENE_GRAPH_OUT_DIR / "scene_graph.json"
    out_path = SCENE_GRAPH_OUT_DIR / "scene_graph_coverage_audit.json"

    points, colors = load_raw_semantic_cloud(pcd_path)
    labels, label_names = classify_colors(colors, categories(), COLOR_TOLERANCE)
    valid = labels >= 0
    points = points[valid]
    colors = colors[valid]
    labels = labels[valid]

    map_counts: dict[str, int] = {}
    for label_idx in np.unique(labels):
        map_counts[label_names[int(label_idx)]] = int((labels == label_idx).sum())

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    down_points, down_colors, down_labels = downsample_labeled_points(
        points,
        colors,
        labels,
        float(BUILD_PARAMS["voxel_size"]),
    )
    _ = down_colors

    node_counts: dict[str, int] = {}
    node_points: dict[str, int] = {}
    for node in graph.get("nodes", []):
        label = str(node.get("label", ""))
        node_counts[label] = node_counts.get(label, 0) + 1
        node_points[label] = node_points.get(label, 0) + int(node.get("point_count", 0))

    rows = []
    missing = []
    instance_mismatch = []
    for label, map_point_count in sorted(map_counts.items(), key=lambda item: item[0]):
        graph_node_count = node_counts.get(label, 0)
        label_idx = label_names.index(label)
        class_down_points = down_points[down_labels == label_idx]
        downsampled_point_count = int(len(class_down_points))
        if label in STRUCTURAL_CATEGORIES:
            expected_node_count = 1 if downsampled_point_count else 0
        else:
            cluster_labels = cluster_points(
                class_down_points,
                float(BUILD_PARAMS["dbscan_eps"]),
                int(BUILD_PARAMS["dbscan_min_points"]),
            )
            expected_node_count = len([cid for cid in np.unique(cluster_labels) if cid != -1])
            expected_node_count = min(expected_node_count, int(BUILD_PARAMS["max_instances_per_class"]))
        row = {
            "label": label,
            "map_point_count": map_point_count,
            "downsampled_point_count": downsampled_point_count,
            "expected_node_count": expected_node_count,
            "graph_node_count": graph_node_count,
            "graph_node_point_count": node_points.get(label, 0),
            "class_covered": graph_node_count > 0,
            "instance_count_matched": graph_node_count >= expected_node_count,
        }
        rows.append(row)
        if graph_node_count == 0:
            missing.append(row)
        if graph_node_count < expected_node_count:
            instance_mismatch.append(row)

    payload = {
        "input_pcd": str(pcd_path),
        "scene_graph": str(graph_path),
        "semantic_class_count": len(map_counts),
        "covered_class_count": len(map_counts) - len(missing),
        "missing_class_count": len(missing),
        "class_coverage": (len(map_counts) - len(missing)) / len(map_counts) if map_counts else None,
        "class_coverage_percent": ((len(map_counts) - len(missing)) / len(map_counts) * 100) if map_counts else None,
        "expected_instance_node_count": sum(row["expected_node_count"] for row in rows),
        "actual_instance_node_count": sum(row["graph_node_count"] for row in rows),
        "instance_mismatch_count": len(instance_mismatch),
        "instance_coverage": (
            sum(min(row["graph_node_count"], row["expected_node_count"]) for row in rows)
            / sum(row["expected_node_count"] for row in rows)
            if rows and sum(row["expected_node_count"] for row in rows)
            else None
        ),
        "instance_coverage_percent": (
            sum(min(row["graph_node_count"], row["expected_node_count"]) for row in rows)
            / sum(row["expected_node_count"] for row in rows)
            * 100
            if rows and sum(row["expected_node_count"] for row in rows)
            else None
        ),
        "missing_classes": missing,
        "instance_mismatches": instance_mismatch,
        "classes": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Scene Graph Coverage Audit]")
    print(f"semantic classes in map : {payload['semantic_class_count']}")
    print(f"covered classes         : {payload['covered_class_count']}")
    print(f"missing classes         : {payload['missing_class_count']}")
    if payload["class_coverage_percent"] is not None:
        print(f"class coverage          : {payload['class_coverage_percent']:.2f}%")
    print(f"expected instance nodes : {payload['expected_instance_node_count']}")
    print(f"actual graph nodes      : {payload['actual_instance_node_count']}")
    print(f"instance mismatches     : {payload['instance_mismatch_count']}")
    if payload["instance_coverage_percent"] is not None:
        print(f"instance coverage       : {payload['instance_coverage_percent']:.2f}%")
    if missing:
        print("missing labels          : " + ", ".join(row["label"] for row in missing))
    if instance_mismatch:
        print("instance mismatch labels: " + ", ".join(row["label"] for row in instance_mismatch))
    print(f"[INFO] audit JSON: {out_path}")
    return 1 if missing or instance_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
