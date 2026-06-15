#!/usr/bin/env python3
"""Build a 3D scene graph from the semantic point cloud produced by the mapping pipeline.

Output graph schema:
  node = semantic object / scene region with bbox, centroid, point count, confidence
  edge = high-confidence spatial relation between two nodes

This script is designed to run after:
  pointcloud_colorized_v2_normal.py
  class_statics_v2.py
  generate_n_pcd_bbox.py / hdbscan_single_label_bbox.py / combine_all_pcds_bbox.py

It can use either a global semantic-colored PCD or a directory containing one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d

SCRIPT_DIR = Path(__file__).resolve().parent
FUSED_DIR = SCRIPT_DIR.parent
REPO_ROOT = FUSED_DIR.parent
for path in (str(FUSED_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from class_statics_config import (  # noqa: E402
    ADE20K_CATEGORIES,
    COLOR_TOLERANCE,
    DEFAULT_MAP_PATH,
    LABEL_CHOICE,
    SCANNET_NYU40_CATEGORIES,
)
from scene_graph_config import BUILD_PARAMS, SCENE_GRAPH_OUT_DIR, SEMANTIC_PCD_PATH  # noqa: E402


STRUCTURAL_CATEGORIES = {
    "wall",
    "floor",
    "ceiling",
    "road",
    "sidewalk",
    "path",
    "earth",
    "building",
    "skyscraper",
    "house",
    "fence",
    "railing",
    "stairs",
    "stairway",
    "runway",
    "field",
    "land",
    "grass",
}

SUPPORT_CATEGORIES = {
    "floor",
    "road",
    "sidewalk",
    "path",
    "earth",
    "table",
    "desk",
    "counter",
    "countertop",
    "shelf",
    "box",
    "cabinet",
    "bench",
}

DEFAULT_RELATIONS = ("adjacent_to", "on", "inside")

STRUCTURAL_ADJACENCY_PAIRS = {
    ("wall", "floor"),
    ("floor", "wall"),
    ("wall", "ceiling"),
    ("ceiling", "wall"),
    ("fence", "floor"),
    ("floor", "fence"),
    ("railing", "floor"),
    ("floor", "railing"),
    ("building", "floor"),
    ("floor", "building"),
    ("building", "road"),
    ("road", "building"),
}


@dataclass
class SceneNode:
    id: str
    label: str
    kind: str
    point_count: int
    confidence: float
    centroid: list[float]
    bbox_min: list[float]
    bbox_max: list[float]
    bbox_center: list[float]
    bbox_extent: list[float]
    volume: float
    color_rgb: list[int]


@dataclass
class SceneEdge:
    id: str
    source: str
    target: str
    relation: str
    confidence: float
    evidence: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a structured scene graph from a semantic-colored point cloud.")
    parser.add_argument("--pcd", type=Path, default=SEMANTIC_PCD_PATH, help="Semantic colored PCD.")
    parser.add_argument("--pcd-dir", type=Path, default=Path(DEFAULT_MAP_PATH), help="Directory containing a semantic PCD.")
    parser.add_argument("--out-dir", type=Path, default=SCENE_GRAPH_OUT_DIR, help="Output directory.")
    parser.add_argument("--scene-name", default=None, help="Scene name. Default: PCD stem.")
    parser.add_argument("--color-tolerance", default=COLOR_TOLERANCE, type=int)
    parser.add_argument("--min-category-points", default=BUILD_PARAMS["min_category_points"], type=int, help="Minimum semantic points to create a category node.")
    parser.add_argument("--min-instance-points", default=BUILD_PARAMS["min_instance_points"], type=int, help="Minimum clustered points to create an object node.")
    parser.add_argument("--voxel-size", default=BUILD_PARAMS["voxel_size"], type=float, help="Voxel downsample size before graph extraction. 0 disables.")
    parser.add_argument("--dbscan-eps", default=BUILD_PARAMS["dbscan_eps"], type=float, help="DBSCAN eps for object/region components in meters.")
    parser.add_argument("--dbscan-min-points", default=BUILD_PARAMS["dbscan_min_points"], type=int)
    parser.add_argument("--max-instances-per-class", default=BUILD_PARAMS["max_instances_per_class"], type=int)
    parser.add_argument("--near-threshold", default=None, type=float, help="BBox distance threshold. Default is scene-adaptive.")
    parser.add_argument("--support-z-threshold", default=0.25, type=float)
    parser.add_argument("--xy-overlap-threshold", default=0.08, type=float)
    parser.add_argument("--relation-confidence", default=BUILD_PARAMS["relation_confidence"], type=float, help="Drop relations below this confidence.")
    parser.add_argument("--min-relation-node-points", default=BUILD_PARAMS["min_relation_node_points"], type=int, help="Only create relation edges between nodes with at least this many points.")
    parser.add_argument(
        "--relations",
        default=",".join(DEFAULT_RELATIONS),
        help=f"Comma-separated relation types from {DEFAULT_RELATIONS}.",
    )
    parser.add_argument(
        "--include-structural-components",
        action="store_true",
        help="Cluster structural classes into components instead of one class-level region node.",
    )
    parser.add_argument("--write-review-templates", dest="write_review_templates", action="store_true", default=True, help="Write node/edge expert review CSV templates.")
    parser.add_argument("--no-review-templates", dest="write_review_templates", action="store_false", help="Do not write review CSV templates.")
    parser.add_argument("--write-csv", action="store_true", help="Also write nodes.csv and edges.csv debug tables.")
    parser.add_argument("--write-dot", action="store_true", help="Also write scene_graph.dot for Graphviz debugging.")
    parser.add_argument("--verbose", action="store_true", help="Print extra label statistics.")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip().lower()) or "unknown"


def categories() -> dict[str, list[int]]:
    if LABEL_CHOICE == "SCANNET_NYU40":
        return SCANNET_NYU40_CATEGORIES
    return ADE20K_CATEGORIES


def resolve_pcd(pcd: Path | None, pcd_dir: Path) -> Path:
    if pcd is not None:
        if not pcd.exists():
            raise FileNotFoundError(pcd)
        return pcd
    if pcd_dir.is_file():
        return pcd_dir
    if not pcd_dir.exists():
        raise FileNotFoundError(pcd_dir)

    preferred = sorted(p for p in pcd_dir.glob("*.pcd") if p.name.endswith("_color.pcd"))
    if preferred:
        return preferred[0]
    pcds = sorted(pcd_dir.glob("*.pcd"))
    if not pcds:
        raise FileNotFoundError(f"No .pcd files found in {pcd_dir}")
    if len(pcds) > 1:
        print(f"[WARN] Multiple PCD files found in {pcd_dir}; using {pcds[0].name}")
    return pcds[0]


def load_semantic_cloud(path: Path, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {path}")
    if not pcd.has_colors():
        raise ValueError(f"Point cloud has no colors: {path}")
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if colors.max() <= 1.0:
        colors = np.rint(colors * 255).astype(np.int16)
    else:
        colors = np.rint(colors).astype(np.int16)
    return points, colors


def load_raw_semantic_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return load_semantic_cloud(path, voxel_size=0.0)


def downsample_labeled_points(
    points: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel downsample per semantic label, preserving the original semantic color."""
    if voxel_size <= 0 or len(points) == 0:
        return points, colors, labels

    out_points: list[np.ndarray] = []
    out_colors: list[np.ndarray] = []
    out_labels: list[np.ndarray] = []
    for label_idx in np.unique(labels):
        mask = labels == label_idx
        class_points = points[mask]
        class_colors = colors[mask]
        if len(class_points) == 0:
            continue

        voxel_keys = np.floor(class_points / voxel_size).astype(np.int64)
        _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
        voxel_count = int(inverse.max()) + 1
        sums = np.zeros((voxel_count, 3), dtype=np.float64)
        np.add.at(sums, inverse, class_points)
        counts = np.bincount(inverse).astype(np.float64)
        class_down_points = sums / counts[:, None]
        class_color = np.rint(class_colors.mean(axis=0)).clip(0, 255).astype(np.int16)

        out_points.append(class_down_points)
        out_colors.append(np.repeat(class_color[None, :], voxel_count, axis=0))
        out_labels.append(np.full((voxel_count,), int(label_idx), dtype=np.int32))

    return np.vstack(out_points), np.vstack(out_colors), np.concatenate(out_labels)


def classify_colors(colors: np.ndarray, palette: dict[str, list[int]], tolerance: int) -> tuple[np.ndarray, list[str]]:
    names = list(palette.keys())
    rgb = np.asarray([palette[name] for name in names], dtype=np.int16)
    if len(colors) == 0:
        return np.empty((0,), dtype=np.int32), names
    unique_colors, inverse = np.unique(colors.astype(np.int16), axis=0, return_inverse=True)
    distances = np.abs(unique_colors[:, None, :] - rgb[None, :, :])
    within = np.all(distances <= tolerance, axis=2)
    l1 = distances.sum(axis=2)
    best = np.argmin(np.where(within, l1, 10_000), axis=1)
    valid = within[np.arange(len(unique_colors)), best]
    unique_labels = np.where(valid, best, -1).astype(np.int32)
    black = np.all(unique_colors == 0, axis=1)
    unique_labels[black] = -1
    labels = unique_labels[inverse]
    return labels, names


def cluster_points(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    if len(points) < min_points:
        return np.full((len(points),), -1, dtype=np.int32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False), dtype=np.int32)
    return labels


def bbox_distance(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    delta = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(delta))


def xy_overlap_ratio(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    inter_x = max(0.0, min(a_max[0], b_max[0]) - max(a_min[0], b_min[0]))
    inter_y = max(0.0, min(a_max[1], b_max[1]) - max(a_min[1], b_min[1]))
    inter = inter_x * inter_y
    area_a = max(1e-9, (a_max[0] - a_min[0]) * (a_max[1] - a_min[1]))
    area_b = max(1e-9, (b_max[0] - b_min[0]) * (b_max[1] - b_min[1]))
    return float(inter / min(area_a, area_b))


def vertical_overlap_ratio(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    inter = max(0.0, min(a_max[2], b_max[2]) - max(a_min[2], b_min[2]))
    height = max(1e-9, min(a_max[2] - a_min[2], b_max[2] - b_min[2]))
    return float(inter / height)


def confidence_from_count(count: int, threshold: int) -> float:
    if threshold <= 0:
        return 1.0
    return float(np.clip(math.log1p(count / threshold) / math.log1p(20.0), 0.35, 1.0))


def create_node(
    node_id: str,
    label: str,
    points: np.ndarray,
    colors: np.ndarray,
    kind: str,
    min_points: int,
) -> SceneNode:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    volume = float(max(extent[0] * extent[1] * extent[2], 0.0))
    color = np.rint(colors.mean(axis=0)).clip(0, 255).astype(int).tolist() if len(colors) else [255, 255, 255]
    return SceneNode(
        id=node_id,
        label=label,
        kind=kind,
        point_count=int(len(points)),
        confidence=confidence_from_count(len(points), min_points),
        centroid=points.mean(axis=0).tolist(),
        bbox_min=bbox_min.tolist(),
        bbox_max=bbox_max.tolist(),
        bbox_center=bbox_center.tolist(),
        bbox_extent=extent.tolist(),
        volume=volume,
        color_rgb=color,
    )


def build_nodes(
    points: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    args: argparse.Namespace,
) -> list[SceneNode]:
    nodes: list[SceneNode] = []
    for label_idx, label_name in enumerate(label_names):
        mask = labels == label_idx
        count = int(mask.sum())
        if count == 0:
            continue
        class_points = points[mask]
        class_colors = colors[mask]
        is_structural = label_name in STRUCTURAL_CATEGORIES
        if count < args.min_instance_points:
            continue
        should_cluster = args.include_structural_components or not is_structural

        if not should_cluster:
            node_id = f"{sanitize_name(label_name)}_region_000"
            nodes.append(create_node(node_id, label_name, class_points, class_colors, "region", args.min_category_points))
            continue

        cluster_labels = cluster_points(class_points, args.dbscan_eps, args.dbscan_min_points)
        unique_clusters = [cid for cid in np.unique(cluster_labels) if cid != -1]
        cluster_records = []
        for cid in unique_clusters:
            idxs = np.where(cluster_labels == cid)[0]
            if len(idxs) >= args.min_instance_points:
                cluster_records.append((cid, idxs))
        cluster_records.sort(key=lambda item: len(item[1]), reverse=True)
        cluster_records = cluster_records[: args.max_instances_per_class]

        if not cluster_records:
            node_id = f"{sanitize_name(label_name)}_instance_000"
            kind = "region" if is_structural else "object"
            nodes.append(create_node(node_id, label_name, class_points, class_colors, kind, args.min_instance_points))
            continue

        for instance_idx, (_, idxs) in enumerate(cluster_records):
            node_id = f"{sanitize_name(label_name)}_instance_{instance_idx:03d}"
            kind = "region" if is_structural else "object"
            nodes.append(
                create_node(
                    node_id,
                    label_name,
                    class_points[idxs],
                    class_colors[idxs],
                    kind,
                    args.min_instance_points,
                )
            )
    return nodes


def relation_confidence(value: float, threshold: float, inverse: bool = False) -> float:
    if threshold <= 0:
        return 1.0
    if inverse:
        score = 1.0 - value / threshold
    else:
        score = value / threshold
    return float(np.clip(score, 0.0, 1.0))


def build_edges(nodes: list[SceneNode], relations: set[str], args: argparse.Namespace) -> list[SceneEdge]:
    relation_nodes = [node for node in nodes if node.point_count >= args.min_relation_node_points]
    if len(relation_nodes) < 2:
        return []
    scene_min = np.min([np.asarray(n.bbox_min) for n in relation_nodes], axis=0)
    scene_max = np.max([np.asarray(n.bbox_max) for n in relation_nodes], axis=0)
    scene_diag = float(np.linalg.norm(scene_max - scene_min))
    near_threshold = args.near_threshold if args.near_threshold is not None else max(0.45, scene_diag * 0.035)

    edges: list[SceneEdge] = []
    edge_idx = 0
    for i, a in enumerate(relation_nodes):
        a_min = np.asarray(a.bbox_min)
        a_max = np.asarray(a.bbox_max)
        a_center = np.asarray(a.bbox_center)
        for j, b in enumerate(relation_nodes):
            if i == j:
                continue
            b_min = np.asarray(b.bbox_min)
            b_max = np.asarray(b.bbox_max)
            b_center = np.asarray(b.bbox_center)
            dist = bbox_distance(a_min, a_max, b_min, b_max)
            xy_overlap = xy_overlap_ratio(a_min, a_max, b_min, b_max)
            z_gap_above = a_min[2] - b_max[2]
            z_gap_below = b_min[2] - a_max[2]
            v_overlap = vertical_overlap_ratio(a_min, a_max, b_min, b_max)

            candidates: list[tuple[str, float, dict]] = []
            if "near" in relations and i < j and dist <= near_threshold:
                conf = relation_confidence(dist, near_threshold, inverse=True) * 0.45 + 0.55
                candidates.append(("near", conf, {"bbox_distance": dist, "near_threshold": near_threshold}))

            if "adjacent_to" in relations and i < j and dist <= near_threshold * 0.6 and v_overlap >= 0.25:
                large_region_pair = a.kind == "region" or b.kind == "region"
                allowed_region_pair = (a.label, b.label) in STRUCTURAL_ADJACENCY_PAIRS
                if (not large_region_pair) or allowed_region_pair:
                    conf = min(1.0, 0.55 + 0.35 * v_overlap + 0.10 * relation_confidence(dist, near_threshold, inverse=True))
                    candidates.append(("adjacent_to", conf, {"bbox_distance": dist, "vertical_overlap": v_overlap}))

            if "on" in relations and b.label in SUPPORT_CATEGORIES and xy_overlap >= args.xy_overlap_threshold:
                strict_support = abs(z_gap_above) <= args.support_z_threshold
                floor_like_support = b.label in {"floor", "road", "sidewalk", "path", "earth"} and a_center[2] > b_center[2] and a_min[2] <= b_max[2] + args.support_z_threshold
                if strict_support or floor_like_support:
                    z_score = relation_confidence(abs(z_gap_above), max(args.support_z_threshold, 1e-6), inverse=True) if strict_support else 0.75
                    conf = min(1.0, 0.55 + 0.30 * xy_overlap + 0.15 * z_score)
                    candidates.append(("on", conf, {"support_z_gap": float(z_gap_above), "xy_overlap": float(xy_overlap), "floor_like_support": bool(floor_like_support)}))

            if "above" in relations and z_gap_above > args.support_z_threshold and xy_overlap >= args.xy_overlap_threshold:
                conf = min(1.0, 0.55 + 0.35 * xy_overlap)
                candidates.append(("above", conf, {"z_gap": z_gap_above, "xy_overlap": xy_overlap}))

            if "below" in relations and z_gap_below > args.support_z_threshold and xy_overlap >= args.xy_overlap_threshold:
                conf = min(1.0, 0.55 + 0.35 * xy_overlap)
                candidates.append(("below", conf, {"z_gap": z_gap_below, "xy_overlap": xy_overlap}))

            if "inside" in relations and b.label not in STRUCTURAL_CATEGORIES:
                inside_margin = np.minimum(a_min - b_min, b_max - a_max)
                if np.all(inside_margin >= -0.05) and np.linalg.norm(a_center - b_center) > 0:
                    size_ratio = float(np.prod(a_max - a_min + 1e-9) / np.prod(b_max - b_min + 1e-9))
                    if size_ratio < 0.75:
                        conf = float(np.clip(0.65 + 0.25 * (1.0 - size_ratio), 0.0, 1.0))
                        candidates.append(("inside", conf, {"size_ratio": size_ratio}))

            for rel, conf, evidence in candidates:
                if conf < args.relation_confidence:
                    continue
                edges.append(
                    SceneEdge(
                        id=f"edge_{edge_idx:05d}",
                        source=a.id,
                        target=b.id,
                        relation=rel,
                        confidence=round(float(conf), 4),
                        evidence=evidence,
                    )
                )
                edge_idx += 1
    return edges


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_nodes_csv(path: Path, nodes: Iterable[SceneNode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(node) for node in nodes]
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [field.name for field in SceneNode.__dataclass_fields__.values()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_edges_csv(path: Path, edges: Iterable[SceneEdge]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for edge in edges:
        row = asdict(edge)
        row["evidence"] = json.dumps(row["evidence"], ensure_ascii=False)
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [field.name for field in SceneEdge.__dataclass_fields__.values()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dot(path: Path, nodes: list[SceneNode], edges: list[SceneEdge]) -> None:
    lines = ["digraph scene_graph {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
    for node in nodes:
        color = "#%02x%02x%02x" % tuple(node.color_rgb)
        font_color = "white" if sum(node.color_rgb) < 360 else "black"
        label = f"{node.id}\\n{node.label}\\npts={node.point_count}"
        lines.append(f'  "{node.id}" [label="{label}", fillcolor="{color}", style="rounded,filled", fontcolor="{font_color}"];')
    for edge in edges:
        label = f"{edge.relation}\\n{edge.confidence:.2f}"
        lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{label}"];')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_templates(out_dir: Path, nodes: list[SceneNode], edges: list[SceneEdge]) -> None:
    node_review = out_dir / "node_review_template.csv"
    edge_review = out_dir / "edge_review_template.csv"
    expected_contents = out_dir / "expected_contents_template.csv"
    expected_relations = out_dir / "expected_relations_template.csv"

    with node_review.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "label", "kind", "point_count", "correct", "gt_label", "note"])
        writer.writeheader()
        for node in nodes:
            writer.writerow({"node_id": node.id, "label": node.label, "kind": node.kind, "point_count": node.point_count, "correct": "", "gt_label": "", "note": ""})

    with edge_review.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["edge_id", "source", "relation", "target", "confidence", "correct", "gt_relation", "note"])
        writer.writeheader()
        for edge in edges:
            writer.writerow({"edge_id": edge.id, "source": edge.source, "relation": edge.relation, "target": edge.target, "confidence": edge.confidence, "correct": "", "gt_relation": "", "note": ""})

    with expected_contents.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["content_id", "expected_label", "required", "matched_node_id", "note"])
        writer.writeheader()

    with expected_relations.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relation_id", "source_label", "relation", "target_label", "required", "matched_edge_id", "note"])
        writer.writeheader()


def main() -> int:
    args = parse_args()
    pcd_path = resolve_pcd(args.pcd, args.pcd_dir)
    scene_name = args.scene_name or pcd_path.stem
    out_dir = args.out_dir or (pcd_path.parent / "scene_graph" / scene_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_points, raw_colors = load_raw_semantic_cloud(pcd_path)
    palette = categories()
    raw_labels, label_names = classify_colors(raw_colors, palette, args.color_tolerance)
    raw_valid_mask = raw_labels >= 0
    raw_points = raw_points[raw_valid_mask]
    raw_colors = raw_colors[raw_valid_mask]
    raw_labels = raw_labels[raw_valid_mask]
    if len(raw_points) == 0:
        raise ValueError("No valid semantic points after color classification.")

    raw_counts_by_label: dict[str, int] = {}
    for label_idx in np.unique(raw_labels):
        raw_counts_by_label[label_names[int(label_idx)]] = int((raw_labels == label_idx).sum())

    points, colors, labels = downsample_labeled_points(raw_points, raw_colors, raw_labels, args.voxel_size)
    valid_mask = labels >= 0
    points = points[valid_mask]
    colors = colors[valid_mask]
    labels = labels[valid_mask]
    if len(points) == 0:
        raise ValueError("No valid semantic points after color classification.")

    nodes = build_nodes(points, colors, labels, label_names, args)
    relations = {item.strip() for item in args.relations.split(",") if item.strip()}
    edges = build_edges(nodes, relations, args)

    counts_by_label = {}
    for node in nodes:
        counts_by_label[node.label] = counts_by_label.get(node.label, 0) + 1

    payload = {
        "scene_name": scene_name,
        "source_pcd": str(pcd_path),
        "label_choice": LABEL_CHOICE,
        "metadata": {
            "raw_valid_semantic_points": int(len(raw_points)),
            "valid_semantic_points": int(len(points)),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "raw_point_counts_by_label": raw_counts_by_label,
            "node_counts_by_label": counts_by_label,
            "params": {
                "voxel_size": args.voxel_size,
                "dbscan_eps": args.dbscan_eps,
                "min_category_points": args.min_category_points,
                "min_instance_points": args.min_instance_points,
                "max_instances_per_class": args.max_instances_per_class,
                "min_relation_node_points": args.min_relation_node_points,
                "relation_confidence": args.relation_confidence,
            },
        },
        "nodes": [asdict(node) for node in nodes],
        "edges": [asdict(edge) for edge in edges],
    }

    write_json(out_dir / "scene_graph.json", payload)
    if args.write_csv:
        write_nodes_csv(out_dir / "nodes.csv", nodes)
        write_edges_csv(out_dir / "edges.csv", edges)
    if args.write_dot:
        write_dot(out_dir / "scene_graph.dot", nodes, edges)
    if args.write_review_templates:
        write_review_templates(out_dir, nodes, edges)

    print(f"[INFO] scene_graph: {out_dir / 'scene_graph.json'}")
    print(f"[INFO] nodes={len(nodes)}, edges={len(edges)}")
    if args.write_review_templates:
        print(f"[INFO] review templates: {out_dir}")
    if args.verbose:
        print("[INFO] top labels:")
        for label, count in sorted(counts_by_label.items(), key=lambda item: item[1], reverse=True)[:20]:
            print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
