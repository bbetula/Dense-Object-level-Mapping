import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d

from class_statics_config import ADE20K_CATEGORIES, COLOR_TOLERANCE

# ===== 配置 =====
INPUT_PATH = Path("results/scannet")
INPUT_PLY = INPUT_PATH / "scene0000_00_vh_clean_2.labels.ply"
OUTPUT_DIR = INPUT_PATH / "color_separated_ply"
MIN_TRIANGLES_PER_COLOR = 50
# ==================================


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"网格为空: {path}")
    if not mesh.has_vertex_colors():
        raise ValueError("缺少顶点颜色，无法拆分。")
    return mesh


def color_to_key(rgb: np.ndarray) -> Tuple[int, int, int]:
    return tuple(int(np.clip(round(v * 255), 0, 255)) for v in rgb)


def sanitize(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip()) or "unknown"


def match_category(color_key: Tuple[int, int, int]) -> str:
    best_name = None
    best_distance = float("inf")
    for name, palette in ADE20K_CATEGORIES.items():
        diffs = [abs(a - b) for a, b in zip(color_key, palette)]
        if any(diff > COLOR_TOLERANCE for diff in diffs):
            continue
        distance = sum(diffs)
        if distance < best_distance:
            best_distance = distance
            best_name = name
    if best_name is None:
        r, g, b = color_key
        return f"unknown_{r:03d}_{g:03d}_{b:03d}"
    return best_name


def split_mesh_by_color(mesh: o3d.geometry.TriangleMesh) -> Dict[str, o3d.geometry.TriangleMesh]:
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None

    color_groups: Dict[str, List[int]] = {}
    for vidx, rgb in enumerate(colors):
        key = color_to_key(rgb)
        cat = match_category(key)
        color_groups.setdefault(cat, []).append(vidx)

    result: Dict[str, o3d.geometry.TriangleMesh] = {}
    for cat, vidxs in color_groups.items():
        vidxs = np.asarray(vidxs, dtype=np.int32)
        vertex_mask = np.zeros(len(vertices), dtype=bool)
        vertex_mask[vidxs] = True
        tri_mask = vertex_mask[triangles].all(axis=1)
        if not tri_mask.any():
            continue
        sub_tris = triangles[tri_mask]
        unique_vidxs, inverse = np.unique(sub_tris, return_inverse=True)
        if len(sub_tris) < MIN_TRIANGLES_PER_COLOR:
            continue

        sub_mesh = o3d.geometry.TriangleMesh()
        sub_mesh.vertices = o3d.utility.Vector3dVector(vertices[unique_vidxs])
        sub_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[unique_vidxs])
        if normals is not None and len(normals):
            sub_mesh.vertex_normals = o3d.utility.Vector3dVector(normals[unique_vidxs])
        sub_mesh.triangles = o3d.utility.Vector3iVector(inverse.reshape(-1, 3))
        result[cat] = sub_mesh
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = load_mesh(INPUT_PLY)
    splits = split_mesh_by_color(mesh)
    if not splits:
        print("没有满足条件的颜色网格。")
        return

    for category, sub_mesh in splits.items():
        out_path = OUTPUT_DIR / f"{sanitize(category)}.ply"
        o3d.io.write_triangle_mesh(
            str(out_path),
            sub_mesh,
            write_ascii=False,
            compressed=False,
            print_progress=False,
        )
        print(f"写入: {out_path}")

    summary = {"categories": sorted(splits.keys())}
    with (INPUT_PATH / "color_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"共输出 {len(splits)} 个类别网格。")


if __name__ == "__main__":
    main()