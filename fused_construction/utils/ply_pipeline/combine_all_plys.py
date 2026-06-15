from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ===== 配置区 =====
SOURCE_DIR = Path("results/scannet/hdbscan_meshes_triangles")
OUTPUT_PLY = SOURCE_DIR.parent / "all_mesh_with_bboxes_merged.ply"
FILE_SUFFIX = "_mesh_with_bboxes.ply"
# ==================================


def _get_count(header: List[str], element_name: str) -> int:
    prefix = f"element {element_name} "
    for h in header:
        if h.startswith(prefix):
            return int(h.split()[2])
    return 0


def _has_property(header: List[str], prop_name: str) -> bool:
    return any(h.strip().endswith(f" {prop_name}") and h.startswith("property ") for h in header)


def read_face_ply_ascii(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        header: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} header 不完整")
            s = line.strip()
            header.append(s)
            if s == "end_header":
                break

        fmt = next((h for h in header if h.startswith("format ")), None)
        if fmt is None or "ascii" not in fmt:
            raise ValueError(f"{path} 仅支持 ASCII PLY。")

        vertex_count = _get_count(header, "vertex")
        face_count = _get_count(header, "face")

        has_r = _has_property(header, "red")
        has_g = _has_property(header, "green")
        has_b = _has_property(header, "blue")
        has_color = has_r and has_g and has_b

        has_nx = _has_property(header, "nx")
        has_ny = _has_property(header, "ny")
        has_nz = _has_property(header, "nz")
        has_normals = has_nx and has_ny and has_nz

        vertices = np.zeros((vertex_count, 3), dtype=np.float64)
        colors = np.zeros((vertex_count, 3), dtype=np.uint8)
        normals = np.zeros((vertex_count, 3), dtype=np.float64) if has_normals else None

        for i in range(vertex_count):
            parts = f.readline().strip().split()
            if len(parts) < 3:
                raise ValueError(f"{path} 顶点行字段不足: {parts}")

            x, y, z = map(float, parts[:3])
            vertices[i] = [x, y, z]

            idx = 3
            if has_color:
                if len(parts) < idx + 3:
                    raise ValueError(f"{path} 顶点颜色字段不足: {parts}")
                r, g, b = map(int, parts[idx:idx + 3])
                colors[i] = [r, g, b]
                idx += 3
            else:
                colors[i] = [255, 255, 255]

            if has_normals:
                if len(parts) < idx + 3:
                    raise ValueError(f"{path} 顶点法向字段不足: {parts}")
                nx, ny, nz = map(float, parts[idx:idx + 3])
                normals[i] = [nx, ny, nz]

        faces = []
        for _ in range(face_count):
            parts = f.readline().strip().split()
            if not parts:
                continue
            n = int(parts[0])
            if n != 3:
                continue
            if len(parts) < 4:
                continue
            i0, i1, i2 = map(int, parts[1:4])
            faces.append([i0, i1, i2])

        return {
            "vertices": vertices,
            "colors": colors,
            "normals": normals,
            "faces": np.asarray(faces, dtype=np.int32),
            "has_normals": np.array([1 if has_normals else 0], dtype=np.int32),
        }


def write_face_ply_ascii(
    out_path: Path,
    vertices: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        if normals is None:
            for v, c in zip(vertices, colors):
                f.write(
                    f"{v[0]:.9f} {v[1]:.9f} {v[2]:.9f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                )
        else:
            for v, c, n in zip(vertices, colors, normals):
                f.write(
                    f"{v[0]:.9f} {v[1]:.9f} {v[2]:.9f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} "
                    f"{n[0]:.9f} {n[1]:.9f} {n[2]:.9f}\n"
                )

        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def main() -> None:
    ply_files = sorted(p for p in SOURCE_DIR.glob("*.ply") if p.name.endswith(FILE_SUFFIX))
    if not ply_files:
        print(f"目录 {SOURCE_DIR} 中未找到后缀为 {FILE_SUFFIX} 的文件。")
        return

    all_vertices: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    all_normals: List[np.ndarray] = []
    has_any_normals = False

    v_offset = 0
    for ply in ply_files:
        data = read_face_ply_ascii(ply)
        verts = data["vertices"]
        cols = data["colors"]
        faces = data["faces"]
        normals = data["normals"]
        has_normals = bool(data["has_normals"][0])

        if len(verts) == 0:
            print(f"跳过空文件: {ply.name}")
            continue

        all_vertices.append(verts)
        all_colors.append(cols)
        if len(faces) > 0:
            all_faces.append(faces + v_offset)

        if has_normals and normals is not None:
            all_normals.append(normals)
            has_any_normals = True
        else:
            all_normals.append(np.zeros((len(verts), 3), dtype=np.float64))

        print(f"加载: {ply.name} -> 顶点 {len(verts)}, 面 {len(faces)}")
        v_offset += len(verts)

    if not all_vertices:
        print("无有效数据。")
        return

    merged_vertices = np.vstack(all_vertices)
    merged_colors = np.vstack(all_colors)
    merged_faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int32)
    merged_normals = np.vstack(all_normals) if has_any_normals else None

    write_face_ply_ascii(
        OUTPUT_PLY,
        merged_vertices,
        merged_colors,
        merged_faces,
        merged_normals,
    )
    print(f"合并完成 -> {OUTPUT_PLY}")


if __name__ == "__main__":
    main()