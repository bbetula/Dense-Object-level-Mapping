from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

# ===== 配置区 =====
INPUT_DIR = Path("/data1/data/scannet/output/label_ply")
OUTPUT_DIR = INPUT_DIR.parent / "pcd_from_ply"
RECURSIVE = False
WRITE_ASCII = False
COMPRESSED = False
# ==================================


def load_ply_as_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """
    优先按点云读取；
    若失败或为空，则按网格读取并将顶点转为点云。
    不做降采样、不做重采样，尽量保留原始实例完整性。
    """
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.is_empty():
        return pcd

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"无法读取 PLY 或内容为空: {path}")

    vertices = np.asarray(mesh.vertices)
    if len(vertices) == 0:
        raise ValueError(f"PLY 中没有可用顶点: {path}")

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(vertices)

    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        if len(colors) == len(vertices):
            out_pcd.colors = o3d.utility.Vector3dVector(colors)

    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)
        if len(normals) == len(vertices):
            out_pcd.normals = o3d.utility.Vector3dVector(normals)

    return out_pcd


def convert_single_file(ply_path: Path, output_dir: Path) -> Optional[Path]:
    try:
        pcd = load_ply_as_point_cloud(ply_path)
        if pcd.is_empty():
            print(f"跳过空文件: {ply_path}")
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{ply_path.stem}.pcd"

        ok = o3d.io.write_point_cloud(
            str(out_path),
            pcd,
            write_ascii=WRITE_ASCII,
            compressed=COMPRESSED,
            print_progress=False,
        )
        if not ok:
            raise RuntimeError("write_point_cloud 返回 False")

        print(f"转换成功: {ply_path.name} -> {out_path.name} | 点数={len(pcd.points)}")
        return out_path
    except Exception as e:
        print(f"转换失败: {ply_path} | {e}")
        return None


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")

    pattern = "**/*.ply" if RECURSIVE else "*.ply"
    ply_files = sorted(INPUT_DIR.glob(pattern))

    if not ply_files:
        print(f"未找到 PLY 文件: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    for ply_path in ply_files:
        out = convert_single_file(ply_path, OUTPUT_DIR)
        if out is not None:
            success += 1

    print(f"完成: 成功转换 {success} / {len(ply_files)} 个文件")


if __name__ == "__main__":
    main()