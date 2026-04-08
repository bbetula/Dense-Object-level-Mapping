import open3d as o3d
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from pathlib import Path


def process_point_cloud(input_path: Path, output_path: Path, visualize: bool = False) -> None:
    print(f"正在加载点云文件: {input_path}")
    pcd = o3d.io.read_point_cloud(str(input_path))
    if pcd.is_empty():
        print("错误：无法读取文件或文件为空。")
        return

    # print("正在进行地面分割...")
    # distance_threshold: 点到平面的最大距离阈值
    # ransac_n: 每次迭代中用于拟合平面的点数
    # num_iterations: RANSAC 算法的迭代次数
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3,
                                             ransac_n=3,
                                             num_iterations=1000)
    off_ground_pcd = pcd.select_by_index(inliers, invert=True)
    print(f"地面点数: {len(inliers)}, 非地面点数: {len(off_ground_pcd.points)}")

    points = np.asarray(off_ground_pcd.points)
    if len(points) == 0:
        print("非地面点为空，跳过此文件。")
        return

    # print("正在进行 HDBSCAN 聚类...")
    # min_cluster_size: 最小簇大小
    # min_samples: 最小样本数
    # gen_min_span_tree: 是否生成最小生成树以便可视化
    clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=10, gen_min_span_tree=False, core_dist_n_jobs=-1)
    cluster_labels = clusterer.fit_predict(points)
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"聚类完成，共发现 {num_clusters} 个簇。")

    max_label = cluster_labels.max()
    colors = np.zeros((len(points), 3))
    cmap = plt.get_cmap("tab20")
    for idx in range(len(points)):
        if cluster_labels[idx] == -1:
            colors[idx] = [0, 0, 0]
        else:
            divisor = max_label if max_label > 0 else 1
            colors[idx] = cmap(cluster_labels[idx] / divisor)[:3]
    off_ground_pcd.colors = o3d.utility.Vector3dVector(colors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), off_ground_pcd)
    print(f"已保存聚类结果到: {output_path}")

    if visualize:
        print("正在显示可视化结果（黑色为噪声，彩色为聚类簇）...")
        o3d.visualization.draw_geometries([off_ground_pcd])


def process_directory(input_dir: Path, visualize: bool = False) -> None:
    input_dir = input_dir.resolve()
    output_dir = input_dir / "hdbscan"
    pcd_files = sorted(input_dir.glob("*.pcd"))

    if not pcd_files:
        print(f"目录 {input_dir} 中未找到 PCD 文件。")
        return

    print(f"共找到 {len(pcd_files)} 个 PCD 文件，输出目录: {output_dir}")
    for pcd_file in pcd_files:
        output_path = output_dir / pcd_file.name
        process_point_cloud(pcd_file, output_path, visualize=visualize)


if __name__ == "__main__":
    # input_directory = Path("/data1/user/data/fastlivo_output_outdoor_1s/lidar")
    # input_directory = Path("results/fastlivo_output_outdoor_1s/")
    input_directory = Path("results/fastlivo_output_indoor_107")
    process_directory(input_directory, visualize=False)