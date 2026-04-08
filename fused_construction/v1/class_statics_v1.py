import os
import sys
import time
import numpy as np
import open3d as o3d
from collections import Counter
from ade20k_config import ADE20K_CATEGORIES, COLOR_TOLERANCE, MAX_IMG_COUNT, DEFAULT_PCD_PATH


def load_category_config(use_bgr: bool = False) -> tuple[np.ndarray, list[str], dict]:
    """
    加载类别配置并转换格式
    :param use_bgr: 是否将RGB转换为BGR格式
    :return: 类别颜色数组, 类别名称列表, 类别计数字典
    """
    # 转换配置为数组和列表（保持顺序一致）
    category_names = list(ADE20K_CATEGORIES.keys())
    category_colors = np.array(list(ADE20K_CATEGORIES.values()), dtype=int)
    
    # 转换为BGR格式（如果需要）
    if use_bgr:
        category_colors = category_colors[:, [2, 1, 0]]
    
    # 初始化计数器
    category_counts = {name: 0 for name in category_names}
    category_counts["其他"] = 0
    
    return category_colors, category_names, category_counts


def validate_pcd_path(pcd_path: str) -> bool:
    """
    校验PCD路径合法性
    :param pcd_path: 点云目录路径
    :return: 合法返回True，否则False
    """
    if not os.path.exists(pcd_path):
        print(f"错误: 路径 {pcd_path} 不存在")
        return False
    if not os.path.isdir(pcd_path):
        print(f"错误: {pcd_path} 不是有效目录")
        return False
    
    # 检查是否有PCD文件
    pcd_files = [f for f in os.listdir(pcd_path) if f.endswith(".pcd")]
    if len(pcd_files) == 0:
        print("错误: 目录中未找到任何PCD文件")
        return False
    return True


def process_single_pcd(file_path: str, 
                       category_colors: np.ndarray, 
                       color_tolerance: int,
                       skip_others: bool) -> dict:
    """
    处理单个PCD文件
    :param file_path: PCD文件路径
    :param category_colors: 类别颜色数组
    :param color_tolerance: 颜色匹配容差
    :param skip_others: 是否跳过其他类别
    :return: 处理结果字典
    """
    result = {
        "valid_points": np.array([]),
        "valid_colors": np.array([]),
        "category_counts": np.zeros(len(category_colors), dtype=int),
        "unclassified_count": 0,
        "black_points_count": 0,
        "other_rgb_counter": Counter(),
        "success": False
    }
    
    try:
        # 读取点云
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_colors():
            print(f"警告: 文件 {os.path.basename(file_path)} 无颜色信息，跳过")
            return result
        
        # 提取点云和颜色
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # 标准化颜色到0-255
        colors_255 = (colors * 255).round().astype(int) if np.max(colors) <= 1.0 else colors.round().astype(int)
        
        # 过滤黑色点（RGB全0）
        is_black = np.all(colors_255 == 0, axis=1)
        result["black_points_count"] = np.sum(is_black)
        non_black_mask = ~is_black
        valid_points = points[non_black_mask]
        valid_colors = colors[non_black_mask]
        valid_colors_255 = colors_255[non_black_mask]
        
        if len(valid_points) == 0:
            return result
        
        # 向量化计算颜色距离（替代循环，提升效率）
        distance_threshold = np.sqrt(3 * color_tolerance**2)
        # 计算每个点到所有类别颜色的距离 (N, C)
        color_distances = np.sqrt(np.sum((valid_colors_255[:, None] - category_colors[None, :])**2, axis=2))
        # 找到每个点的最小距离类别
        min_distances = np.min(color_distances, axis=1)
        point_categories = np.where(min_distances <= distance_threshold, np.argmin(color_distances, axis=1), -1)
        
        # 统计各类别点数
        for i in range(len(category_colors)):
            result["category_counts"][i] = np.sum(point_categories == i)
        # 统计未分类点
        unclassified_mask = point_categories == -1
        result["unclassified_count"] = np.sum(unclassified_mask)
        
        # 收集未分类点的RGB值
        for rgb in valid_colors_255[unclassified_mask]:
            result["other_rgb_counter"][tuple(rgb)] += 1
        
        # 筛选需要保留的点（跳过/保留其他类别）
        if skip_others:
            keep_mask = ~unclassified_mask
            result["valid_points"] = valid_points[keep_mask]
            result["valid_colors"] = valid_colors[keep_mask]
        else:
            result["valid_points"] = valid_points
            result["valid_colors"] = valid_colors
        
        result["success"] = True
        
    except Exception as e:
        print(f"处理文件 {os.path.basename(file_path)} 出错: {str(e)}")
    
    return result


def print_statistics(category_counts: dict, 
                     total_points: int, 
                     skipped_black: int, 
                     skipped_other: int,
                     file_count: int,
                     total_files: int,
                     process_time: float):
    """
    打印统计结果
    """
    print("\n" + "-" * 60)
    print("类别统计结果")
    print("-" * 60)
    print(f"{'类别':<15} {'点数':<15} {'占比':<10}")
    print("-" * 60)
    
    # 按点数降序打印
    for category in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
        count = category_counts[category]
        if count == 0 and category != "其他":
            continue
        percentage = (count / total_points) * 100 if total_points > 0 else 0
        print(f"{category:<15} {count:<15} {percentage:.2f}%")
    
    print("-" * 60)
    print(f"有效点总数: {total_points}")
    print(f"跳过的黑色点总数: {skipped_black}")
    print(f"跳过的其他类别点总数: {skipped_other}")
    print(f"处理的文件数: {file_count}/{total_files}")
    print(f"总处理时间: {process_time:.2f} 秒")


def print_other_rgb_stats(other_rgb_counter: Counter, other_total: int, use_bgr: bool):
    """
    打印其他类别RGB值统计
    """
    if other_total == 0:
        return
    
    print("\n" + "-" * 60)
    print("其他类别的RGB值统计 (前20个最常见)")
    print("-" * 60)
    print(f"{'RGB值':<20} {'出现次数':<15} {'占比(%)':<10}")
    print("-" * 60)
    
    for rgb, count in other_rgb_counter.most_common(20):
        # 转换回RGB格式（如果是BGR）
        display_rgb = rgb if not use_bgr else (rgb[2], rgb[1], rgb[0])
        display_rgb = tuple(int(x) for x in display_rgb)
        percentage = (count / other_total) * 100
        print(f"{str(display_rgb):<20} {count:<15} {percentage:.2f}")


def save_merged_pcd(all_points: list, all_colors: list, pcd_path: str):
    """
    合并并保存所有有效点云
    """
    if not all_points:
        return
    
    try:
        # 创建输出目录
        res_dir = os.path.join(os.path.dirname(pcd_path), "res")
        os.makedirs(res_dir, exist_ok=True)
        output_path = os.path.join(res_dir, "all_seg.pcd")
        
        # 合并点云和颜色
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        
        # 保存点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)
        o3d.io.write_point_cloud(output_path, pcd)
        
        print(f"\n合并后的点云已保存到: {output_path}")
        print(f"保存的点数: {len(merged_points)}")
    except Exception as e:
        print(f"保存点云失败: {str(e)}")


def main(pcd_path: str):
    """
    主函数：处理PCD文件并统计类别
    """
    # 配置参数
    use_bgr = False  # 是否使用BGR颜色格式
    skip_others = True  # 是否跳过其他类别
    
    # 初始化统计变量
    start_time = time.time()
    total_points = 0
    file_count = 0
    skipped_black_points = 0
    skipped_other_points = 0
    all_points = []
    all_colors = []
    other_rgb_counter = Counter()
    
    # 1. 校验路径
    if not validate_pcd_path(pcd_path):
        return
    
    # 2. 加载类别配置
    category_colors, category_names, category_counts = load_category_config(use_bgr)
    print(f"使用{'BGR' if use_bgr else 'RGB'}格式处理颜色，容差±{COLOR_TOLERANCE}")
    print(f"跳过黑色点: 是 | 跳过其他类别点: {'是' if skip_others else '否'}")
    
    # 3. 获取PCD文件列表
    pcd_files = [f for f in os.listdir(pcd_path) if f.endswith(".pcd")]
    total_files = len(pcd_files)
    print(f"\n开始处理 {total_files} 个PCD文件 (最大处理数: {MAX_IMG_COUNT})...")
    
    # 4. 遍历处理每个PCD文件
    for idx, filename in enumerate(pcd_files):
        if idx >= MAX_IMG_COUNT:
            break
        
        file_path = os.path.join(pcd_path, filename)
        result = process_single_pcd(file_path, category_colors, COLOR_TOLERANCE, skip_others)
        
        if not result["success"]:
            continue
        
        # 更新全局统计
        skipped_black_points += result["black_points_count"]
        skipped_other_points += result["unclassified_count"] if skip_others else 0
        total_points += len(result["valid_points"])
        
        # 更新类别计数
        for i, name in enumerate(category_names):
            category_counts[name] += result["category_counts"][i]
        category_counts["其他"] += result["unclassified_count"]
        
        # 收集其他类别RGB统计
        other_rgb_counter.update(result["other_rgb_counter"])
        
        # 收集点云和颜色
        if len(result["valid_points"]) > 0:
            all_points.append(result["valid_points"])
            all_colors.append(result["valid_colors"])
        
        file_count += 1
        # 打印进度
        progress_info = (
            f"处理进度: [{file_count}/{total_files}] "
            f"文件: {filename} | 有效点数: {len(result['valid_points'])} "
            f"跳过黑色点: {result['black_points_count']} "
            f"跳过其他点: {result['unclassified_count'] if skip_others else 0}"
        )
        print(progress_info)
    
    # 5. 打印统计结果
    process_time = time.time() - start_time
    if total_points > 0:
        print_statistics(category_counts, total_points, skipped_black_points, 
                         skipped_other_points, file_count, total_files, process_time)
        print_other_rgb_stats(other_rgb_counter, category_counts["其他"], use_bgr)
        save_merged_pcd(all_points, all_colors, pcd_path)
    else:
        print("\n未找到有效点云数据")


if __name__ == "__main__":
    # 解析命令行参数（优先使用传入的路径，否则用默认路径）
    pcd_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PCD_PATH
    main(pcd_path)