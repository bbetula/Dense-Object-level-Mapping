import os
import sys
import time
import numpy as np
import open3d as o3d
from collections import Counter
from class_statics_config import LABEL_CHOICE, COLOR_TOLERANCE, DEFAULT_MAP_PATH

if LABEL_CHOICE == "ADE20K":
    from class_statics_config import ADE20K_CATEGORIES
    CATEGORIES = ADE20K_CATEGORIES
elif LABEL_CHOICE == "SCANNET_NYU40":
    from class_statics_config import SCANNET_NYU40_CATEGORIES
    CATEGORIES = SCANNET_NYU40_CATEGORIES

INFO_TXT_PATH = os.path.join(os.path.dirname(__file__), "txt/info.txt")
def load_category_config() -> tuple[np.ndarray, list[str], dict]:
    """
    加载类别配置并转换格式
    """
    category_names = list(CATEGORIES.keys())
    category_colors = np.array(list(CATEGORIES.values()), dtype=int)
    category_counts = {name: 0 for name in category_names}
    category_counts["其他"] = 0
    return category_colors, category_names, category_counts


def resolve_single_pcd_file(pcd_input: str) -> str | None:
    """
    解析输入路径，支持直接传入单个PCD文件，或传入仅包含一个全局彩色PCD的目录
    """
    if not pcd_input:
        print("错误: 未提供PCD输入路径")
        return None

    if not os.path.exists(pcd_input):
        print(f"错误: 路径 {pcd_input} 不存在")
        return None

    if os.path.isfile(pcd_input):
        if not pcd_input.endswith(".pcd"):
            print(f"错误: {pcd_input} 不是PCD文件")
            return None
        return pcd_input

    if not os.path.isdir(pcd_input):
        print(f"错误: {pcd_input} 既不是文件也不是目录")
        return None

    color_pcd_files = sorted(
        f for f in os.listdir(pcd_input)
        if f.endswith("_color.pcd")
    )
    if len(color_pcd_files) == 1:
        return os.path.join(pcd_input, color_pcd_files[0])
    if len(color_pcd_files) > 1:
        print(f"错误: 目录 {pcd_input} 下找到多个全局彩色PCD，请显式指定文件路径")
        for filename in color_pcd_files:
            print(f"  - {filename}")
        return None

    pcd_files = sorted(
        f for f in os.listdir(pcd_input)
        if f.endswith(".pcd")
    )
    if len(pcd_files) == 1:
        return os.path.join(pcd_input, pcd_files[0])

    print(f"错误: 目录 {pcd_input} 中未找到唯一的PCD文件")
    return None


def process_single_pcd(file_path: str,
                       category_colors: np.ndarray,
                       color_tolerance: int) -> dict:
    """
    处理单个PCD文件
    """
    result = {
        "category_counts": np.zeros(len(category_colors), dtype=int),
        "classified_count": 0,
        "unclassified_count": 0,
        "black_points_count": 0,
        "other_rgb_counter": Counter(),
        "success": False
    }

    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_colors():
            print(f"警告: 文件 {os.path.basename(file_path)} 无颜色信息，跳过")
            return result

        colors = np.asarray(pcd.colors)
        colors_255 = (colors * 255).round().astype(int) if np.max(colors) <= 1.0 else colors.round().astype(int)

        is_black = np.all(colors_255 == 0, axis=1)
        result["black_points_count"] = int(np.sum(is_black))
        non_black_mask = ~is_black
        valid_colors_255 = colors_255[non_black_mask]

        if len(valid_colors_255) == 0:
            return result

        distance_threshold = np.sqrt(3 * color_tolerance ** 2)
        color_distances = np.sqrt(
            np.sum((valid_colors_255[:, None] - category_colors[None, :]) ** 2, axis=2)
        )
        min_distances = np.min(color_distances, axis=1)
        point_categories = np.where(
            min_distances <= distance_threshold,
            np.argmin(color_distances, axis=1),
            -1
        )

        valid_cats = point_categories[point_categories >= 0]
        result["category_counts"] = np.bincount(valid_cats, minlength=len(category_colors))
        result["classified_count"] = int(np.sum(result["category_counts"]))
        unclassified_mask = point_categories == -1
        result["unclassified_count"] = int(np.sum(unclassified_mask))
        result["other_rgb_counter"] = Counter(map(tuple, valid_colors_255[unclassified_mask].tolist()))
        result["success"] = True

    except Exception as e:
        print(f"处理文件 {os.path.basename(file_path)} 出错: {str(e)}")

    return result


def format_statistics(category_counts: dict,
                      total_points: int,
                      skipped_black: int,
                      skipped_other: int,
                      process_time: float) -> list[str]:
    """
    格式化统计结果
    """
    lines = [
        "-" * 60,
        "类别统计结果",
        "-" * 60,
        f"{'类别':<25} {'点数':<15} {'占比':<10}",
        "-" * 60,
    ]

    for category in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
        count = category_counts[category]
        if count == 0 and category != "其他":
            continue
        percentage = (count / total_points) * 100 if total_points > 0 else 0
        lines.append(f"{category:<25} {count:<15} {percentage:.2f}%")

    lines.extend([
        "-" * 60,
        f"有效点总数: {total_points}",
        f"跳过的黑色点总数: {skipped_black}",
        f"跳过的其他类别点总数: {skipped_other}",
        f"总处理时间: {process_time:.2f} 秒",
    ])
    return lines


def format_other_rgb_stats(other_rgb_counter: Counter, other_total: int) -> list[str]:
    """
    格式化其他类别RGB值统计
    """
    if other_total == 0:
        return []

    lines = [
        "-" * 60,
        "其他类别的RGB值统计 (前20个最常见)",
        "-" * 60,
        f"{'RGB值':<20} {'出现次数':<15} {'占比(%)':<10}",
        "-" * 60,
    ]

    for rgb, count in other_rgb_counter.most_common(20):
        display_rgb = tuple(int(x) for x in rgb)
        percentage = (count / other_total) * 100
        lines.append(f"{str(display_rgb):<20} {count:<15} {percentage:.2f}")
    return lines


def write_report(report_lines: list[str]) -> None:
    """
    将统计结果写入 info.txt
    """
    report_text = "\n".join(report_lines) + "\n"
    with open(INFO_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text, end="")
    print(f"\n统计结果已保存到: {INFO_TXT_PATH}")


def main(pcd_input: str):
    """
    主函数：直接处理单个全局彩色PCD文件并统计类别
    """
    start_time = time.time()
    pcd_file = resolve_single_pcd_file(pcd_input)
    if pcd_file is None:
        return

    category_colors, category_names, category_counts = load_category_config()
    print(f"使用RGB格式处理颜色，容差±{COLOR_TOLERANCE}")
    print("输入模式: 单个全局彩色点云")
    print(f"输入文件: {pcd_file}")

    result = process_single_pcd(pcd_file, category_colors, COLOR_TOLERANCE)
    if not result["success"]:
        print("\n未找到有效点云数据")
        return

    total_points = result["classified_count"]
    skipped_black_points = result["black_points_count"]
    skipped_other_points = result["unclassified_count"]

    for i, name in enumerate(category_names):
        category_counts[name] += int(result["category_counts"][i])
    category_counts["其他"] += result["unclassified_count"]

    process_time = time.time() - start_time
    report_lines = [
        f"输入文件: {pcd_file}",
        f"颜色容差: ±{COLOR_TOLERANCE}",
    ]
    report_lines.extend(format_statistics(
        category_counts,
        total_points,
        skipped_black_points,
        skipped_other_points,
        process_time
    ))
    other_rgb_lines = format_other_rgb_stats(result["other_rgb_counter"], category_counts["其他"])
    if other_rgb_lines:
        report_lines.append("")
        report_lines.extend(other_rgb_lines)
    write_report(report_lines)


if __name__ == "__main__":
    pcd_input = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MAP_PATH
    main(pcd_input)
