import cv2
import numpy as np
from typing import Dict, Tuple
from class_statics_config import ADE20K_CATEGORIES

def create_ade20k_color_card_5col(
    categories: Dict[str, Tuple[int, int, int]],
    output_path: str = "ade20k_color_card.png",
    item_width: int = 350,    # 每列宽度
    item_height: int = 50,    # 每个类别的高度
    font_scale: float = 0.6,
    font_thickness: int = 1,
    margin: int = 20,
    cols: int = 5             # 固定为5列
) -> None:
    """生成三列布局的ADE20K色卡（解决显示不全+排版紧凑）"""
    # 按字母排序类别（便于对齐）
    sorted_categories = sorted(categories.items())
    num_items = len(sorted_categories)
    
    # 计算行列数
    rows = (num_items + cols - 1) // cols  # 向上取整
    
    # 计算图片尺寸
    img_width = cols * item_width + (cols + 1) * margin
    img_height = rows * item_height + margin * 2 + 700  # 预留标题高度
    
    # 创建白色背景
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 遍历每个类别，按三列网格布局绘制
    for idx, (name, rgb) in enumerate(sorted_categories):
        # 计算当前类别所在的列和行
        col_idx = idx % cols
        row_idx = idx // cols
        
        # 计算位置（颜色块+文字）
        x_start = margin + (item_width + margin) * col_idx
        y_start = margin + 50 + (item_height + margin) * row_idx  # 50是标题高度
        y_end = y_start + item_height

        # 1. 绘制颜色块（RGB转BGR）
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(
            img,
            (x_start, y_start),
            (x_start + 60, y_end),  # 颜色块宽度固定60
            bgr,
            -1
        )

        # 2. 绘制类别名称
        text_name = f"{name}"
        cv2.putText(
            img,
            text_name,
            (x_start + 70, y_start + item_height//2 + 8),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA
        )

        # 3. 绘制完整RGB值（解决显示不全）
        text_rgb = f"RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}"
        cv2.putText(
            img,
            text_rgb,
            (x_start + 70, y_start + item_height//2 + 30),
            font,
            font_scale * 0.8,
            (50, 50, 50),
            font_thickness,
            cv2.LINE_AA
        )

    # 添加标题
    cv2.putText(
        img,
        "ADE20K Category Color Card",
        (margin, margin + 30),
        cv2.FONT_HERSHEY_COMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # 保存图片
    cv2.imwrite(output_path, img)
    print(f"五列色卡已生成：{output_path}")

if __name__ == "__main__":
    create_ade20k_color_card_5col(ADE20K_CATEGORIES)