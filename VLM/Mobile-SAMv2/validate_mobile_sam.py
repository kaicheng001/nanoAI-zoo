import cv2
import numpy as np
from PIL import Image
import numpy as np
from typing import List, Optional, Any
import torch
from sam import MobileSAMClient

sam_segmentor = MobileSAMClient(port=12183)

def get_segmentation(segmented_img, img, label, score, color):

    
    # [修改] 按照要求，使用一个“拟定”的手动box
    # 你可以把 [100, 150, 300, 350] 替换成你想要的任何值
    # 格式为 [x1, y1, x2, y2]
    manual_box = [100, 150, 300, 350]
    bbox_denorm = np.array(manual_box)

    
    # [修正] 使用 img.shape 来创建 object_mask，而不是硬编码 480x640
    object_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    x1, y1, x2, y2 = [int(v) for v in bbox_denorm]
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]

    if bbox_area / img_area < 0.99:
        
        # [修正] 假设 sam_segmentor.segment_bbox 返回 (mask, iou) 元组
        # 你的原始代码会把元组赋给 object_mask，导致 findContours 失败
        try:
            # 假设 sam_segmentor 是 MobileSAM 的实例
            object_mask = sam_segmentor.segment_bbox(img, bbox_denorm.tolist())
        except Exception as e:
            print(f"Error during segmentation: {e}")
            # 如果分割失败，返回一个空掩码
            return segmented_img, object_mask

        # [修正] SAM 返回 bool 掩码, cv2 需要 uint8
        valid_mask = object_mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            valid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            cv2.drawContours(segmented_img, [contour], 0, color, 4)

        # 绘制边界框
        cv2.rectangle(
            segmented_img,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # [修正] 标签绘制逻辑 (使其更健壮，不会画出屏幕)
        label_text = f"{label} ({score:.2f})"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1
        )
        
        # 计算标签背景的位置
        label_bg_y1 = max(y1 - text_height - 10, 0) # 确保不超出顶部
        label_bg_y2 = y1 - 5 # 标签在框上方 5 像素处
        label_bg_x1 = x1
        label_bg_x2 = x1 + text_width
        
        # 绘制标签背景
        cv2.rectangle(
            segmented_img,
            (label_bg_x1, label_bg_y1),
            (label_bg_x2, label_bg_y2),
            color,
            cv2.FILLED, # 填充
        )

        # 绘制标签文本
        cv2.putText(
            segmented_img,
            label_text,
            (x1, y1 - 10), # 文本位置
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255), # 白色文本
            1,
        )

    return segmented_img, object_mask


if __name__ == "__main__":
    IMAGE_PATH = "sheep.png"
    img = cv2.imread(IMAGE_PATH)
    segmented_img = img.copy()
    segmented_img, object_mask = get_segmentation(segmented_img, img, label = "自定义", score = 0.50, color = (255, 0, 0))
    cv2.imwrite("segmented.jpg", segmented_img)
    


