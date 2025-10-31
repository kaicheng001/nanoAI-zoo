import cv2
import numpy as np
from coco_classes import COCO_CLASSES
from PIL import Image
import numpy as np
from typing import List, Optional, Any
import torch
from sam import MobileSAMClient
# 假设这些是你已有的导入
from detections import ObjectDetections 
from grounding_dino import GroundingDINOClient

dino_detector = GroundingDINOClient(port=12181)
##可选：将目标检测后的box传给mobile sam进行分割
# from sam import MobileSAMClient
# sam_segmentor = MobileSAMClient(port=12183)


def detect_with_grounding_dino(
    dino_detector: Any,  # 传入你的 dino_detector 实例
    image: np.ndarray,
    classes_to_detect: str, # DINO 必须指定要检测的类别
    box_threshold: float,
    text_threshold: float
) -> ObjectDetections:
    """
    使用 Grounding DINO 检测一组指定的自定义类别。

    与 YOLO 不同，DINO 是一种开集检测器。
    它只会检测 'classes_to_detect' 中指定的类别。

    Args:
        dino_detector: 已初始化的 Grounding DINO 检测器实例。
        image: 输入的Numpy图像 (H, W, C)。
        classes_to_detect: (必需) 一个字符串列表，指定要检测的类别。
                           例如: ["person", "sofa", "a red box"]。
        box_threshold: DINO 的 BBox 置信度阈值。
        text_threshold: DINO 的文本-图像对齐阈值。

    Returns:
        一个 ObjectDEtections 对象，仅包含检测到的物体。
    """
    
    # 1. 检查输入：DINO 必须有类别才能检测
    if not classes_to_detect:
        print("警告: Grounding DINO 必须提供 'classes_to_detect' 列表。返回空检测。")
        # 返回一个标准的空 ObjectDetections 对象
        empty_boxes = torch.empty((0, 4), dtype=torch.float32)
        empty_logits = torch.empty((0,), dtype=torch.float32)
        return ObjectDetections(
            boxes=empty_boxes,
            logits=empty_logits,
            phrases=[],
            image_source=image,
            fmt="xyxy"
        )

    # 2. 构建 Grounding DINO 的 caption
    # 经典的 DINO caption 格式是用 " . " 分隔
    caption = classes_to_detect

    # 3. 运行 DINO 检测
    # 假设你的 dino_detector.predict(...) 已经
    # 按照约定返回了一个 ObjectDetections 对象。
    detections = dino_detector.predict(
        image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    # 4. 直接返回结果
    # DINO 的 `predict` 已经完成了所有工作，不需要额外的 Python 过滤。
    return detections


# --- 函数 2: 批量获取Masks ---

def _segment_single_bbox(sam_segmentor: Any, image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    (辅助函数) 从你的旧代码中提取的单个物体分割逻辑。
    
    注意: 'sam_segmentor' 是一个假设的对象，你需要将其传入。
    """
    object_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # 检查BBox面积，避免分割整个图像
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = image.shape[0] * image.shape[1]

    if bbox_area / img_area < 0.99:
        # 假设 sam_segmentor 有一个 .segment_bbox 方法
        object_mask = sam_segmentor.segment_bbox(image, bbox)
    
    return object_mask

def get_object_masks(
    sam_segmentor: Any,
    image: np.ndarray,
    detections: ObjectDetections
) -> List[np.ndarray]:
    """
    为 'detections' 中的每一个物体生成一个分割mask。

    Args:
        sam_segmentor: 你用来分割的分割器实例 (例如 SAM)。
        image: 原始图像。
        detections: YOLOv7返回的 ObjectDetections 对象。

    Returns:
        一个Numpy掩码(mask)的列表，每个掩码对应一个detection。
    """
    masks_list = []
    
    # 获取图像尺寸用于反归一化
    img_h, img_w = image.shape[:2]
    
    for box in detections.boxes:
        # 1. 反归一化 (Denormalize) BBox 坐标
        # 你的YOLOv7 predict函数最后几行在归一化，所以这里要反归一化
        bbox_denorm = box * np.array([img_w, img_h, img_w, img_h])
        bbox_int = [int(v) for v in bbox_denorm]

        # 2. 为每个BBox生成Mask
        object_mask = _segment_single_bbox(sam_segmentor, image, bbox_int)
        masks_list.append(object_mask)
        
    return masks_list



def draw_detections(
    image: np.ndarray,
    detections: ObjectDetections,
    masks: Optional[List[np.ndarray]] = None,
    box_color: tuple = (255, 0, 0),    # BGR: 蓝色, 用于"普通"的框和标签
    contour_color: tuple = (0, 255, 0) # BGR: 绿色, 专门用于掩码
) -> np.ndarray:
    """
    在图像上绘制检测框、标签，以及可选的分割掩码。

    Args:
        image: 要绘制的原始图像。
        detections: ObjectDetections 对象。
        masks: (可选) 一个与detections对应的Numpy掩码列表。
        box_color: (可选) 绘制BBox和标签背景的颜色。
        contour_color: (可选) 绘制掩码轮廓的颜色。

    Returns:
        一个绘制了所有可视化结果的新图像。
    """
    # 创建一个图像副本进行绘制，不修改原图
    vis_image = image.copy()
    img_h, img_w = vis_image.shape[:2]

    # 确保 masks 列表和 detections 列表长度一致
    if masks and len(masks) != len(detections.boxes):
        print("警告: 掩码数量与检测数量不匹配。将忽略掩码。")
        masks = None

    for idx in range(len(detections.boxes)):
        # 1. 获取检测信息
        box = detections.boxes[idx]
        score = detections.logits[idx].item()
        label = detections.phrases[idx]
        # 注意: 'color' 变量被移除了，我们现在直接使用 box_color 和 contour_color
        
        # 2. 反归一化BBox
        bbox_denorm = box * np.array([img_w, img_h, img_w, img_h])
        x1, y1, x2, y2 = [int(v) for v in bbox_denorm]

        # 3. (可选) 绘制分割掩码的轮廓
        if masks:
            object_mask = masks[idx]
            # 查找轮廓并绘制
            contours, _ = cv2.findContours(
                object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                # *** 修改: 只对轮廓使用 contour_color ***
                cv2.drawContours(vis_image, [contour], 0, contour_color, 4)

        # 4. 绘制 Bounding Box
        # *** 修改: 对BBox使用 box_color ***
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), box_color, 2)

        # 5. 绘制标签和置信度
        label_text = f"{label} ({score:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
        )
        label_y = y1 - 10 # 留出一点空间
        
        # 绘制标签背景
        # *** 修改: 对标签背景使用 box_color ***
        cv2.rectangle(
            vis_image,
            (x1, label_y - text_height - 5),
            (x1 + text_width, label_y + 5),
            box_color, # 使用 box_color
            cv2.FILLED
        )
        
        # 绘制标签文字
        cv2.putText(
            vis_image,
            label_text,
            (x1, label_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255), # 白色文字
            1,
        )

    return vis_image



if __name__ == "__main__":
    img = Image.open("RR.png")
    # 确保图片是RGB格式
#    如果原始图片是RGBA（带透明通道）或L（灰度）等，
#    .convert('RGB') 会将其转换为标准的RGB格式。
    img_rgb = img.convert('RGB')
# 将PIL Image对象转换为Numpy数组
    img = np.array(img_rgb)
    
    CLASSES = "flower ."
    detections = detect_with_grounding_dino(
        dino_detector=dino_detector, # 传入你的DINO检测器
        image=img,
        classes_to_detect=CLASSES,
        box_threshold=0.35,
        text_threshold=0.25
    )
    print(f"检测到 {len(detections.boxes)} 个目标, 正在分割...")
    if len(detections.boxes) > 0:
        # object_masks = get_object_masks(
        #     sam_segmentor,
        #     img,
        #     detections_filtered
        # )
        #如果需要分割，则把下面的masks=None改成masks=object_masks
        
        # 3. 绘制结果
        boxed_image = draw_detections(
            img,
            detections,
            masks=None,
        )
        boxed_image = boxed_image[:, :, ::-1]
        cv2.imwrite("dino_boxed.jpg", boxed_image)
        print("已保存分割图像。")
    