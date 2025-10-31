import cv2
from PIL import Image
import numpy as np
from typing import List, Optional, Any
import torch
# 假设这些是你已有的导入
from detections import ObjectDetections
from yolov7 import YOLOv7Client
# from sam_segmentor import sam_segmentor # 假设你的SAM分割器在这里初始化

# --- 函数 1: 检测与过滤 ---
yolov7_detector = YOLOv7Client(port=12184)

##可选：将目标检测后的box传给mobile sam进行分割
# from sam import MobileSAMClient
# sam_segmentor = MobileSAMClient(port=12183)

def detect_and_filter_objects(
    image: np.ndarray,
    classes_to_detect: Optional[List[str]] = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    agnostic_nms: bool = False,
) -> ObjectDetections:
    """
    运行YOLOv7检测，并可选择性地只保留指定类别的物体。

    Args:
        yolov7_detector: 已初始化的YOLOv7检测器实例。
        image: 输入的Numpy图像 (H, W, C)。
        classes_to_detect: (可选) 一个字符串列表，指定只保留哪些类别。
                           例如: ["person", "sofa"]。
                           如果为 None，则返回所有检测到的物体。
        conf_thres: 置信度阈值。
        iou_thres: IOU阈值。
        agnostic_nms: 是否使用agnostic NMS。

    Returns:
        一个 ObjectDetections 对象，仅包含被过滤后的物体。
    """
    # 1. 运行YOLOv7检测，获取所有结果
    # 这一步返回的 all_detections.boxes 已经是 "xyxy" 格式
    all_detections = yolov7_detector.predict(
        image,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        agnostic_nms=agnostic_nms,
    )

    # 2. 如果不需要过滤 (classes_to_detect 为 None)，则直接返回所有结果
    if not classes_to_detect:
        return all_detections

    # 3. 如果需要过滤，则遍历所有检测结果
    target_classes_set = set(classes_to_detect)
    
    filtered_indices = []
    filtered_phrases = []

    for idx, phrase in enumerate(all_detections.phrases):
        if phrase in target_classes_set:
            filtered_indices.append(idx)
            filtered_phrases.append(phrase)

    if not filtered_indices:
        # --- FIX 1: 处理空检测 ---
        # 之前用了 np.empty，导致了 Tensor.unbind() 错误
        # 现在改用 torch.empty，并且必须指定 fmt="xyxy"
        empty_boxes = torch.empty((0, 4), dtype=torch.float32)
        empty_logits = torch.empty((0,), dtype=torch.float32)
        return ObjectDetections(
            boxes=empty_boxes,
            logits=empty_logits,
            phrases=[],
            image_source=image,
            fmt="xyxy"  # <--- 关键修正！
        )

    # 4. 根据过滤后的索引，创建新的 ObjectDetections 对象
    # (假设Detections.boxes 是一个 Tensor，支持索引)
    filtered_boxes = all_detections.boxes[filtered_indices]
    filtered_logits = all_detections.logits[filtered_indices]
    
    # --- FIX 2: 处理过滤后的检测 ---
    # 之前没有传递 fmt，导致 __init__ 默认 fmt="cxcywh" 并错误调用 box_convert
    filtered_detections = ObjectDetections(
        boxes=filtered_boxes,
        logits=filtered_logits,
        phrases=filtered_phrases,
        image_source=all_detections.image_source,
        fmt="xyxy"  # <--- 关键修正！
    )
    
    return filtered_detections

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


# --- 函数 3: 可视化绘制 ---

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
    img = Image.open("sheep.png")

# 确保图片是RGB格式
# 如果原始图片是RGBA（带透明通道）或L（灰度）等，
# convert('RGB') 会将其转换为标准的RGB格式。
    img_rgb = img.convert('RGB')
# 将PIL Image对象转换为Numpy数组
    img = np.array(img_rgb)

    print("running main")


    # --- 场景 1: 只想检测 "sheep"，并获取语义分割后的图像 ---

    # 1. 只检测sheep
    print("正在检测 'sheep'...")
    detections_filtered = detect_and_filter_objects(
        img,
        classes_to_detect=["sheep"],
        conf_thres=0.3 # cfg.yolo.confidence_threshold_yolo
    )

    # 2. 获取这些物体的Masks
    print(f"检测到 {len(detections_filtered.boxes)} 个目标, 正在分割...")
    if len(detections_filtered.boxes) > 0:
        # object_masks = get_object_masks(
        #     sam_segmentor,
        #     img,
        #     detections_filtered
        # )
        #如果需要分割，则把下面的masks=None改成masks=object_masks
        
        # 3. 绘制结果
        segmented_image = draw_detections(
            img,
            detections_filtered,
            masks=None,
        )
        segmented_image = segmented_image[:, :, ::-1]
        cv2.imwrite("sheep_boxed.jpg", segmented_image)
        print("已保存分割图像。")


    # --- 场景 2: 检测所有COCO物体 ---

    print("正在检测所有物体...")
    # 1. 检测所有物体 (classes_to_detect=None)
    all_detections = detect_and_filter_objects(
        img,
        conf_thres=0.25
    )
    
    # 2. 只获取Masks
    print(f"检测到 {len(all_detections.boxes)} 个物体...")
    if len(all_detections.boxes) > 0:
        # all_masks = get_object_masks(
        #     sam_segmentor,
        #     img,
        #     all_detections
        # )
        #如果需要分割，则把下面的masks=None改成masks=all_masks
        
        # 3. 绘制结果
        segmented_image = draw_detections(
            img,
            all_detections,
            masks=None,
        )
        segmented_image = segmented_image[:, :, ::-1]
        cv2.imwrite("boxed.jpg", segmented_image)
        print("已保存分割图像。")
    

#         print(f"成功获取 {len(all_masks)} 个Masks。")
#         # all_masks 是一个 [mask1, mask2, ...] 的列表，你可以单独处理它们






#     # --- 场景 3: 只检测 "person"，并只绘制BBox（不分割） ---

#     print("正在检测 'person'...")
#     # 1. 只检测人
#     detections_person = detect_and_filter_objects(
#         yolov7_detector,
#         img,
#         classes_to_detect=["person"],
#         conf_thres=0.5
#     )

#     # 2. 绘制结果 (不传入 'masks' 参数)
#     print(f"检测到 {len(detections_person.boxes)} 个 'person'...")
#     bbox_image = draw_detections(
#         img,
#         detections_person,
#         masks=None, # 不传入Masks
#         default_color=(255, 0, 0) # 蓝色
#     )
#     cv2.imwrite("person_bboxes.jpg", bbox_image)
#     print("已保存BBox图像。")