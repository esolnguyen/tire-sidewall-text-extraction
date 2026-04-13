"""Shared evaluation utilities for detection evaluation scripts."""

from typing import List, Tuple

import numpy as np

BoundingBox = List[float]


def polygon_to_bbox(polygon: List[float]) -> BoundingBox:
    if not polygon:
        return [0, 0, 0, 0]
    xs = polygon[::2]
    ys = polygon[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def polygons_to_bboxes(polygons: List[List[float]]) -> List[BoundingBox]:
    return [polygon_to_bbox(p) for p in polygons]


def xywh_to_xyxy(bbox: List[float]) -> BoundingBox:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_width = max(0, xi_max - xi_min)
    inter_height = max(0, yi_max - yi_min)
    inter_area = inter_width * inter_height

    area1 = max(0.0, (x1_max - x1_min)) * max(0.0, (y1_max - y1_min))
    area2 = max(0.0, (x2_max - x2_min)) * max(0.0, (y2_max - y2_min))
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_detections(
    gt_boxes: List[BoundingBox],
    pred_boxes: List[BoundingBox],
    iou_threshold: float,
) -> Tuple[int, int, int]:
    """Class-agnostic matching. Returns (TP, FP, FN)."""
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)
    if not gt_boxes:
        return 0, len(pred_boxes), 0

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred_box, gt_box)

    matched_gt_indices = set()
    true_positives = 0

    for i in range(len(pred_boxes)):
        best_gt_idx = -1
        max_iou = iou_threshold
        for j in range(len(gt_boxes)):
            if j not in matched_gt_indices and iou_matrix[i, j] > max_iou:
                max_iou = iou_matrix[i, j]
                best_gt_idx = j
        if best_gt_idx != -1:
            true_positives += 1
            matched_gt_indices.add(best_gt_idx)

    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives
    return true_positives, false_positives, false_negatives
