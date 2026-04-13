import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
import csv

from eval_utils import (
    BoundingBox,
    polygon_to_bbox,
    polygons_to_bboxes,
    xywh_to_xyxy,
    compute_iou,
    match_detections,
)

warnings.filterwarnings('ignore')

DEFAULT_RESIZE_SIZE = (640, 6400)


def match_detections_class_aware(
    gt_boxes: List[BoundingBox],
    gt_classes: List[int],
    pred_boxes: List[BoundingBox],
    pred_classes: List[int],
    iou_threshold: float,
) -> Tuple[int, int, int]:
    """Only match boxes if IoU passes AND classes match."""
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)
    if not gt_boxes:
        return 0, len(pred_boxes), 0

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, pbox in enumerate(pred_boxes):
        for j, gbox in enumerate(gt_boxes):
            if pred_classes[i] != gt_classes[j]:
                iou_matrix[i, j] = 0.0
            else:
                iou_matrix[i, j] = compute_iou(pbox, gbox)

    matched_gt = set()
    tp = 0
    for i in range(len(pred_boxes)):
        best_j = -1
        best_iou = iou_threshold
        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue
            if iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        if best_j != -1:
            tp += 1
            matched_gt.add(best_j)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def load_annotations(annotations_file: str) -> Tuple[Dict[int, List[BoundingBox]], List[Dict], Dict[int, int]]:
    """
    Returns:
      gt_map: image_id -> list of xyxy boxes
      images: the images array from COCO json
      gt_cls_map: image_id -> list of category_ids aligned with gt_map order
    """
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    gt_map = defaultdict(list)
    gt_cls_map = defaultdict(list)
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        bbox_xyxy = xywh_to_xyxy(ann["bbox"])
        gt_map[image_id].append(bbox_xyxy)
        gt_cls_map[image_id].append(ann.get("category_id", 0))

    return gt_map, coco_data["images"], gt_cls_map


def append_results_to_csv(csv_path: Path, results: Dict[str, Any]):
    file_exists = csv_path.is_file()
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
        print(f"Successfully appended results to {csv_path}")
    except IOError as e:
        print(f"Error: Could not write to CSV file {csv_path}: {e}")


class YOLO_Predictor:
    def __init__(self, checkpoint: str, device: str, conf: float = 0.25, nms_iou: float = 0.7, imgsz: Optional[Tuple[int, int]] = None, rect: bool = True):
        print("Initializing YOLO model...")
        self.model = YOLO(checkpoint)
        self.device = device
        self.conf = conf
        self.nms_iou = nms_iou
        self.imgsz = imgsz
        self.rect = rect

    def __call__(self, image_path: str):
        """
        Returns:
          pred_bboxes: List[xyxy]
          pred_classes: List[int]
          result: Ultralytics Result
        """
        results = self.model.predict(
            source=str(image_path),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.nms_iou,
            rect=self.rect,
            device=self.device
        )
        r = results[0]

        pred_bboxes: List[BoundingBox] = []
        pred_classes: List[int] = []

        # Segmentation predictions (instance segmentation masks)
        if getattr(r, "masks", None) is not None and r.masks is not None:
            # Method 1: Extract bounding boxes from segmentation masks
            # r.masks.xy contains polygon coordinates for each mask
            # r.masks.data contains binary masks (H, W) for each instance
            if hasattr(r.masks, 'xy') and r.masks.xy is not None:
                # Polygon-based segmentation (e.g., SAM, YOLOv8-seg)
                # List of numpy arrays, each shape (N, 2)
                polygons = r.masks.xy
                cls = r.boxes.cls.detach().cpu().numpy().astype(
                    int) if r.boxes is not None and r.boxes.cls is not None else np.zeros(len(polygons), dtype=int)

                for k, poly in enumerate(polygons):
                    if len(poly) > 0:
                        xs = poly[:, 0]
                        ys = poly[:, 1]
                        x_min, y_min, x_max, y_max = float(xs.min()), float(
                            ys.min()), float(xs.max()), float(ys.max())
                        pred_bboxes.append([x_min, y_min, x_max, y_max])
                        pred_classes.append(int(cls[k]) if k < len(cls) else 0)

            elif hasattr(r.masks, 'data') and r.masks.data is not None:
                # Binary mask-based segmentation
                masks = r.masks.data.detach().cpu().numpy()  # Shape: (N, H, W)
                cls = r.boxes.cls.detach().cpu().numpy().astype(
                    int) if r.boxes is not None and r.boxes.cls is not None else np.zeros(len(masks), dtype=int)

                for k, mask in enumerate(masks):
                    # Find bounding box from binary mask
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if rows.any() and cols.any():
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        pred_bboxes.append([float(x_min), float(
                            y_min), float(x_max), float(y_max)])
                        pred_classes.append(int(cls[k]) if k < len(cls) else 0)

            # If masks exist but we also have boxes, prefer using the boxes directly
            if not pred_bboxes and r.boxes is not None and r.boxes.xyxy is not None:
                boxes = r.boxes.xyxy.detach().cpu().numpy()
                cls = r.boxes.cls.detach().cpu().numpy().astype(
                    int) if r.boxes.cls is not None else np.zeros(len(boxes), dtype=int)
                for k, box in enumerate(boxes):
                    pred_bboxes.append([float(box[0]), float(
                        box[1]), float(box[2]), float(box[3])])
                    pred_classes.append(int(cls[k]))

        # OBB predictions (Oriented Bounding Boxes)
        elif getattr(r, "obb", None) is not None and r.obb is not None and r.obb.xyxyxyxy is not None:
            polys = r.obb.xyxyxyxy.detach().cpu().numpy()
            cls = r.obb.cls.detach().cpu().numpy().astype(
                int) if r.obb.cls is not None else np.zeros(len(polys), dtype=int)
            for k, poly in enumerate(polys):
                xs = poly[:, 0]
                ys = poly[:, 1]
                x_min, y_min, x_max, y_max = float(xs.min()), float(
                    ys.min()), float(xs.max()), float(ys.max())
                pred_bboxes.append([x_min, y_min, x_max, y_max])
                pred_classes.append(int(cls[k]))

        # Axis-aligned boxes (standard object detection)
        elif getattr(r, "boxes", None) is not None and r.boxes is not None and r.boxes.xyxy is not None:
            boxes = r.boxes.xyxy.detach().cpu().numpy()
            cls = r.boxes.cls.detach().cpu().numpy().astype(
                int) if r.boxes.cls is not None else np.zeros(len(boxes), dtype=int)
            for k, box in enumerate(boxes):
                pred_bboxes.append([float(box[0]), float(
                    box[1]), float(box[2]), float(box[3])])
                pred_classes.append(int(cls[k]))

        return pred_bboxes, pred_classes, r


def evaluate(
    predictor: YOLO_Predictor,
    data_dir: str,
    annotations_file: str,
    eval_iou_threshold: float,
    use_class: bool = False,
) -> Dict[str, Any]:
    gt_bboxes_map, images_info, gt_cls_map = load_annotations(annotations_file)

    total_tp, total_fp, total_fn = 0, 0, 0
    total_boxes = 0

    dummy = torch.randn(1, 3, 640, 6400).to("cpu")

    for _ in range(5):
        _ = predictor.model(dummy)

    per_image_rows = []
    t0 = time.time()

    for image_info in tqdm(images_info, desc="Evaluating Images"):
        image_id = image_info["id"]
        file_path = f"{data_dir}/{image_info['file_name']}"

        if not os.path.exists(file_path):
            print(f"Warning: Image file not found, skipping: {file_path}")
            continue

        pred_boxes, pred_classes, _ = predictor(file_path)
        gt_boxes = gt_bboxes_map.get(image_id, [])
        gt_classes = gt_cls_map.get(image_id, [])
        total_boxes += len(gt_boxes)

        if use_class:
            tp, fp, fn = match_detections_class_aware(
                gt_boxes, gt_classes, pred_boxes, pred_classes, eval_iou_threshold
            )
        else:
            tp, fp, fn = match_detections(
                gt_boxes, pred_boxes, eval_iou_threshold)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_image_rows.append({
            "image_id": image_id,
            "file_name": image_info['file_name'],
            "gt": len(gt_boxes),
            "pred": len(pred_boxes),
            "tp": tp, "fp": fp, "fn": fn
        })

    elapsed = time.time() - t0
    images_done = len(per_image_rows)
    ips = images_done / elapsed if elapsed > 0 else 0.0

    precision = total_tp / \
        (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / \
        (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0.0
    accuracy = total_tp / total_boxes if total_boxes > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "images": images_done,
        "time_sec": elapsed,
        "images_per_sec": ips,
        "per_image_rows": per_image_rows,
    }


def print_results(metrics: Dict[str, Any], iou_threshold: float, use_class: bool):
    """Prints evaluation results in a formatted table."""
    print("\n" + "=" * 38)
    print("         EVALUATION RESULTS")
    print("=" * 38)
    print(f"Eval IoU Threshold:   {iou_threshold:.2f}")
    print(f"Class-aware match:    {use_class}")
    print("-" * 38)
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"Precision:            {metrics['precision']:.4f}")
    print(f"Recall:               {metrics['recall']:.4f}")
    print(f"F1-Score:             {metrics['f1_score']:.4f}")
    print("-" * 38)
    print(f"True Positives:       {metrics['true_positives']}")
    print(f"False Positives:      {metrics['false_positives']}")
    print(f"False Negatives:      {metrics['false_negatives']}")
    print("-" * 38)
    print(f"Images:               {metrics['images']}")
    print(f"Time (s):             {metrics['time_sec']:.2f}")
    print(f"Images/sec:           {metrics['images_per_sec']:.2f}")
    print("=" * 38)


def run_evaluation(
    checkpoint: str,
    data_dir: str,
    ann_file: str,
    eval_iou: float,
    device: Optional[str],
    conf: float,
    nms_iou: float,
    use_class: bool,
    imgsz_w: Optional[int],
    imgsz_h: Optional[int],
    output_csv: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    imgsz = (imgsz_h, imgsz_w) if (imgsz_w and imgsz_h) else None

    predictor = YOLO_Predictor(
        checkpoint=checkpoint,
        device=device,
        conf=conf,
        nms_iou=nms_iou,
        imgsz=imgsz,
        rect=True
    )

    metrics = evaluate(
        predictor=predictor,
        data_dir=data_dir,
        annotations_file=ann_file,
        eval_iou_threshold=eval_iou,
        use_class=use_class,
    )

    # Save per-image metrics to CSV
    csv_filename = f"textdet_eval.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        if metrics['per_image_rows']:
            fieldnames = metrics['per_image_rows'][0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics['per_image_rows'])
    print(f"\nPer-image metrics saved to: {csv_filename}")

    print_results(metrics, eval_iou, use_class)

    if output_csv:
        results_data = {
            "model": "YOLOv11n-OCiLP-ASPP-P3",
            "iou_threshold": eval_iou,
            "accuracy": f"{metrics['accuracy']:.4f}",
            "precision": f"{metrics['precision']:.4f}",
            "recall": f"{metrics['recall']:.4f}",
            "f1_score": f"{metrics['f1_score']:.4f}",
            "images": metrics['images'],
            "time_sec": f"{metrics['time_sec']:.2f}",
            "images_per_sec": f"{metrics['images_per_sec']:.2f}",
        }
        append_results_to_csv(Path(output_csv), results_data)


if __name__ == "__main__":
    run_evaluation(
        checkpoint="eval_results/TextDet/YOLOv11n-OCiLP-QG-ASPP-C5_E100/weights/yolov11n_oclip_asp_c4_sppf2.pt",
        data_dir="data/tire_dataset/test",
        ann_file="data/tire_dataset/test.json",
        eval_iou=0.5,
        device="cpu",
        conf=0.25,
        nms_iou=0.7,
        use_class=False,
        imgsz_w=3200,
        imgsz_h=640,
        output_csv="data/textdet_mmocr_eval_results.csv"
    )
