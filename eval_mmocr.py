import argparse
import csv
import json
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')

DEFAULT_RESIZE_SIZE = (640, 6400)

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

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def match_detections(
        gt_boxes: List[BoundingBox], pred_boxes: List[BoundingBox], iou_threshold: float
) -> Tuple[int, int, int]:
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)
    if not gt_boxes:
        return 0, len(pred_boxes), 0

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
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


def load_annotations(annotations_file: str) -> Tuple[Dict[int, List[BoundingBox]], List[Dict]]:
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    gt_map = defaultdict(list)
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        bbox_xyxy = xywh_to_xyxy(ann["bbox"])
        gt_map[image_id].append(bbox_xyxy)

    return gt_map, coco_data["images"]


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


class MMOCR_Predictor:
    def __init__(self, cfg_file: str, checkpoint: str, device: str):
        from mmocr.apis import TextDetInferencer
        print("Initializing MMOCR inferencer...")
        self.inferencer = TextDetInferencer(
            model=cfg_file, weights=checkpoint, device=device)

    def __call__(self, image_path: Path) -> List[BoundingBox]:
        result = self.inferencer(str(image_path), return_vis=False)
        pred_polygons = result["predictions"][0].get("polygons", [])
        pred_bboxes = []
        for poly in pred_polygons:
            if len(poly) >= 4:
                xs = poly[::2]
                ys = poly[1::2]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                pred_bboxes.append([x_min, y_min, x_max, y_max])
        return pred_bboxes


def evaluate(
        predictor,
        data_dir: str,
        annotations_file: str,
        iou_threshold: float,
) -> Dict[str, Any]:
    gt_bboxes_map, images_info = load_annotations(annotations_file)

    total_tp, total_fp, total_fn = 0, 0, 0
    total_boxes = 0

    per_image_rows = []
    t0 = time.time()

    for image_info in tqdm(images_info, desc="Evaluating Images"):
        image_id = image_info["id"]
        file_path = f"{data_dir}/{image_info['file_name']}"

        if not os.path.exists(file_path):
            print(f"Warning: Image file not found, skipping: {file_path}")
            continue

        pred_boxes = predictor(file_path)
        gt_boxes = gt_bboxes_map.get(image_id, [])
        total_boxes += len(gt_boxes)

        tp, fp, fn = match_detections(gt_boxes, pred_boxes, iou_threshold)
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


def print_results(metrics: Dict[str, Any], iou_threshold: float):
    """Prints evaluation results in a formatted table."""
    print("\n" + "=" * 38)
    print("         EVALUATION RESULTS")
    print("=" * 38)
    print(f"Eval IoU Threshold:   {iou_threshold:.2f}")
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
    config: str,
    checkpoint: str,
    data_dir: str,
    ann_file: str,
    iou: float,
    output_csv: str,
    device: str
):
    if device:
        device = device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    predictor = MMOCR_Predictor(config, checkpoint, device)

    metrics = evaluate(predictor, data_dir, ann_file, iou)

    # Save per-image metrics to CSV
    csv_filename = f"textdet_eval_mmocr.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        if metrics['per_image_rows']:
            fieldnames = metrics['per_image_rows'][0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics['per_image_rows'])
    print(f"\nPer-image metrics saved to: {csv_filename}")

    print_results(metrics, iou)

    if output_csv:
        results_data = {
            "model": Path(checkpoint).name,
            "iou_threshold": iou,
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
    model = {
        "DBNetPP": {
            "config": "mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py",
            "checkpoint": "models/mmocr/DBNetPP.pth",
        },
        "DRRG": {
            "config": "mmocr/configs/textdet/drrg/drrg_resnet50-oclip_fpn-unet_1200e_ctw1500.py",
            "checkpoint": "models/mmocr/DRRG.pth",
        },
        "PANNet": {
            "config": "mmocr/configs/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_ctw1500.py",
            "checkpoint": "models/mmocr/PANNet.pth",
        }
    }
    run_evaluation(
        config="mmocr/configs/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500.py",
        checkpoint="models/mmocr/TextSnake.pth",
        data_dir="tire_dataset/test",
        ann_file="tire_dataset/test.json",
        iou=0.5,
        output_csv="data/textdet_mmocr_eval_results.csv",
        device="cpu"
    )
