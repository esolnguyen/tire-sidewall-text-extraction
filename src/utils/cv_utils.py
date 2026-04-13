import json
import logging
import os
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def contour_to_bbox(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """Converts a contour to a bounding box (x1, y1, x2, y2)."""
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, x + w, y + h


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

    # Compute difference between points; top-right will have smallest diff,
    # bottom-left will have largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Applies a perspective warp to extract a quadrilateral region from an image.
    Input `pts` should be a 4x2 numpy array of (x, y) coordinates.
    """
    if pts.shape[0] != 4:
        # print(f"Warning: four_point_transform requires 4 points, got {pts.shape[0]}.")
        return None

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth <= 0 or maxHeight <= 0:
        # print("Warning: Degenerate polygon for four_point_transform resulted in zero width/height.")
        return None

    # Construct the destination points for the "birds-eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and warp the perspective
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from: {file_path}")
        return None


def save_json_file(data, file_path):
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created directory {output_dir}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to: {file_path}")
    except IOError as e:
        logger.error(f"Could not write data to {file_path}: {e}")
