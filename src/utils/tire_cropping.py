import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO

from config import TireExtractionConfig
from utils.cv_utils import contour_to_bbox

logger = logging.getLogger(__name__)


def estimate_wheel_parameters(wheel_bbox, rim_bbox):
    if not wheel_bbox or not rim_bbox:
        logger.warning("Invalid bounding boxes provided.")
        return None

    wheel_x_min, wheel_y_min, wheel_x_max, wheel_y_max = wheel_bbox
    cx = (wheel_x_min + wheel_x_max) / 2
    cy = (wheel_y_min + wheel_y_max) / 2

    wheel_width = wheel_x_max - wheel_x_min
    wheel_height = wheel_y_max - wheel_y_min
    r_outer = (wheel_width + wheel_height) / 4

    rim_x_min, rim_y_min, rim_x_max, rim_y_max = rim_bbox
    rim_width = rim_x_max - rim_x_min
    rim_height = rim_y_max - rim_y_min
    r_inner = (rim_width + rim_height) / 4
    r_inner *= 0.8
    if r_inner >= r_outer or r_inner <= 0 or r_outer <= 0:
        logger.warning(
            f"Estimated radii are invalid (r_inner={r_inner}, r_outer={r_outer}).")
        return None

    return cx, cy, r_inner, r_outer


def flatten_sidewall(
        image: np.ndarray,
        cx: float,
        cy: float,
        r_inner: float,
        r_outer: float,
        output_height: int,
        output_width: Optional[int] = None,
        max_output_width: Optional[int] = None,
        interpolation=cv2.INTER_LINEAR,
        angle_offset_degrees: float = 0,
        angle_crop_percent: float = 0.1,
) -> np.ndarray:
    """Flatten a circular tire sidewall into a rectangular strip."""
    if r_inner >= r_outer or r_inner <= 0 or r_outer <= 0:
        return np.array([])
    if output_height <= 0:
        return np.array([])

    _actual_output_width: int

    if output_width is not None:
        if output_width <= 0:
            return np.array([])
        _actual_output_width = output_width
    else:
        avg_radius = (r_inner + r_outer) / 2
        effective_angle_proportion = 1.0 - angle_crop_percent
        if effective_angle_proportion <= 0:
            return np.array([])

        calculated_width = int(2 * np.pi * avg_radius * effective_angle_proportion)
        if calculated_width <= 0:
            return np.array([])
        _actual_output_width = calculated_width

    if max_output_width is not None and max_output_width > 0:
        _actual_output_width = min(_actual_output_width, max_output_width)

    if _actual_output_width <= 0:
        return np.array([])

    row_indices = np.arange(output_height, dtype=np.float32)
    radial_range = r_outer - r_inner
    if radial_range <= 0:
        return np.array([])

    radial_bin_width = radial_range / output_height
    radius = r_inner + (row_indices + 0.5) * radial_bin_width

    angle_offset_rad = np.deg2rad(angle_offset_degrees)
    total_angle_rad = 2 * np.pi * (1.0 - angle_crop_percent)
    if total_angle_rad <= 0:
        return np.array([])

    theta_start = angle_offset_rad + (2 * np.pi * angle_crop_percent / 2.0)
    theta_end = theta_start + total_angle_rad
    theta = np.linspace(theta_start, theta_end, _actual_output_width, endpoint=False)

    radius_grid = radius[:, np.newaxis]
    theta_grid = theta[np.newaxis, :]

    map_x = cx + radius_grid * np.cos(theta_grid)
    map_y = cy + radius_grid * np.sin(theta_grid)

    remapped_image = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if remapped_image is None or remapped_image.size == 0:
        return np.array([])
    return remapped_image


def detect_tire_and_rim(
    yolo_model: YOLO,
    image: np.ndarray,
    save_path: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Detect wheel and rim contours using YOLO segmentation."""
    results = yolo_model.predict(image, verbose=False)

    if not results or not hasattr(results[0], "masks") or results[0].masks is None:
        return None

    masks_xy = results[0].masks.xy
    if len(masks_xy) < 2:
        return None

    sorted_masks = sorted(
        masks_xy,
        key=lambda x: cv2.contourArea(np.array(x, dtype=np.int32)),
        reverse=True,
    )

    wheel_contour = np.array(sorted_masks[0], dtype=np.int32)
    rim_contour = np.array(sorted_masks[1], dtype=np.int32)

    if save_path is not None:
        h, w = image.shape[:2]
        wheel_mask = np.zeros((h, w), dtype=np.uint8)
        rim_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(wheel_mask, [wheel_contour], 255)
        cv2.fillPoly(rim_mask, [rim_contour], 255)
        combined_mask = cv2.bitwise_or(wheel_mask, rim_mask)

        overlay = image.copy()
        overlay[wheel_mask == 255] = (0, 255, 0)
        overlay[rim_mask == 255] = (0, 0, 255)
        seg_overlay = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        dark_background = (image.astype(np.float32) * 0.3).astype(np.uint8)
        vis_image = dark_background.copy()
        vis_image[combined_mask == 255] = seg_overlay[combined_mask == 255]
        cv2.imwrite(save_path, vis_image)

    return wheel_contour, rim_contour


def get_flattened_sidewall_image(
    image: np.ndarray,
    wheel_contour: np.ndarray,
    rim_contour: np.ndarray,
    config: TireExtractionConfig,
) -> Optional[np.ndarray]:
    """Flatten the tire sidewall from a circular image to a rectangular strip."""
    wheel_bbox = contour_to_bbox(wheel_contour)
    rim_bbox = contour_to_bbox(rim_contour)

    params = estimate_wheel_parameters(wheel_bbox, rim_bbox)
    if not params:
        return None

    cx, cy, r_inner, r_outer = params
    if r_outer <= r_inner:
        return None

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [wheel_contour], 255)
    cv2.fillPoly(mask, [rim_contour], 0)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    flattened = flatten_sidewall(
        masked_image, cx, cy, r_inner, r_outer,
        output_height=config.FLATTEN_OUTPUT_HEIGHT,
        max_output_width=config.FLATTEN_OUTPUT_WIDTH,
        angle_offset_degrees=config.FLATTEN_ANGLE_OFFSET_DEGREES,
        angle_crop_percent=config.FLATTEN_ANGLE_CROP_PERCENT,
    )
    if flattened is None or flattened.size == 0:
        return None

    flattened = cv2.flip(flattened, 0)

    if len(flattened.shape) == 3:
        flat_mask = np.any(flattened != 0, axis=2).astype(np.uint8) * 255
    else:
        flat_mask = (flattened != 0).astype(np.uint8) * 255

    return cv2.bitwise_and(flattened, flattened, mask=flat_mask)
