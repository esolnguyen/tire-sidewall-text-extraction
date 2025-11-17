import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import TireExtractionConfig
from utils.cv_utils import contour_to_bbox


def estimate_wheel_parameters(wheel_bbox, rim_bbox):
    if not wheel_bbox or not rim_bbox:
        print("Warning: Invalid bounding boxes provided.")
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
        print(
            f"Warning: Estimated radii are invalid (r_inner={r_inner}, r_outer={r_outer}). Check bounding box accuracy.")
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
        angle_crop_percent: float = 0.1
) -> np.ndarray:
    if r_inner >= r_outer or r_inner <= 0 or r_outer <= 0:
        print(
            f"Error: Invalid radii provided (r_inner={r_inner}, r_outer={r_outer}).")
        return np.array([])
    if output_height <= 0:
        print(f"Error: output_height must be positive.")
        return np.array([])

    # Determine the width to be used for processing
    _actual_output_width: int

    if output_width is not None:  # User provided a specific output_width
        if output_width <= 0:
            print(
                f"Error: Provided output_width ({output_width}) must be positive.")
            return np.array([])
        _actual_output_width = output_width
        print(f"Using provided output_width: {_actual_output_width}")
    else:  # output_width is None, calculate it based on average radius and angle
        avg_radius = (r_inner + r_outer) / 2
        effective_angle_proportion = 1.0 - angle_crop_percent
        if effective_angle_proportion <= 0:
            print(
                f"Error: angle_crop_percent ({angle_crop_percent}) is too high, results in non-positive angle.")
            return np.array([])

        calculated_width = int(2 * np.pi * avg_radius *
                               effective_angle_proportion)

        if calculated_width <= 0:
            print(
                f"Error: Calculated output_width ({calculated_width}) must be positive. Check radii or angle_crop_percent.")
            return np.array([])
        _actual_output_width = calculated_width
        print(f"Calculated initial output_width: {_actual_output_width}")

    # Apply max_output_width constraint
    if max_output_width is not None:
        if max_output_width <= 0:
            print(
                f"Warning: Invalid max_output_width ({max_output_width}) provided. It must be positive. Ignoring constraint.")
        elif _actual_output_width > max_output_width:
            print(
                f"Capping output_width from {_actual_output_width} to max_output_width {max_output_width}.")
            _actual_output_width = max_output_width

    # Final check for the determined width after potential capping
    if _actual_output_width <= 0:
        print(
            f"Error: Final determined output_width ({_actual_output_width}) is not positive.")
        return np.array([])

    print(f"Final output_width for processing: {_actual_output_width}")

    row_indices = np.arange(output_height, dtype=np.float32)

    radial_range = r_outer - r_inner
    if radial_range <= 0:
        print(
            f"Error: radial_range is not positive (r_outer - r_inner = {radial_range})")
        return np.array([])
    radial_bin_width = radial_range / output_height
    radius = r_inner + (row_indices + 0.5) * radial_bin_width

    angle_offset_rad = np.deg2rad(angle_offset_degrees)

    total_angle_rad = 2 * np.pi * (1.0 - angle_crop_percent)
    if total_angle_rad <= 0:
        print(f"Error: angle_crop_percent results in zero or negative angular sweep.")
        return np.array([])

    theta_start = angle_offset_rad + (2 * np.pi * angle_crop_percent / 2.0)
    theta_end = theta_start + total_angle_rad

    # Use the final, potentially capped, _actual_output_width here
    theta = np.linspace(theta_start, theta_end, _actual_output_width,
                        endpoint=False)

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
        borderValue=(0, 0, 0)
    )
    if remapped_image is None or remapped_image.size == 0:
        print("Error: cv2.remap failed or produced an empty image.")
        return np.array([])
    return remapped_image


def detect_tire_and_rim(yolo_model: YOLO, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    results = yolo_model.predict(image, verbose=False)

    if not results or not results[0].masks:
        print("No masks detected by YOLO.")
        return None

    masks_xy = results[0].masks.xy

    if len(masks_xy) < 2:
        print(
            f"Not enough masks detected. Found {len(masks_xy)}, need at least 2 for wheel and rim.")
        return None

    sorted_masks = sorted(masks_xy, key=lambda x: cv2.contourArea(
        np.array(x, dtype=np.int32)), reverse=True)

    wheel_contour = np.array(sorted_masks[0], dtype=np.int32)
    rim_contour = np.array(sorted_masks[1], dtype=np.int32)

    return wheel_contour, rim_contour


def get_flattened_sidewall_image(image: np.ndarray, wheel_contour: np.ndarray, rim_contour: np.ndarray, config: TireExtractionConfig) -> Optional[
        np.ndarray]:
    wheel_bbox = contour_to_bbox(wheel_contour)
    rim_bbox = contour_to_bbox(rim_contour)

    params = estimate_wheel_parameters(wheel_bbox, rim_bbox)

    if not params:
        print("Could not estimate valid wheel parameters from bounding boxes.")
        return None

    cx, cy, r_inner, r_outer = params
    print(
        f"Estimated params: cx={cx:.2f}, cy={cy:.2f}, r_inner={r_inner:.2f}, r_outer={r_outer:.2f}")

    if r_outer <= r_inner:
        print(
            f"Invalid radii: r_outer ({r_outer}) must be greater than r_inner ({r_inner}).")
        return None

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.fillPoly(mask, [wheel_contour], 255)

    cv2.fillPoly(mask, [rim_contour], 0)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    flattened_sidewall_img = flatten_sidewall(
        masked_image,
        cx,
        cy,
        r_inner,
        r_outer,
        output_height=config.flatten_output_height,
        max_output_width=config.flatten_output_width,
        angle_offset_degrees=config.flatten_angle_offset_degrees,
        angle_crop_percent=config.flatten_angle_crop_percent
    )

    if flattened_sidewall_img is None or flattened_sidewall_img.size == 0:
        print("Failed to flatten sidewall.")
        return None

    flattened_sidewall_img = cv2.flip(
        flattened_sidewall_img, 0)  # Flip vertically

    if len(flattened_sidewall_img.shape) == 3:
        flattened_mask = np.any(flattened_sidewall_img !=
                                0, axis=2).astype(np.uint8) * 255
    else:
        flattened_mask = (flattened_sidewall_img != 0).astype(np.uint8) * 255

    if len(flattened_sidewall_img.shape) == 3:
        flattened_sidewall_img = cv2.bitwise_and(
            flattened_sidewall_img, flattened_sidewall_img, mask=flattened_mask)
    else:
        flattened_sidewall_img = cv2.bitwise_and(
            flattened_sidewall_img, flattened_sidewall_img, mask=flattened_mask)

    return flattened_sidewall_img


def rotate_image(image, angle=0):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M


def crop_image(image, wheel_mask, rim_mask, angle):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [wheel_mask], 255)
    cv2.fillPoly(mask, [rim_mask], 0)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    x1, y1, w1, h1 = cv2.boundingRect(wheel_mask)
    cropped_wheel = masked_image[y1:y1 + h1, x1:x1 + w1]

    rotated_image, _ = rotate_image(cropped_wheel, angle)

    x2, y2, w2, h2 = cv2.boundingRect(rim_mask)
    outer_radius = max(w1, h1) // 2
    inner_radius = min(w2, h2) // 2
    crop_height = outer_radius - inner_radius

    crop_height = min(crop_height, rotated_image.shape[0])
    cropped = rotated_image[:crop_height + 100, :]
    return cropped


def polygon_to_bbox(polygon: List[float]) -> List[int]:
    """
    Converts a flat list of polygon coordinates to a bounding box [xmin, ymin, xmax, ymax].
    Example polygon: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    # This check is good practice, but for this function, we assume a flat list.
    if not polygon:
        return [0, 0, 0, 0]

    # Efficiently slice the list to get all x and y coordinates
    xs = polygon[0::2]
    ys = polygon[1::2]

    # Find the min and max of the coordinates
    xmin = int(min(xs))
    ymin = int(min(ys))
    xmax = int(max(xs))
    ymax = int(max(ys))

    return [xmin, ymin, xmax, ymax]


def crop_text_from_polygons(image_path, polygons, save_dir="cropped_pan"):
    """
    Crops text regions from an image using bounding boxes and saves them if a folder is given.

    Args:
        image_path (str): Path to the image.
        polygons (list): List of polygons. Each polygon is [x1, y1, x2, y2, ...].
        save_dir (str, optional): Folder to save crops. If None, do not save.

    Returns:
        list: List of cropped image regions (as NumPy arrays).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return []

    H, W = image.shape[:2]
    print(f"Loaded image: {W} x {H}")

    cropped_images = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, poly in enumerate(polygons):
        xmin, ymin, xmax, ymax = polygon_to_bbox(poly)

        # Clamp to image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(W, xmax)
        ymax = min(H, ymax)

        if xmax <= xmin or ymax <= ymin:
            print(
                f"Skipping polygon {i}: Invalid bounding box [{xmin}, {ymin}, {xmax}, {ymax}]")
            continue

        crop = image[ymin:ymax, xmin:xmax]
        cropped_images.append(crop)

        if save_dir:
            crop_filename = f"crop_{i:04d}.jpg"
            crop_path = os.path.join(save_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            print(f"Saved: {crop_path}")

    return cropped_images
