import cv2
import os
import numpy as np
from ultralytics import YOLO
from common import logger

from config import *
from utils.tire_cropping import detect_tire_and_rim, get_flattened_sidewall_image

TIRE_DATA_PATH = "tire"

yolo_model = YOLO("models/yolo.pt")

config = TireExtractionConfig()

# Desired output size
TARGET_WIDTH = 6400
TARGET_HEIGHT = 640


def run_pipeline(image_path: str, file_name: str):
    if not os.path.exists(image_path):
        logger.info(f"Error: Image not found at {image_path}")
        return

    logger.info(f"Processing image: {image_path}")
    original_image = cv2.imread(image_path)
    if original_image is None:
        logger.info(f"Error: Could not read image {image_path}")
        return

    output = detect_tire_and_rim(yolo_model, original_image)
    if not output:
        logger.info(f"Error: cannot detect tire or rim image {image_path}")
        return

    wheel_contour, rim_contour = output
    logger.info("\n--- Flattening Sidewall ---")
    flattened_image = get_flattened_sidewall_image(
        original_image,
        wheel_contour,
        rim_contour,
        config
    )
    if flattened_image is None:
        logger.info(
            "Pipeline stopped: Could not generate flattened sidewall image.")
        return

    # Apply the resizing logic to flattened_image
    # Get grayscale and threshold to find non-black regions
    gray = cv2.cvtColor(flattened_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Get bounding box of the sidewall
    x, y, w, h = cv2.boundingRect(mask)

    # Crop the sidewall region
    sidewall = flattened_image[y:y+h, x:x+w]

    # Resize the sidewall directly to the target size
    resized_sidewall = cv2.resize(
        sidewall, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Create CLAHE object with specified parameters
    clahe = cv2.createCLAHE(
        # Controls contrast enhancement (1.0-4.0 typical range)
        clipLimit=4.0,
        tileGridSize=(8, 8)  # Size of grid for histogram equalization
    )

    # Convert to grayscale and apply CLAHE
    gray_sidewall = cv2.cvtColor(resized_sidewall, cv2.COLOR_BGR2GRAY)
    enhanced_sidewall = clahe.apply(gray_sidewall)

    new_file_name = file_name.split('.')[0]
    cv2.imwrite(f"fatten_{new_file_name}.jpg", enhanced_sidewall)
    logger.info(
        "Sidewall flattened, resized to 6400x640, and enhanced with CLAHE.")


if __name__ == '__main__':
    run_pipeline("data/tire_test_final/IMG_2794.jpg", "IMG_2794.jpg")
