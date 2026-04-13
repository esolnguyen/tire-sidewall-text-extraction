import cv2
import os
import sys
import logging
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import TireExtractionConfig
from utils.tire_cropping import detect_tire_and_rim, get_flattened_sidewall_image
from utils.image_preprocessing import clahe_enhancement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

config = TireExtractionConfig()
yolo_model = YOLO(config.YOLO_MODEL_PATH)

TARGET_WIDTH = config.FLATTEN_OUTPUT_WIDTH
TARGET_HEIGHT = config.FLATTEN_OUTPUT_HEIGHT


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
    logger.info("Flattening Sidewall...")
    flattened_image = get_flattened_sidewall_image(
        original_image, wheel_contour, rim_contour, config
    )
    if flattened_image is None:
        logger.info("Pipeline stopped: Could not generate flattened sidewall image.")
        return

    gray = cv2.cvtColor(flattened_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    sidewall = flattened_image[y : y + h, x : x + w]

    resized_sidewall = cv2.resize(
        sidewall, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR
    )

    gray_sidewall = cv2.cvtColor(resized_sidewall, cv2.COLOR_BGR2GRAY)
    enhanced_sidewall = clahe_enhancement(
        gray_sidewall,
        clip_limit=config.CLAHE_CLIP_LIMIT,
        tile_grid_size=config.CLAHE_TILE_GRID_SIZE,
    )

    new_file_name = file_name.split(".")[0]
    cv2.imwrite(f"fatten_{new_file_name}.jpg", enhanced_sidewall)
    logger.info("Sidewall flattened, resized, and enhanced with CLAHE.")


if __name__ == "__main__":
    run_pipeline("data/tire_test_final/IMG_2794.jpg", "IMG_2794.jpg")
