from src.utils.image_preprocessing import preprocess_image
from src.utils.tire_cropping import detect_tire_and_rim, get_flattened_sidewall_image
from models_types.tire_info import TireInfo
from services.gemini_service import extract_tire_information
from models.text_recognition import TextRecognitionModel
from models.text_detection import TextDetectionModel
from config import TireExtractionConfig
from ultralytics import YOLO
import cv2
import os
import sys
import numpy as np
import requests
import logging
from io import BytesIO
from PIL import Image
from typing import Optional

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TireImageProcessingPipeline:
    """Complete pipeline for tire information extraction from images"""

    def __init__(self, config: Optional[TireExtractionConfig] = None):
        """Initialize the pipeline with models and configuration

        Args:
            config: Configuration object with model paths and settings
        """
        self.config = config or TireExtractionConfig()

        logger.info("Loading models...")
        self.yolo_model = YOLO(self.config.YOLO_MODEL_PATH)
        self.text_detection_model = TextDetectionModel(
            self.config.TEXT_DETECTION_MODEL_PATH,
            device=self.config.DEVICE,
            imgsz=(self.config.TARGET_HEIGHT, self.config.TARGET_WIDTH)
        )
        self.text_recognition_model = TextRecognitionModel(
            self.config.TEXT_RECOGNITION_MODEL_PATH,
            device=self.config.DEVICE,
            img_h=self.config.TEXT_RECOGNITION_IMG_HEIGHT,
            img_w=self.config.TEXT_RECOGNITION_IMG_WIDTH,
            charset=self.config.TEXT_RECOGNITION_CHARSET,
            batch_max_length=self.config.TEXT_RECOGNITION_BATCH_MAX_LENGTH
        )
        logger.info("Models loaded successfully")

    def load_image_from_url(self, image_url: str) -> np.ndarray:
        """Download and load image from URL

        Args:
            image_url: URL of the image

        Returns:
            Image as numpy array in BGR format
        """
        logger.info(f"Downloading image from URL: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        # Convert to PIL Image then to OpenCV format
        pil_image = Image.open(BytesIO(response.content))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return image

    def load_image(self, image_source: str) -> np.ndarray:
        """Load image from file path or URL

        Args:
            image_source: File path or URL of the image

        Returns:
            Image as numpy array in BGR format
        """
        if image_source.startswith('http://') or image_source.startswith('https://'):
            return self.load_image_from_url(image_source)

        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image not found at {image_source}")

        file_ext = os.path.splitext(image_source)[1].lower()

        if file_ext in ['.heic', '.heif']:
            logger.info(f"Detected HEIC/HEIF format, converting to JPEG...")
            if not HEIF_AVAILABLE:
                raise ValueError(
                    "HEIC/HEIF support not available. Please install pillow-heif: pip install pillow-heif"
                )
            try:
                pil_image = Image.open(image_source)

                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')

                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                logger.info(f"Successfully converted HEIC to BGR format")
                return image
            except Exception as e:
                raise ValueError(f"Failed to convert HEIC image: {e}")

        if file_ext == '.jpeg':
            logger.info(f"Detected JPEG format (will process as JPG)")

        image = cv2.imread(image_source)
        if image is None:
            raise ValueError(f"Could not read image from {image_source}")

        return image

    def preprocess_tire_image(self, flattened_image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to flattened tire image

        Args:
            flattened_image: Flattened sidewall image

        Returns:
            Preprocessed image ready for text detection
        """
        # Crop to non-zero region
        gray = cv2.cvtColor(flattened_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        sidewall = flattened_image[y:y+h, x:x+w]

        # Apply selected preprocessing method from config
        preprocessed = preprocess_image(
            sidewall,
            method=self.config.PREPROCESSING_METHOD,
            clip_limit=self.config.CLAHE_CLIP_LIMIT,
            tile_grid_size=self.config.CLAHE_TILE_GRID_SIZE,
            min_percentile=self.config.LINEAR_MIN_PERCENTILE,
            max_percentile=self.config.LINEAR_MAX_PERCENTILE
        )

        return preprocessed

    def run_pipeline(self, image_source: str, save_debug: bool = False) -> TireInfo:
        """Run the complete tire extraction pipeline

        Args:
            image_source: Image file path or URL
            save_debug: Whether to save intermediate images for debugging

        Returns:
            TireInfo object with extracted information
        """
        logger.info(f"Starting pipeline for: {image_source}")

        # Step 1: Load image
        original_image = self.load_image(image_source)
        logger.info(f"Image loaded: {original_image.shape}")

        # Step 2: Detect tire and rim
        logger.info("Detecting tire and rim...")
        output_path = os.path.join("debug1", "00_tire_rim_detection.jpg")
        detection_result = detect_tire_and_rim(
            self.yolo_model, original_image, output_path)
        if detection_result is None:
            raise RuntimeError("Tire or rim detection failed")

        wheel_contour, rim_contour = detection_result

        # Step 3: Flatten sidewall
        logger.info("Flattening sidewall...")

        # Create a config object with the flattening parameters
        from config import TireExtractionConfig as RootConfig
        root_config = RootConfig()
        root_config.flatten_output_height = self.config.FLATTEN_OUTPUT_HEIGHT
        root_config.flatten_output_width = self.config.FLATTEN_OUTPUT_WIDTH
        root_config.flatten_angle_offset_degrees = self.config.FLATTEN_ANGLE_OFFSET_DEGREES
        root_config.flatten_angle_crop_percent = self.config.FLATTEN_ANGLE_CROP_PERCENT
        root_config.rim_radius_scale_factor = self.config.RIM_RADIUS_SCALE_FACTOR

        flattened_image = get_flattened_sidewall_image(
            original_image,
            wheel_contour,
            rim_contour,
            root_config
        )

        if flattened_image is None:
            raise RuntimeError("Failed to flatten sidewall")

        # Step 4: Preprocess flattened image
        logger.info("Preprocessing flattened image...")
        preprocessed_image = self.preprocess_tire_image(flattened_image)

        # Create debug directory if needed
        debug_dir = None
        if save_debug:
            debug_dir = "debug1"
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(
                debug_dir, "01_flattened.jpg"), flattened_image)
            cv2.imwrite(os.path.join(
                debug_dir, "02_preprocessed.jpg"), preprocessed_image)
            logger.info(f"Debug images saved to {debug_dir}/")

        # Step 5: Detect text regions
        logger.info("Detecting text regions...")
        detections = self.text_detection_model.detect_text(
            preprocessed_image, self.config.CONF_THRESHOLD)
        logger.info(f"Found {len(detections)} text regions")

        if len(detections) == 0:
            logger.warning("No text detected on tire sidewall")

        # Step 6: Extract text regions
        text_crops = self.text_detection_model.crop_text_regions(
            preprocessed_image,
            detections,
            debug_dir=debug_dir  # Pass debug directory to save crops
        )

        # Save detection visualization if debugging
        if save_debug and len(detections) > 0:
            # Save bounding boxes on preprocessed image
            self.text_detection_model.visualize_detections(
                preprocessed_image,
                detections,
                save_path=os.path.join(
                    debug_dir, "detections_preprocessed.jpg")
            )

            # Save bounding boxes on flattened image (original tire unwrapped)
            self.text_detection_model.visualize_detections(
                flattened_image,
                detections,
                save_path=os.path.join(
                    debug_dir, "detections_flattened.jpg")
            )
            logger.info(
                f"Detection visualizations saved (on both preprocessed and flattened images)")

        # Step 7: Recognize text
        logger.info("Recognizing text...")
        recognized_texts = self.text_recognition_model.recognize_batch(
            text_crops)
        logger.info(f"Recognized {len(recognized_texts)} text strings")

        # Combine recognized texts with their bounding box information
        texts_with_bboxes = []
        for i, (text, detection) in enumerate(zip(recognized_texts, detections)):
            # Extract bbox coordinates from detection
            # Detection format varies, but typically has 'bbox' or coordinates
            if hasattr(detection, 'bbox'):
                bbox = detection.bbox
            elif isinstance(detection, (list, tuple)) and len(detection) >= 4:
                bbox = detection[:4]
            elif isinstance(detection, dict) and 'bbox' in detection:
                bbox = detection['bbox']
            else:
                bbox = [0, 0, 0, 0]  # Fallback

            # Format: "Text: {text} | BBox: (x1={x1}, y1={y1}, x2={x2}, y2={y2})"
            bbox_info = f"Text: {text} | BBox: (x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f})"
            texts_with_bboxes.append(bbox_info)
            logger.info(f"  Text {i+1}: {bbox_info}")

        # Step 8: Extract structured information using LLM
        logger.info("Extracting tire information using Gemini...")

        if not self.config.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not configured. Please set it in environment variables.")

        # Convert flattened image to bytes for LLM
        flattened_image_bytes = None
        try:
            success, buffer = cv2.imencode('.jpg', flattened_image)
            if success:
                flattened_image_bytes = buffer.tobytes()
                logger.info(
                    f"Encoded flattened image ({len(flattened_image_bytes)} bytes) for LLM")
        except Exception as e:
            logger.warning(f"Failed to encode flattened image for LLM: {e}")

        # Pass texts with bbox information to LLM
        tire_info_dict = extract_tire_information(
            model=self.config.GEMINI_MODEL,
            ocr_texts=texts_with_bboxes,  # Now includes bbox information
            api_key=self.config.GEMINI_API_KEY,
            flattened_image=flattened_image_bytes
        )

        # Convert to TireInfo object
        tire_info = TireInfo.from_dict(tire_info_dict)
        logger.info("Pipeline completed successfully")

        return tire_info


def main(image_source: str, save_debug: bool = False) -> TireInfo:
    """Main entry point for the pipeline

    Args:
        image_source: Image file path or URL
        save_debug: Whether to save debug images

    Returns:
        TireInfo object with extracted information
    """
    config = TireExtractionConfig()
    pipeline = TireImageProcessingPipeline(config)
    tire_info = pipeline.run_pipeline(image_source, save_debug=save_debug)
    return tire_info


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path_or_url> [--debug]")
        print("Example: python pipeline.py data/input/tire.jpg")
        print("Example: python pipeline.py https://example.com/tire.jpg --debug")
        sys.exit(1)

    image_source = sys.argv[1]
    save_debug = "--debug" in sys.argv

    try:
        tire_info = main(image_source, save_debug=save_debug)
        print("\n" + "="*50)
        print(tire_info)
        print("="*50)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
