import os
import string
from dotenv import load_dotenv

load_dotenv()


class TireExtractionConfig:
    # Model paths
    YOLO_MODEL_PATH = "models/tire_det.pt"
    TEXT_DETECTION_MODEL_PATH = "models/yolo_textdetv3.pt"
    TEXT_RECOGNITION_MODEL_PATH = "models/text_rec.pth"

    # Text recognition model configuration
    TEXT_RECOGNITION_IMG_HEIGHT = 32
    # Must match training parameter (affects TPS grid size)
    TEXT_RECOGNITION_IMG_WIDTH = 100
    # Character set for text recognition (case-sensitive alphanumeric + symbols)
    # Using string.printable[:-6] to match the training configuration with --sensitive flag
    # This includes: 0-9, a-z, A-Z, and common symbols (94 characters total)
    TEXT_RECOGNITION_CHARSET = string.printable[:-6]
    TEXT_RECOGNITION_BATCH_MAX_LENGTH = 40  # Must match training parameter

    # Target dimensions for image processing (flattened sidewall)
    TARGET_WIDTH = 3200
    TARGET_HEIGHT = 320
    CONF_THRESHOLD = 0.2

    # Flattening parameters
    FLATTEN_OUTPUT_HEIGHT = 640
    FLATTEN_OUTPUT_WIDTH = 6400
    FLATTEN_ANGLE_OFFSET_DEGREES = -90
    FLATTEN_ANGLE_CROP_PERCENT = -0.2
    RIM_RADIUS_SCALE_FACTOR = 0.6

    # Image preprocessing parameters
    # Preprocessing method: 'linear', 'histogram_eq', 'clahe', or 'none'
    # For thesis comparison of three methods: Linear Stretching, HE, CLAHE
    PREPROCESSING_METHOD = "clahe"

    # CLAHE parameters (used when PREPROCESSING_METHOD = 'clahe')
    CLAHE_CLIP_LIMIT = 4.0
    CLAHE_TILE_GRID_SIZE = (8, 8)

    # Linear Stretching parameters (used when PREPROCESSING_METHOD = 'linear')
    LINEAR_MIN_PERCENTILE = 2.0
    LINEAR_MAX_PERCENTILE = 98.0

    # Input and output directories
    INPUT_DIR = os.path.join("data", "input")
    OUTPUT_DIR = os.path.join("data", "output")

    # Logging configuration
    LOGGING_LEVEL = "INFO"

    # Google Gemini API configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash"

    # GEMINI_MODEL = "gemini-3-pro-preview"

    # Other constants
    MAX_TEXT_LENGTH = 100  # Maximum length of text to process
    # Supported image formats for input
    SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png')

    # Device configuration
    DEVICE = "cuda" if os.getenv("USE_CUDA", "").lower() == "true" else "cpu"
