"""Core tire image processing pipeline."""

import os
import time
import logging
from typing import Optional, Dict, List, Iterator
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO

from config import TireExtractionConfig
from schemas.tire_info import TireInfo
from services.gemini import extract_tire_information, extract_tire_information_raw
from models.text_recognition import TextRecognitionModel
from models.text_detection import TextDetectionModel
from utils.image_preprocessing import preprocess_image
from utils.tire_cropping import detect_tire_and_rim, get_flattened_sidewall_image
from pipeline.types import TextDetectionResult, PipelineResult, PipelineStepEvent
from exceptions import (
    ImageLoadError,
    TireDetectionError,
    SidewallFlatteningError,
    LLMExtractionError,
)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

logger = logging.getLogger(__name__)


class TireImageProcessingPipeline:
    """Complete pipeline for tire information extraction from images."""

    def __init__(self, config: Optional[TireExtractionConfig] = None):
        self.config = config or TireExtractionConfig()

        logger.info("Loading models...")
        self.yolo_model = YOLO(self.config.YOLO_MODEL_PATH)
        self.text_detection_model = TextDetectionModel(
            self.config.TEXT_DETECTION_MODEL_PATH,
            device=self.config.DEVICE,
            imgsz=(self.config.TARGET_HEIGHT, self.config.TARGET_WIDTH),
        )
        self.text_recognition_model = TextRecognitionModel(
            self.config.TEXT_RECOGNITION_MODEL_PATH,
            device=self.config.DEVICE,
            img_h=self.config.TEXT_RECOGNITION_IMG_HEIGHT,
            img_w=self.config.TEXT_RECOGNITION_IMG_WIDTH,
            charset=self.config.TEXT_RECOGNITION_CHARSET,
            batch_max_length=self.config.TEXT_RECOGNITION_BATCH_MAX_LENGTH,
        )
        logger.info("Models loaded successfully")

    def load_image(self, image_source: str) -> np.ndarray:
        """Load image from file path or URL."""
        if image_source.startswith("http://") or image_source.startswith("https://"):
            return self._load_image_from_url(image_source)

        if not os.path.exists(image_source):
            raise ImageLoadError(f"Image not found at {image_source}")

        file_ext = os.path.splitext(image_source)[1].lower()

        if file_ext in [".heic", ".heif"]:
            if not HEIF_AVAILABLE:
                raise ImageLoadError(
                    "HEIC/HEIF support not available. Install pillow-heif."
                )
            pil_image = Image.open(image_source)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        image = cv2.imread(image_source)
        if image is None:
            raise ImageLoadError(f"Could not read image from {image_source}")
        return image

    def _load_image_from_url(self, image_url: str) -> np.ndarray:
        logger.info(f"Downloading image from URL: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def preprocess_tire_image(self, flattened_image: np.ndarray) -> np.ndarray:
        """Crop to non-zero region and apply configured preprocessing."""
        gray = cv2.cvtColor(flattened_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        sidewall = flattened_image[y : y + h, x : x + w]

        return preprocess_image(
            sidewall,
            method=self.config.PREPROCESSING_METHOD,
            clip_limit=self.config.CLAHE_CLIP_LIMIT,
            tile_grid_size=self.config.CLAHE_TILE_GRID_SIZE,
            min_percentile=self.config.LINEAR_MIN_PERCENTILE,
            max_percentile=self.config.LINEAR_MAX_PERCENTILE,
        )

    def run_pipeline(
        self,
        image_source: str,
        output_dir: Optional[str] = None,
        save_debug: bool = False,
    ) -> PipelineResult:
        """Run the complete tire extraction pipeline (blocking)."""
        output_files: Dict[str, str] = {}
        text_detection_results: List[TextDetectionResult] = []
        tire_info: Optional[TireInfo] = None

        for event in self.run_pipeline_streaming(
            image_source, output_dir=output_dir, save_debug=save_debug
        ):
            output_files.update(event.output_files)
            if event.text_detections:
                text_detection_results = event.text_detections
            if event.tire_info is not None:
                tire_info = event.tire_info

        if tire_info is None:
            raise LLMExtractionError("Pipeline finished without tire info")

        return PipelineResult(
            tire_info=tire_info,
            text_detections=text_detection_results,
            output_files=output_files,
        )

    def run_pipeline_streaming(
        self,
        image_source: str,
        output_dir: Optional[str] = None,
        save_debug: bool = False,
    ) -> Iterator[PipelineStepEvent]:
        """Run the pipeline as a generator that yields after each step."""
        logger.info(f"Starting pipeline for: {image_source}")

        if output_dir is None:
            base = os.path.splitext(os.path.basename(image_source))[0]
            output_dir = os.path.join(self.config.OUTPUT_DIR, base)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Load image
        t0 = time.time()
        original_image = self.load_image(image_source)
        logger.info(f"Image loaded: {original_image.shape}")
        yield PipelineStepEvent(name="Load image", duration=time.time() - t0)

        # Step 2: Detect tire and rim
        t0 = time.time()
        logger.info("Detecting tire and rim...")
        seg_save = os.path.join(output_dir, "tire_rim_detection.jpg")
        detection_result = detect_tire_and_rim(
            self.yolo_model, original_image, seg_save
        )
        if detection_result is None:
            raise TireDetectionError("Tire or rim detection failed")
        wheel_contour, rim_contour = detection_result
        yield PipelineStepEvent(
            name="Detect tire & rim",
            duration=time.time() - t0,
            output_files={"tire_rim_detection": seg_save},
        )

        # Step 3: Flatten sidewall
        t0 = time.time()
        logger.info("Flattening sidewall...")
        flattened_image = get_flattened_sidewall_image(
            original_image, wheel_contour, rim_contour, self.config,
        )
        if flattened_image is None:
            raise SidewallFlatteningError("Failed to flatten sidewall")
        flattened_path = os.path.join(output_dir, "flattened.jpg")
        cv2.imwrite(flattened_path, flattened_image)
        yield PipelineStepEvent(
            name="Flatten sidewall",
            duration=time.time() - t0,
            output_files={"flattened": flattened_path},
        )

        # Step 4: Preprocess
        t0 = time.time()
        logger.info("Preprocessing flattened image...")
        preprocessed_image = self.preprocess_tire_image(flattened_image)
        preprocessed_path = os.path.join(output_dir, "preprocessed.jpg")
        cv2.imwrite(preprocessed_path, preprocessed_image)
        yield PipelineStepEvent(
            name="Preprocess",
            duration=time.time() - t0,
            output_files={"preprocessed": preprocessed_path},
        )

        # Step 5: Detect text regions
        t0 = time.time()
        logger.info("Detecting text regions...")
        detections = self.text_detection_model.detect_text(
            preprocessed_image, self.config.CONF_THRESHOLD
        )
        logger.info(f"Found {len(detections)} text regions")

        detection_files: Dict[str, str] = {}
        if len(detections) > 0:
            vis_path = os.path.join(output_dir, "detections_preprocessed.jpg")
            self.text_detection_model.visualize_detections(
                preprocessed_image, detections, save_path=vis_path
            )
            detection_files["detections_preprocessed"] = vis_path
        yield PipelineStepEvent(
            name=f"Detect text regions ({len(detections)} found)",
            duration=time.time() - t0,
            output_files=detection_files,
        )

        # Step 6: Crop text regions
        t0 = time.time()
        crop_dir = os.path.join(output_dir, "crops") if save_debug else None
        text_crops = self.text_detection_model.crop_text_regions(
            preprocessed_image, detections, debug_dir=crop_dir
        )
        crop_files: Dict[str, str] = {}
        if crop_dir and os.path.isdir(crop_dir):
            crop_files["text_crops_dir"] = crop_dir
        yield PipelineStepEvent(
            name="Crop text regions",
            duration=time.time() - t0,
            output_files=crop_files,
        )

        # Step 7: Recognize text
        t0 = time.time()
        logger.info("Recognizing text...")
        recognized_texts = self.text_recognition_model.recognize_batch(text_crops)
        texts_with_bboxes, text_detection_results = self._build_text_results(
            recognized_texts, detections
        )
        yield PipelineStepEvent(
            name="Recognize text (OCR)",
            duration=time.time() - t0,
            text_detections=text_detection_results,
        )

        # Step 8: Extract structured information using LLM
        t0 = time.time()
        tire_info = self._extract_with_llm(texts_with_bboxes, flattened_image)
        logger.info("Pipeline completed successfully")
        yield PipelineStepEvent(
            name="LLM extraction",
            duration=time.time() - t0,
            tire_info=tire_info,
        )

    def run_llm_only(self, image_source: str) -> TireInfo:
        """Send image directly to Gemini LLM without any preprocessing."""
        if not self.config.GEMINI_API_KEY:
            raise LLMExtractionError("GEMINI_API_KEY not configured.")

        original_image = self.load_image(image_source)
        ok, buf = cv2.imencode(".jpg", original_image)
        if not ok:
            raise ImageLoadError("Failed to encode image")

        result = extract_tire_information_raw(
            model=self.config.GEMINI_MODEL,
            api_key=self.config.GEMINI_API_KEY,
            image_bytes=buf.tobytes(),
        )
        if result is None:
            raise LLMExtractionError("LLM returned no result")
        return TireInfo.from_dict(result)

    # ── Private helpers ───────────────────────────────────────────────

    def _build_text_results(self, recognized_texts, detections):
        """Pair recognized texts with their bounding boxes."""
        texts_with_bboxes: List[str] = []
        text_detection_results: List[TextDetectionResult] = []

        for i, (text, detection) in enumerate(zip(recognized_texts, detections)):
            if hasattr(detection, "bbox"):
                bbox = detection.bbox
            elif isinstance(detection, (list, tuple)) and len(detection) >= 4:
                bbox = detection[:4]
            elif isinstance(detection, dict) and "bbox" in detection:
                bbox = detection["bbox"]
            else:
                bbox = [0, 0, 0, 0]

            conf = (
                detection.get("confidence", 0.0) if isinstance(detection, dict) else 0.0
            )

            text_detection_results.append(
                TextDetectionResult(
                    text=text,
                    bbox=[int(b) for b in bbox],
                    confidence=round(conf, 4),
                )
            )

            bbox_info = (
                f"Text: {text} | BBox: (x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, "
                f"x2={bbox[2]:.1f}, y2={bbox[3]:.1f})"
            )
            texts_with_bboxes.append(bbox_info)
            logger.info(f"  Text {i + 1}: {bbox_info}")

        return texts_with_bboxes, text_detection_results

    def _extract_with_llm(self, texts_with_bboxes, flattened_image) -> TireInfo:
        """Call Gemini LLM to extract tire info from OCR results."""
        if not self.config.GEMINI_API_KEY:
            raise LLMExtractionError("GEMINI_API_KEY not configured.")

        flattened_image_bytes = None
        try:
            success, buffer = cv2.imencode(".jpg", flattened_image)
            if success:
                flattened_image_bytes = buffer.tobytes()
        except Exception as e:
            logger.warning(f"Failed to encode flattened image for LLM: {e}")

        tire_info_dict = extract_tire_information(
            model=self.config.GEMINI_MODEL,
            ocr_texts=texts_with_bboxes,
            api_key=self.config.GEMINI_API_KEY,
            flattened_image=flattened_image_bytes,
        )
        return TireInfo.from_dict(tire_info_dict)
