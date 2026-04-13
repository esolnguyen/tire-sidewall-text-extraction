import cv2
import numpy as np
import os
import logging
from ultralytics import YOLO
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class TextDetectionModel:
    def __init__(self, model_path: str, device: str = "cpu", imgsz: Tuple[int, int] = (640, 640)):
        """Initialize text detection model.

        Args:
            model_path: Path to the YOLO text detection model (.pt file)
            device: Device to run inference on ("cpu" or "cuda")
            imgsz: Image size for inference (height, width). Default is (640, 640)
        """
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.model.to(device)

    def detect_text(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[dict]:
        """Detect text regions in the image.

        Args:
            image: Input image as numpy array (BGR format)
            conf_threshold: Confidence threshold for detections

        Returns:
            List of detection dictionaries with bbox, confidence, and class
        """
        results = self.model.predict(
            image,
            conf=conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
            rect=True
        )
        detections = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": conf,
                        "class": cls
                    })

        return detections

    def crop_text_regions(self, image: np.ndarray, detections: List[dict],
                          debug_dir: Optional[str] = None) -> List[np.ndarray]:
        """Crop detected text regions from the image.

        Args:
            image: Input image as numpy array
            detections: List of detection dictionaries from detect_text()
            debug_dir: Optional directory to save cropped regions for debugging

        Returns:
            List of cropped image regions
        """
        cropped_regions = []

        # Create debug directory if specified
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]

            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 > x1 and y2 > y1:
                cropped = image[y1:y2, x1:x2]
                cropped_regions.append(cropped)

                # Save cropped region to debug folder if specified
                if debug_dir:
                    crop_filename = os.path.join(
                        debug_dir,
                        f"text_crop_{idx:03d}_conf{detection['confidence']:.2f}.jpg"
                    )
                    cv2.imwrite(crop_filename, cropped)

        return cropped_regions

    def visualize_detections(self, image: np.ndarray, detections: List[dict],
                             save_path: Optional[str] = None) -> np.ndarray:
        """Visualize detection bounding boxes on the image.

        Args:
            image: Input image as numpy array
            detections: List of detection dictionaries from detect_text()
            save_path: Optional path to save the visualization

        Returns:
            Image with bounding boxes drawn
        """
        vis_image = image.copy()

        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]

            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence score
            label = f"{idx}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Draw background for text
            cv2.rectangle(vis_image,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1),
                          (0, 255, 0), -1)

            # Draw text
            cv2.putText(vis_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, vis_image)

        return vis_image
